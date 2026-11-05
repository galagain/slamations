#!/usr/bin/env python3
"""Run an online DynEoMT sequence and benchmark it on an NVIDIA Jetson."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config

from videomt import add_videomt_config


DYNAMIC_BGR = np.asarray((55, 65, 235), dtype=np.float32)
STATIC_BGR = np.asarray((75, 190, 80), dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--short-side", type=int, default=720)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fast-exit",
        action="store_true",
        help="Skip native-library teardown after all outputs have been flushed.",
    )
    return parser.parse_args()


def build_cfg(config: Path, checkpoint: Path):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(str(config))
    cfg.merge_from_list(
        [
            "MODEL.WEIGHTS",
            str(checkpoint),
            "INPUT.SAMPLING_FRAME_NUM",
            "1",
            "DATALOADER.NUM_WORKERS",
            "0",
            "TEST.AUG.ENABLED",
            "False",
        ]
    )
    cfg.freeze()
    return cfg


def image_paths(input_dir: Path, max_frames: int) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in suffixes)
    if max_frames > 0:
        paths = paths[:max_frames]
    if not paths:
        raise FileNotFoundError(f"No images found in {input_dir}")
    return paths


def resize_for_model(frame_bgr: np.ndarray, short_side: int, max_side: int) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    scale = short_side / min(height, width)
    if max(height, width) * scale > max_side:
        scale = max_side / max(height, width)
    target = (int(round(width * scale)), int(round(height * scale)))
    if target == (width, height):
        return frame_bgr
    return cv2.resize(frame_bgr, target, interpolation=cv2.INTER_LINEAR)


def input_record(frame_bgr: np.ndarray, keep: bool, short_side: int, max_side: int) -> dict:
    original_height, original_width = frame_bgr.shape[:2]
    resized = resize_for_model(frame_bgr, short_side, max_side)
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.as_tensor(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))).float()
    return {
        "image": [tensor],
        "height": original_height,
        "width": original_width,
        "keep": keep,
    }


def render_prediction(
    frame_bgr: np.ndarray,
    output: dict,
    dynamic_threshold: float,
) -> tuple[np.ndarray, list[dict]]:
    rendered = frame_bgr.astype(np.float32).copy()
    panoptic = output["pred_masks"][0].numpy()
    dynamic_scores = output.get("dynamic_scores")
    records = []
    for index, segment in enumerate(output.get("segments_infos", [])):
        probability = None
        if dynamic_scores is not None and index < len(dynamic_scores):
            probability = float(dynamic_scores[index, 0])
        is_dynamic = probability is not None and probability >= dynamic_threshold
        color = DYNAMIC_BGR if is_dynamic else STATIC_BGR
        mask = panoptic == int(segment["id"])
        rendered[mask] = 0.45 * rendered[mask] + 0.55 * color
        records.append(
            {
                "segment_id": int(segment["id"]),
                "category_id": int(segment["category_id"]),
                "isthing": bool(segment["isthing"]),
                "dynamic_probability": probability,
                "dynamic": is_dynamic,
                "area": int(mask.sum()),
            }
        )

    dynamic_count = sum(record["dynamic"] for record in records)
    label = f"DynEoMT  dynamic={dynamic_count}  static={len(records) - dynamic_count}"
    cv2.rectangle(rendered, (0, 0), (min(frame_bgr.shape[1], 620), 38), (18, 18, 18), -1)
    cv2.putText(
        rendered,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.clip(rendered, 0, 255).astype(np.uint8), records


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.config.resolve(), args.checkpoint.resolve())
    device = torch.device(args.device)
    model = build_model(cfg).eval().to(device)
    DetectionCheckpointer(model).load(str(args.checkpoint.resolve()))
    model.window_inference = True
    model.window_size = 1

    paths = image_paths(args.input_dir, args.max_frames)
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read {paths[0]}")
    height, width = first.shape[:2]
    video_path = args.output_dir / "dyneomt_jetson.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {video_path}")

    torch.cuda.reset_peak_memory_stats(device)
    timings_ms: list[float] = []
    frame_results = []
    try:
        with torch.inference_mode():
            for frame_index, path in enumerate(paths):
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Cannot read {path}")
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                batch = [
                    input_record(
                        frame,
                        keep=frame_index > 0,
                        short_side=args.short_side,
                        max_side=args.max_side,
                    )
                ]
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = model(batch)
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                if frame_index >= args.warmup_frames:
                    timings_ms.append(elapsed_ms)
                rendered, segments = render_prediction(
                    frame, output, args.dynamic_threshold
                )
                writer.write(rendered)
                frame_results.append(
                    {
                        "frame": path.name,
                        "latency_ms": elapsed_ms,
                        "timed": frame_index >= args.warmup_frames,
                        "segments": segments,
                    }
                )
                print(
                    f"[{frame_index + 1:03d}/{len(paths):03d}] {path.name} "
                    f"{elapsed_ms:.1f} ms segments={len(segments)}",
                    flush=True,
                )
    finally:
        writer.release()

    if not timings_ms:
        raise RuntimeError("No timed frames; reduce --warmup-frames")
    mean_ms = statistics.mean(timings_ms)
    report = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "input_dir": str(args.input_dir.resolve()),
        "input_resolution": [height, width],
        "model_short_side": args.short_side,
        "model_max_side": args.max_side,
        "frames": len(paths),
        "warmup_frames": args.warmup_frames,
        "timed_frames": len(timings_ms),
        "latency_ms_mean": mean_ms,
        "latency_ms_median": statistics.median(timings_ms),
        "latency_ms_p95": percentile(timings_ms, 95),
        "fps": 1000.0 / mean_ms,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
        "video": str(video_path.resolve()),
        "frame_results": frame_results,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "frame_results"}, indent=2))
    if args.fast_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
