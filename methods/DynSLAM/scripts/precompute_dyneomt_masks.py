#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute DynEOMT dynamic masks for a TUM RGB-D association file.")
    parser.add_argument("--repo", required=True, help="DynEOMT repository root")
    parser.add_argument("--config", required=True, help="DynEOMT yaml config")
    parser.add_argument("--checkpoint", required=True, help="DynEOMT checkpoint")
    parser.add_argument("--sequence", required=True, help="TUM RGB-D sequence directory")
    parser.add_argument("--association", required=True, help="TUM association file")
    parser.add_argument("--output-dir", required=True, help="Directory receiving frame_XXXX_dynamic.png masks")
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1, help="Run DynEOMT once every N associated frames")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_associations(path, max_frames):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            records.append({
                "source_frame": len(records) + 1,
                "rgb_timestamp": float(parts[0]),
                "rgb": parts[1],
                "depth_timestamp": float(parts[2]),
                "depth": parts[3],
            })
            if max_frames > 0 and len(records) >= max_frames:
                break
    return records


def resize_keep_aspect(image, short_side, max_side):
    height, width = image.shape[:2]
    scale = float(short_side) / float(min(height, width))
    if max(height, width) * scale > max_side:
        scale = float(max_side) / float(max(height, width))
    if abs(scale - 1.0) < 1e-6:
        return image, (height, width)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)
    return resized, (height, width)


def to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def build_model(repo, config_path, checkpoint_path, device_arg):
    sys.path.insert(0, str(repo))

    from detectron2.config import get_cfg
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model as detectron2_build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from videomt import add_videomt_config

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list([
        "MODEL.WEIGHTS",
        str(checkpoint_path),
        "INPUT.SAMPLING_FRAME_NUM",
        "1",
        "DATALOADER.NUM_WORKERS",
        "0",
        "TEST.AUG.ENABLED",
        "False",
    ])
    cfg.freeze()

    device = torch.device(device_arg if device_arg.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = detectron2_build_model(cfg).eval().to(device)
    DetectionCheckpointer(model).load(str(checkpoint_path))
    model.window_inference = True
    model.window_size = 1
    return model, device


def output_to_dynamic_mask(output, original_shape, threshold):
    if isinstance(output, (list, tuple)):
        output = output[0]

    if "pred_masks" not in output or "segments_infos" not in output:
        return np.zeros(original_shape, dtype=np.uint8)

    pred_masks = output["pred_masks"]
    if isinstance(pred_masks, (list, tuple)):
        panoptic = to_numpy(pred_masks[0])
    else:
        panoptic = to_numpy(pred_masks)
    if panoptic is None:
        return np.zeros(original_shape, dtype=np.uint8)
    if panoptic.ndim == 3:
        panoptic = panoptic[0]

    dynamic_scores = to_numpy(output.get("dynamic_scores"))
    segments = output.get("segments_infos", [])
    dynamic = np.zeros(panoptic.shape[:2], dtype=np.uint8)

    for idx, segment in enumerate(segments):
        score = 0.0
        if dynamic_scores is not None and idx < dynamic_scores.shape[0]:
            score = float(np.ravel(dynamic_scores[idx])[0])
        if score >= threshold:
            dynamic[panoptic == segment["id"]] = 255

    if dynamic.shape[:2] != original_shape:
        dynamic = cv2.resize(dynamic, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    return dynamic


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    sequence = Path(args.sequence).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_associations(args.association, args.max_frames)
    if not records:
        raise RuntimeError(f"No associated RGB-D frames found in {args.association}")

    frame_stride = max(1, int(args.frame_stride))
    inferred_frames = sum(1 for record in records if (int(record["source_frame"]) - 1) % frame_stride == 0)

    print(f"DynEOMT precompute: {inferred_frames} inferred frames")
    print(f"  cache frames: {len(records)}")
    print(f"  source frames: {len(records)}")
    print(f"  frame stride: {frame_stride}")
    print(f"  repo: {repo}")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  output: {output_dir}")

    model, device = build_model(repo, Path(args.config), Path(args.checkpoint), args.device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    if device_type == "cuda":
        torch.cuda.synchronize(device)

    manifest = []
    pixel_index = {}
    latencies = []
    last_dynamic = None
    last_inferred_frame = 0
    actual_inferred_frames = 0
    wall_start = time.perf_counter()

    for frame_idx, record in enumerate(records, start=1):
        rgb_path = sequence / record["rgb"]
        image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read RGB image: {rgb_path}")

        source_frame = int(record["source_frame"])
        pixel_hash = hashlib.sha1(image.tobytes()).hexdigest()
        should_infer = ((source_frame - 1) % frame_stride == 0) or last_dynamic is None
        latency = 0.0

        if should_infer:
            resized, original_shape = resize_keep_aspect(image, args.short_side, args.max_side)
            resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_record = {
                "file_name": str(rgb_path),
                "image": [torch.as_tensor(np.ascontiguousarray(resized_rgb.transpose(2, 0, 1))).float()],
                "height": original_shape[0],
                "width": original_shape[1],
                "keep": source_frame > 1,
            }

            if device_type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                if device_type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model([input_record])
                else:
                    output = model([input_record])
            if device_type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - start
            latencies.append(latency)

            dynamic = output_to_dynamic_mask(output, original_shape, args.dynamic_threshold)
            last_dynamic = dynamic.copy()
            last_inferred_frame = source_frame
            actual_inferred_frames += 1
        else:
            dynamic = last_dynamic.copy()

        mask_name = f"frame_{int(record['source_frame'])}_dynamic.png"
        cv2.imwrite(str(output_dir / mask_name), dynamic)
        pixel_index[pixel_hash] = mask_name

        manifest.append({
            "frame": int(record["source_frame"]),
            "rgb": record["rgb"],
            "rgb_timestamp": record["rgb_timestamp"],
            "mask": mask_name,
            "inferred": should_infer,
            "source_mask_frame": last_inferred_frame,
            "latency_seconds": latency,
            "dynamic_pixels": int(np.count_nonzero(dynamic)),
        })
        if frame_idx == 1 or frame_idx % 25 == 0 or frame_idx == len(records):
            fps = frame_idx / (time.perf_counter() - wall_start)
            inferred = "infer" if should_infer else f"reuse:{last_inferred_frame}"
            print(f"  {frame_idx:5d}/{len(records)} masks, {inferred}, latency {latency:.4f}s, throughput {fps:.2f} fps")

    wall_seconds = time.perf_counter() - wall_start
    perf = {
        "frames": len(records),
        "source_frames": len(records),
        "cache_frames": len(records),
        "inferred_frames": actual_inferred_frames,
        "frame_stride": frame_stride,
        "wall_seconds": wall_seconds,
        "throughput_fps": len(records) / wall_seconds if wall_seconds > 0 else 0.0,
        "inference_throughput_fps": actual_inferred_frames / wall_seconds if wall_seconds > 0 else 0.0,
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "median_latency_seconds": float(np.median(latencies)) if latencies else 0.0,
        "p95_latency_seconds": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "dynamic_threshold": args.dynamic_threshold,
        "short_side": args.short_side,
        "max_side": args.max_side,
        "device": device_type,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with open(output_dir / "index.json", "w", encoding="utf-8") as handle:
        json.dump({"pixel_sha1_to_mask": pixel_index}, handle, indent=2)
    with open(output_dir / "performance.json", "w", encoding="utf-8") as handle:
        json.dump(perf, handle, indent=2)

    print(json.dumps(perf, indent=2))


if __name__ == "__main__":
    main()
