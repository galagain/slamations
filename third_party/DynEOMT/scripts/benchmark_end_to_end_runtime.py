#!/usr/bin/env python3
"""Benchmark model-side end-to-end inference for VidEoMT and DynEoMT.

The timed region is `model(batched_inputs)` on preloaded dataloader batches.
It includes model forward, query propagation, prediction heads, and model
post-processing, but excludes disk I/O, dataloader work, metric evaluation, and
visualization.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from torch.amp.autocast_mode import autocast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from videomt import (
    PanopticDatasetVideoMapper,
    SemanticDatasetVideoMapper,
    YTVISDatasetMapper,
    add_videomt_config,
    build_detection_test_loader,
)


@dataclass(frozen=True)
class Experiment:
    dataset: str
    backbone: str
    method: str
    config: str
    weights: str


def repo_root() -> Path:
    return ROOT


def existing(*candidates: str) -> str:
    root = repo_root()
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return candidate
        absolute = Path(candidate)
        if absolute.exists():
            return str(absolute)
    return candidates[0]


def paper_suite() -> list[Experiment]:
    """Rows used by the paper tables.

    Runtime depends mainly on architecture and output task, but we still point
    to the final checkpoints used in the experiments whenever available.
    """

    rows: list[Experiment] = []
    datasets = [
        ("VIPSeg", "vipseg"),
        ("OVIS", "ovis"),
        ("YTVIS22", "ytvis2022"),
        ("VSPW", "vspw"),
    ]
    backbones = [
        ("S", "vit-small", "vits"),
        ("B", "vit-base", "vitb"),
        ("L", "vit-large", "vitl"),
    ]
    cfg_dataset = {
        "vipseg": "VIPSeg",
        "ovis": "ovis",
        "ytvis2022": "ytvis22",
        "vspw": "VSPW",
    }

    for dataset_label, dataset_key in datasets:
        for backbone_label, backbone_dir, backbone_tag in backbones:
            cfg_part = cfg_dataset[dataset_key]
            rows.append(
                Experiment(
                    dataset=dataset_label,
                    backbone=backbone_label,
                    method="VidEoMT",
                    config=(
                        f"configs/VidEoMT/{cfg_part}/dinov2/"
                        f"{backbone_dir}/videomt_online_{backbone_tag}.yaml"
                    ),
                    weights=existing(
                        f"outputs/videomt_retrain/{dataset_key}/{backbone_dir}/model_final.pth",
                        f"outputs/videomt_retrain_same_protocol_finalweights/{dataset_key}/{backbone_dir}/model_final.pth",
                    ),
                )
            )

            dyn_weights = [
                f"outputs/dyneomt/{dataset_key}/{backbone_dir}/model_final.pth",
                f"outputs/dyneomt_corrected_temporal_targets/{dataset_key}/{backbone_dir}/model_final.pth",
                f"../outputs/dyneomt_models/{dataset_key}/model_final.pth",
            ]
            if dataset_key in {"ovis", "ytvis2022"}:
                dyn_weights = [
                    f"outputs/dyneomt_corrected_temporal_targets/{dataset_key}/{backbone_dir}/model_final.pth",
                    f"outputs/dyneomt/{dataset_key}/{backbone_dir}/model_final.pth",
                    f"../outputs/dyneomt_models/{dataset_key}/model_final.pth",
                ]
            rows.append(
                Experiment(
                    dataset=dataset_label,
                    backbone=backbone_label,
                    method="DynEoMT",
                    config=(
                        f"configs/DynEoMT/{cfg_part}/dinov2/"
                        f"{backbone_dir}/dyneomt_online_{backbone_tag}.yaml"
                    ),
                    weights=existing(*dyn_weights),
                )
            )
    return rows


def setup_cfg(config_file: str, weights: str):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(
        [
            "MODEL.WEIGHTS",
            weights,
            "DATALOADER.NUM_WORKERS",
            "0",
            "TEST.AUG.ENABLED",
            "False",
        ]
    )
    cfg.freeze()
    setup_logger()
    return cfg


def build_test_loader(cfg):
    mapper_dict = {
        "video_instance": YTVISDatasetMapper,
        "video_panoptic": PanopticDatasetVideoMapper,
        "video_semantic": SemanticDatasetVideoMapper,
    }
    dataset_type = cfg.DATASETS.DATASET_TYPE_TEST[0]
    mapper = mapper_dict[dataset_type](cfg, is_train=False)
    return build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=mapper)


def count_frames(batch) -> int:
    item = batch[0]
    if "file_names" in item:
        return len(item["file_names"])
    if "image" in item:
        return len(item["image"])
    return 1


def preload_batches(data_loader: Iterable, total_batches: int) -> list:
    batches = []
    for i, batch in enumerate(data_loader):
        if i >= total_batches:
            break
        batches.append(batch)
    return batches


def benchmark_one(exp: Experiment, args) -> dict:
    root = repo_root()
    config_path = str(root / exp.config)
    weights_path = str((root / exp.weights).resolve()) if not Path(exp.weights).is_absolute() else exp.weights
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Missing checkpoint for {exp}: {weights_path}")

    cfg = setup_cfg(config_path, weights_path)
    model = build_model(cfg)
    model.eval()
    model.to(torch.device("cuda"))
    model.window_inference = True
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    loader = build_test_loader(cfg)
    batches = preload_batches(loader, args.warmup_batches + args.measure_batches)
    if len(batches) <= args.warmup_batches:
        raise RuntimeError(f"Not enough batches for {exp}")

    torch.cuda.reset_peak_memory_stats()
    timed_ms: list[float] = []
    timed_frames = 0

    with torch.inference_mode(), autocast(dtype=torch.float16, device_type="cuda"):
        for idx, batch in enumerate(batches):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if idx >= args.warmup_batches:
                timed_ms.append(elapsed_ms)
                timed_frames += count_frames(batch)

    total_ms = sum(timed_ms)
    fps = 1000.0 * timed_frames / total_ms
    ms_per_frame = total_ms / timed_frames
    result = {
        "dataset": exp.dataset,
        "backbone": exp.backbone,
        "method": exp.method,
        "config": exp.config,
        "weights": exp.weights,
        "clips": len(timed_ms),
        "frames": timed_frames,
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "clip_ms_mean": statistics.mean(timed_ms),
        "clip_ms_std": statistics.pstdev(timed_ms) if len(timed_ms) > 1 else 0.0,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024**2),
    }
    del model
    torch.cuda.empty_cache()
    return result


def write_results(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with output.with_suffix(".json").open("w") as handle:
        json.dump(rows, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["paper"], default="paper")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--backbones", nargs="*", default=None, choices=["S", "B", "L"])
    parser.add_argument("--methods", nargs="*", default=None, choices=["VidEoMT", "DynEoMT"])
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=3)
    parser.add_argument("--output", default="outputs/runtime_end_to_end/paper_runtime.csv")
    args = parser.parse_args()

    experiments = paper_suite()
    if args.datasets:
        wanted = {x.lower() for x in args.datasets}
        experiments = [e for e in experiments if e.dataset.lower() in wanted]
    if args.backbones:
        wanted = set(args.backbones)
        experiments = [e for e in experiments if e.backbone in wanted]
    if args.methods:
        wanted = set(args.methods)
        experiments = [e for e in experiments if e.method in wanted]

    rows = []
    for exp in experiments:
        print(f"[runtime] {exp.dataset} {exp.backbone} {exp.method}", flush=True)
        row = benchmark_one(exp, args)
        print(
            f"  fps={row['fps']:.2f}, ms/frame={row['ms_per_frame']:.2f}, "
            f"frames={row['frames']}, peak_mem={row['peak_mem_mb']:.0f} MB",
            flush=True,
        )
        rows.append(row)
        write_results(rows, repo_root() / args.output)

    print(f"Wrote {repo_root() / args.output}")


if __name__ == "__main__":
    main()
