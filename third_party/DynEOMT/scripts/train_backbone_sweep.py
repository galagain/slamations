#!/usr/bin/env python3
"""Train DynEoMT backbone variants sequentially for one dataset and GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "vipseg": "VIPSeg",
    "ovis": "ovis",
    "ytvis2022": "ytvis22",
    "vspw": "VSPW",
}
BACKBONES = {
    "vit-small": "dyneomt_online_vits.yaml",
    "vit-base": "dyneomt_online_vitb.yaml",
    "vit-large": "dyneomt_online_vitl.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument(
        "--backbones",
        nargs="+",
        choices=BACKBONES,
        default=["vit-small", "vit-base"],
    )
    parser.add_argument(
        "--device",
        required=True,
        help="Physical CUDA device exposed to the training subprocess",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "data",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=ROOT.parent / "outputs" / "dynamic_general_full" / "labels",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "dyneomt",
        help="Root directory for dataset/backbone training outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Train even when model_final.pth already exists",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing last_checkpoint file and start from initialization",
    )
    return parser.parse_args()


def config_path(dataset: str, backbone: str) -> Path:
    backbone_dir = backbone
    return (
        ROOT
        / "configs"
        / "DynEoMT"
        / DATASETS[dataset]
        / "dinov2"
        / backbone_dir
        / BACKBONES[backbone]
    )


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    dataset_targets = args.target_root / args.dataset
    if not dataset_targets.is_dir():
        raise FileNotFoundError(
            f"Dynamic targets do not exist for {args.dataset}: {dataset_targets}"
        )
    for backbone in args.backbones:
        path = config_path(args.dataset, backbone)
        if not path.is_file():
            raise FileNotFoundError(f"Training config does not exist: {path}")


def train(args: argparse.Namespace, backbone: str) -> int:
    output_dir = args.output_root / args.dataset / backbone
    final_checkpoint = output_dir / "model_final.pth"
    if final_checkpoint.is_file() and not args.force:
        print(f"[skip] {args.dataset}/{backbone}: {final_checkpoint} exists", flush=True)
        return 0

    command = [
        sys.executable,
        str(ROOT / "train_net_video.py"),
        "--num-gpus",
        "1",
        "--config-file",
        str(config_path(args.dataset, backbone)),
    ]
    if (output_dir / "last_checkpoint").is_file() and not args.no_resume:
        command.append("--resume")
    command.extend(
        [
            "MODEL.DYNAMIC.TARGET_ROOT",
            str(args.target_root.resolve()),
            "OUTPUT_DIR",
            str(output_dir),
        ]
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["DETECTRON2_DATASETS"] = str(args.data_root.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"[start] {args.dataset}/{backbone} on physical GPU {args.device}\n"
        f"        output: {output_dir}\n"
        f"        command: {' '.join(command)}",
        flush=True,
    )
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode == 0:
        print(f"[done] {args.dataset}/{backbone}", flush=True)
    else:
        print(
            f"[failed] {args.dataset}/{backbone}: exit code {result.returncode}",
            flush=True,
        )
    return result.returncode


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.target_root = args.target_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    validate_inputs(args)
    for backbone in args.backbones:
        return_code = train(args, backbone)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
