#!/usr/bin/env python3
"""Train VidEoMT baselines sequentially for one dataset and physical GPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "vipseg": "VIPSeg",
    "ovis": "ovis",
    "ytvis2022": "ytvis22",
    "vspw": "VSPW",
}
BACKBONES = {
    "vit-small": "videomt_online_vits.yaml",
    "vit-base": "videomt_online_vitb.yaml",
    "vit-large": "videomt_online_vitl.yaml",
}
FINAL_WEIGHTS = {
    "vipseg": "checkpoints/vipseg_vit_large_55.2.pth",
    "ovis": "checkpoints/ovis_vit_large_52.5.pth",
    "ytvis2022": "checkpoints/yt_2022_vit_large_42.6.pth",
    "vspw": "checkpoints/vspw_vit_large_95.0_64.9.pth",
}
SAME_PROTOCOL = {
    "vipseg": {
        "SOLVER.CHECKPOINT_PERIOD": "5000",
        "TEST.EVAL_PERIOD": "5000",
    },
    "ovis": {
        "DATASETS.TRAIN": "('ovis_train',)",
        "DATASETS.TEST": "('ovis_val',)",
        "DATASETS.DATASET_RATIO": "[1.0]",
        "DATASETS.DATASET_NEED_MAP": "[False]",
        "DATASETS.DATASET_TYPE": "['video_instance']",
        "DATASETS.DATASET_TYPE_TEST": "['video_instance']",
        "SOLVER.CHECKPOINT_PERIOD": "10000",
        "TEST.EVAL_PERIOD": "10000",
    },
    "ytvis2022": {
        "DATASETS.TRAIN": "('ytvis_2022_train',)",
        "DATASETS.TEST": "('ytvis_2022_val',)",
        "DATASETS.DATASET_RATIO": "[1.0]",
        "DATASETS.DATASET_NEED_MAP": "[False]",
        "DATASETS.DATASET_TYPE": "['video_instance']",
        "DATASETS.DATASET_TYPE_TEST": "['video_instance']",
        "SOLVER.CHECKPOINT_PERIOD": "10000",
        "TEST.EVAL_PERIOD": "10000",
    },
    "vspw": {
        "SOLVER.CHECKPOINT_PERIOD": "5000",
        "TEST.EVAL_PERIOD": "5000",
    },
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
    parser.add_argument("--device", required=True, help="Physical CUDA device")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT.parent / "data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "videomt_retrain",
    )
    parser.add_argument(
        "--same-protocol-final-weights",
        action="store_true",
        help=(
            "For ViT-L baselines, start from the public final VidEoMT weights "
            "and mirror the DynEoMT training protocol."
        ),
    )
    parser.add_argument(
        "--wait-for-screen",
        help="Delay training until this GNU screen session has exited",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def config_path(dataset: str, backbone: str) -> Path:
    return (
        ROOT
        / "configs"
        / "VidEoMT"
        / DATASETS[dataset]
        / "dinov2"
        / backbone
        / BACKBONES[backbone]
    )


def screen_exists(name: str) -> bool:
    result = subprocess.run(
        ["screen", "-ls"],
        capture_output=True,
        text=True,
        check=False,
    )
    return f".{name}\t" in result.stdout


def wait_for_screen(name: str) -> None:
    while screen_exists(name):
        print(f"[wait] screen {name!r} is still active", flush=True)
        time.sleep(60)
    print(f"[ready] screen {name!r} has exited", flush=True)


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    for backbone in args.backbones:
        config = config_path(args.dataset, backbone)
        if not config.is_file():
            raise FileNotFoundError(f"Training config does not exist: {config}")
        if args.same_protocol_final_weights and backbone != "vit-large":
            raise ValueError("--same-protocol-final-weights is only defined for vit-large")
    if args.same_protocol_final_weights:
        weights = ROOT / FINAL_WEIGHTS[args.dataset]
        if not weights.is_file():
            raise FileNotFoundError(f"Final VidEoMT weights do not exist: {weights}")


def output_root(args: argparse.Namespace) -> Path:
    root = args.output_root.expanduser()
    if not root.is_absolute():
        root = ROOT / root
    return root


def same_protocol_opts(dataset: str) -> list[str]:
    opts = {
        "MODEL.WEIGHTS": FINAL_WEIGHTS[dataset],
        "SOLVER.IMS_PER_BATCH": "1",
        "SOLVER.BASE_LR": "0.0000125",
        "INPUT.SAMPLING_FRAME_NUM": "3",
        "INPUT.SAMPLING_FRAME_RANGE": "1",
        "INPUT.REVERSE_AGU": "False",
    }
    opts.update(SAME_PROTOCOL[dataset])
    flattened: list[str] = []
    for key, value in opts.items():
        flattened.extend([key, value])
    return flattened


def train(args: argparse.Namespace, backbone: str) -> int:
    output_dir = output_root(args) / args.dataset / backbone
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
    command.extend(["OUTPUT_DIR", str(output_dir)])
    if args.same_protocol_final_weights:
        command.extend(same_protocol_opts(args.dataset))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.device)
    env["DETECTRON2_DATASETS"] = str(args.data_root.resolve())
    env["PYTHONUNBUFFERED"] = "1"
    print(
        f"[start] VidEoMT {args.dataset}/{backbone} on physical GPU {args.device}\n"
        f"        output: {output_dir}\n"
        f"        command: {' '.join(command)}",
        flush=True,
    )
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    state = "done" if result.returncode == 0 else "failed"
    print(f"[{state}] VidEoMT {args.dataset}/{backbone}", flush=True)
    return result.returncode


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    validate_inputs(args)
    if args.wait_for_screen:
        wait_for_screen(args.wait_for_screen)
    for backbone in args.backbones:
        return_code = train(args, backbone)
        if return_code != 0:
            return return_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
