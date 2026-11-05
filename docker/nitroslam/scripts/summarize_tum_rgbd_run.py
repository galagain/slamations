#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(errors="ignore").splitlines() if line.strip() and not line.startswith("#"))


def load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def parse_tracking_times(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    out: dict[str, float] = {}
    for name in ("median", "mean"):
        match = re.search(rf"{name} tracking time\s*:\s*([0-9.eE+-]+)\s*(ms)?", text, re.IGNORECASE)
        if not match:
            continue
        value = float(match.group(1))
        seconds = value / 1000.0 if match.group(2) else value
        out[f"tracking_{name}_s"] = seconds
        out[f"tracking_{name}_ms"] = seconds * 1000.0
    mean_s = out.get("tracking_mean_s")
    if mean_s and mean_s > 0:
        out["tracking_fps"] = 1.0 / mean_s
    return out


def finite(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def fmt(value: object, suffix: str = "") -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}{suffix}"


def write_md(summary: dict[str, object], path: Path) -> None:
    evaluation = summary.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    ate = evaluation.get("ate_m") if isinstance(evaluation.get("ate_m"), dict) else {}
    rpe = evaluation.get("rpe_trans_m") if isinstance(evaluation.get("rpe_trans_m"), dict) else {}

    lines = [
        f"# {summary['method']} on {summary['sequence_name']}",
        "",
        f"- exit: `{summary['run_exit_code']}`",
        f"- wall time: {fmt(summary.get('wall_seconds'), ' s')}",
        f"- input frames: `{summary.get('input_rgb_frames')}`",
        f"- estimated poses: `{summary.get('estimated_poses')}`",
        f"- wall FPS: {fmt(summary.get('fps_input_frames'))}",
        f"- tracking mean: {fmt(summary.get('tracking_mean_ms'), ' ms')}",
        f"- tracking FPS: {fmt(summary.get('tracking_fps'))}",
        f"- ATE RMSE: {fmt(ate.get('rmse'), ' m')}",
        f"- RPE trans RMSE: {fmt(rpe.get('rmse'), ' m')}",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--run-exit-code", required=True, type=int)
    args = parser.parse_args()

    evaluation = load_json(args.run_dir / "evaluation.json")
    log_times = parse_tracking_times(args.run_dir / "run.log")
    input_frames = count_rows(args.sequence / "rgb.txt")
    estimated_poses = count_rows(args.run_dir / "CameraTrajectory.txt")
    summary: dict[str, object] = {
        "method": args.method,
        "sequence": str(args.sequence),
        "sequence_name": args.sequence.name,
        "run_dir": str(args.run_dir),
        "run_exit_code": args.run_exit_code,
        "wall_seconds": args.wall_seconds,
        "input_rgb_frames": input_frames,
        "estimated_poses": estimated_poses,
        "fps_input_frames": input_frames / args.wall_seconds if args.wall_seconds > 0 else None,
        "fps_estimated_poses": estimated_poses / args.wall_seconds if args.wall_seconds > 0 else None,
        "evaluation": evaluation,
    }
    summary.update(log_times)

    json_path = args.run_dir / "profile_summary.json"
    md_path = args.run_dir / "profile_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_md(summary, md_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
