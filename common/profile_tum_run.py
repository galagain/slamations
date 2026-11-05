#!/usr/bin/env python3
"""Summarize a TUM run directory with wall-time, FPS and trajectory metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional


def count_tum_rows(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            total += 1
    return total


def read_float_lines(path: Path) -> List[float]:
    values: List[float] = []
    if not path.exists():
        return values
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = re.split(r"[\s,;]+", line)
        numeric = []
        for field in fields:
            try:
                numeric.append(float(field))
            except ValueError:
                pass
        if len(numeric) == 1 and math.isfinite(numeric[0]):
            values.append(numeric[0])
    return values


def sanitize_header(label: str) -> str:
    label = label.strip().lstrip("#").strip()
    label = label.replace("[ms]", "_ms").replace("[#]", "_count")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return label


def read_timing_table(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith("#"):
        return {}

    headers = [sanitize_header(part) for part in lines[0].split(",")]
    if len(headers) < 2:
        return {}

    columns: Dict[str, List[float]] = {header: [] for header in headers}
    for raw in lines[1:]:
        fields = re.split(r"\s*,\s*", raw)
        if len(fields) < 2:
            continue
        for idx, header in enumerate(headers):
            if idx >= len(fields):
                continue
            try:
                value = float(fields[idx])
            except ValueError:
                continue
            if math.isfinite(value):
                columns[header].append(value)

    column_stats = {name: stats(values) for name, values in columns.items() if values}
    if not column_stats:
        return {}

    return {
        "columns": column_stats,
        "rows": max((len(values) for values in columns.values()), default=0),
    }


def stats(values: Iterable[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {}
    n = len(vals)
    p95_index = min(n - 1, max(0, int(math.ceil(0.95 * n)) - 1))
    return {
        "count": n,
        "mean": mean(vals),
        "median": median(vals),
        "p95": vals[p95_index],
        "min": vals[0],
        "max": vals[-1],
    }


def parse_log_times(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    out: Dict[str, float] = {}
    patterns = {
        "tracking_median_s": r"median tracking time\s*[:=]\s*([0-9.eE+-]+)",
        "tracking_mean_s": r"mean tracking time\s*[:=]\s*([0-9.eE+-]+)",
        "tracking_median_ms": r"median tracking time\s*[:=]\s*([0-9.eE+-]+)\s*ms",
        "tracking_mean_ms": r"mean tracking time\s*[:=]\s*([0-9.eE+-]+)\s*ms",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                out[key] = float(match.group(1))
            except ValueError:
                pass
    return out


def parse_tegrastats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"samples": 0}
    cpu_values: List[float] = []
    gpu_values: List[float] = []
    ram_values: List[float] = []
    for line in path.read_text(errors="ignore").splitlines():
        cpu = re.search(r"CPU \[([^\]]+)\]", line)
        if cpu:
            per_core = []
            for entry in cpu.group(1).split(","):
                match = re.search(r"(\d+(?:\.\d+)?)%", entry)
                if match:
                    per_core.append(float(match.group(1)))
            if per_core:
                cpu_values.append(mean(per_core))
        gpu = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        if gpu:
            gpu_values.append(float(gpu.group(1)))
        ram = re.search(r"RAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB", line)
        if ram:
            ram_values.append(float(ram.group(1)))
    return {
        "samples": max(len(cpu_values), len(gpu_values), len(ram_values)),
        "cpu_percent": stats(cpu_values),
        "gpu_gr3d_percent": stats(gpu_values),
        "ram_used_mb": stats(ram_values),
    }


def load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def fmt(value: object, suffix: str = "") -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.3f}{suffix}"
    return "n/a"


def write_markdown(summary: Dict[str, object], path: Path) -> None:
    eval_data = summary.get("evaluation") or {}
    ate = ((eval_data.get("ate_m") or {}) if isinstance(eval_data, dict) else {})
    rpe = ((eval_data.get("rpe_trans_m") or {}) if isinstance(eval_data, dict) else {})
    tegra = summary.get("tegrastats") or {}
    cpu = ((tegra.get("cpu_percent") or {}) if isinstance(tegra, dict) else {})
    gpu = ((tegra.get("gpu_gr3d_percent") or {}) if isinstance(tegra, dict) else {})
    ram = ((tegra.get("ram_used_mb") or {}) if isinstance(tegra, dict) else {})

    lines = [
        f"# {summary.get('method', 'method')} on {summary.get('sequence_name', 'sequence')}",
        "",
        f"- run exit code: `{summary.get('run_exit_code')}`",
        f"- wall time: {fmt(summary.get('wall_seconds'), ' s')}",
        f"- input RGB frames: `{summary.get('input_rgb_frames')}`",
        f"- estimated poses: `{summary.get('estimated_poses')}`",
        f"- wall FPS over RGB frames: {fmt(summary.get('fps_input_frames'))}",
        f"- wall FPS over estimated poses: {fmt(summary.get('fps_estimated_poses'))}",
        f"- ATE RMSE: {fmt(ate.get('rmse'), ' m')}",
        f"- RPE translation RMSE: {fmt(rpe.get('rmse'), ' m')}",
        f"- CPU mean: {fmt(cpu.get('mean'), '%')}",
        f"- GR3D mean: {fmt(gpu.get('mean'), '%')}",
        f"- RAM mean: {fmt(ram.get('mean'), ' MB')}",
    ]

    log_times = summary.get("log_times") or {}
    if isinstance(log_times, dict) and log_times:
        lines += ["", "## Parsed Log Times"]
        for key, value in sorted(log_times.items()):
            lines.append(f"- {key}: {fmt(value)}")

    time_files = summary.get("time_files") or {}
    if isinstance(time_files, dict) and time_files:
        lines += ["", "## Numeric Time Files"]
        for name, data in sorted(time_files.items()):
            if isinstance(data, dict):
                lines.append(
                    f"- `{name}`: mean {fmt(data.get('mean'))}, median {fmt(data.get('median'))}, p95 {fmt(data.get('p95'))}, n={data.get('count')}"
                )

    time_tables = summary.get("time_tables") or {}
    if isinstance(time_tables, dict) and time_tables:
        lines += ["", "## Timing Tables"]
        for name, table in sorted(time_tables.items()):
            if not isinstance(table, dict):
                continue
            columns = table.get("columns")
            if not isinstance(columns, dict):
                continue
            highlights = []
            for column_name in ("total_ms", "orb_ext_ms", "lm_track_ms", "lba_ms"):
                column = columns.get(column_name)
                if isinstance(column, dict) and column.get("count"):
                    highlights.append(
                        f"{column_name} mean {fmt(column.get('mean'))}, p95 {fmt(column.get('p95'))}"
                    )
            if highlights:
                lines.append(f"- `{name}`: " + "; ".join(highlights))

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--wall-seconds", required=True, type=float)
    parser.add_argument("--run-exit-code", required=True, type=int)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    camera_traj = run_dir / "CameraTrajectory.txt"
    evaluation = load_json(run_dir / "evaluation.json")
    input_frames = count_tum_rows(args.sequence / "rgb.txt")
    estimated_poses = count_tum_rows(camera_traj)

    time_files: Dict[str, object] = {}
    time_tables: Dict[str, object] = {}
    for path in sorted(run_dir.glob("*Time*.txt")) + sorted(run_dir.glob("*time*.txt")):
        table = read_timing_table(path)
        if table:
            time_tables[path.name] = table
        values = read_float_lines(path)
        if values:
            time_files[path.name] = stats(values)

    log_times = parse_log_times(args.log) if args.log else {}
    tegrastats = parse_tegrastats(run_dir / "tegrastats.log")

    summary: Dict[str, object] = {
        "method": args.method,
        "sequence": str(args.sequence),
        "sequence_name": args.sequence.name,
        "run_dir": str(run_dir),
        "run_exit_code": args.run_exit_code,
        "wall_seconds": args.wall_seconds,
        "input_rgb_frames": input_frames,
        "estimated_poses": estimated_poses,
        "fps_input_frames": (input_frames / args.wall_seconds) if args.wall_seconds > 0 else None,
        "fps_estimated_poses": (estimated_poses / args.wall_seconds) if args.wall_seconds > 0 else None,
        "evaluation": evaluation,
        "log_times": log_times,
        "time_files": time_files,
        "time_tables": time_tables,
        "tegrastats": tegrastats,
    }

    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(summary, args.output_md)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
