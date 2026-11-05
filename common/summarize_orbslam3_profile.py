#!/usr/bin/env python3
"""Summarize ORB-SLAM3 timing files and Jetson tegrastats logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple


NumberMap = Dict[str, List[float]]


def sanitize_header(label: str) -> str:
    label = label.strip().lstrip("#").strip()
    label = label.replace("[ms]", "_ms").replace("[#]", "_count")
    label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
    return label


def read_timing_csv(path: Path, warmup: int = 0) -> Tuple[List[str], NumberMap]:
    if not path.exists():
        return [], {}
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return [], {}

    header_line = lines[0]
    headers = [sanitize_header(part) for part in header_line.split(",")]
    rows = list(csv.reader(lines[1:]))
    if warmup > 0:
        rows = rows[warmup:]

    columns: NumberMap = {header: [] for header in headers}
    for row in rows:
        for idx, header in enumerate(headers):
            if idx >= len(row):
                continue
            try:
                value = float(row[idx])
            except ValueError:
                continue
            columns[header].append(value)
    return headers, columns


def parse_float(value: str) -> float:
    value = value.strip()
    if value.lower() == "nan":
        return float("nan")
    return float(value)


def sanitize_section(label: str) -> str:
    label = re.sub(r"\(.*?\)", "", label)
    return sanitize_header(label)


def parse_exec_mean(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    sections: Dict[str, Dict[str, object]] = {}
    current = "global"
    sections[current] = {}

    mean_std_re = re.compile(r"^([^:]+):\s*([+-]?(?:nan|\d+(?:\.\d+)?))\s*\$\\pm\$\s*([+-]?(?:nan|\d+(?:\.\d+)?))")
    value_re = re.compile(r"^([^:]+):\s*([+-]?(?:nan|\d+(?:\.\d+)?))$")

    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or set(line) == {"-"}:
            continue
        if line.startswith("TIME STATS") or line.startswith("OpenCV version"):
            continue

        mean_std = mean_std_re.match(line)
        if mean_std:
            key = sanitize_header(mean_std.group(1))
            sections.setdefault(current, {})[key] = {
                "mean": parse_float(mean_std.group(2)),
                "std": parse_float(mean_std.group(3)),
            }
            continue

        value = value_re.match(line)
        if value:
            key = sanitize_header(value.group(1))
            sections.setdefault(current, {})[key] = parse_float(value.group(2))
            continue

        if ":" not in line:
            current = sanitize_section(line)
            sections.setdefault(current, {})

    return {"sections": sections}


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if math.isfinite(float(v))]


def percentile(values: List[float], pct: float) -> Optional[float]:
    values = sorted(finite(values))
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (pct / 100.0) * (len(values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def stats(values: Iterable[float]) -> Dict[str, float]:
    values = finite(values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": percentile(values, 50.0),
        "p90": percentile(values, 90.0),
        "p95": percentile(values, 95.0),
        "p99": percentile(values, 99.0),
        "min": min(values),
        "max": max(values),
    }


def histogram(values: Iterable[float]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in finite(values):
        key = str(int(round(value)))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def dominant(columns: NumberMap, names: Iterable[str]) -> Dict[str, object]:
    candidates = []
    for name in names:
        col_stats = stats(columns.get(name, []))
        if col_stats.get("count", 0):
            candidates.append((col_stats["mean"], name, col_stats))
    if not candidates:
        return {}
    candidates.sort(reverse=True)
    mean_value, name, col_stats = candidates[0]
    return {"name": name, "mean_ms": mean_value, "stats": col_stats}


def parse_tegrastats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    cpu_values: List[float] = []
    gpu_values: List[float] = []
    ram_used: List[float] = []
    ram_total: List[float] = []
    power: Dict[str, List[float]] = {}
    samples = 0

    for line in path.read_text(errors="ignore").splitlines():
        samples += 1
        cpu = re.search(r"CPU \[([^\]]+)\]", line)
        if cpu:
            core_values = []
            for token in cpu.group(1).split(","):
                match = re.search(r"(\d+(?:\.\d+)?)%(@|$)", token)
                if match:
                    core_values.append(float(match.group(1)))
            if core_values:
                cpu_values.append(mean(core_values))

        gpu = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        if gpu:
            gpu_values.append(float(gpu.group(1)))

        ram = re.search(r"RAM\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)MB", line)
        if ram:
            ram_used.append(float(ram.group(1)))
            ram_total.append(float(ram.group(2)))

        for rail, value in re.findall(r"([A-Z0-9_]+)\s+(\d+(?:\.\d+)?)mW(?:/\d+(?:\.\d+)?mW)?", line):
            power.setdefault(rail, []).append(float(value))

    result: Dict[str, object] = {"samples": samples}
    if cpu_values:
        result["cpu_percent_mean_per_sample"] = stats(cpu_values)
    if gpu_values:
        result["gpu_gr3d_percent"] = stats(gpu_values)
    if ram_used:
        result["ram_used_mb"] = stats(ram_used)
        result["ram_total_mb_mean"] = mean(ram_total)
    if power:
        result["power_mw"] = {rail: stats(values) for rail, values in sorted(power.items())}
    return result


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def summarize(run_dir: Path, frame_period_ms: float, warmup: int) -> Dict[str, object]:
    tracking_headers, tracking_cols = read_timing_csv(run_dir / "TrackingTimeStats.txt", warmup)
    local_headers, local_cols = read_timing_csv(run_dir / "LocalMapTimeStats.txt", 0)
    lba_headers, lba_cols = read_timing_csv(run_dir / "LBA_Stats.txt", 0)

    tracking_components = [
        "image_rect_ms",
        "image_resize_ms",
        "orb_ext_ms",
        "stereo_match_ms",
        "imu_preint_ms",
        "pose_pred_ms",
        "lm_track_ms",
        "kf_dec_ms",
    ]
    local_components = [
        "kf_insertion_ms",
        "mp_culling_ms",
        "mp_creation_ms",
        "lba_ms",
        "kf_culling_ms",
    ]

    total = tracking_cols.get("total_ms", [])
    over_budget = [v for v in total if math.isfinite(v) and v > frame_period_ms]

    result: Dict[str, object] = {
        "run_dir": str(run_dir),
        "frame_period_ms": frame_period_ms,
        "warmup_frames_ignored": warmup,
        "tracking": {
            "headers": tracking_headers,
            "columns": {name: stats(values) for name, values in tracking_cols.items()},
            "dominant_mean_component": dominant(tracking_cols, tracking_components),
            "frames_over_budget": len(over_budget),
            "frames_over_budget_ratio": (len(over_budget) / len(total)) if total else None,
            "state_histogram": histogram(tracking_cols.get("tracking_state_count", [])),
        },
        "local_mapping": {
            "headers": local_headers,
            "columns": {name: stats(values) for name, values in local_cols.items()},
            "dominant_mean_component": dominant(local_cols, local_components),
        },
        "lba": {
            "headers": lba_headers,
            "columns": {name: stats(values) for name, values in lba_cols.items()},
        },
        "global_modules": parse_exec_mean(run_dir / "ExecMean.txt"),
        "tegrastats": parse_tegrastats(run_dir / "tegrastats.log"),
        "evaluation": load_json(run_dir / "evaluation.json"),
        "session_info": (run_dir / "SessionInfo.txt").read_text(errors="ignore") if (run_dir / "SessionInfo.txt").exists() else "",
    }
    return result


def format_ms(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def format_percent(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}%"


def share(value: Optional[float], total: Optional[float]) -> Optional[float]:
    if value is None or total is None:
        return None
    if not math.isfinite(value) or not math.isfinite(total) or total == 0:
        return None
    return 100.0 * value / total


def stat_column(columns: Dict[str, Dict[str, float]], name: str) -> Dict[str, float]:
    return columns.get(name, {})


def exec_metric(summary: Dict[str, object], section: str, name: str) -> Dict[str, float]:
    modules = summary.get("global_modules", {})
    sections = modules.get("sections", {}) if isinstance(modules, dict) else {}
    section_data = sections.get(section, {}) if isinstance(sections, dict) else {}
    metric = section_data.get(name, {}) if isinstance(section_data, dict) else {}
    return metric if isinstance(metric, dict) else {}


def exec_value(summary: Dict[str, object], section: str, name: str) -> Optional[float]:
    modules = summary.get("global_modules", {})
    sections = modules.get("sections", {}) if isinstance(modules, dict) else {}
    section_data = sections.get(section, {}) if isinstance(sections, dict) else {}
    value = section_data.get(name) if isinstance(section_data, dict) else None
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def markdown(summary: Dict[str, object]) -> str:
    tracking = summary["tracking"]
    local_mapping = summary["local_mapping"]
    lba = summary["lba"]
    tegra = summary.get("tegrastats", {})
    evaluation = summary.get("evaluation", {})

    lines = [
        "# ORB-SLAM3 Jetson Bottleneck Summary",
        "",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Frame period budget: {summary['frame_period_ms']:.3f} ms",
        f"- Warmup frames ignored in tracking summary: {summary['warmup_frames_ignored']}",
        "",
    ]

    dom_track = tracking.get("dominant_mean_component") or {}
    dom_map = local_mapping.get("dominant_mean_component") or {}
    if dom_track:
        lines.append(f"- Dominant tracking component by mean: `{dom_track['name']}` ({dom_track['mean_ms']:.3f} ms)")
    if dom_map:
        lines.append(f"- Dominant local-mapping component by mean: `{dom_map['name']}` ({dom_map['mean_ms']:.3f} ms)")
    ratio = tracking.get("frames_over_budget_ratio")
    if ratio is not None:
        lines.append(f"- Tracking frames over budget: {tracking['frames_over_budget']} ({100.0 * ratio:.1f}%)")

    lines.extend([
        "",
        "## Geometric SLAM Blocks",
        "",
        "| Block | Source metric | Mean ms | P95 ms | Max ms | Count | Share |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    tracking_total = stat_column(tracking["columns"], "total_ms").get("mean")
    local_total = stat_column(local_mapping["columns"], "total_ms").get("mean")
    block_rows = [
        ("Input rectification", "tracking.image_rect_ms", "tracking", "image_rect_ms", tracking_total),
        ("Image resize", "tracking.image_resize_ms", "tracking", "image_resize_ms", tracking_total),
        ("Visual front-end ORB", "tracking.orb_ext_ms", "tracking", "orb_ext_ms", tracking_total),
        ("Stereo/RGB-D association", "tracking.stereo_match_ms", "tracking", "stereo_match_ms", tracking_total),
        ("IMU preintegration", "tracking.imu_preint_ms", "tracking", "imu_preint_ms", tracking_total),
        ("Pose prediction", "tracking.pose_pred_ms", "tracking", "pose_pred_ms", tracking_total),
        ("Local map tracking", "tracking.lm_track_ms", "tracking", "lm_track_ms", tracking_total),
        ("Keyframe decision", "tracking.kf_dec_ms", "tracking", "kf_dec_ms", tracking_total),
        ("Tracking total", "tracking.total_ms", "tracking", "total_ms", tracking_total),
        ("Keyframe insertion", "local_mapping.kf_insertion_ms", "local_mapping", "kf_insertion_ms", local_total),
        ("Map point culling", "local_mapping.mp_culling_ms", "local_mapping", "mp_culling_ms", local_total),
        ("Map point creation", "local_mapping.mp_creation_ms", "local_mapping", "mp_creation_ms", local_total),
        ("Local bundle adjustment", "local_mapping.lba_ms", "local_mapping", "lba_ms", local_total),
        ("Keyframe culling", "local_mapping.kf_culling_ms", "local_mapping", "kf_culling_ms", local_total),
        ("Local mapping total", "local_mapping.total_ms", "local_mapping", "total_ms", local_total),
    ]
    for block, metric_label, family, name, total_mean in block_rows:
        columns = tracking["columns"] if family == "tracking" else local_mapping["columns"]
        col = columns.get(name, {})
        mean_value = col.get("mean")
        lines.append(
            f"| {block} | `{metric_label}` | {format_ms(mean_value)} | {format_ms(col.get('p95'))} | "
            f"{format_ms(col.get('max'))} | {col.get('count', 0)} | {format_percent(share(mean_value, total_mean))} |"
        )

    pr_total = exec_metric(summary, "place_recognition", "total_place_recognition")
    pr_query = exec_metric(summary, "place_recognition", "database_query")
    pr_se3 = exec_metric(summary, "place_recognition", "se3_estimation")
    if pr_total:
        lines.append(
            f"| Place recognition | `exec_mean.total_place_recognition` | {format_ms(pr_total.get('mean'))} | n/a | n/a | mean only | n/a |"
        )
        lines.append(
            f"| Place recognition DB query | `exec_mean.database_query` | {format_ms(pr_query.get('mean'))} | n/a | n/a | mean only | n/a |"
        )
        lines.append(
            f"| Place recognition SE3 | `exec_mean.se3_estimation` | {format_ms(pr_se3.get('mean'))} | n/a | n/a | mean only | n/a |"
        )

    lines.extend(["", "## Tracking Details", "", "| Metric | Mean ms | P95 ms | Max ms | Count |", "|---|---:|---:|---:|---:|"])
    for name in ["image_rect_ms", "image_resize_ms", "orb_ext_ms", "stereo_match_ms", "imu_preint_ms", "pose_pred_ms", "lm_track_ms", "kf_dec_ms", "total_ms"]:
        col = tracking["columns"].get(name, {})
        lines.append(
            f"| `{name}` | {format_ms(col.get('mean'))} | {format_ms(col.get('p95'))} | {format_ms(col.get('max'))} | {col.get('count', 0)} |"
        )

    lines.extend(["", "## Tracking Signals", "", "| Metric | Mean | P95 | Max | Count |", "|---|---:|---:|---:|---:|"])
    for name in ["features_count", "inliers_count", "tracked_mps_count"]:
        col = tracking["columns"].get(name, {})
        lines.append(
            f"| `{name}` | {format_ms(col.get('mean'))} | {format_ms(col.get('p95'))} | {format_ms(col.get('max'))} | {col.get('count', 0)} |"
        )
    state_hist = tracking.get("state_histogram", {})
    if state_hist:
        rendered = ", ".join(f"{state}: {count}" for state, count in state_hist.items())
        lines.append("")
        lines.append(f"- Tracking state histogram: {rendered}")

    lines.extend(["", "## Local Mapping", "", "| Metric | Mean ms | P95 ms | Max ms | Count |", "|---|---:|---:|---:|---:|"])
    for name in ["kf_insertion_ms", "mp_culling_ms", "mp_creation_ms", "lba_ms", "kf_culling_ms", "total_ms", "kf_queue_before_processing_count"]:
        col = local_mapping["columns"].get(name, {})
        lines.append(
            f"| `{name}` | {format_ms(col.get('mean'))} | {format_ms(col.get('p95'))} | {format_ms(col.get('max'))} | {col.get('count', 0)} |"
        )

    lines.extend([
        "",
        "## LBA Complexity",
        "",
        "This table uses `LBA_Stats.txt`, so it describes executed LBAs only. The `local_mapping.lba_ms` row above is per local-mapping iteration and includes iterations without LBA.",
        "",
        "| Metric | Mean | P95 | Max | Count |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ["lba_time_ms", "kf_opt_count", "kf_fixed_count", "mp_count", "edges_count"]:
        col = lba["columns"].get(name, {})
        lines.append(
            f"| `{name}` | {format_ms(col.get('mean'))} | {format_ms(col.get('p95'))} | {format_ms(col.get('max'))} | {col.get('count', 0)} |"
        )

    modules = summary.get("global_modules", {})
    sections = modules.get("sections", {}) if isinstance(modules, dict) else {}
    if sections:
        loop_exec = exec_value(summary, "loop_closing", "numb_exec")
        merge_exec = exec_value(summary, "map_merging", "numb_exec")
        gba_exec = exec_value(summary, "full_gba", "numb_exec")
        lines.extend(["", "## Global Modules", ""])
        if loop_exec is not None:
            lines.append(f"- Loop closing executions: {int(loop_exec)}")
        if merge_exec is not None:
            lines.append(f"- Map merging executions: {int(merge_exec)}")
        if gba_exec is not None:
            lines.append(f"- Full GBA executions: {int(gba_exec)}")
        if not any(v for v in [loop_exec, merge_exec, gba_exec] if isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
            lines.append("- No loop closing, map merging, or full GBA execution was triggered in this run.")

        lines.extend(["", "| Module block | Mean ms | Std ms | Executions / size |", "|---|---:|---:|---:|"])
        global_rows = [
            ("Loop fusion", "loop_closing", "loop_fusion", loop_exec),
            ("Essential graph", "loop_closing", "essential_graph", loop_exec),
            ("Total loop closing", "loop_closing", "total_loop_closing", loop_exec),
            ("Map merge", "map_merging", "merge_maps", merge_exec),
            ("Welding BA", "map_merging", "welding_ba", merge_exec),
            ("Map merge essential graph", "map_merging", "optimization_ess", merge_exec),
            ("Total map merging", "map_merging", "total_map_merging", merge_exec),
            ("Full GBA", "full_gba", "gba", gba_exec),
            ("Full GBA map update", "full_gba", "map_update", gba_exec),
            ("Total full GBA", "full_gba", "total_full_gba", gba_exec),
        ]
        for label, section, metric_name, executions in global_rows:
            metric = exec_metric(summary, section, metric_name)
            if metric or executions:
                lines.append(
                    f"| {label} | {format_ms(metric.get('mean'))} | {format_ms(metric.get('std'))} | "
                    f"{int(executions) if executions is not None else 'n/a'} |"
                )

        full_gba_kfs = exec_metric(summary, "full_gba", "number_of_kfs").get("mean")
        full_gba_mps = exec_metric(summary, "full_gba", "number_of_mps").get("mean")
        if full_gba_kfs is not None or full_gba_mps is not None:
            lines.extend(["", "Full GBA problem size:"])
            if full_gba_kfs is not None and math.isfinite(full_gba_kfs):
                lines.append(f"- Keyframes: {full_gba_kfs:.0f}")
            if full_gba_mps is not None and math.isfinite(full_gba_mps):
                lines.append(f"- Map points: {full_gba_mps:.0f}")

    if tegra:
        lines.extend(["", "## Jetson System", ""])
        cpu = tegra.get("cpu_percent_mean_per_sample", {})
        gpu = tegra.get("gpu_gr3d_percent", {})
        ram = tegra.get("ram_used_mb", {})
        lines.append(f"- `tegrastats` samples: {tegra.get('samples', 0)}")
        if cpu:
            lines.append(f"- CPU mean per-sample utilization: {format_ms(cpu.get('mean'))}% mean, {format_ms(cpu.get('p95'))}% p95")
        if gpu:
            lines.append(f"- GR3D GPU utilization: {format_ms(gpu.get('mean'))}% mean, {format_ms(gpu.get('p95'))}% p95")
        if ram:
            lines.append(f"- RAM used: {format_ms(ram.get('mean'))} MB mean, {format_ms(ram.get('max'))} MB max")

    if evaluation:
        ate = evaluation.get("ate_m", {})
        rpe = evaluation.get("rpe_trans_m", {})
        lines.extend(["", "## Accuracy", ""])
        lines.append(f"- Matched pose pairs: {evaluation.get('matched_pairs', 'n/a')}")
        if ate:
            lines.append(f"- ATE RMSE: {ate.get('rmse', float('nan')):.6f} m")
        if rpe:
            lines.append(f"- RPE translation RMSE: {rpe.get('rmse', float('nan')):.6f} m")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--frame-period-ms", type=float, default=33.333)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    summary = summarize(args.run_dir, args.frame_period_ms, args.warmup)
    json_text = json.dumps(summary, indent=2, sort_keys=True)
    print(json_text)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json_text + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
