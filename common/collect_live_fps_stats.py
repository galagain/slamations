#!/usr/bin/env python3
"""Collect in-process, per-frame live FPS statistics for TUM RGB-D runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional


SEQUENCES = [
    "rgbd_dataset_freiburg1_xyz",
    "rgbd_dataset_freiburg2_large_with_loop",
    "rgbd_dataset_freiburg3_walking_xyz",
]


def finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def percentile(values: Iterable[float], pct: float) -> Optional[float]:
    vals = sorted(finite(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (pct / 100.0) * (len(vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (rank - lo)


def safe_fps(latency_ms: Optional[float]) -> Optional[float]:
    if latency_ms is None or not math.isfinite(latency_ms) or latency_ms <= 0:
        return None
    return 1000.0 / latency_ms


def fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.{digits}f}"
    if value is None:
        return "n/a"
    return str(value)


def run_timestamp_key(summary_path: Path) -> str:
    match = re.search(r"_(\d{8}_\d{6})$", summary_path.parent.name)
    return match.group(1) if match else ""


def latest_summary(method_dir: Path, sequence_name: str) -> Optional[Path]:
    candidates = sorted(method_dir.glob(f"results/profiles/{sequence_name}_*/profile_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=run_timestamp_key)


def load_json(path: Path) -> Dict[str, object]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def nested(data: Dict[str, object], dotted: str) -> object:
    current: object = data
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def log_tracking_ms(summary: Dict[str, object]) -> Dict[str, Optional[float]]:
    log_times = summary.get("log_times")
    if not isinstance(log_times, dict):
        return {"mean_ms": None, "median_ms": None}

    mean_ms = None
    median_ms = None
    if isinstance(log_times.get("tracking_mean_ms"), (int, float)):
        mean_ms = float(log_times["tracking_mean_ms"])
    elif isinstance(log_times.get("tracking_mean_s"), (int, float)):
        mean_ms = 1000.0 * float(log_times["tracking_mean_s"])

    if isinstance(log_times.get("tracking_median_ms"), (int, float)):
        median_ms = float(log_times["tracking_median_ms"])
    elif isinstance(log_times.get("tracking_median_s"), (int, float)):
        median_ms = 1000.0 * float(log_times["tracking_median_s"])

    return {"mean_ms": mean_ms, "median_ms": median_ms}


def parse_frame_track_times(path: Path) -> List[float]:
    values: List[float] = []
    if not path.exists():
        return values
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        try:
            values.append(float(parts[2]))
        except ValueError:
            continue
    return values


def parse_tracking_time_stats(path: Path) -> List[float]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    headers = [part.strip().lower() for part in lines[0].lstrip("#").split(",")]
    total_idx = None
    for idx, header in enumerate(headers):
        if header.startswith("total"):
            total_idx = idx
            break
    if total_idx is None:
        return []
    values: List[float] = []
    for row in csv.reader(lines[1:]):
        if total_idx >= len(row):
            continue
        try:
            values.append(float(row[total_idx]))
        except ValueError:
            continue
    return values


def summarize_latencies(values: List[float], source: str) -> Dict[str, object]:
    vals = finite(values)
    if not vals:
        return {"source": source, "frames": 0}
    mean_ms = mean(vals)
    median_ms = median(vals)
    p90_ms = percentile(vals, 90.0)
    p95_ms = percentile(vals, 95.0)
    p99_ms = percentile(vals, 99.0)
    max_ms = max(vals)
    return {
        "source": source,
        "frames": len(vals),
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": max_ms,
        "mean_fps": safe_fps(mean_ms),
        "median_fps": safe_fps(median_ms),
        "p95_safe_fps": safe_fps(p95_ms),
        "p99_safe_fps": safe_fps(p99_ms),
        "max_safe_fps": safe_fps(max_ms),
        "over_30hz_ratio": sum(1 for value in vals if value > 33.333333) / len(vals),
        "over_60hz_ratio": sum(1 for value in vals if value > 16.666667) / len(vals),
    }


def summarize_run(summary_path: Path) -> Dict[str, object]:
    summary = load_json(summary_path)
    run_dir = summary_path.parent

    frame_values = parse_frame_track_times(run_dir / "FrameTrackTimes.txt")
    if frame_values:
        live = summarize_latencies(frame_values, "FrameTrackTimes.txt")
    else:
        tracking_values = parse_tracking_time_stats(run_dir / "TrackingTimeStats.txt")
        if tracking_values:
            live = summarize_latencies(tracking_values, "TrackingTimeStats.txt")
        else:
            log_stats = log_tracking_ms(summary)
            mean_ms = log_stats["mean_ms"]
            median_ms = log_stats["median_ms"]
            live = {
                "source": "log mean/median only",
                "frames": None,
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "p90_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
                "mean_fps": safe_fps(mean_ms),
                "median_fps": safe_fps(median_ms),
                "p95_safe_fps": None,
                "p99_safe_fps": None,
                "max_safe_fps": None,
                "over_30hz_ratio": None,
                "over_60hz_ratio": None,
            }

    live.update(
        {
            "method": summary.get("method"),
            "sequence_name": summary.get("sequence_name"),
            "run_dir": str(run_dir),
            "exit": summary.get("run_exit_code"),
            "poses": summary.get("estimated_poses"),
            "ate_rmse_m": nested(summary, "evaluation.ate_m.rmse"),
            "rpe_trans_rmse_m": nested(summary, "evaluation.rpe_trans_m.rmse"),
        }
    )
    return live


def collect(methods_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for method_dir in sorted(path for path in methods_root.iterdir() if path.is_dir() and path.name != "comparison"):
        for sequence_name in SEQUENCES:
            summary_path = latest_summary(method_dir, sequence_name)
            if summary_path:
                rows.append(summarize_run(summary_path))
    return rows


def sort_key(row: Dict[str, object]) -> float:
    for key in ("p95_safe_fps", "mean_fps"):
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return -math.inf


def md_table(headers: List[str], body: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def write_markdown(rows: List[Dict[str, object]], output: Path) -> None:
    lines = [
        "# Live FPS TUM RGB-D",
        "",
        "Ces statistiques excluent le demarrage Docker, le chargement du vocabulaire, la sauvegarde de trajectoire et l'evaluation.",
        "Elles utilisent les temps mesures a l'interieur du binaire SLAM, image par image quand `FrameTrackTimes.txt` ou `TrackingTimeStats.txt` est disponible.",
        "Quand seule la sortie standard est disponible, seules les moyenne/mediane internes imprimees par le binaire sont reportees.",
        "",
    ]

    headers = [
        "Methode",
        "Source",
        "Frames",
        "FPS moyen",
        "FPS p95",
        "FPS p99",
        "Lat. moy ms",
        "Lat. med ms",
        "P95 ms",
        "P99 ms",
        ">33.3ms",
        ">16.7ms",
        "ATE m",
        "Poses",
    ]
    for sequence_name in SEQUENCES:
        sequence_rows = sorted(
            [row for row in rows if row.get("sequence_name") == sequence_name],
            key=sort_key,
            reverse=True,
        )
        lines.extend([f"## {sequence_name}", ""])
        body = []
        for row in sequence_rows:
            body.append(
                [
                    str(row.get("method", "")),
                    str(row.get("source", "")),
                    fmt(row.get("frames"), 0),
                    fmt(row.get("mean_fps")),
                    fmt(row.get("p95_safe_fps")),
                    fmt(row.get("p99_safe_fps")),
                    fmt(row.get("mean_ms")),
                    fmt(row.get("median_ms")),
                    fmt(row.get("p95_ms")),
                    fmt(row.get("p99_ms")),
                    fmt(100.0 * row["over_30hz_ratio"], 1) + "%" if isinstance(row.get("over_30hz_ratio"), float) else "n/a",
                    fmt(100.0 * row["over_60hz_ratio"], 1) + "%" if isinstance(row.get("over_60hz_ratio"), float) else "n/a",
                    fmt(row.get("ate_rmse_m")),
                    fmt(row.get("poses"), 0),
                ]
            )
        lines.extend(md_table(headers, body))
        lines.append("")

    lines.extend(
        [
            "## Lecture",
            "",
        "- `FPS moyen` vaut `1000 / latence moyenne par image`; c'est le debit moyen maximal si les images arrivent sans attente artificielle.",
        "- `FPS p95` vaut `1000 / p95(latence)`: c'est plus conservateur, car 95% des images sont plus rapides que cette latence.",
        "- Les methodes sont triees par `FPS p95` lorsque les temps image par image sont disponibles, puis par `FPS moyen` en repli.",
        "- Les colonnes `>33.3ms` et `>16.7ms` indiquent la proportion d'images qui depassent respectivement un budget 30 Hz et 60 Hz.",
            "- Les lignes `log mean/median only` sont moins riches: elles viennent des statistiques imprimees par le binaire, mais les percentiles par image n'etaient pas sauvegardes dans ces anciens runs.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-root", type=Path, default=Path("methods"))
    parser.add_argument("--output-md", type=Path, default=Path("methods/comparison/tum_rgbd_live_fps_summary.md"))
    parser.add_argument("--output-json", type=Path, default=Path("methods/comparison/tum_rgbd_live_fps_summary.json"))
    args = parser.parse_args()

    rows = collect(args.methods_root)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(rows, args.output_md)
    args.output_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_md)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
