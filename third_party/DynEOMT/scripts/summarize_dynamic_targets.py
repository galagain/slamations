#!/usr/bin/env python3
"""Validate and summarize DynEoMT schema-v2 pseudo-label files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


VALID_LABELS = {"dynamic", "static", "uncertain"}


def validate_payload(payload: dict, path: Path) -> list[str]:
    errors = []
    required = {
        "schema_version",
        "dataset",
        "split",
        "sequence_id",
        "frame_count",
        "transition_count",
        "frames",
    }
    missing = required - payload.keys()
    if missing:
        return [f"{path}: missing fields {sorted(missing)}"]
    if payload["schema_version"] != 2:
        errors.append(f"{path}: expected schema_version=2")
    if payload["transition_count"] != max(payload["frame_count"] - 1, 0):
        errors.append(f"{path}: transition count does not match frame count")
    if len(payload["frames"]) != payload["transition_count"]:
        errors.append(f"{path}: frames list does not match transition count")
    for frame in payload["frames"]:
        for record in frame.get("masks", []):
            if record.get("label") not in VALID_LABELS:
                errors.append(f"{path}: invalid label {record.get('label')!r}")
            probability = record.get("probability")
            if probability is None or not 0.0 <= float(probability) <= 1.0:
                errors.append(f"{path}: probability outside [0, 1]")
    return errors


def summarize(root: Path) -> tuple[dict, list[str]]:
    summary: dict[tuple[str, str], dict] = defaultdict(
        lambda: {
            "sequences": 0,
            "frames": 0,
            "transitions": 0,
            "dynamic": 0,
            "static": 0,
            "uncertain": 0,
        }
    )
    errors = []
    for path in sorted(root.glob("*/*/targets.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue
        errors.extend(validate_payload(payload, path))
        key = (str(payload.get("dataset", "unknown")), str(payload.get("split", "unknown")))
        row = summary[key]
        row["sequences"] += 1
        row["frames"] += int(payload.get("frame_count", 0))
        row["transitions"] += int(payload.get("transition_count", 0))
        for frame in payload.get("frames", []):
            for record in frame.get("masks", []):
                label = record.get("label")
                if label in VALID_LABELS:
                    row[label] += 1
    return dict(summary), errors


def print_table(summary: dict) -> None:
    header = (
        f"{'dataset':<12} {'split':<8} {'seq':>6} {'frames':>9} "
        f"{'dynamic':>11} {'static':>11} {'uncertain':>11} {'labeled %':>10}"
    )
    print(header)
    print("-" * len(header))
    for (dataset, split), row in sorted(summary.items()):
        total = row["dynamic"] + row["static"] + row["uncertain"]
        labeled = row["dynamic"] + row["static"]
        labeled_ratio = 100.0 * labeled / total if total else 0.0
        print(
            f"{dataset:<12} {split:<8} {row['sequences']:>6,d} "
            f"{row['frames']:>9,d} {row['dynamic']:>11,d} "
            f"{row['static']:>11,d} {row['uncertain']:>11,d} "
            f"{labeled_ratio:>9.2f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/dynamic_targets/labels")
    )
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--strict", action="store_true", help="Return a non-zero status on validation errors"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, errors = summarize(args.root)
    print_table(summary)
    print(f"\nValidation errors: {len(errors)}")
    for error in errors[:20]:
        print(f"  {error}")
    if args.json_output:
        payload = {
            "splits": {
                f"{dataset}/{split}": row
                for (dataset, split), row in sorted(summary.items())
            },
            "errors": errors,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.strict and errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
