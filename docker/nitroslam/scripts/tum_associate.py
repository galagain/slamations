#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def read_file_list(path: Path) -> dict[float, str]:
    rows: dict[float, str] = {}
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows[float(fields[0])] = fields[1]
    return rows


def associate(first: dict[float, str], second: dict[float, str], max_difference: float) -> list[tuple[float, float]]:
    potential: list[tuple[float, float, float]] = []
    for a in first:
        for b in second:
            diff = abs(a - b)
            if diff < max_difference:
                potential.append((diff, a, b))
    potential.sort()

    used_a: set[float] = set()
    used_b: set[float] = set()
    matches: list[tuple[float, float]] = []
    for _, a, b in potential:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        matches.append((a, b))
    return sorted(matches)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb", type=Path)
    parser.add_argument("depth", type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02)
    args = parser.parse_args()

    rgb = read_file_list(args.rgb)
    depth = read_file_list(args.depth)
    for a, b in associate(rgb, depth, args.max_difference):
        print(f"{a:.6f} {rgb[a]} {b:.6f} {depth[b]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
