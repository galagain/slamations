#!/usr/bin/env python3
"""Create TUM RGB-D association files.

Output format:
  rgb_timestamp rgb/path depth_timestamp depth/path
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_file_list(path: Path) -> Dict[float, List[str]]:
    data = {}  # type: Dict[float, List[str]]
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        data[float(parts[0])] = parts[1:]
    return data


def associate(
    first: Dict[float, List[str]],
    second: Dict[float, List[str]],
    offset: float,
    max_difference: float,
) -> List[Tuple[float, float]]:
    potential = [
        (abs(a - (b + offset)), a, b)
        for a in first
        for b in second
        if abs(a - (b + offset)) < max_difference
    ]
    potential.sort()
    matches = []  # type: List[Tuple[float, float]]
    used_first = set()  # type: Set[float]
    used_second = set()  # type: Set[float]
    for _, a, b in potential:
        if a in used_first or b in used_second:
            continue
        used_first.add(a)
        used_second.add(b)
        matches.append((a, b))
    matches.sort()
    return matches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_txt", type=Path)
    parser.add_argument("depth_txt", type=Path)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--max-difference", type=float, default=0.02)
    args = parser.parse_args()

    rgb = read_file_list(args.rgb_txt)
    depth = read_file_list(args.depth_txt)
    for rgb_t, depth_t in associate(rgb, depth, args.offset, args.max_difference):
        print(
            f"{rgb_t:.6f} {' '.join(rgb[rgb_t])} "
            f"{depth_t:.6f} {' '.join(depth[depth_t])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
