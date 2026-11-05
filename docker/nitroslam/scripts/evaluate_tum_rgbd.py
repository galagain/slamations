#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def pose_from_tq(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    qx, qy, qz, qw = q / np.linalg.norm(q)
    rot = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = t
    return pose


def read_tum(path: Path) -> dict[float, np.ndarray]:
    poses: dict[float, np.ndarray] = {}
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            continue
        ts = float(fields[0])
        t = np.array([float(x) for x in fields[1:4]], dtype=float)
        q = np.array([float(x) for x in fields[4:8]], dtype=float)
        poses[ts] = pose_from_tq(t, q)
    return poses


def associate(first: list[float], second: list[float], max_difference: float) -> list[tuple[float, float]]:
    potential: list[tuple[float, float, float]] = []
    second_sorted = sorted(second)
    for a in sorted(first):
        for b in second_sorted:
            diff = abs(a - b)
            if diff < max_difference:
                potential.append((diff, a, b))
            elif b > a + max_difference:
                break
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


def se3_align(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    h = (src - src_mean).T @ (dst - dst_mean)
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = dst_mean - rot @ src_mean
    return rot, trans


def rotation_angle_deg(rot: np.ndarray) -> float:
    value = min(1.0, max(-1.0, float((np.trace(rot) - 1.0) / 2.0)))
    return math.degrees(math.acos(value))


def stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "rmse": float(np.sqrt(np.mean(values * values))),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate(gt: dict[float, np.ndarray], est: dict[float, np.ndarray], max_difference: float) -> dict[str, object]:
    matches = associate(list(est), list(gt), max_difference)
    if len(matches) < 2:
        return {"matched_pairs": len(matches), "error": "not enough timestamp matches"}

    est_points = np.array([est[e][:3, 3] for e, _ in matches])
    gt_points = np.array([gt[g][:3, 3] for _, g in matches])
    rot, trans = se3_align(est_points, gt_points)
    aligned_points = (rot @ est_points.T).T + trans
    ate = np.linalg.norm(aligned_points - gt_points, axis=1)

    align_tf = np.eye(4)
    align_tf[:3, :3] = rot
    align_tf[:3, 3] = trans
    aligned_est = [align_tf @ est[e] for e, _ in matches]
    gt_matched = [gt[g] for _, g in matches]

    rpe_trans = []
    rpe_rot = []
    for i in range(len(matches) - 1):
        est_delta = np.linalg.inv(aligned_est[i]) @ aligned_est[i + 1]
        gt_delta = np.linalg.inv(gt_matched[i]) @ gt_matched[i + 1]
        err = np.linalg.inv(gt_delta) @ est_delta
        rpe_trans.append(float(np.linalg.norm(err[:3, 3])))
        rpe_rot.append(rotation_angle_deg(err[:3, :3]))

    return {
        "matched_pairs": len(matches),
        "ate_m": stats(ate),
        "rpe_trans_m": stats(np.array(rpe_trans)),
        "rpe_rot_deg": stats(np.array(rpe_rot)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--max-difference", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(read_tum(args.groundtruth), read_tum(args.estimate), args.max_difference)
    result["groundtruth"] = str(args.groundtruth)
    result["estimate"] = str(args.estimate)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
