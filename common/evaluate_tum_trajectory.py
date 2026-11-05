#!/usr/bin/env python3
"""Evaluate TUM-format trajectories against TUM ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


PoseMap = Dict[float, np.ndarray]


def read_tum_poses(path: Path) -> PoseMap:
    poses: PoseMap = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        ts = float(parts[0])
        tx, ty, tz = map(float, parts[1:4])
        qx, qy, qz, qw = map(float, parts[4:8])
        poses[ts] = pose_from_tq(np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
    return poses


def read_association_timestamps(path: Path) -> List[float]:
    stamps: List[float] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts:
            stamps.append(float(parts[0]))
    return stamps


def read_vdo_matrices(path: Path, associations: Path, invert: bool) -> PoseMap:
    stamps = read_association_timestamps(associations)
    poses: PoseMap = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 17:
            continue
        index = int(float(parts[0]))
        if index < 0 or index >= len(stamps):
            continue
        values = np.array([float(v) for v in parts[1:17]], dtype=float)
        pose = values.reshape((4, 4))
        if invert:
            pose = np.linalg.inv(pose)
        poses[stamps[index]] = pose
    return poses


def read_vins_csv(path: Path) -> PoseMap:
    poses: PoseMap = {}
    for raw in path.read_text().splitlines():
        line = raw.strip().rstrip(",")
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.split(",") if p]
        if len(parts) < 8:
            continue
        ts = float(parts[0]) / 1e9
        tx, ty, tz = map(float, parts[1:4])
        qw, qx, qy, qz = map(float, parts[4:8])
        poses[ts] = pose_from_tq(np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]))
    return poses


def pose_from_tq(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    q = q.astype(float)
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError("zero quaternion")
    qx, qy, qz, qw = q / norm
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


def associate_by_time(first: Iterable[float], second: Iterable[float], max_difference: float) -> List[Tuple[float, float]]:
    first_list = sorted(first)
    second_list = sorted(second)
    potential = []
    for a in first_list:
        for b in second_list:
            diff = abs(a - b)
            if diff < max_difference:
                potential.append((diff, a, b))
            elif b > a + max_difference:
                break
    potential.sort()
    used_a = set()
    used_b = set()
    matches = []
    for _, a, b in potential:
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        matches.append((a, b))
    matches.sort()
    return matches


def se3_align(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1
        rot = vt.T @ u.T
    trans = dst_mean - rot @ src_mean
    return rot, trans


def rotation_angle_deg(rot: np.ndarray) -> float:
    value = (np.trace(rot) - 1.0) / 2.0
    value = min(1.0, max(-1.0, float(value)))
    return math.degrees(math.acos(value))


def summarize(values: np.ndarray) -> Dict[str, float]:
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


def evaluate(gt: PoseMap, est: PoseMap, max_difference: float) -> Dict[str, object]:
    matches = associate_by_time(est.keys(), gt.keys(), max_difference)
    if len(matches) < 2:
        return {"matched_pairs": len(matches), "error": "not enough timestamp matches"}

    est_points = np.array([est[e][:3, 3] for e, _ in matches])
    gt_points = np.array([gt[g][:3, 3] for _, g in matches])
    align_rot, align_trans = se3_align(est_points, gt_points)
    aligned_est_points = (align_rot @ est_points.T).T + align_trans
    ate = np.linalg.norm(aligned_est_points - gt_points, axis=1)

    aligned_est: List[np.ndarray] = []
    gt_matched: List[np.ndarray] = []
    align_tf = np.eye(4)
    align_tf[:3, :3] = align_rot
    align_tf[:3, 3] = align_trans
    for e, g in matches:
        aligned_est.append(align_tf @ est[e])
        gt_matched.append(gt[g])

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
        "ate_m": summarize(ate),
        "rpe_trans_m": summarize(np.array(rpe_trans)),
        "rpe_rot_deg": summarize(np.array(rpe_rot)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", required=True, type=Path)
    parser.add_argument("--estimate", required=True, type=Path)
    parser.add_argument("--format", choices=("tum", "vdo", "vins"), default="tum")
    parser.add_argument("--associations", type=Path)
    parser.add_argument("--invert-estimate", action="store_true")
    parser.add_argument("--max-difference", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    gt = read_tum_poses(args.groundtruth)
    if args.format == "tum":
        est = read_tum_poses(args.estimate)
    elif args.format == "vins":
        est = read_vins_csv(args.estimate)
    else:
        if args.associations is None:
            parser.error("--associations is required for --format vdo")
        est = read_vdo_matrices(args.estimate, args.associations, args.invert_estimate)

    result = evaluate(gt, est, args.max_difference)
    result["groundtruth"] = str(args.groundtruth)
    result["estimate"] = str(args.estimate)
    result["format"] = args.format
    result["inverted_estimate"] = bool(args.invert_estimate)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
