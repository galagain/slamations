"""Camera-compensated, mask-level dynamicity pseudo-label estimation."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .datasets import MaskRecord, build_adapter


class FlowStore:
    def __init__(self, sequence_root: Path) -> None:
        self.root = sequence_root
        metadata_path = sequence_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing MemFlow metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        frames = metadata["frames"]
        self.pairs = {
            (frames[index], frames[index + 1]): sequence_root
            / "npz"
            / f"flow_{index:04d}_to_{index + 1:04d}.npz"
            for index in range(len(frames) - 1)
        }

    def load(
        self, from_frame: str, to_frame: str, shape: tuple[int, int]
    ) -> np.ndarray:
        path = self.pairs.get((from_frame, to_frame))
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"No MemFlow transition {from_frame} -> {to_frame} in {self.root}"
            )
        flow = np.load(path)["flow"].astype(np.float32)
        target_height, target_width = shape
        source_height, source_width = flow.shape[:2]
        if (source_height, source_width) != (target_height, target_width):
            flow = cv2.resize(
                flow, (target_width, target_height), interpolation=cv2.INTER_LINEAR
            )
            flow[..., 0] *= target_width / source_width
            flow[..., 1] *= target_height / source_height
        return flow


def forward_backward_valid(
    flow: np.ndarray,
    backward: np.ndarray | None,
    absolute_tolerance: float = 1.5,
    relative_tolerance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = flow.shape[:2]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    finite = np.isfinite(flow).all(axis=2)
    if backward is None:
        return finite, np.zeros((height, width), dtype=np.float32)
    map_x = xx + flow[..., 0]
    map_y = yy + flow[..., 1]
    inside = (
        (map_x >= 0)
        & (map_x <= width - 1)
        & (map_y >= 0)
        & (map_y <= height - 1)
    )
    back_x = cv2.remap(
        backward[..., 0],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    back_y = cv2.remap(
        backward[..., 1],
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=np.nan,
    )
    error = np.hypot(flow[..., 0] + back_x, flow[..., 1] + back_y)
    tolerance = absolute_tolerance + relative_tolerance * np.linalg.norm(flow, axis=2)
    valid = finite & inside & np.isfinite(error) & (error <= tolerance)
    return valid, np.nan_to_num(error, nan=1e6, posinf=1e6, neginf=1e6)


def _apply_model(
    model: dict, x: np.ndarray, y: np.ndarray, width: int, height: int
) -> np.ndarray:
    if model["kind"] == "translation":
        return np.broadcast_to(
            np.asarray(model["offset"], dtype=np.float64), (x.size, 2)
        ).copy()
    if model["kind"] == "affine":
        matrix = np.asarray(model["matrix"], dtype=np.float64)
        points = np.stack([x, y, np.ones_like(x)], axis=1)
        return points @ matrix.T - np.stack([x, y], axis=1)
    if model["kind"] == "homography":
        matrix = np.asarray(model["matrix"], dtype=np.float64)
        points = np.stack([x, y, np.ones_like(x)], axis=1)
        projected = points @ matrix.T
        denominator = projected[:, 2:3]
        denominator = np.where(
            np.abs(denominator) < 1e-8, np.sign(denominator) * 1e-8 + 1e-8, denominator
        )
        return projected[:, :2] / denominator - np.stack([x, y], axis=1)
    x_normalized = 2.0 * x / max(width - 1, 1) - 1.0
    y_normalized = 2.0 * y / max(height - 1, 1) - 1.0
    design = np.stack(
        [
            np.ones_like(x_normalized),
            x_normalized,
            y_normalized,
            x_normalized**2,
            x_normalized * y_normalized,
            y_normalized**2,
        ],
        axis=1,
    )
    return design @ np.asarray(model["coefficients"], dtype=np.float64)


def _fit_quadratic(
    x: np.ndarray, y: np.ndarray, flow: np.ndarray, width: int, height: int
) -> dict | None:
    x_normalized = 2.0 * x / max(width - 1, 1) - 1.0
    y_normalized = 2.0 * y / max(height - 1, 1) - 1.0
    design = np.stack(
        [
            np.ones_like(x_normalized),
            x_normalized,
            y_normalized,
            x_normalized**2,
            x_normalized * y_normalized,
            y_normalized**2,
        ],
        axis=1,
    )
    keep = np.ones(x.size, dtype=bool)
    coefficients = None
    for _ in range(5):
        if keep.sum() < 30:
            return None
        coefficients = np.linalg.lstsq(design[keep], flow[keep], rcond=None)[0]
        residual = np.linalg.norm(flow - design @ coefficients, axis=1)
        median = float(np.median(residual[keep]))
        sigma = 1.4826 * float(np.median(np.abs(residual[keep] - median)))
        updated = residual <= median + 2.75 * max(sigma, 0.25)
        if np.array_equal(updated, keep):
            break
        keep = updated
    return {
        "kind": "quadratic",
        "coefficients": coefficients.tolist(),
        "parameters": 12,
    }


def _fit_models(
    x: np.ndarray, y: np.ndarray, flow: np.ndarray, width: int, height: int
) -> list[dict]:
    source = np.stack([x, y], axis=1).astype(np.float32)
    target = (source + flow).astype(np.float32)
    models = [
        {
            "kind": "translation",
            "offset": np.median(flow, axis=0).tolist(),
            "parameters": 2,
        }
    ]
    cv2.setRNGSeed(1234)
    affine, _ = cv2.estimateAffine2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.5,
        maxIters=4000,
        confidence=0.995,
        refineIters=20,
    )
    if affine is not None and np.isfinite(affine).all():
        models.append({"kind": "affine", "matrix": affine.tolist(), "parameters": 6})
    homography, _ = cv2.findHomography(
        source, target, cv2.RANSAC, 2.5, maxIters=5000, confidence=0.995
    )
    if homography is not None and np.isfinite(homography).all():
        models.append(
            {"kind": "homography", "matrix": homography.tolist(), "parameters": 8}
        )
    quadratic = _fit_quadratic(x, y, flow, width, height)
    if quadratic is not None:
        models.append(quadratic)
    return models


def fit_camera_flow(
    flow: np.ndarray,
    valid: np.ndarray,
    preferred_background: np.ndarray,
    tail_weight: float = 0.20,
    parameter_penalty: float = 0.035,
    residual_sigma_multiplier: float = 3.0,
    residual_sigma_floor: float = 0.20,
    residual_pixel_floor: float = 1.25,
    residual_diagonal_fraction: float = 0.0015,
) -> tuple[np.ndarray, dict]:
    height, width = flow.shape[:2]
    reference = valid & preferred_background
    source = "unannotated_background"
    if reference.sum() < max(1000, int(0.08 * height * width)):
        reference = valid
        source = "robust_all_pixels"
    yy, xx = np.nonzero(reference)
    if xx.size < 50:
        offset = np.nanmedian(flow[valid], axis=0) if valid.any() else np.zeros(2)
        return np.broadcast_to(offset, flow.shape).astype(np.float32).copy(), {
            "model": "translation_fallback",
            "source": source,
            "threshold_px": max(
                1.5,
                residual_pixel_floor,
                residual_diagonal_fraction * math.hypot(width, height),
            ),
            "noise_median_px": 0.0,
            "noise_sigma_px": 0.0,
            "inlier_fraction": 0.0,
            "candidate_scores": {},
        }

    if xx.size > 24000:
        selected = np.random.default_rng(1234).choice(xx.size, 24000, replace=False)
        xx, yy = xx[selected], yy[selected]
    sampled_flow = flow[yy, xx].astype(np.float64)
    models = _fit_models(
        xx.astype(np.float64), yy.astype(np.float64), sampled_flow, width, height
    )
    candidate_scores = {}
    for model in models:
        prediction = _apply_model(model, xx, yy, width, height)
        residual = np.linalg.norm(sampled_flow - prediction, axis=1)
        robust_score = float(
            np.median(residual) + tail_weight * np.quantile(residual, 0.90)
        )
        model["selection_score"] = robust_score + parameter_penalty * int(
            model["parameters"]
        )
        candidate_scores[model["kind"]] = round(model["selection_score"], 4)
    model = min(models, key=lambda candidate: candidate["selection_score"])

    dense_y, dense_x = np.mgrid[:height, :width]
    camera = _apply_model(
        model, dense_x.reshape(-1), dense_y.reshape(-1), width, height
    ).reshape(height, width, 2)
    residual = np.linalg.norm(
        sampled_flow - _apply_model(model, xx, yy, width, height), axis=1
    )
    noise_median = float(np.median(residual))
    noise_sigma = 1.4826 * float(np.median(np.abs(residual - noise_median)))
    threshold = max(
        noise_median
        + residual_sigma_multiplier * max(noise_sigma, residual_sigma_floor),
        residual_pixel_floor,
        residual_diagonal_fraction * math.hypot(width, height),
    )
    return camera.astype(np.float32), {
        "model": model["kind"],
        "source": source,
        "threshold_px": round(float(threshold), 4),
        "noise_median_px": round(noise_median, 4),
        "noise_sigma_px": round(noise_sigma, 4),
        "inlier_fraction": round(float(np.mean(residual <= threshold)), 4),
        "candidate_scores": candidate_scores,
    }


def _centroid(mask: np.ndarray) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    return np.asarray([xx.mean(), yy.mean()], dtype=np.float32)


def assign_vspw_tracks(
    current: list[MaskRecord],
    previous: list[MaskRecord] | None,
    flow: np.ndarray | None,
    next_ids: dict[int, int],
) -> None:
    if not previous or flow is None:
        for record in current:
            next_ids[record.category_id] += 1
            record.track_id = f"{record.category_id}:{next_ids[record.category_id]}"
        return
    diagonal = math.hypot(*flow.shape[:2])
    for category_id in sorted({record.category_id for record in current}):
        old = [record for record in previous if record.category_id == category_id]
        new = [record for record in current if record.category_id == category_id]
        if not old:
            for record in new:
                next_ids[category_id] += 1
                record.track_id = f"{category_id}:{next_ids[category_id]}"
            continue
        cost = np.full((len(old), len(new)), 100.0, dtype=np.float32)
        for old_index, old_record in enumerate(old):
            motion = flow[old_record.mask]
            delta = np.median(motion, axis=0) if motion.size else np.zeros(2)
            predicted_centroid = _centroid(old_record.mask) + delta
            for new_index, new_record in enumerate(new):
                distance = np.linalg.norm(predicted_centroid - _centroid(new_record.mask))
                distance /= max(0.08 * diagonal, 1.0)
                area_change = abs(
                    math.log(max(new_record.area, 1) / max(old_record.area, 1))
                )
                cost[old_index, new_index] = distance + 0.25 * area_change
        rows, columns = linear_sum_assignment(cost)
        assigned = set()
        for row, column in zip(rows, columns):
            if cost[row, column] <= 1.75:
                new[column].track_id = old[row].track_id
                assigned.add(column)
        for index, record in enumerate(new):
            if index not in assigned:
                next_ids[category_id] += 1
                record.track_id = f"{category_id}:{next_ids[category_id]}"


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    area = int(mask.sum())
    radius = int(np.clip(round(math.sqrt(max(area, 1)) / 90.0), 1, 6))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    return mask if eroded.sum() < max(20, int(0.20 * area)) else eroded


def mask_motion_record(
    record: MaskRecord,
    residual_flow: np.ndarray,
    valid: np.ndarray,
    fb_error: np.ndarray,
    threshold: float,
    camera_confidence: float,
) -> dict:
    interior = _erode_mask(record.mask)
    usable = interior & valid
    valid_pixels = int(usable.sum())
    valid_ratio = valid_pixels / max(int(interior.sum()), 1)
    result = {
        "source_id": record.source_id,
        "track_id": record.track_id,
        "category_id": record.category_id,
        "category_name": record.category_name,
        "isthing": record.isthing,
        "area": record.area,
        "valid_pixels": valid_pixels,
        "valid_ratio": round(valid_ratio, 4),
    }
    if valid_pixels < max(20, int(0.03 * record.area)):
        return {
            **result,
            "probability_raw": 0.5,
            "probability": 0.5,
            "label": "uncertain",
            "confidence": 0.0,
            "residual_median_px": None,
            "residual_p75_px": None,
            "residual_p90_px": None,
            "moving_fraction": None,
            "direction_coherence": None,
            "fb_error_median_px": None,
        }
    vectors = residual_flow[usable]
    magnitude = np.linalg.norm(vectors, axis=1)
    median = float(np.median(magnitude))
    p75 = float(np.quantile(magnitude, 0.75))
    p90 = float(np.quantile(magnitude, 0.90))
    moving_fraction = float(np.mean(magnitude >= threshold))
    coherence = float(
        np.linalg.norm(vectors.mean(axis=0)) / max(float(np.mean(magnitude)), 1e-6)
    )
    evidence = 0.55 * (median / threshold - 1.0)
    evidence += 0.30 * (p75 / (1.35 * threshold) - 1.0)
    evidence += 0.80 * (moving_fraction - 0.35)
    evidence += 0.08 * max(coherence - 0.5, 0.0)
    raw_probability = 1.0 / (
        1.0 + math.exp(-2.5 * float(np.clip(evidence, -8.0, 8.0)))
    )
    support = min(1.0, math.sqrt(valid_pixels / 180.0))
    support *= min(1.0, valid_ratio / 0.45)
    confidence = float(np.clip(support * camera_confidence, 0.0, 1.0))
    probability = 0.5 + (raw_probability - 0.5) * confidence
    return {
        **result,
        "probability_raw": round(raw_probability, 5),
        "probability": round(probability, 5),
        "label": "uncertain",
        "confidence": round(confidence, 5),
        "residual_median_px": round(median, 4),
        "residual_p75_px": round(p75, 4),
        "residual_p90_px": round(p90, 4),
        "moving_fraction": round(moving_fraction, 4),
        "direction_coherence": round(coherence, 4),
        "fb_error_median_px": round(float(np.median(fb_error[usable])), 4),
    }


def temporal_smoothing(
    frames: list[dict],
    dynamic_threshold: float = 0.70,
    static_threshold: float = 0.30,
) -> None:
    tracks: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for record in frame["masks"]:
            tracks[record["track_id"]].append((frame_index, record))
    for observations in tracks.values():
        source_probabilities = [record["probability"] for _, record in observations]
        for position, (frame_index, record) in enumerate(observations):
            values = [(source_probabilities[position], 0.60)]
            if position > 0 and frame_index - observations[position - 1][0] <= 2:
                values.append((source_probabilities[position - 1], 0.20))
            if (
                position + 1 < len(observations)
                and observations[position + 1][0] - frame_index <= 2
            ):
                values.append((source_probabilities[position + 1], 0.20))
            probability = sum(value * weight for value, weight in values) / sum(
                weight for _, weight in values
            )
            record["probability"] = round(float(probability), 5)
            if probability >= dynamic_threshold:
                record["label"] = "dynamic"
            elif probability <= static_threshold:
                record["label"] = "static"
            else:
                record["label"] = "uncertain"


def build_sequence_targets(
    dataset: str,
    split: str,
    sequence_id: str,
    data_root: Path,
    forward_dir: Path,
    backward_dir: Path,
    output_root: Path,
    save_artifacts: bool = False,
) -> Path:
    adapter = build_adapter(dataset, split, sequence_id, data_root)
    output_dir = output_root / adapter.dataset / adapter.sequence_id
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "artifacts"
    if save_artifacts:
        artifact_dir.mkdir(exist_ok=True)
    forward = FlowStore(forward_dir)
    backward = FlowStore(backward_dir)
    frames = []
    previous_masks = None
    previous_flow = None
    next_track_ids: dict[int, int] = defaultdict(int)

    for frame_index in range(len(adapter.frame_paths) - 1):
        image = cv2.imread(str(adapter.frame_paths[frame_index]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read {adapter.frame_paths[frame_index]}")
        height, width = image.shape[:2]
        from_frame = adapter.frame_keys[frame_index]
        to_frame = adapter.frame_keys[frame_index + 1]
        flow = forward.load(from_frame, to_frame, (height, width))
        backward_flow = backward.load(to_frame, from_frame, (height, width))
        valid, fb_error = forward_backward_valid(flow, backward_flow)
        masks = adapter.load_masks(frame_index)
        if adapter.dataset == "vspw":
            assign_vspw_tracks(masks, previous_masks, previous_flow, next_track_ids)
        union = np.zeros((height, width), dtype=bool)
        for record in masks:
            union |= record.mask
        camera_flow, camera_info = fit_camera_flow(flow, valid, ~union)
        residual = flow - camera_flow
        threshold = float(camera_info["threshold_px"])
        camera_confidence = float(
            np.clip(camera_info["inlier_fraction"] / 0.80, 0.15, 1.0)
        )
        frames.append(
            {
                "index": frame_index,
                "frame": from_frame,
                "next_frame": to_frame,
                "camera": camera_info,
                "masks": [
                    mask_motion_record(
                        record,
                        residual,
                        valid,
                        fb_error,
                        threshold,
                        camera_confidence,
                    )
                    for record in masks
                ],
            }
        )
        if save_artifacts:
            np.savez_compressed(
                artifact_dir / f"{frame_index:05d}.npz",
                camera_flow=camera_flow.astype(np.float16),
                residual_magnitude=np.linalg.norm(residual, axis=2).astype(np.float16),
                valid=valid.astype(np.uint8),
            )
        previous_masks = masks
        previous_flow = flow

    temporal_smoothing(frames)
    counts: dict[str, int] = defaultdict(int)
    for frame in frames:
        for record in frame["masks"]:
            counts[record["label"]] += 1
    payload = {
        "schema_version": 2,
        "method": (
            "bidirectional MemFlow, forward-backward filtering, adaptive robust "
            "camera field, and temporally aggregated mask residuals"
        ),
        "dataset": adapter.dataset,
        "split": adapter.split,
        "sequence_id": adapter.sequence_id,
        "frame_count": len(adapter.frame_paths),
        "transition_count": len(frames),
        "labels": {"dynamic": 1, "static": 0, "uncertain": -1},
        "decision_thresholds": {"dynamic": 0.70, "static": 0.30},
        "summary": dict(counts),
        "frames": frames,
    }
    output_path = output_dir / "targets.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)
    return output_path
