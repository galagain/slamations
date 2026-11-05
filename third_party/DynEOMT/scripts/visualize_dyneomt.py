#!/usr/bin/env python3
"""Render DynEoMT predictions for one sequence from each supported dataset."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dynamic_targets.datasets import build_adapter


ROOT = Path(__file__).resolve().parents[1]
DATASET_NAMES = ("vipseg", "ovis", "ytvis2022", "vspw")
BACKBONE_CONFIGS = {
    "vit-small": "dyneomt_online_vits.yaml",
    "vit-base": "dyneomt_online_vitb.yaml",
    "vit-large": "dyneomt_online_vitl.yaml",
}
DYNAMIC_COLOR = (55, 65, 235)
STATIC_COLOR = (75, 190, 80)
UNCERTAIN_COLOR = (150, 150, 150)
PANEL_BACKGROUND = (16, 19, 24)

DATASETS = {
    "vipseg": {
        "config_dataset": "VIPSeg",
        "mapper": "vps",
        "split": "val",
        "sequence": "516_5bqIhLCjTzE",
    },
    "ovis": {
        "config_dataset": "ovis",
        "mapper": "vis",
        "split": "valid",
        "sequence": "2ca20519",
    },
    "ytvis2022": {
        "config_dataset": "ytvis22",
        "mapper": "vis",
        "split": "valid",
        "sequence": "c289d17243",
    },
    "vspw": {
        "config_dataset": "VSPW",
        "mapper": "vss",
        "split": "val",
        "sequence": "2368_x3rgdHiDN7U",
    },
}


@dataclass
class Runtime:
    DetectionCheckpointer: type
    DatasetCatalog: object
    MetadataCatalog: object
    Trainer: type
    add_deeplab_config: object
    add_videomt_config: object
    get_cfg: object
    mappers: dict[str, type]


@dataclass
class DrawRecord:
    mask: np.ndarray
    name: str
    object_id: str
    color: tuple[int, int, int]
    score: float | None = None
    dynamic_probability: float | None = None
    dynamic_label: str = "uncertain"

    @property
    def area(self) -> int:
        return int(self.mask.sum())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_name",
        nargs="?",
        choices=("all", *DATASET_NAMES),
        help="Dataset to render; defaults to all",
    )
    parser.add_argument(
        "sequence_name",
        nargs="?",
        help="Sequence ID when one positional dataset is selected",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_option",
        choices=("all", *DATASET_NAMES),
        help="Option-style alias for the positional dataset argument",
    )
    parser.add_argument(
        "--mode",
        choices=("grid", "prediction", "clean"),
        default="grid",
        help=(
            "grid: 2x2 prediction/ground truth; prediction: predicted outputs only; "
            "clean: three separate original/tracking/dynamicity videos without text"
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=tuple(BACKBONE_CONFIGS),
        default="vit-large",
        help="DynEoMT backbone/configuration variant",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        help=(
            "Sequence ID for one dataset, or DATASET=SEQUENCE when --dataset all. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        help=(
            "Checkpoint path for one dataset, or DATASET=PATH when --dataset all. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Raw dataset root; automatically discovered when omitted",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        help="Dynamic target root; automatically discovered when omitted",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Root containing <dataset>/model_final.pth; automatically discovered",
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "visualizations"
    )
    parser.add_argument("--output", type=Path, help="Exact output path for one dataset")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--min-area", type=int, default=90)
    parser.add_argument("--max-labels", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--cell-width", type=int, default=720)
    parser.add_argument("--cell-height", type=int, default=405)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve datasets, sequences, targets, and checkpoints without inference",
    )
    return parser.parse_args()


def resolve_data_root(
    explicit_root: Path | None, selected_datasets: list[str]
) -> Path:
    markers = {
        "vipseg": Path("VIPSeg/VIPSeg_720P"),
        "ovis": Path("ovis"),
        "ytvis2022": Path("ytvis_2022"),
        "vspw": Path("VSPW_480p"),
    }
    candidates = (
        [explicit_root]
        if explicit_root is not None
        else [ROOT / "datasets", ROOT.parent / "data"]
    )
    for candidate in candidates:
        if candidate is not None and all(
            (candidate / markers[dataset]).is_dir() for dataset in selected_datasets
        ):
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        f"Cannot locate raw data for {selected_datasets}; searched: {searched}"
    )


def resolve_checkpoint(
    dataset: str,
    backbone: str,
    custom_checkpoint: str | None,
    checkpoint_root: Path | None,
) -> Path:
    if custom_checkpoint is not None:
        path = Path(custom_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Custom checkpoint does not exist: {path}")
        return path

    roots = (
        [checkpoint_root]
        if checkpoint_root is not None
        else [
            ROOT / "outputs" / "dyneomt_models",
            ROOT / "outputs" / "dyneomt",
            ROOT / "checkpoints" / "dyneomt",
            ROOT.parent / "outputs" / "dyneomt_models",
            ROOT.parent / "outputs" / "dyneomt",
        ]
    )
    candidates = [root / dataset / backbone / "model_final.pth" for root in roots]
    if backbone == "vit-large":
        candidates.extend(root / dataset / "model_final.pth" for root in roots)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Cannot find the trained checkpoint for "
        f"{dataset}. Searched: {', '.join(str(path) for path in candidates)}. "
        "Use --checkpoint for a custom model."
    )


def resolve_target_path(
    dataset: str,
    sequence_id: str,
    target_root: Path | None,
) -> Path:
    roots = (
        [target_root]
        if target_root is not None
        else [
            ROOT / "outputs" / "dynamic_targets" / "labels",
            ROOT / "outputs" / "dynamic_general_full" / "labels",
            ROOT.parent / "outputs" / "dynamic_targets" / "labels",
            ROOT.parent / "outputs" / "dynamic_general_full" / "labels",
        ]
    )
    candidates = [root / dataset / sequence_id / "targets.json" for root in roots]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Cannot find dynamic targets for "
        f"{dataset}/{sequence_id}. Searched: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def import_runtime(data_root: Path) -> Runtime:
    os.environ["DETECTRON2_DATASETS"] = str(data_root.resolve())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config

    from train_net_video import Trainer
    from videomt import (
        PanopticDatasetVideoMapper,
        SemanticDatasetVideoMapper,
        YTVISDatasetMapper,
        add_videomt_config,
    )

    return Runtime(
        DetectionCheckpointer=DetectionCheckpointer,
        DatasetCatalog=DatasetCatalog,
        MetadataCatalog=MetadataCatalog,
        Trainer=Trainer,
        add_deeplab_config=add_deeplab_config,
        add_videomt_config=add_videomt_config,
        get_cfg=get_cfg,
        mappers={
            "vps": PanopticDatasetVideoMapper,
            "vis": YTVISDatasetMapper,
            "vss": SemanticDatasetVideoMapper,
        },
    )


def _parse_dataset_values(
    values: list[str] | None, selected_datasets: list[str], option: str
) -> dict[str, str]:
    if not values:
        return {}
    result = {}
    for value in values:
        if "=" in value:
            dataset, item = value.split("=", 1)
            if dataset not in selected_datasets:
                raise ValueError(f"{option}: dataset {dataset!r} is not selected")
        elif len(selected_datasets) == 1:
            dataset, item = selected_datasets[0], value
        else:
            raise ValueError(
                f"{option}: use DATASET=VALUE when generating multiple datasets"
            )
        if dataset in result:
            raise ValueError(f"{option}: duplicate value for {dataset}")
        result[dataset] = item
    return result


def build_cfg(
    runtime: Runtime,
    dataset: str,
    backbone: str,
    checkpoint: Path,
    device: str,
):
    cfg = runtime.get_cfg()
    runtime.add_deeplab_config(cfg)
    runtime.add_videomt_config(cfg)
    config = (
        ROOT
        / "configs"
        / "DynEoMT"
        / DATASETS[dataset]["config_dataset"]
        / "dinov2"
        / backbone
        / BACKBONE_CONFIGS[backbone]
    )
    cfg.merge_from_file(str(config))
    cfg.MODEL.WEIGHTS = str(checkpoint)
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def record_sequence_id(record: dict, dataset: str) -> str:
    if dataset in {"vipseg", "vspw"}:
        return str(record["video_id"])
    return Path(record["file_names"][0]).parent.name


def find_record(
    records: list[dict], dataset: str, query: str
) -> tuple[dict, str] | None:
    records_by_id = {record_sequence_id(record, dataset): record for record in records}
    if query in records_by_id:
        return copy.deepcopy(records_by_id[query]), query
    compact_query = "".join(char.lower() for char in query if char.isalnum())
    matches = [
        sequence_id
        for sequence_id in records_by_id
        if compact_query
        in "".join(char.lower() for char in sequence_id if char.isalnum())
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"Cannot resolve sequence {query!r}; matches={matches[:10]}")
    return copy.deepcopy(records_by_id[matches[0]]), matches[0]


def select_catalog_record(
    runtime: Runtime, cfg, dataset: str, query: str
) -> tuple[dict, str, str, str]:
    catalogs = [
        *((name, DATASETS[dataset]["split"]) for name in cfg.DATASETS.TEST),
        *((name, "train") for name in cfg.DATASETS.TRAIN),
    ]
    searched = []
    for catalog_name, split in catalogs:
        searched.append(catalog_name)
        found = find_record(runtime.DatasetCatalog.get(catalog_name), dataset, query)
        if found is not None:
            record, sequence_id = found
            return record, sequence_id, catalog_name, split
    raise RuntimeError(
        f"Cannot resolve sequence {query!r} for {dataset}; catalogs={searched}"
    )


def truncate_record(record: dict, max_frames: int) -> dict:
    if max_frames <= 0:
        return record
    for key in (
        "file_names",
        "annotations",
        "pan_seg_file_names",
        "segments_infos",
        "sem_mask_names",
    ):
        if key in record:
            record[key] = record[key][:max_frames]
    if "length" in record:
        record["length"] = min(int(record["length"]), max_frames)
    return record


def stable_color(key: str) -> tuple[int, int, int]:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    hsv = np.uint8(
        [[[int(digest[0]), 170 + digest[1] % 70, 205 + digest[2] % 45]]]
    )
    return tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def as_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != shape:
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask.astype(bool)


def class_name(metadata, class_id: int, task: str) -> str:
    if task == "vps":
        thing_names = metadata.thing_classes or []
        stuff_names = metadata.stuff_classes or []
        if class_id < len(thing_names):
            return str(thing_names[class_id])
        stuff_id = class_id - len(thing_names)
        if 0 <= stuff_id < len(stuff_names):
            return str(stuff_names[stuff_id])
        return f"class_{class_id}"
    names = metadata.thing_classes if task == "vis" else metadata.stuff_classes
    if names is not None and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class_{class_id}"


def prediction_records(
    output: dict,
    metadata,
    frame_index: int,
    shape: tuple[int, int],
    dynamic_threshold: float,
    score_threshold: float,
    min_area: int,
) -> tuple[list[DrawRecord], list[DrawRecord]]:
    task = output["task"]
    dynamic_scores = as_numpy(output.get("dynamic_scores", np.empty((0, 0))))
    if dynamic_scores.ndim != 2:
        raise RuntimeError(
            "The model output has no query-level dynamic scores. Check that the "
            "DynEoMT config and checkpoint are being used."
        )

    if task == "vis":
        scores = [float(value) for value in output["pred_scores"]]
        labels = [int(value) for value in output["pred_labels"]]
        track_ids = [int(value) for value in output["pred_ids"]]
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        records = []
        used_tracks = set()
        for index in order:
            if scores[index] < score_threshold or track_ids[index] in used_tracks:
                continue
            mask = resize_mask(
                as_numpy(output["pred_masks"][index][frame_index]), shape
            )
            if int(mask.sum()) < min_area:
                continue
            if index >= dynamic_scores.shape[0] or frame_index >= dynamic_scores.shape[1]:
                continue
            used_tracks.add(track_ids[index])
            probability = float(dynamic_scores[index, frame_index])
            records.append(
                DrawRecord(
                    mask=mask,
                    name=class_name(metadata, labels[index], task),
                    object_id=f"q{track_ids[index]}",
                    color=stable_color(f"pred-track-{track_ids[index]}"),
                    score=scores[index],
                    dynamic_probability=probability,
                    dynamic_label=(
                        "dynamic" if probability >= dynamic_threshold else "static"
                    ),
                )
            )
        return records, records

    if task == "vps":
        panoptic = as_numpy(output["pred_masks"])[frame_index]
        track_ids = [int(as_numpy(value)) for value in output["pred_ids"]]
        records = []
        for index, segment in enumerate(output["segments_infos"]):
            if index >= len(track_ids) or index >= dynamic_scores.shape[0]:
                continue
            mask = resize_mask(panoptic == int(segment["id"]), shape)
            if int(mask.sum()) < min_area:
                continue
            category_id = int(segment["category_id"])
            track_id = track_ids[index]
            probability = float(dynamic_scores[index, frame_index])
            color_key = (
                f"pred-track-{track_id}"
                if segment["isthing"]
                else f"pred-class-{category_id}"
            )
            records.append(
                DrawRecord(
                    mask=mask,
                    name=class_name(metadata, category_id, task),
                    object_id=f"q{track_id}",
                    color=stable_color(color_key),
                    dynamic_probability=probability,
                    dynamic_label=(
                        "dynamic" if probability >= dynamic_threshold else "static"
                    ),
                )
            )
        return records, records

    semantic = as_numpy(output["pred_masks"])[frame_index]
    segmentation = []
    for category_id in np.unique(semantic):
        category_id = int(category_id)
        mask = resize_mask(semantic == category_id, shape)
        if int(mask.sum()) >= min_area:
            segmentation.append(
                DrawRecord(
                    mask=mask,
                    name=class_name(metadata, category_id, task),
                    object_id=f"c{category_id}",
                    color=stable_color(f"pred-class-{category_id}"),
                )
            )

    dynamic_masks = as_numpy(output.get("dynamic_masks", np.empty((0, 0, 0, 0))))
    dynamic_labels = as_numpy(
        output.get("dynamic_labels", np.empty((0,), dtype=np.int64))
    )
    dynamic = []
    for index, category_id in enumerate(dynamic_labels):
        if index >= dynamic_scores.shape[0]:
            continue
        mask = resize_mask(dynamic_masks[index, frame_index], shape)
        if int(mask.sum()) < min_area:
            continue
        category_id = int(category_id)
        probability = float(dynamic_scores[index, frame_index])
        dynamic.append(
            DrawRecord(
                mask=mask,
                name=class_name(metadata, category_id, task),
                object_id=f"q{index}",
                color=stable_color(f"pred-query-{index}"),
                dynamic_probability=probability,
                dynamic_label=(
                    "dynamic" if probability >= dynamic_threshold else "static"
                ),
            )
        )
    return segmentation, dynamic


def target_frames(payload: dict) -> dict[str, list[dict]]:
    frames = payload.get("frames", [])
    if isinstance(frames, dict):
        return {
            Path(frame_name).stem: list(records.values())
            for frame_name, records in frames.items()
        }
    return {
        Path(frame["frame"]).stem: frame.get("masks", []) for frame in frames
    }


def ground_truth_records(
    dataset: str,
    adapter,
    adapter_index: int,
    frame_targets: list[dict],
    shape: tuple[int, int],
    dynamic_threshold: float,
) -> tuple[list[DrawRecord], list[DrawRecord]]:
    masks = adapter.load_masks(adapter_index)
    targets = {str(record["source_id"]): record for record in frame_targets}

    if dataset == "vspw":
        masks_by_category: dict[int, list] = {}
        for record in masks:
            masks_by_category.setdefault(int(record.category_id), []).append(record)
        targets_by_category: dict[int, list[dict]] = {}
        for record in frame_targets:
            if str(record.get("label", "uncertain")) in {"static", "dynamic"}:
                targets_by_category.setdefault(
                    int(record["category_id"]), []
                ).append(record)

        segmentation, dynamic = [], []
        for category_id, category_masks in masks_by_category.items():
            mask = np.zeros(shape, dtype=bool)
            for record in category_masks:
                mask |= resize_mask(record.mask, shape)
            if not mask.any():
                continue
            name = category_masks[0].category_name
            color = stable_color(f"gt-class-{category_id}")
            segmentation.append(
                DrawRecord(mask, name, f"c{category_id}", color)
            )
            valid_targets = targets_by_category.get(category_id, [])
            if valid_targets:
                total_area = sum(
                    max(float(item.get("area", 1)), 1.0) for item in valid_targets
                )
                probability = sum(
                    float(str(item.get("label")) == "dynamic")
                    * max(float(item.get("area", 1)), 1.0)
                    for item in valid_targets
                ) / total_area
                label = (
                    "dynamic" if probability >= dynamic_threshold else "static"
                )
            else:
                probability, label = None, "uncertain"
            dynamic.append(
                DrawRecord(
                    mask=mask,
                    name=name,
                    object_id=f"c{category_id}",
                    color=color,
                    dynamic_probability=probability,
                    dynamic_label=label,
                )
            )
        return segmentation, dynamic

    segmentation, dynamic = [], []
    for record in masks:
        mask = resize_mask(record.mask, shape)
        if not mask.any():
            continue
        color_key = (
            f"gt-track-{record.track_id}"
            if record.isthing
            else f"gt-class-{record.category_id}"
        )
        color = stable_color(color_key)
        target = targets.get(str(record.source_id))
        label = str(target.get("label", "uncertain")) if target else "uncertain"
        probability = None
        if target and label in {"static", "dynamic"}:
            probability = float(target.get("probability", label == "dynamic"))
        segmentation.append(
            DrawRecord(
                mask=mask,
                name=record.category_name,
                object_id=f"id{record.track_id}",
                color=color,
            )
        )
        dynamic.append(
            DrawRecord(
                mask=mask,
                name=record.category_name,
                object_id=f"id{record.track_id}",
                color=color,
                dynamic_probability=probability,
                dynamic_label=label,
            )
        )
    return segmentation, dynamic


def fit_records(
    image: np.ndarray,
    records: list[DrawRecord],
    width: int,
    height: int,
) -> tuple[np.ndarray, list[DrawRecord]]:
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    target_width = max(1, int(round(source_width * scale)))
    target_height = max(1, int(round(source_height * scale)))
    offset_x = (width - target_width) // 2
    offset_y = (height - target_height) // 2
    canvas = np.full(
        (height, width, 3), PANEL_BACKGROUND, dtype=np.uint8
    )
    canvas[
        offset_y : offset_y + target_height,
        offset_x : offset_x + target_width,
    ] = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    fitted = []
    for record in records:
        resized = cv2.resize(
            record.mask.astype(np.uint8),
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = np.zeros((height, width), dtype=bool)
        mask[
            offset_y : offset_y + target_height,
            offset_x : offset_x + target_width,
        ] = resized.astype(bool)
        fitted.append(
            DrawRecord(
                mask=mask,
                name=record.name,
                object_id=record.object_id,
                color=record.color,
                score=record.score,
                dynamic_probability=record.dynamic_probability,
                dynamic_label=record.dynamic_label,
            )
        )
    return canvas, fitted


def dynamic_color(record: DrawRecord) -> tuple[int, int, int]:
    if record.dynamic_label == "dynamic":
        return DYNAMIC_COLOR
    if record.dynamic_label == "static":
        return STATIC_COLOR
    return UNCERTAIN_COLOR


def record_text(record: DrawRecord, mode: str) -> str:
    if mode == "segmentation":
        suffix = f" s={record.score:.2f}" if record.score is not None else ""
        return f"{record.name} | {record.object_id}{suffix}"
    label = {"dynamic": "DYN", "static": "STATIC", "uncertain": "?"}[
        record.dynamic_label
    ]
    probability = (
        ""
        if record.dynamic_probability is None
        else f" p={record.dynamic_probability:.2f}"
    )
    return f"{record.name} | {record.object_id} | {label}{probability}"


def fit_text(
    text: str,
    font: int,
    initial_scale: float,
    thickness: int,
    max_width: int,
    minimum_scale: float = 0.30,
) -> tuple[str, float, int, int]:
    scale = initial_scale
    width, height = cv2.getTextSize(text, font, scale, thickness)[0]
    while width > max_width and scale > minimum_scale:
        scale = max(minimum_scale, scale - 0.03)
        width, height = cv2.getTextSize(text, font, scale, thickness)[0]
    if width > max_width:
        suffix = "..."
        while text and cv2.getTextSize(
            text + suffix, font, scale, thickness
        )[0][0] > max_width:
            text = text[:-1]
        text += suffix
        width, height = cv2.getTextSize(text, font, scale, thickness)[0]
    return text, scale, width, height


def draw_labels(
    panel: np.ndarray, records: list[DrawRecord], mode: str, max_labels: int
) -> None:
    if max_labels <= 0:
        return
    if mode == "dynamic":
        records = sorted(
            records,
            key=lambda item: (item.dynamic_label == "dynamic", item.area),
            reverse=True,
        )
    else:
        records = sorted(records, key=lambda item: item.area, reverse=True)

    occupied = []
    for record in records[:max_labels]:
        ys, xs = np.nonzero(record.mask)
        if not len(xs):
            continue
        text = record_text(record, mode)
        font, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1
        text, scale, text_width, text_height = fit_text(
            text,
            font,
            0.43,
            thickness,
            panel.shape[1] - 16,
        )
        maximum_x = max(2, panel.shape[1] - text_width - 10)
        x = int(np.clip(np.median(xs), 2, maximum_x))
        y = int(np.clip(np.median(ys), 54, panel.shape[0] - 8))
        box = [x, y - text_height - 8, x + text_width + 8, y + 4]
        for attempt in range(6):
            overlaps = any(
                not (
                    box[2] < previous[0]
                    or box[0] > previous[2]
                    or box[3] < previous[1]
                    or box[1] > previous[3]
                )
                for previous in occupied
            )
            if not overlaps:
                break
            y += text_height + 12
            if y > panel.shape[0] - 8:
                y = max(54, y - 2 * (text_height + 12))
                x = min(maximum_x, x + 36 * (attempt + 1))
            box = [x, y - text_height - 8, x + text_width + 8, y + 4]
        color = record.color if mode == "segmentation" else dynamic_color(record)
        cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), (5, 7, 10), -1)
        cv2.rectangle(panel, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv2.putText(
            panel,
            text,
            (x + 4, y),
            font,
            scale,
            (250, 250, 250),
            thickness,
            cv2.LINE_AA,
        )
        occupied.append(box)


def render_panel(
    image: np.ndarray,
    records: list[DrawRecord],
    mode: str,
    title: str,
    args: argparse.Namespace,
) -> np.ndarray:
    panel, fitted_records = fit_records(
        image, records, args.cell_width, args.cell_height
    )
    overlay = panel.copy()
    covered = np.zeros(panel.shape[:2], dtype=bool)
    for record in sorted(fitted_records, key=lambda item: item.area, reverse=True):
        color = record.color if mode == "segmentation" else dynamic_color(record)
        overlay[record.mask] = color
        covered |= record.mask
    blended = cv2.addWeighted(panel, 0.54, overlay, 0.46, 0)
    panel[covered] = blended[covered]
    for record in fitted_records:
        color = record.color if mode == "segmentation" else dynamic_color(record)
        contours, _ = cv2.findContours(
            record.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
    draw_labels(panel, fitted_records, mode, args.max_labels)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 36), PANEL_BACKGROUND, -1)
    cv2.putText(
        panel,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (248, 248, 248),
        1,
        cv2.LINE_AA,
    )
    return panel


def render_clean_overlay(
    image: np.ndarray,
    records: list[DrawRecord],
    mode: str,
) -> np.ndarray:
    """Overlay masks on the source frame without resizing or adding annotations."""
    panel = image.copy()
    overlay = image.copy()
    covered = np.zeros(image.shape[:2], dtype=bool)
    for record in sorted(records, key=lambda item: item.area, reverse=True):
        color = record.color if mode == "segmentation" else dynamic_color(record)
        overlay[record.mask] = color
        covered |= record.mask
    blended = cv2.addWeighted(panel, 0.54, overlay, 0.46, 0)
    panel[covered] = blended[covered]
    for record in records:
        color = record.color if mode == "segmentation" else dynamic_color(record)
        contours, _ = cv2.findContours(
            record.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)
    return panel


def even_frame(frame: np.ndarray) -> np.ndarray:
    """Keep VideoWriter-compatible dimensions without introducing a visible frame."""
    return frame[: frame.shape[0] - frame.shape[0] % 2, : frame.shape[1] - frame.shape[1] % 2]


def compose_frame(
    image: np.ndarray,
    prediction_segmentation: list[DrawRecord],
    prediction_dynamic: list[DrawRecord],
    ground_truth_segmentation: list[DrawRecord],
    ground_truth_dynamic: list[DrawRecord],
    dataset: str,
    sequence_id: str,
    frame_name: str,
    frame_index: int,
    frame_count: int,
    args: argparse.Namespace,
) -> np.ndarray:
    prediction_row = np.hstack(
        [
            render_panel(
                image,
                prediction_segmentation,
                "segmentation",
                "PREDICTION | SEGMENTATION",
                args,
            ),
            render_panel(
                image,
                prediction_dynamic,
                "dynamic",
                "PREDICTION | DYNAMICITY",
                args,
            ),
        ]
    )
    content = prediction_row
    if args.mode == "grid":
        ground_truth_row = np.hstack(
            [
                render_panel(
                    image,
                    ground_truth_segmentation,
                    "segmentation",
                    "GROUND TRUTH | SEGMENTATION",
                    args,
                ),
                render_panel(
                    image,
                    ground_truth_dynamic,
                    "dynamic",
                    "PSEUDO-GROUND TRUTH | DYNAMICITY",
                    args,
                ),
            ]
        )
        content = np.vstack([prediction_row, ground_truth_row])

    header = np.full((46, content.shape[1], 3), PANEL_BACKGROUND, dtype=np.uint8)
    header_text = (
        f"DynEoMT | {dataset} | {sequence_id} | {frame_name} | "
        f"{frame_index + 1}/{frame_count}"
    )
    header_text, header_scale, _, _ = fit_text(
        header_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        2,
        header.shape[1] - 28,
        minimum_scale=0.42,
    )
    cv2.putText(
        header,
        header_text,
        (14, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        header_scale,
        (248, 248, 248),
        2,
        cv2.LINE_AA,
    )
    legend = np.full((44, content.shape[1], 3), PANEL_BACKGROUND, dtype=np.uint8)
    legend_items = [
        ("DYNAMIC", DYNAMIC_COLOR),
        ("STATIC", STATIC_COLOR),
        ("UNCERTAIN GT", UNCERTAIN_COLOR),
    ]
    x = 14
    for label, color in legend_items:
        cv2.rectangle(legend, (x, 11), (x + 24, 32), color, -1)
        label_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1
        )[0][0]
        cv2.putText(
            legend,
            label,
            (x + 32, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        x += 44 + label_width
    explanation = (
        f"threshold={args.dynamic_threshold:.2f} | stable colors by track/query"
    )
    available_width = content.shape[1] - x - 24
    if available_width >= 220:
        explanation, explanation_scale, _, _ = fit_text(
            explanation,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            1,
            available_width,
            minimum_scale=0.38,
        )
        cv2.putText(
            legend,
            explanation,
            (x + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            explanation_scale,
            (210, 214, 220),
            1,
            cv2.LINE_AA,
        )
    frame = np.vstack([header, content, legend])
    if frame.shape[0] % 2 or frame.shape[1] % 2:
        frame = cv2.copyMakeBorder(
            frame,
            0,
            frame.shape[0] % 2,
            0,
            frame.shape[1] % 2,
            cv2.BORDER_CONSTANT,
            value=PANEL_BACKGROUND,
        )
    return frame


def infer(model, sample: dict, device: str) -> dict:
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model([sample])
        return model([sample])


def render_dataset(
    runtime: Runtime,
    dataset: str,
    sequence_query: str,
    checkpoint: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path] | None:
    spec = DATASETS[dataset]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {dataset}: {checkpoint}")
    cfg = build_cfg(runtime, dataset, args.backbone, checkpoint, args.device)
    record, sequence_id, catalog_name, split = select_catalog_record(
        runtime, cfg, dataset, sequence_query
    )
    record = truncate_record(record, args.max_frames)
    target_path = (
        resolve_target_path(dataset, sequence_id, args.target_root)
        if args.mode == "grid"
        else None
    )

    print(
        json.dumps(
            {
                "dataset": dataset,
                "sequence": sequence_id,
                "frames": len(record["file_names"]),
                "split": split,
                "catalog": catalog_name,
                "mode": args.mode,
                "backbone": args.backbone,
                "checkpoint": str(checkpoint),
                "targets": str(target_path) if target_path is not None else None,
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return None

    mapper = runtime.mappers[spec["mapper"]](cfg, is_train=False)
    sample = mapper(record)
    model = runtime.Trainer.build_model(cfg)
    runtime.DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    output = infer(model, sample, args.device)
    metadata = runtime.MetadataCatalog.get(catalog_name)

    adapter = None
    adapter_by_stem = {}
    targets_by_stem = {}
    if args.mode == "grid":
        adapter = build_adapter(dataset, split, sequence_id, args.data_root)
        adapter_by_stem = {
            Path(path).stem: index for index, path in enumerate(adapter.frame_paths)
        }
        targets_by_stem = target_frames(json.loads(target_path.read_text()))

    if args.output is not None:
        output_path = args.output
    else:
        output_path = (
            args.output_root
            / f"{dataset}_{sequence_id}_{args.backbone}_{args.mode}.mp4"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path = output_path.with_suffix(".jpg")
    frame_indices = list(sample["frame_idx"])
    source_file_names = record["file_names"]
    writer = None
    clean_writers = {}
    clean_paths = {}
    best_preview = None

    progress = tqdm(
        enumerate(frame_indices),
        total=len(frame_indices),
        desc=f"render {dataset}/{sequence_id}",
        unit="frame",
    )
    for output_index, original_index in progress:
        image_path = Path(source_file_names[original_index])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        prediction_segmentation, prediction_dynamic = prediction_records(
            output,
            metadata,
            output_index,
            image.shape[:2],
            args.dynamic_threshold,
            args.score_threshold,
            args.min_area,
        )
        ground_truth_segmentation, ground_truth_dynamic = [], []
        if args.mode == "grid":
            adapter_index = adapter_by_stem.get(image_path.stem)
            if adapter_index is None:
                raise KeyError(
                    f"Frame {image_path.stem} is missing from the {dataset} adapter"
                )
            ground_truth_segmentation, ground_truth_dynamic = ground_truth_records(
                dataset,
                adapter,
                adapter_index,
                targets_by_stem.get(image_path.stem, []),
                image.shape[:2],
                args.dynamic_threshold,
            )
        if args.mode == "clean":
            clean_frames = {
                "original": even_frame(image),
                "tracking": even_frame(
                    render_clean_overlay(image, prediction_segmentation, "segmentation")
                ),
                "dynamicity": even_frame(
                    render_clean_overlay(image, prediction_dynamic, "dynamic")
                ),
            }
            if not clean_writers:
                base = output_path.with_suffix("")
                for clean_name, clean_frame in clean_frames.items():
                    clean_path = base.parent / f"{base.name}_{clean_name}.mp4"
                    clean_writer = cv2.VideoWriter(
                        str(clean_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.fps,
                        (clean_frame.shape[1], clean_frame.shape[0]),
                    )
                    if not clean_writer.isOpened():
                        raise RuntimeError(f"Cannot open video writer for {clean_path}")
                    clean_paths[clean_name] = clean_path
                    clean_writers[clean_name] = clean_writer
            for clean_name, clean_frame in clean_frames.items():
                clean_writers[clean_name].write(clean_frame)
            dynamic_count = sum(
                record.dynamic_label == "dynamic" for record in prediction_dynamic
            )
            progress.set_postfix_str(
                f"masks={len(prediction_segmentation)} dynamic={dynamic_count}"
            )
            continue

        frame = compose_frame(
            image,
            prediction_segmentation,
            prediction_dynamic,
            ground_truth_segmentation,
            ground_truth_dynamic,
            dataset,
            sequence_id,
            image_path.name,
            output_index,
            len(frame_indices),
            args,
        )
        if writer is None:
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (frame.shape[1], frame.shape[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open video writer for {output_path}")
        writer.write(frame)
        dynamic_count = sum(
            record.dynamic_label == "dynamic" for record in prediction_dynamic
        )
        progress.set_postfix_str(
            f"masks={len(prediction_segmentation)} dynamic={dynamic_count}"
        )
        preview_score = dynamic_count + 0.01 * len(prediction_segmentation)
        if best_preview is None or preview_score > best_preview[0]:
            best_preview = (preview_score, frame.copy())

    if writer is not None:
        writer.release()
    for clean_writer in clean_writers.values():
        clean_writer.release()
    if args.mode == "clean":
        del output, model, sample
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        for clean_name, clean_path in clean_paths.items():
            print(f"{clean_name}={clean_path}")
        return tuple(clean_paths.values())
    if best_preview is not None:
        cv2.imwrite(str(preview_path), best_preview[1])
    del output, model, sample
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    print(f"video={output_path}")
    print(f"preview={preview_path}")
    return output_path, preview_path


def main() -> None:
    args = parse_args()
    if args.dataset_name and args.dataset_option:
        raise ValueError("Specify the dataset positionally or with --dataset, not both")
    dataset_selection = args.dataset_name or args.dataset_option or "all"
    selected_datasets = (
        list(DATASET_NAMES)
        if dataset_selection == "all"
        else [dataset_selection]
    )
    if args.output is not None and len(selected_datasets) != 1:
        raise ValueError("--output can only be used with one dataset")
    sequence_values = list(args.sequence or [])
    if args.sequence_name is not None:
        if len(selected_datasets) != 1:
            raise ValueError("A positional sequence requires one positional dataset")
        if sequence_values:
            raise ValueError("Specify the sequence positionally or with --sequence, not both")
        sequence_values.append(args.sequence_name)
    sequence_overrides = _parse_dataset_values(
        sequence_values, selected_datasets, "--sequence"
    )
    checkpoint_overrides = _parse_dataset_values(
        args.checkpoint, selected_datasets, "--checkpoint"
    )
    args.data_root = resolve_data_root(args.data_root, selected_datasets)
    runtime = import_runtime(args.data_root)

    for dataset in selected_datasets:
        sequence_query = sequence_overrides.get(dataset, DATASETS[dataset]["sequence"])
        checkpoint = resolve_checkpoint(
            dataset,
            args.backbone,
            checkpoint_overrides.get(dataset),
            args.checkpoint_root,
        )
        render_dataset(
            runtime,
            dataset,
            sequence_query,
            checkpoint,
            args,
        )


if __name__ == "__main__":
    main()
