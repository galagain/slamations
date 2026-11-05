"""Dataset adapters exposing frames and annotated masks through one interface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


@dataclass
class MaskRecord:
    source_id: str
    track_id: str
    category_id: int
    category_name: str
    mask: np.ndarray
    isthing: bool

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class SequenceAdapter:
    dataset: str
    split: str
    sequence_id: str
    frame_paths: list[Path]
    frame_keys: list[str]

    def load_masks(self, frame_index: int) -> list[MaskRecord]:
        raise NotImplementedError


def _resolve_sequence(query: str, candidates: list[str]) -> str:
    if query in candidates:
        return query
    compact = "".join(char.lower() for char in query if char.isalnum())
    matches = [
        candidate
        for candidate in candidates
        if compact in "".join(char.lower() for char in candidate if char.isalnum())
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Cannot resolve sequence {query!r}; matches={matches[:10]}")
    return matches[0]


def _decode_vis_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rle = mask_utils.frPyObjects(segmentation, height, width)
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return decoded.astype(bool)
    if isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        segmentation = mask_utils.frPyObjects(segmentation, height, width)
    decoded = mask_utils.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return decoded.astype(bool)


class VISAdapter(SequenceAdapter):
    def __init__(
        self,
        dataset: str,
        split: str,
        sequence_query: str,
        annotation_path: Path,
        image_root: Path,
    ) -> None:
        payload = json.loads(annotation_path.read_text())
        names = [video["file_names"][0].split("/")[0] for video in payload["videos"]]
        sequence_id = _resolve_sequence(sequence_query, names)
        self.video = next(
            video
            for video in payload["videos"]
            if video["file_names"][0].split("/")[0] == sequence_id
        )
        self.annotations = [
            annotation
            for annotation in payload.get("annotations", [])
            if annotation["video_id"] == self.video["id"]
        ]
        self.categories = {int(category["id"]): category for category in payload["categories"]}
        self.dataset = dataset
        self.split = split
        self.sequence_id = sequence_id
        self.frame_paths = [image_root / name for name in self.video["file_names"]]
        self.frame_keys = [path.name for path in self.frame_paths]
        self.height = int(self.video["height"])
        self.width = int(self.video["width"])

    def load_masks(self, frame_index: int) -> list[MaskRecord]:
        records = []
        for annotation in self.annotations:
            segmentations = annotation.get("segmentations", [])
            if frame_index >= len(segmentations) or segmentations[frame_index] is None:
                continue
            mask = _decode_vis_segmentation(
                segmentations[frame_index], self.height, self.width
            )
            if not mask.any():
                continue
            category_id = int(annotation["category_id"])
            category = self.categories[category_id]
            track_id = str(annotation["id"])
            records.append(
                MaskRecord(
                    source_id=track_id,
                    track_id=track_id,
                    category_id=category_id,
                    category_name=str(category["name"]),
                    mask=mask,
                    isthing=True,
                )
            )
        return records


class VIPSegAdapter(SequenceAdapter):
    def __init__(self, root: Path, split: str, sequence_query: str) -> None:
        payload = json.loads((root / f"panoptic_gt_VIPSeg_{split}.json").read_text())
        sequence_id = _resolve_sequence(
            sequence_query, [video["video_id"] for video in payload["videos"]]
        )
        sequence_annotation = next(
            annotation
            for annotation in payload["annotations"]
            if annotation["video_id"] == sequence_id
        )
        self.frame_annotations = sequence_annotation["annotations"]
        self.categories = {
            int(category["id"]): category for category in payload["categories"]
        }
        self.dataset = "vipseg"
        self.split = split
        self.sequence_id = sequence_id
        self.image_root = root / "images" / sequence_id
        self.mask_root = root / "panomasksRGB" / sequence_id
        self.frame_keys = [
            Path(annotation["file_name"]).with_suffix(".jpg").name
            for annotation in self.frame_annotations
        ]
        self.frame_paths = [self.image_root / key for key in self.frame_keys]

    def load_masks(self, frame_index: int) -> list[MaskRecord]:
        annotation = self.frame_annotations[frame_index]
        rgb = np.asarray(
            Image.open(self.mask_root / annotation["file_name"]).convert("RGB")
        )
        panoptic = (
            rgb[..., 0].astype(np.int64)
            + 256 * rgb[..., 1].astype(np.int64)
            + 65536 * rgb[..., 2].astype(np.int64)
        )
        records = []
        for segment in annotation["segments_info"]:
            if int(segment.get("iscrowd", 0)):
                continue
            segment_id = int(segment["id"])
            category_id = int(segment["category_id"])
            category = self.categories[category_id]
            mask = panoptic == segment_id
            if mask.any():
                records.append(
                    MaskRecord(
                        source_id=str(segment_id),
                        track_id=str(segment_id),
                        category_id=category_id,
                        category_name=str(category["name"]),
                        mask=mask,
                        isthing=bool(category.get("isthing", 0)),
                    )
                )
        return records


class VSPWAdapter(SequenceAdapter):
    def __init__(
        self,
        root: Path,
        split: str,
        sequence_query: str,
        min_component_area: int = 64,
    ) -> None:
        candidates = [
            line.strip()
            for line in (root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        sequence_id = _resolve_sequence(sequence_query, candidates)
        self.dataset = "vspw"
        self.split = split
        self.sequence_id = sequence_id
        self.sequence_root = root / "data" / sequence_id
        self.frame_paths = sorted((self.sequence_root / "origin").glob("*.jpg"))
        masks_by_stem = {
            path.stem: path for path in (self.sequence_root / "mask").glob("*.png")
        }
        self.mask_paths = [masks_by_stem.get(path.stem) for path in self.frame_paths]
        self.frame_keys = [path.name for path in self.frame_paths]
        names_to_ids = json.loads((root / "label_num_dic_final.json").read_text())
        self.id_to_name = {
            int(category_id): str(name) for name, category_id in names_to_ids.items()
        }
        self.min_component_area = int(min_component_area)

    def load_masks(self, frame_index: int) -> list[MaskRecord]:
        path = self.mask_paths[frame_index]
        if path is None:
            return []
        semantic = np.asarray(Image.open(path))
        records = []
        for category_id in np.unique(semantic):
            category_id = int(category_id)
            if category_id in (0, 255):
                continue
            binary = (semantic == category_id).astype(np.uint8)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            component_index = 0
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < self.min_component_area:
                    continue
                component_index += 1
                source_id = f"{category_id}:{component_index}"
                records.append(
                    MaskRecord(
                        source_id=source_id,
                        track_id=source_id,
                        category_id=category_id,
                        category_name=self.id_to_name.get(
                            category_id, f"class_{category_id}"
                        ),
                        mask=labels == label,
                        isthing=True,
                    )
                )
        return records


def build_adapter(
    dataset: str, split: str, sequence_id: str, data_root: Path
) -> SequenceAdapter:
    dataset = dataset.lower()
    if dataset == "vipseg":
        return VIPSegAdapter(
            data_root / "VIPSeg" / "VIPSeg_720P", split, sequence_id
        )
    if dataset == "ovis":
        return VISAdapter(
            dataset,
            split,
            sequence_id,
            data_root / "ovis" / "annotations" / f"annotations_{split}.json",
            data_root / "ovis" / split,
        )
    if dataset in {"ytvis", "ytvis2022", "ytvis_2022"}:
        return VISAdapter(
            "ytvis2022",
            split,
            sequence_id,
            data_root / "ytvis_2022" / f"{split}.json",
            data_root / "ytvis_2022" / split / "JPEGImages",
        )
    if dataset == "vspw":
        return VSPWAdapter(data_root / "VSPW_480p", split, sequence_id)
    raise ValueError(f"Unsupported dataset: {dataset}")
