"""Dataset-independent evaluation for query-level dynamicity."""

import math
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from panopticapi.utils import rgb2id

from .dynamic_targets import DynamicTargetStore


def _as_numpy(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _decode_vis_mask(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_util.frPyObjects(segmentation, height, width)
        segmentation = mask_util.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        segmentation = mask_util.frPyObjects(segmentation, height, width)
    decoded = mask_util.decode(segmentation)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return decoded.astype(bool)


def _auc(labels, scores):
    """AUROC from average ranks, including ties."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = ranks[labels == 1].sum()
    return float(
        (rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def dynamic_metrics(stats, labels, scores):
    """Compute conditional dynamicity and end-to-end DynSeg metrics."""

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    tp, tn, fp, fn = (stats[key] for key in ("tp", "tn", "fp", "fn"))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    f1 = ratio(2 * precision * recall, precision + recall)
    mcc_denominator = math.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = ratio(tp * tn - fp * fn, mcc_denominator)

    correct = stats["joint_correct"]
    dynseg_precision = ratio(correct, stats["pred"])
    dynseg_recall = ratio(correct, stats["gt"])
    dynseg_f1 = ratio(
        2 * dynseg_precision * dynseg_recall,
        dynseg_precision + dynseg_recall,
    )
    return {
        "Dyn-Acc": 100.0 * ratio(tp + tn, stats["matched"]),
        "Dyn-BAcc": 50.0 * (recall + specificity),
        "Dyn-Precision": 100.0 * precision,
        "Dyn-Recall": 100.0 * recall,
        "Dyn-F1": 100.0 * f1,
        "Dyn-MCC": 100.0 * mcc,
        "Dyn-AUROC": 100.0 * _auc(labels, scores),
        "MaskCoverage": 100.0 * ratio(stats["matched"], stats["gt"]),
        "Matched-IoU": 100.0 * ratio(stats["iou_sum"], stats["matched"]),
        "DynSeg-Precision": 100.0 * dynseg_precision,
        "DynSeg-Recall": 100.0 * dynseg_recall,
        "DynSeg-F1": 100.0 * dynseg_f1,
    }


class DynamicEvaluator(DatasetEvaluator):
    """Evaluate dynamicity after class-aware mask matching."""

    def __init__(self, dataset_name, cfg, distributed=True):
        self.dataset_name = dataset_name
        self.dataset = cfg.MODEL.DYNAMIC.DATASET.lower()
        self.store = DynamicTargetStore(
            cfg.MODEL.DYNAMIC.TARGET_ROOT,
            self.dataset,
            cfg.MODEL.DYNAMIC.REQUIRED,
        )
        self.iou_threshold = float(cfg.MODEL.DYNAMIC.EVAL_IOU_THRESHOLD)
        self.distributed = distributed
        self.metadata = MetadataCatalog.get(dataset_name)
        self.reset()

    def reset(self):
        self.stats = defaultdict(float)
        self.labels = []
        self.scores = []

    def _sequence_id(self, sample):
        if self.dataset in {"vipseg", "vspw"}:
            return str(sample["video_id"])
        return str(sample["file_names"][0]).replace("\\", "/").split("/")[-2]

    def _predictions(self, output, frame_index):
        scores = output.get("dynamic_scores")
        if scores is None:
            return []
        scores = _as_numpy(scores)
        if scores.ndim != 2 or frame_index >= scores.shape[1]:
            return []

        predictions = []
        if output["task"] == "vis":
            for index, label in enumerate(output["pred_labels"]):
                mask = _as_numpy(output["pred_masks"][index][frame_index]).astype(bool)
                if mask.any():
                    predictions.append((mask, int(label), float(scores[index, frame_index])))
        elif output["task"] == "vps":
            panoptic = _as_numpy(output["pred_masks"][frame_index])
            for index, segment in enumerate(output["segments_infos"]):
                mask = panoptic == int(segment["id"])
                if mask.any():
                    predictions.append(
                        (mask, int(segment["category_id"]), float(scores[index, frame_index]))
                    )
        elif output["task"] == "vss":
            masks = _as_numpy(output.get("dynamic_masks", []))
            labels = _as_numpy(output.get("dynamic_labels", []))
            for index, label in enumerate(labels):
                mask = masks[index, frame_index].astype(bool)
                if mask.any():
                    predictions.append((mask, int(label), float(scores[index, frame_index])))
        return predictions

    def _vps_class(self, raw_category):
        raw_category = int(raw_category)
        thing_map = self.metadata.thing_dataset_id_to_contiguous_id
        stuff_map = self.metadata.stuff_dataset_id_to_contiguous_id
        if raw_category in thing_map:
            return sorted(thing_map.values()).index(thing_map[raw_category])
        return len(thing_map) + sorted(stuff_map.values()).index(stuff_map[raw_category])

    def _ground_truth(self, sample, targets, frame_index):
        file_name = sample["file_names"][frame_index]
        records = self.store.frame_records(targets, file_name)
        valid_records = {item["source_id"]: item for item in records if item["valid"]}
        ground_truth = []

        if self.dataset in {"ovis", "ytvis2022"}:
            annotations = sample.get("dynamic_annotations", [])[frame_index]
            for annotation in annotations:
                record = valid_records.get(str(annotation["id"]))
                if record is None:
                    continue
                mask = _decode_vis_mask(
                    annotation["segmentation"],
                    int(sample["height"]),
                    int(sample["width"]),
                )
                ground_truth.append(
                    (mask, int(annotation["category_id"]), int(record["target"]))
                )
            return ground_truth

        if self.dataset == "vipseg":
            panoptic = rgb2id(
                np.asarray(
                    Image.open(sample["pan_seg_file_names"][frame_index]).convert("RGB")
                )
            )
            info_by_id = {
                str(item["id"]): item for item in sample["segments_infos"][frame_index]
            }
            for source_id, record in valid_records.items():
                info = info_by_id.get(source_id)
                if info is None:
                    continue
                mask = panoptic == int(source_id)
                if mask.any():
                    ground_truth.append(
                        (mask, self._vps_class(info["category_id"]), int(record["target"]))
                    )
            return ground_truth

        semantic = np.asarray(Image.open(sample["sem_mask_names"][frame_index]))
        records_by_category = defaultdict(list)
        for record in records:
            if record["valid"]:
                records_by_category[int(record["category_id"])].append(record)
        for category_id, category_records in records_by_category.items():
            mask = semantic == category_id
            if not mask.any():
                continue
            contiguous_id = self.metadata.stuff_dataset_id_to_contiguous_id.get(
                category_id - 1
            )
            if contiguous_id is None:
                continue
            total_area = sum(max(item["area"], 1.0) for item in category_records)
            target = sum(
                item["target"] * max(item["area"], 1.0)
                for item in category_records
            ) / total_area
            ground_truth.append((mask, int(contiguous_id), int(target >= 0.5)))
        return ground_truth

    def _accumulate_frame(self, predictions, ground_truth):
        self.stats["gt"] += len(ground_truth)
        self.stats["pred"] += len(predictions)
        candidates = []
        for pred_index, (pred_mask, pred_class, _) in enumerate(predictions):
            pred_area = pred_mask.sum()
            for gt_index, (gt_mask, gt_class, _) in enumerate(ground_truth):
                if pred_class != gt_class:
                    continue
                intersection = np.logical_and(pred_mask, gt_mask).sum()
                union = pred_area + gt_mask.sum() - intersection
                iou = float(intersection / union) if union else 0.0
                if iou >= self.iou_threshold:
                    candidates.append((iou, pred_index, gt_index))

        used_predictions, used_ground_truth = set(), set()
        for iou, pred_index, gt_index in sorted(candidates, reverse=True):
            if pred_index in used_predictions or gt_index in used_ground_truth:
                continue
            used_predictions.add(pred_index)
            used_ground_truth.add(gt_index)
            score = predictions[pred_index][2]
            target = ground_truth[gt_index][2]
            decision = int(score >= 0.5)
            self.stats["matched"] += 1
            self.stats["iou_sum"] += iou
            self.stats["joint_correct"] += int(decision == target)
            self.stats["tp"] += int(decision == 1 and target == 1)
            self.stats["tn"] += int(decision == 0 and target == 0)
            self.stats["fp"] += int(decision == 1 and target == 0)
            self.stats["fn"] += int(decision == 0 and target == 1)
            self.labels.append(target)
            self.scores.append(score)

    def process(self, inputs, outputs):
        assert len(inputs) == 1
        sample = inputs[0]
        targets = self.store.load(self._sequence_id(sample))
        for output_index, frame_index in enumerate(sample["frame_idx"]):
            predictions = self._predictions(outputs, output_index)
            ground_truth = self._ground_truth(sample, targets, frame_index)
            self._accumulate_frame(predictions, ground_truth)

    def evaluate(self):
        payload = {
            "stats": dict(self.stats),
            "labels": self.labels,
            "scores": self.scores,
        }
        if self.distributed:
            comm.synchronize()
            gathered = comm.gather(payload, dst=0)
            if not comm.is_main_process():
                return {}
        else:
            gathered = [payload]

        stats = defaultdict(float)
        labels, scores = [], []
        for item in gathered:
            for key, value in item["stats"].items():
                stats[key] += value
            labels.extend(item["labels"])
            scores.extend(item["scores"])
        return {"dynamic": dynamic_metrics(stats, labels, scores)}
