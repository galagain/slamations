#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Render DynEOMT semantic and dynamic overlays for a TUM RGB-D sequence.")
    parser.add_argument("--repo", required=True, help="DynEOMT repository root")
    parser.add_argument("--config", required=True, help="DynEOMT yaml config")
    parser.add_argument("--checkpoint", required=True, help="DynEOMT checkpoint")
    parser.add_argument("--sequence", required=True, help="TUM RGB-D sequence directory")
    parser.add_argument("--association", required=True, help="TUM association file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--short-side", type=int, default=384)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--dynamic-threshold", type=float, default=0.5)
    parser.add_argument("--min-label-area", type=int, default=1800)
    parser.add_argument("--max-labels", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_associations(path, max_frames):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            records.append({"rgb_timestamp": float(parts[0]), "rgb": parts[1]})
            if max_frames > 0 and len(records) >= max_frames:
                break
    return records


def resize_keep_aspect(image, short_side, max_side):
    height, width = image.shape[:2]
    scale = float(short_side) / float(min(height, width))
    if max(height, width) * scale > max_side:
        scale = float(max_side) / float(max(height, width))
    if abs(scale - 1.0) < 1e-6:
        return image
    return cv2.resize(image, (int(round(width * scale)), int(round(height * scale))), interpolation=cv2.INTER_LINEAR)


def as_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def stable_color(key):
    import hashlib

    digest = hashlib.sha1(key.encode("utf-8")).digest()
    hsv = np.uint8([[[int(digest[0]), 155 + digest[1] % 80, 205 + digest[2] % 50]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def text_box(image, text, org, scale=0.45, color=(255, 255, 255), thickness=1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    x = max(2, min(x, image.shape[1] - tw - 6))
    y = max(th + 4, min(y, image.shape[0] - base - 4))
    cv2.rectangle(image, (x - 3, y - th - 4), (x + tw + 3, y + base + 3), (15, 18, 22), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def fit_text(text, max_chars=34):
    text = text.replace("_", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "."


def build_model(repo, config_path, checkpoint_path, device_arg):
    sys.path.insert(0, str(repo))
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model as d2_build_model
    from detectron2.projects.deeplab import add_deeplab_config
    from videomt import add_videomt_config
    from videomt.data_video.datasets.vps import get_metadata

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_videomt_config(cfg)
    cfg.merge_from_file(str(config_path))
    cfg.merge_from_list([
        "MODEL.WEIGHTS",
        str(checkpoint_path),
        "INPUT.SAMPLING_FRAME_NUM",
        "1",
        "DATALOADER.NUM_WORKERS",
        "0",
        "TEST.AUG.ENABLED",
        "False",
    ])
    cfg.freeze()

    device = torch.device(device_arg if device_arg.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = d2_build_model(cfg).eval().to(device)
    DetectionCheckpointer(model).load(str(checkpoint_path))
    model.window_inference = True
    model.window_size = 1
    return model, device, get_metadata(None)


def class_name(metadata, class_id, isthing):
    thing_names = metadata.get("thing_classes") or []
    stuff_names = metadata.get("stuff_classes") or []
    if class_id < len(thing_names):
        return str(thing_names[class_id])
    stuff_id = class_id - len(thing_names)
    if 0 <= stuff_id < len(stuff_names):
        return str(stuff_names[stuff_id])
    category = (metadata.get("categories") or {}).get(class_id)
    if category:
        return str(category["name"])
    kind = "thing" if isthing else "stuff"
    return f"{kind}_{class_id}"


def input_record(frame_bgr, keep, short_side, max_side):
    original_height, original_width = frame_bgr.shape[:2]
    resized = resize_keep_aspect(frame_bgr, short_side, max_side)
    frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.as_tensor(np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))).float()
    return {
        "image": [tensor],
        "height": original_height,
        "width": original_width,
        "keep": keep,
    }


def extract_records(output, metadata, shape, dynamic_threshold):
    if isinstance(output, (list, tuple)):
        output = output[0]
    panoptic = as_numpy(output["pred_masks"])
    if panoptic.ndim == 3:
        panoptic = panoptic[0]
    if panoptic.shape != shape:
        panoptic = cv2.resize(panoptic.astype(np.int32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    dynamic_scores = as_numpy(output.get("dynamic_scores"))
    pred_ids = output.get("pred_ids") or []
    records = []
    dynamic_mask = np.zeros(shape, dtype=np.uint8)

    for index, segment in enumerate(output.get("segments_infos", [])):
        segment_id = int(segment["id"])
        category_id = int(segment["category_id"])
        isthing = bool(segment["isthing"])
        mask = panoptic == segment_id
        area = int(mask.sum())
        if area <= 0:
            continue
        probability = None
        if dynamic_scores is not None and index < dynamic_scores.shape[0]:
            probability = float(np.ravel(dynamic_scores[index])[0])
        dynamic = probability is not None and probability >= dynamic_threshold
        if dynamic:
            dynamic_mask[mask] = 255
        track_id = None
        if index < len(pred_ids):
            track_id = int(as_numpy(pred_ids[index]).ravel()[0])
        name = class_name(metadata, category_id, isthing)
        color_key = f"track-{track_id}" if isthing and track_id is not None else f"class-{category_id}"
        records.append({
            "segment_id": segment_id,
            "category_id": category_id,
            "class": name,
            "isthing": isthing,
            "track_id": track_id,
            "dynamic_probability": probability,
            "dynamic": dynamic,
            "area": area,
            "mask": mask,
            "color": stable_color(color_key),
        })
    return records, dynamic_mask


def render_semantic_panel(image, records, args):
    panel = image.astype(np.float32).copy()
    for record in sorted(records, key=lambda item: item["area"], reverse=True):
        color = np.asarray(record["color"], dtype=np.float32)
        panel[record["mask"]] = 0.50 * panel[record["mask"]] + 0.50 * color
    panel = np.clip(panel, 0, 255).astype(np.uint8)

    for record in records:
        contours, _ = cv2.findContours(record["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, record["color"], 1, cv2.LINE_AA)

    label_records = [r for r in records if r["area"] >= args.min_label_area]
    label_records.sort(key=lambda item: item["area"], reverse=True)
    for record in label_records[: args.max_labels]:
        moments = cv2.moments(record["mask"].astype(np.uint8))
        if abs(moments["m00"]) < 1e-6:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        label = fit_text(record["class"])
        if record["track_id"] is not None and record["isthing"]:
            label = f"{label} q{record['track_id']}"
        text_box(panel, label, (cx - 20, cy), scale=0.42)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (16, 19, 24), -1)
    cv2.putText(panel, "Semantique: classes et segments suivis", (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def orb_points(image, dynamic_mask):
    orb = cv2.ORB_create(nfeatures=1200, fastThreshold=12)
    keypoints = orb.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)
    kept = []
    removed = []
    height, width = dynamic_mask.shape[:2]
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        if dynamic_mask[y, x] > 0:
            removed.append((x, y))
        else:
            kept.append((x, y))
    return kept, removed


def render_dynamic_panel(image, dynamic_mask, kept, removed, records):
    panel = image.copy()
    dynamic = dynamic_mask > 0
    panel[dynamic] = (0.35 * panel[dynamic] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
    contours, _ = cv2.findContours(dynamic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, (0, 0, 180), 2, cv2.LINE_AA)

    for x, y in kept[:550]:
        cv2.circle(panel, (x, y), 1, (60, 240, 90), -1, cv2.LINE_AA)
    for x, y in removed[:300]:
        cv2.drawMarker(panel, (x, y), (20, 30, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)

    dynamic_records = [r for r in records if r["dynamic"]]
    dynamic_records.sort(key=lambda item: item["area"], reverse=True)
    for record in dynamic_records[:5]:
        moments = cv2.moments(record["mask"].astype(np.uint8))
        if abs(moments["m00"]) < 1e-6:
            continue
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        probability = record["dynamic_probability"] or 0.0
        text_box(panel, f"{fit_text(record['class'], 22)} dyn {probability:.2f}", (cx - 30, cy), scale=0.40)

    cv2.rectangle(panel, (0, 0), (panel.shape[1], 54), (16, 19, 24), -1)
    pct = 100.0 * float(np.count_nonzero(dynamic_mask)) / float(dynamic_mask.size)
    title = f"Dynamique + features: gardees {len(kept)} | filtrees {len(removed)} | pixels dyn {pct:.1f}%"
    cv2.putText(panel, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, "verts = candidates gardees, rouges = candidates supprimees", (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 225, 230), 1, cv2.LINE_AA)
    return panel


def footer(width, records):
    panel = np.full((62, width, 3), (16, 19, 24), dtype=np.uint8)
    classes = {}
    for record in records:
        item = classes.setdefault(record["class"], {"area": 0, "dynamic": False})
        item["area"] += record["area"]
        item["dynamic"] = item["dynamic"] or record["dynamic"]
    ordered = sorted(classes.items(), key=lambda item: item[1]["area"], reverse=True)[:8]
    x = 12
    y = 25
    for name, info in ordered:
        color = (0, 0, 230) if info["dynamic"] else stable_color(name)
        label = fit_text(name, 20)
        cv2.rectangle(panel, (x, y - 12), (x + 16, y + 3), color, -1)
        cv2.putText(panel, label, (x + 22, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        x += 30 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
        if x > width - 160:
            x = 12
            y = 51
    return panel


def compose_frame(image, records, dynamic_mask, kept, removed, frame_index, frame_count, timestamp, args):
    left = render_semantic_panel(image, records, args)
    right = render_dynamic_panel(image, dynamic_mask, kept, removed, records)
    content = np.hstack([left, right])
    header = np.full((42, content.shape[1], 3), (16, 19, 24), dtype=np.uint8)
    title = f"DynSLAM / DynEOMT ViT-L | fr3 walking xyz | frame {frame_index}/{frame_count} | t={timestamp:.3f}"
    cv2.putText(header, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (248, 248, 248), 1, cv2.LINE_AA)
    return np.vstack([header, content, footer(content.shape[1], records)])


def main():
    args = parse_args()
    sequence = Path(args.sequence).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_associations(args.association, args.max_frames)
    if not records:
        raise RuntimeError(f"No frames in {args.association}")

    model, device, metadata = build_model(Path(args.repo).resolve(), Path(args.config).resolve(), Path(args.checkpoint).resolve(), args.device)
    first = cv2.imread(str(sequence / records[0]["rgb"]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Cannot read {sequence / records[0]['rgb']}")

    frame_h, frame_w = first.shape[:2]
    out_w, out_h = frame_w * 2, frame_h + 42 + 62
    video_path = output_dir / "dyneomt_semantic_dynamic_overlay.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {video_path}")

    report = []
    latencies = []
    start_wall = time.perf_counter()
    try:
        for frame_index, record in enumerate(records, start=1):
            image = cv2.imread(str(sequence / record["rgb"]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read {sequence / record['rgb']}")
            if image.shape[:2] != (frame_h, frame_w):
                image = cv2.resize(image, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)

            batch = [input_record(image, frame_index > 1, args.short_side, args.max_side)]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        output = model(batch)
                else:
                    output = model(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency = time.perf_counter() - t0
            latencies.append(latency)

            segment_records, dynamic_mask = extract_records(output, metadata, image.shape[:2], args.dynamic_threshold)
            kept, removed = orb_points(image, dynamic_mask)
            frame = compose_frame(image, segment_records, dynamic_mask, kept, removed, frame_index, len(records), record["rgb_timestamp"], args)
            writer.write(frame)

            report.append({
                "frame": frame_index,
                "rgb": record["rgb"],
                "rgb_timestamp": record["rgb_timestamp"],
                "latency_seconds": latency,
                "dynamic_pixels": int(np.count_nonzero(dynamic_mask)),
                "orb_candidates_kept": len(kept),
                "orb_candidates_filtered": len(removed),
                "segments": [
                    {k: v for k, v in segment.items() if k not in {"mask", "color"}}
                    for segment in segment_records
                ],
            })
            if frame_index == 1 or frame_index % 25 == 0 or frame_index == len(records):
                elapsed = time.perf_counter() - start_wall
                fps = frame_index / elapsed if elapsed > 0 else math.nan
                print(f"{frame_index:5d}/{len(records)} frames | latency {latency:.3f}s | render {fps:.2f} fps", flush=True)
    finally:
        writer.release()

    summary = {
        "video": str(video_path),
        "frames": len(records),
        "fps": args.fps,
        "size": [out_w, out_h],
        "mean_latency_seconds": float(np.mean(latencies)) if latencies else 0.0,
        "throughput_fps": len(records) / (time.perf_counter() - start_wall),
        "dynamic_frames": sum(1 for item in report if item["dynamic_pixels"] > 0),
        "mean_dynamic_pixels": float(np.mean([item["dynamic_pixels"] for item in report])) if report else 0.0,
        "mean_orb_candidates_kept": float(np.mean([item["orb_candidates_kept"] for item in report])) if report else 0.0,
        "mean_orb_candidates_filtered": float(np.mean([item["orb_candidates_filtered"] for item in report])) if report else 0.0,
    }
    (output_dir / "dyneomt_semantic_dynamic_overlay_report.json").write_text(json.dumps({"summary": summary, "frames": report}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
