#!/usr/bin/env python3
"""Build DynEoMT pseudo-labels for all supported video datasets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path
from queue import Empty

from tqdm import tqdm

from dynamic_targets import build_sequence_targets
from dynamic_targets import memflow


ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_DATASETS = ("vipseg", "ovis", "ytvis2022", "vspw")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _vis_tasks(dataset: str, data_root: Path) -> list[dict]:
    tasks = []
    for split in ("train", "valid"):
        if dataset == "ovis":
            annotation_path = (
                data_root / "ovis" / "annotations" / f"annotations_{split}.json"
            )
            image_root = data_root / "ovis" / split
        else:
            annotation_path = data_root / "ytvis_2022" / f"{split}.json"
            image_root = data_root / "ytvis_2022" / split / "JPEGImages"
        if not annotation_path.is_file():
            continue
        payload = _read_json(annotation_path)
        annotated_ids = {
            annotation["video_id"] for annotation in payload.get("annotations", [])
        }
        for video in payload.get("videos", []):
            if video["id"] not in annotated_ids or not video.get("file_names"):
                continue
            sequence_id = video["file_names"][0].split("/")[0]
            tasks.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "sequence_id": sequence_id,
                    "frame_dir": str(image_root / sequence_id),
                    "frame_count": len(video["file_names"]),
                }
            )
    return tasks


def _vipseg_tasks(data_root: Path) -> list[dict]:
    tasks = []
    root = data_root / "VIPSeg" / "VIPSeg_720P"
    for split in ("train", "val"):
        annotation_path = root / f"panoptic_gt_VIPSeg_{split}.json"
        if not annotation_path.is_file():
            continue
        payload = _read_json(annotation_path)
        annotations = {
            annotation["video_id"]: annotation["annotations"]
            for annotation in payload["annotations"]
        }
        for video in payload["videos"]:
            sequence_id = video["video_id"]
            tasks.append(
                {
                    "dataset": "vipseg",
                    "split": split,
                    "sequence_id": sequence_id,
                    "frame_dir": str(root / "images" / sequence_id),
                    "frame_count": len(annotations[sequence_id]),
                }
            )
    return tasks


def _vspw_tasks(data_root: Path) -> list[dict]:
    tasks = []
    root = data_root / "VSPW_480p"
    for split in ("train", "val"):
        split_path = root / f"{split}.txt"
        if not split_path.is_file():
            continue
        sequence_ids = [
            line.strip() for line in split_path.read_text().splitlines() if line.strip()
        ]
        for sequence_id in sequence_ids:
            frame_dir = root / "data" / sequence_id / "origin"
            tasks.append(
                {
                    "dataset": "vspw",
                    "split": split,
                    "sequence_id": sequence_id,
                    "frame_dir": str(frame_dir),
                    "frame_count": len(list(frame_dir.glob("*.jpg"))),
                }
            )
    return tasks


def collect_tasks(data_root: Path, datasets: list[str]) -> list[dict]:
    tasks = []
    if "vipseg" in datasets:
        tasks.extend(_vipseg_tasks(data_root))
    if "ovis" in datasets:
        tasks.extend(_vis_tasks("ovis", data_root))
    if "ytvis2022" in datasets:
        tasks.extend(_vis_tasks("ytvis2022", data_root))
    if "vspw" in datasets:
        tasks.extend(_vspw_tasks(data_root))
    return sorted(
        tasks,
        key=lambda task: (
            -task["frame_count"],
            task["dataset"],
            task["sequence_id"],
        ),
    )


def target_path(output_root: Path, task: dict) -> Path:
    return (
        output_root
        / "labels"
        / task["dataset"]
        / task["sequence_id"]
        / "targets.json"
    )


def target_complete(path: Path, task: dict) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == 2
        and payload.get("dataset") == task["dataset"]
        and payload.get("split") == task["split"]
        and payload.get("sequence_id") == task["sequence_id"]
        and payload.get("transition_count") == max(task["frame_count"] - 1, 0)
    )


def _worker(
    worker_id: int,
    device_name: str,
    task_queue,
    result_queue,
    settings: dict,
) -> None:
    device = memflow.torch.device(device_name)
    memflow.torch.cuda.set_device(device)
    runtime = memflow.load_model(
        Path(settings["memflow_repo"]), Path(settings["checkpoint"]), device
    )
    output_root = Path(settings["output_root"])
    temporary_root = output_root / "tmp" / f"worker_{worker_id}"
    while True:
        task = task_queue.get()
        if task is None:
            break
        started = time.monotonic()
        key = f"{task['dataset']}/{task['split']}/{task['sequence_id']}"
        task_temporary = temporary_root / task["dataset"] / task["sequence_id"]
        try:
            forward_dir = task_temporary / "forward"
            backward_dir = task_temporary / "backward"
            memflow.run_sequence(
                Path(task["frame_dir"]),
                forward_dir,
                runtime,
                device,
                memflow.FlowOptions(
                    max_side=settings["max_side"], dtype=settings["dtype"]
                ),
            )
            memflow.run_sequence(
                Path(task["frame_dir"]),
                backward_dir,
                runtime,
                device,
                memflow.FlowOptions(
                    max_side=settings["max_side"],
                    dtype=settings["dtype"],
                    reverse=True,
                ),
            )
            output_path = build_sequence_targets(
                dataset=task["dataset"],
                split=task["split"],
                sequence_id=task["sequence_id"],
                data_root=Path(settings["data_root"]),
                forward_dir=forward_dir,
                backward_dir=backward_dir,
                output_root=output_root / "labels",
                save_artifacts=settings["save_artifacts"],
            )
            result_queue.put(
                {
                    "status": "completed",
                    "key": key,
                    "device": device_name,
                    "output": str(output_path),
                    "seconds": time.monotonic() - started,
                }
            )
        except Exception as error:
            result_queue.put(
                {
                    "status": "failed",
                    "key": key,
                    "device": device_name,
                    "error": f"{type(error).__name__}: {error}",
                    "seconds": time.monotonic() - started,
                }
            )
            memflow.cleanup_device_cache(device)
        finally:
            if not settings["keep_flows"]:
                shutil.rmtree(task_temporary, ignore_errors=True)
    result_queue.put(
        {"status": "worker_done", "device": device_name, "worker_id": worker_id}
    )


def _write_progress(
    path: Path, total: int, counts: dict, failures: list[dict], devices: list[str]
) -> None:
    payload = {
        "total_sequences": total,
        "counts": counts,
        "devices": devices,
        "failures": failures,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", nargs="+", choices=SUPPORTED_DATASETS, default=SUPPORTED_DATASETS
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        help="Optional exact sequence IDs to process (across the selected datasets)",
    )
    parser.add_argument("--devices", nargs="+", default=["0"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "datasets")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "dynamic_targets"
    )
    parser.add_argument(
        "--memflow-repo", type=Path, default=ROOT / "external" / "MemFlow"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "external" / "MemFlow" / "ckpts" / "MemFlowNet_sintel.pth",
    )
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--keep-flows", action="store_true")
    parser.add_argument("--save-artifacts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = memflow.normalize_cuda_devices(args.devices)
    tasks = collect_tasks(args.data_root, list(args.datasets))
    if args.sequences:
        selected_sequences = set(args.sequences)
        tasks = [task for task in tasks if task["sequence_id"] in selected_sequences]
    discovered = len(tasks)
    if not args.overwrite:
        tasks = [
            task
            for task in tasks
            if not target_complete(target_path(args.output_root, task), task)
        ]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    print(f"Discovered sequences: {discovered}")
    print(f"Sequences to process: {len(tasks)}")
    for dataset in args.datasets:
        count = sum(task["dataset"] == dataset for task in tasks)
        print(f"  {dataset}: {count}")
    print(f"Devices: {', '.join(devices)}")
    print(f"Output: {args.output_root}")
    if args.dry_run or not tasks:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    task_queue = context.Queue()
    result_queue = context.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in devices:
        task_queue.put(None)

    settings = vars(args).copy()
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in settings.items()
    }
    workers = [
        context.Process(
            target=_worker,
            args=(index, device, task_queue, result_queue, settings),
            name=f"dynamic-{device}",
        )
        for index, device in enumerate(devices)
    ]
    for process in workers:
        process.start()

    counts = {"completed": 0, "failed": 0}
    failures = []
    done_workers = 0
    state_path = args.output_root / "progress.json"
    _write_progress(state_path, len(tasks), counts, failures, devices)
    progress = tqdm(total=len(tasks), desc="dynamic targets", unit="seq")
    while done_workers < len(workers):
        try:
            message = result_queue.get(timeout=10)
        except Empty:
            crashed = [
                process
                for process in workers
                if process.exitcode is not None and process.exitcode != 0
            ]
            if crashed:
                raise RuntimeError(
                    f"Workers crashed: {[(p.name, p.exitcode) for p in crashed]}"
                )
            continue
        if message["status"] == "worker_done":
            done_workers += 1
            continue
        counts[message["status"]] += 1
        if message["status"] == "failed":
            failures.append(message)
            tqdm.write(
                f"FAILED {message['key']} on {message['device']}: {message['error']}"
            )
        progress.update(1)
        progress.set_postfix_str(
            f"{message['device']} {message['key']} failed={counts['failed']}"
        )
        _write_progress(state_path, len(tasks), counts, failures, devices)
    progress.close()
    for process in workers:
        process.join()
    _write_progress(state_path, len(tasks), counts, failures, devices)
    if failures:
        (args.output_root / "failures.json").write_text(
            json.dumps(failures, indent=2) + "\n"
        )
        raise SystemExit(f"Pseudo-label generation failed for {len(failures)} sequences")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
