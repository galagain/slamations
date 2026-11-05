"""Minimal MemFlow inference wrapper for pseudo-label generation."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FlowOptions:
    max_side: int | None = 1280
    dtype: str = "float16"
    reverse: bool = False


def normalize_cuda_devices(devices: list[str]) -> list[str]:
    normalized = []
    for device in devices:
        value = str(device).strip()
        if value.isdigit():
            normalized.append(f"cuda:{value}")
        elif value.startswith("cuda:"):
            normalized.append(value)
        elif value:
            raise ValueError(f"Expected a CUDA index or cuda:N, got {device!r}")
    if not normalized:
        raise ValueError("At least one CUDA device is required")
    return normalized


def load_model(repo: Path, checkpoint: Path, device: torch.device):
    if not (repo / "core").is_dir():
        raise FileNotFoundError(f"MemFlow repository not found: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"MemFlow checkpoint not found: {checkpoint}")
    sys.path.insert(0, str(repo / "core"))
    sys.path.insert(0, str(repo))

    from configs.sintel_memflownet import get_cfg
    from core.Networks import build_network
    from core.utils.utils import InputPadder, forward_interpolate
    from inference import inference_core_skflow

    cfg = get_cfg()
    cfg.restore_ckpt = str(checkpoint)
    cfg.MemFlowNet_skflow.pretrain = False
    cfg.val_decoder_depth = min(int(cfg.val_decoder_depth), 15)

    model = build_network(cfg)
    loaded = torch.load(cfg.restore_ckpt, map_location="cpu")
    state = loaded.get("model", loaded) if isinstance(loaded, dict) else loaded
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    return cfg, model, InputPadder, forward_interpolate, inference_core_skflow


def _frame_paths(sequence_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sequence_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _target_size(path: Path, max_side: int | None) -> tuple[int, int] | None:
    if max_side is None:
        return None
    with Image.open(path) as image:
        width, height = image.size
    if max(width, height) <= max_side:
        return None
    scale = max_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _load_frame(
    path: Path, resize: tuple[int, int] | None
) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_size = image.size
        if resize is not None:
            image = image.resize(resize, Image.Resampling.LANCZOS)
        array = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(array).permute(2, 0, 1).float(), original_size


def _reset_output(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "npz").mkdir(parents=True)


def run_sequence(
    sequence_dir: Path,
    output_dir: Path,
    runtime,
    device: torch.device,
    options: FlowOptions,
) -> int:
    """Predict adjacent flow while retaining MemFlow memory within the sequence."""
    cfg, model, input_padder_cls, forward_interpolate, inference_core = runtime
    frames = _frame_paths(sequence_dir)
    if options.reverse:
        frames.reverse()
    if len(frames) < 2:
        raise RuntimeError(f"Need at least two frames in {sequence_dir}")

    resize = _target_size(frames[0], options.max_side)
    first_frame, source_size = _load_frame(frames[0], resize)
    processed_size = (first_frame.shape[-1], first_frame.shape[-2])
    _reset_output(output_dir)
    processor = inference_core.InferenceCore(model, config=cfg)
    previous_frame = first_frame
    previous_flow = None

    with torch.inference_mode():
        for index in range(len(frames) - 1):
            next_frame, next_source_size = _load_frame(frames[index + 1], resize)
            if next_source_size != source_size:
                raise RuntimeError(
                    f"Frame size changed at {frames[index + 1]}: "
                    f"{source_size} -> {next_source_size}"
                )
            images = torch.stack([previous_frame, next_frame]).unsqueeze(0).to(device)
            padder = input_padder_cls(images.shape)
            images = 2.0 * (padder.pad(images) / 255.0) - 1.0
            flow_low, flow_prediction = processor.step(
                images,
                end=index == len(frames) - 2,
                add_pe=("rope" in cfg and cfg.rope),
                flow_init=previous_flow,
            )
            flow = padder.unpad(flow_prediction[0]).permute(1, 2, 0).cpu().numpy()
            flow = flow.astype(np.float16 if options.dtype == "float16" else np.float32)
            np.savez(
                output_dir / "npz" / f"flow_{index:04d}_to_{index + 1:04d}.npz",
                flow=flow,
            )
            if "warm_start" in cfg and cfg.warm_start:
                previous_flow = forward_interpolate(flow_low[0]).to(device)[None]
            previous_frame = next_frame

    metadata = {
        "sequence_id": sequence_dir.name,
        "model": "MemFlow",
        "direction": "backward" if options.reverse else "forward",
        "frames": [path.name for path in frames],
        "frame_count": len(frames),
        "source_frame_size": list(source_size),
        "processed_frame_size": list(processed_size),
        "flow_coordinate_system": "processed_frame_pixels",
        "npz_dtype": options.dtype,
        "max_side": options.max_side,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return len(frames) - 1


def cleanup_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
