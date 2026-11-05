#!/usr/bin/env python3
"""Export DynEoMT/VidEoMT backbones to ONNX and TensorRT.

This exporter intentionally targets the neural backbone only. The Detectron2
meta-architecture and dataset evaluators still perform Python post-processing,
so these engines are meant for runtime profiling or later integration in a
TensorRT-backed backbone wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from videomt import add_videomt_config


@dataclass(frozen=True)
class ModelSpec:
    family: str
    dataset: str
    backbone: str
    config: Path
    weights: Path

    @property
    def name(self) -> str:
        return f"{self.family}_{self.dataset}_{self.backbone}"


CONFIG_DATASET = {
    "vipseg": "VIPSeg",
    "ovis": "ovis",
    "ytvis2022": "ytvis22",
    "vspw": "VSPW",
}

CONFIG_STEM = {
    "dyneomt": {
        "vit-small": "dyneomt_online_vits.yaml",
        "vit-base": "dyneomt_online_vitb.yaml",
        "vit-large": "dyneomt_online_vitl.yaml",
    },
    "videomt": {
        "vit-small": "videomt_online_vits.yaml",
        "vit-base": "videomt_online_vitb.yaml",
        "vit-large": "videomt_online_vitl.yaml",
    },
}


class BackboneExportWrapper(torch.nn.Module):
    """Return a tensor tuple instead of the backbone output dictionary."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = self.backbone(image, resume=False)
        tensors = [outputs["pred_logits"], outputs["pred_masks"]]
        if "pred_dynamic_logits" in outputs:
            tensors.append(outputs["pred_dynamic_logits"])
        return tuple(tensors)


def discover_specs(output_roots: Iterable[Path]) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for output_root in output_roots:
        root_name = output_root.name
        family = "videomt" if root_name.startswith("videomt") else "dyneomt"
        weight_paths = [(p, None) for p in sorted(output_root.glob("*/*/model_final.pth"))]
        weight_paths.extend((p, "vit-large") for p in sorted(output_root.glob("*/model_final.pth")))
        seen_weights: set[Path] = set()
        for weights, forced_backbone in weight_paths:
            if weights in seen_weights:
                continue
            seen_weights.add(weights)
            if forced_backbone is None:
                dataset = weights.parents[1].name
                backbone = weights.parents[0].name
            else:
                dataset = weights.parents[0].name
                backbone = forced_backbone
            cfg_dataset = CONFIG_DATASET.get(dataset)
            cfg_stem = CONFIG_STEM[family].get(backbone)
            if cfg_dataset is None or cfg_stem is None:
                continue
            config = (
                REPO_ROOT
                / "configs"
                / ("VidEoMT" if family == "videomt" else "DynEoMT")
                / cfg_dataset
                / "dinov2"
                / backbone
                / cfg_stem
            )
            if not config.exists():
                continue
            specs.append(ModelSpec(root_name, dataset, backbone, config, weights))
    return specs


def build_cfg(config: Path, weights: Path):
    cfg = get_cfg()
    add_videomt_config(cfg)
    cfg.merge_from_file(str(config))
    cfg.merge_from_list(["MODEL.WEIGHTS", str(weights)])
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def load_wrapper(spec: ModelSpec, device: str) -> tuple[BackboneExportWrapper, int]:
    cfg = build_cfg(spec.config, spec.weights)
    model = build_model(cfg).eval().to(device)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.backbone.eval()
    return BackboneExportWrapper(model.backbone).eval().to(device), int(cfg.MODEL.BACKBONE.IMG_SIZE)


def export_onnx(wrapper: torch.nn.Module, image: torch.Tensor, onnx_path: Path) -> list[list[int]]:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        outputs = wrapper(image)
    output_names = ["pred_logits", "pred_masks"]
    if len(outputs) == 3:
        output_names.append("pred_dynamic_logits")
    torch.onnx.export(
        wrapper,
        image,
        str(onnx_path),
        input_names=["image"],
        output_names=output_names,
        opset_version=20,
        dynamo=True,
    )
    return [list(t.shape) for t in outputs]


def build_tensorrt_engine(onnx_path: Path, engine_path: Path, workspace_gb: int) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("\n".join(errors))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned an empty serialized engine")
    engine_path.write_bytes(serialized)


def benchmark_torch(wrapper: torch.nn.Module, image: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(warmup):
            wrapper(image)
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(iters):
            start.record()
            wrapper(image)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    mean_ms = float(sum(times) / len(times))
    return {"torch_ms": mean_ms, "torch_fps": 1000.0 / mean_ms}


def measure_gflops(wrapper: torch.nn.Module, image: torch.Tensor) -> dict[str, float | str]:
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(wrapper, image)
        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        return {"gflops": float(flops.total() / 1e9)}
    except Exception as exc:
        return {"gflops_error": repr(exc)}


def benchmark_trt(engine_path: Path, image: torch.Tensor, warmup: int, iters: int) -> dict[str, float]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    bindings: dict[str, torch.Tensor] = {}
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            bindings[name] = image
        else:
            shape = tuple(context.get_tensor_shape(name))
            bindings[name] = torch.empty(shape, device=image.device, dtype=torch.float32)
        context.set_tensor_address(name, bindings[name].data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        context.execute_async_v3(stream)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start.record()
        context.execute_async_v3(stream)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    mean_ms = float(sum(times) / len(times))
    return {"trt_ms": mean_ms, "trt_fps": 1000.0 / mean_ms}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-roots", nargs="+", default=["outputs/dyneomt_corrected_temporal_targets", "outputs/dyneomt"])
    parser.add_argument("--export-root", default="../outputs/tensorrt_exports")
    parser.add_argument("--only", nargs="*", default=None, help="Substrings that must match the model name.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-export", action="store_true", help="Only run PyTorch timing/GFLOPs; do not export ONNX/TRT.")
    parser.add_argument("--no-flops", action="store_true", help="Skip fvcore GFLOPs analysis.")
    args = parser.parse_args()

    os.environ.setdefault("DETECTRON2_DATASETS", str(REPO_ROOT / "datasets"))
    os.chdir(REPO_ROOT)
    export_root = (REPO_ROOT / args.export_root).resolve()
    report_path = export_root / "backbone_tensorrt_report.jsonl"
    specs = discover_specs(REPO_ROOT / Path(root) for root in args.output_roots)
    if args.only:
        specs = [s for s in specs if all(token in s.name for token in args.only)]

    export_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        model_dir = export_root / spec.family / spec.dataset / spec.backbone
        onnx_path = model_dir / "backbone_firstframe.onnx"
        engine_path = model_dir / "backbone_firstframe.trt"
        record = {
            "model": spec.name,
            "config": str(spec.config.relative_to(REPO_ROOT)),
            "weights": str(spec.weights.relative_to(REPO_ROOT)),
            "onnx": str(onnx_path),
            "engine": str(engine_path),
            "scope": "backbone_first_frame_fixed_native_resolution",
        }
        try:
            wrapper, input_size = load_wrapper(spec, args.device)
            image = torch.randn(1, 3, input_size, input_size, device=args.device)
            record["input_shape"] = list(image.shape)
            with torch.no_grad():
                record["output_shapes"] = [list(t.shape) for t in wrapper(image)]
            if not args.no_flops:
                record.update(measure_gflops(wrapper, image))
            if not args.no_export:
                if not (args.skip_existing and onnx_path.exists()):
                    record["output_shapes"] = export_onnx(wrapper, image, onnx_path)
            record.update(benchmark_torch(wrapper, image, args.warmup, args.iters))
            if not args.no_export:
                input_shape = tuple(image.shape)
                del wrapper
                del image
                torch.cuda.empty_cache()
                if not (args.skip_existing and engine_path.exists()):
                    build_tensorrt_engine(onnx_path, engine_path, args.workspace_gb)
                image = torch.randn(*input_shape, device=args.device)
                record.update(benchmark_trt(engine_path, image, args.warmup, args.iters))
            record["status"] = "ok"
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = repr(exc)
        with report_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
