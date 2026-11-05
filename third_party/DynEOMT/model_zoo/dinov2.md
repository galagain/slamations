# VidEoMT Initialization Model Zoo - DINOv2

This publication repository retains configurations only for the four datasets
evaluated with DynEoMT. Reported values below are the original VidEoMT results.
FPS was measured by the authors on an NVIDIA H100 with `torch.compile`.

## Video Instance Segmentation

| Dataset | Config | AP | AR10 | FPS | Model weights |
|---|---|---:|---:|---:|---|
| YouTube-VIS 2022 | [VidEoMT-L](../configs/VidEoMT/ytvis22/dinov2/vit-large/videomt_online_vitl.yaml) | 42.6 | 48.1 | 161 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2022_vit_large_42.6.pth?download=true) |
| OVIS | [VidEoMT-L](../configs/VidEoMT/ovis/dinov2/vit-large/videomt_online_vitl.yaml) | 52.5 | 57.5 | 115 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/ovis_vit_large_52.5.pth?download=true) |

## Video Panoptic Segmentation

| Dataset | Config | VPQ | STQ | FPS | Model weights |
|---|---|---:|---:|---:|---|
| VIPSeg | [VidEoMT-L](../configs/VidEoMT/VIPSeg/dinov2/vit-large/videomt_online_vitl.yaml) | 55.2 | 48.9 | 75 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/vipseg_vit_large_55.2.pth?download=true) |

## Video Semantic Segmentation

| Dataset | Config | mVC16 | mIoU | FPS | Model weights |
|---|---|---:|---:|---:|---|
| VSPW | [VidEoMT-L](../configs/VidEoMT/VSPW/dinov2/vit-large/videomt_online_vitl.yaml) | 95.0 | 64.9 | 73 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/vspw_vit_large_95.0_64.9.pth?download=true) |

## ViT-S/B Transfer Initialization

No dataset-specific VidEoMT-S/B weights are published for the four benchmarks
above. DynEoMT-S/B starts from the official YouTube-VIS 2019 video checkpoints so
that the temporal-query mechanism, not only the image segmenter, is initialized:

| Backbone | Source AP | Model weights |
|---|---:|---|
| ViT-S | 52.8 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_small_52.8.pth?download=true) |
| ViT-B | 58.2 | [download](https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_base_58.2.pth?download=true) |
