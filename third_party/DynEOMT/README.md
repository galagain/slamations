# DynEoMT

DynEoMT extends [VidEoMT](https://github.com/tue-mps/videomt) with online,
query-level scene dynamicity prediction. One shared ViT predicts video masks,
semantic classes, track identities, and whether each tracked region is dynamic or
static.

The dynamic head uses the same query representation as the classification head.
At inference step `t`, its context is strictly limited to the current RGB image
`I_t`, learned queries, and track queries propagated from `t-1`. Past or future RGB
frames, optical flow, and depth are not model inputs. Optical flow is used only
offline to construct training and evaluation pseudo-labels.

## Method

For every decoder query, DynEoMT adds a small binary MLP head:

```text
current RGB frame + previous track queries
                    |
              shared ViT encoder
                    |
        +-----------+-----------+
        |           |           |
     classes       masks    dynamicity logit
```

The original online VidEoMT propagation remains unchanged. Training clips contain
three adjacent frames to unroll the recurrent query propagation, but each step
processes only its current frame and the preceding queries. The new head is trained
jointly with the original classification, mask, dice, and tracking objectives.

For Hungarian-matched queries with a confident pseudo-label `y in {0, 1}`, the
additional objective is weighted binary cross entropy:

```text
L = L_VidEoMT + lambda_dyn * BCEWithLogits(z_dyn, y; pos_weight)
```

Unmatched queries and targets marked `uncertain` are excluded from this loss.

## Installation

The following setup follows the original VidEoMT environment:

```bash
conda create -n dyneomt python=3.12.3 -y
conda activate dyneomt

pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt
```

Set the dataset root before training or evaluation:

```bash
export DETECTRON2_DATASETS="$PWD/datasets"
```

Weights & Biases is supported by the training code. Run `wandb login` to enable
remote experiment tracking.

## Raw Datasets

DynEoMT supports VIPSeg, OVIS, YouTube-VIS 2022, and VSPW. Download each dataset
from its official source and arrange it as follows:

```text
datasets/
  VIPSeg/VIPSeg_720P/
    images/<sequence>/*.jpg
    panomasksRGB/<sequence>/*.png
    panoptic_gt_VIPSeg_train.json
    panoptic_gt_VIPSeg_val.json
  ovis/
    annotations/annotations_train.json
    annotations/annotations_valid.json
    train/<sequence>/*
    valid/<sequence>/*
  ytvis_2022/
    train.json
    valid.json
    train/JPEGImages/<sequence>/*
    valid/JPEGImages/<sequence>/*
  VSPW_480p/
    data/<sequence>/origin/*.jpg
    data/<sequence>/mask/*.png
    train.txt
    val.txt
    label_num_dic_final.json
```

The original VidEoMT download and conversion notes are retained in
[`datasets/README.md`](datasets/README.md).

## Dynamic Target Dataset

The released target builder applies one common protocol to all four datasets:

1. Predict forward and backward dense optical flow with MemFlow. A complete
   sequence stays on one GPU so that MemFlow's temporal memory is preserved.
2. Reject inconsistent and occluded pixels using forward-backward consistency.
3. Fit translation, affine, homography, and quadratic camera-flow candidates with
   robust estimation, then select the best penalized model.
4. Subtract the selected camera field from the dense flow.
5. Erode each annotated mask and aggregate robust residual-motion statistics over
   its reliable interior pixels.
6. Convert the evidence to a calibrated probability, smooth it along track IDs,
   and assign `dynamic`, `static`, or `uncertain`.

The per-transition threshold is adaptive:

```text
tau_t = max(median_camera_error + 3 * robust_sigma,
            1.25 pixels,
            0.0015 * image_diagonal)
```

This combines a noise-dependent term, an absolute lower bound, and a
resolution-dependent lower bound. A target is dynamic at probability `>= 0.70`,
static at probability `<= 0.30`, and uncertain otherwise.

### 1. Install MemFlow

```bash
git clone https://github.com/DQiaole/MemFlow.git external/MemFlow
mkdir -p external/MemFlow/ckpts
```

Download `MemFlowNet_sintel.pth` from the official
[MemFlow v1.0.0 release](https://github.com/DQiaole/MemFlow/releases/tag/v1.0.0)
and place it at:

```text
external/MemFlow/ckpts/MemFlowNet_sintel.pth
```

### 2. Validate Dataset Discovery

The dry run parses all annotations and reports the work queue without loading a
network or writing outputs:

```bash
python scripts/build_dynamic_targets.py \
  --data-root datasets \
  --datasets vipseg ovis ytvis2022 vspw \
  --devices 0 1 2 3 \
  --dry-run
```

Add `--sequences <sequence_id> [...]` to inspect or regenerate exact sequences.

### 3. Generate All Targets

```bash
python scripts/build_dynamic_targets.py \
  --data-root datasets \
  --output-root outputs/dynamic_targets \
  --memflow-repo external/MemFlow \
  --checkpoint external/MemFlow/ckpts/MemFlowNet_sintel.pth \
  --datasets vipseg ovis ytvis2022 vspw \
  --devices 0 1 2 3
```

One worker is started per listed GPU and whole sequences are dispatched across
workers. The longest sequences are scheduled first. The command is resumable:
complete schema-v2 files are skipped unless `--overwrite` is specified. Dense
forward/backward flow is temporary by default; use `--keep-flows` for debugging or
`--save-artifacts` to retain camera and residual fields.

Progress is shown with `tqdm` and persisted to
`outputs/dynamic_targets/progress.json`. Failed sequences are listed in
`outputs/dynamic_targets/failures.json`.

### 4. Validate the Generated Dataset

```bash
python scripts/summarize_dynamic_targets.py \
  --root outputs/dynamic_targets/labels \
  --json-output outputs/dynamic_targets/summary.json \
  --strict
```

The generated structure is:

```text
outputs/dynamic_targets/labels/
  vipseg/<sequence>/targets.json
  ovis/<sequence>/targets.json
  ytvis2022/<sequence>/targets.json
  vspw/<sequence>/targets.json
```

Each schema-v2 file stores sequence metadata and, for every annotated mask and
transition, the source/track/category IDs, binary label, probability, confidence,
valid support, camera model, and residual-flow statistics. These files are read
directly by the training mappers.

## Initialization Weights

The ViT-L recipes initialize the shared network from the final, dataset-specific
VidEoMT weights and randomly initialize only the new dynamic head:

```bash
mkdir -p checkpoints

wget -O checkpoints/vipseg_vit_large_55.2.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/vipseg_vit_large_55.2.pth?download=true'
wget -O checkpoints/ovis_vit_large_52.5.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/ovis_vit_large_52.5.pth?download=true'
wget -O checkpoints/yt_2022_vit_large_42.6.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2022_vit_large_42.6.pth?download=true'
wget -O checkpoints/vspw_vit_large_95.0_64.9.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/vspw_vit_large_95.0_64.9.pth?download=true'
```

VidEoMT does not publish dataset-specific ViT-S or ViT-B models for these four
benchmarks. The S/B recipes therefore use the published temporally trained
YouTube-VIS 2019 checkpoints as initialization. Incompatible dataset-specific
classification heads are skipped by the Detectron2 checkpointer; the shared ViT,
decoder, mask head, and temporal-query parameters are transferred.

```bash
mkdir -p checkpoints/videomt/ytvis2019

wget -O checkpoints/videomt/ytvis2019/yt_2019_vit_small_52.8.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_small_52.8.pth?download=true'
wget -O checkpoints/videomt/ytvis2019/yt_2019_vit_base_58.2.pth \
  'https://huggingface.co/tue-mps/VidEoMT/resolve/main/yt_2019_vit_base_58.2.pth?download=true'
```

## Configuration Layout

Only the four datasets evaluated for DynEoMT are retained in the publication
configuration tree. Original baselines and proposed-method recipes are separated:

```text
configs/
  VidEoMT/{VIPSeg,ovis,ytvis22,VSPW}/dinov2/{vit-small,vit-base,vit-large}/
  DynEoMT/{VIPSeg,ovis,ytvis22,VSPW}/dinov2/{vit-small,vit-base,vit-large}/
```

Files under `VidEoMT/` contain the unmodified segmentation/tracking task setup.
Files under `DynEoMT/` inherit the corresponding baseline and add the dynamic
head, pseudo-label source, dynamic loss, sampling protocol, and output directory.

## Training

For example, train the ViT-L VIPSeg model on one GPU with:

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file \
  configs/DynEoMT/VIPSeg/dinov2/vit-large/dyneomt_online_vitl.yaml
```

The four ViT-L recipes are:

```text
configs/DynEoMT/VIPSeg/dinov2/vit-large/dyneomt_online_vitl.yaml
configs/DynEoMT/ovis/dinov2/vit-large/dyneomt_online_vitl.yaml
configs/DynEoMT/ytvis22/dinov2/vit-large/dyneomt_online_vitl.yaml
configs/DynEoMT/VSPW/dinov2/vit-large/dyneomt_online_vitl.yaml
```

Replace `vit-large` and the filename suffix with `vit-small/..._vits.yaml` or
`vit-base/..._vitb.yaml` for the smaller backbones. To queue S then B on one
physical GPU with automatic checkpoint resume:

```bash
python scripts/train_backbone_sweep.py vipseg \
  --device 0 \
  --data-root datasets \
  --target-root outputs/dynamic_targets/labels
```

Dataset-specific iteration counts, class-imbalance weights, evaluation periods,
and output directories are defined in the YAML files. The sweep outputs are
written under `outputs/dyneomt/<dataset>/<backbone>/`.

Use `--output-root` to isolate a replication or corrected rerun without
overwriting an existing experiment:

```bash
python scripts/train_backbone_sweep.py ovis \
  --backbones vit-small vit-base vit-large \
  --device 0 \
  --data-root datasets \
  --target-root outputs/dynamic_targets/labels \
  --output-root outputs/dyneomt_corrected
```

For VIS clips, track classes are read from the first sampled frame in which the
track ID is visible. Missing-frame sentinels use
`MODEL.BACKBONE.NUM_CLASSES`. This prevents an object that enters, leaves, or
is occluded within a clip from receiving the no-object class for its complete
trajectory; `tests/test_temporal_targets.py` covers this behavior.

### Matched VidEoMT baselines

The ViT-S/B files under `configs/VidEoMT/` are matched comparison recipes. They
use the same initialization, three-frame clips, batch size, learning rate,
schedule, and data as DynEoMT, with the dynamic head and dynamic loss disabled.
Queue both backbones for one dataset with:

```bash
python scripts/train_videomt_backbone_sweep.py vipseg \
  --device 0 \
  --data-root datasets
```

Results are written to `outputs/videomt_retrain/<dataset>/<backbone>/`. The
launcher skips completed models and resumes from `last_checkpoint` by default.

## Evaluation

Evaluate a trained model with both the original task metrics and the dynamicity
metrics:

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file \
  configs/DynEoMT/VIPSeg/dinov2/vit-large/dyneomt_online_vitl.yaml \
  --eval-only \
  MODEL.WEIGHTS outputs/dyneomt/vipseg/vit-large/model_final.pth \
  OUTPUT_DIR outputs/dyneomt/vipseg/vit-large/evaluation
```

Predicted and ground-truth masks are greedily class-matched at IoU `>= 0.5`.
DynEoMT reports:

- `Dyn-MCC`: Matthews correlation coefficient on matched masks. This is the main
  class-imbalance-resistant dynamicity metric.
- `Dyn-BAcc`: mean of dynamic recall and static recall on matched masks.
- `Dyn-F1` and `Dyn-AUROC`: conditional dynamicity quality on matched masks.
- `MaskCoverage`: fraction of labeled ground-truth masks matched by segmentation.
- `Matched-IoU`: mean IoU of those matches.
- `DynSeg-F1`: end-to-end joint score. A true positive requires a class-consistent
  mask match and a correct dynamic/static decision. With `C` jointly correct
  matches, `P` predicted masks, and `G` labeled ground-truth masks, it is
  `2C / (P + G)`.

Original VidEoMT metrics are evaluated in parallel: VPQ/STQ for VIPSeg, video
instance AP for OVIS and YouTube-VIS 2022, and semantic video segmentation metrics
for VSPW.

## TensorRT Export

The current TensorRT exporter targets the DynEoMT backbone at fixed native input
resolution and exports the first-frame path:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/export_tensorrt_backbones.py \
  --only dyneomt vipseg vit-small \
  --device cuda:0
```

It writes ONNX graphs, TensorRT engines, and a JSONL runtime report under
`../outputs/tensorrt_exports/`. The exported tensors are `pred_logits`,
`pred_masks`, and `pred_dynamic_logits`.

This is not yet a full replacement for evaluation. The complete benchmark
metrics still require Detectron2 post-processing and online query-state
integration around the TensorRT engine. Until that wrapper is connected,
post-export metrics should be reported as PyTorch metrics plus TensorRT backbone
latency, not as full TensorRT VPQ/AP/mIoU/DynSeg evaluations.

## Visualization

[`scripts/visualize_dyneomt.py`](scripts/visualize_dyneomt.py) generates one
representative validation sequence for each supported dataset. Models are loaded
one at a time, so `--dataset all` does not keep four ViT-L backbones in GPU memory.

The `grid` mode creates a 2x2 video: predicted segmentation and dynamicity on the
top row, then segmentation ground truth and dynamicity pseudo-ground truth on the
bottom row.

```bash
python scripts/visualize_dyneomt.py all --mode grid --device cuda:0
```

The `prediction` mode keeps only the two predicted panels and does not require
dynamic-target files:

```bash
python scripts/visualize_dyneomt.py all --mode prediction --device cuda:0
```

Use `--backbone vit-small`, `--backbone vit-base`, or `--backbone vit-large` to
select the matching configuration and automatically discover its checkpoint.

For one sequence, provide only its dataset and sequence ID. The final checkpoint
trained for that dataset is discovered automatically:

```bash
python scripts/visualize_dyneomt.py \
  vipseg 4_X9nPISgpz9Q \
  --mode grid \
  --device cuda:0
```

The automatic checkpoint search covers `outputs/dyneomt_models/`,
`outputs/dyneomt/`, `checkpoints/dyneomt/`, and the corresponding output folders
in the parent project. A custom model always takes priority when explicitly given:

```bash
python scripts/visualize_dyneomt.py \
  vipseg 4_X9nPISgpz9Q \
  --checkpoint /path/to/custom_model.pth \
  --mode prediction
```

The script searches both train and validation catalogs, so any annotated sequence
can be selected. For `all`, overrides use `DATASET=VALUE` and may be repeated, for
example `--sequence vipseg=4_X9nPISgpz9Q --sequence vspw=25_kjuAwy1voxM`.
Videos and representative JPEG previews are written to
`outputs/visualizations/`. Use `--dry-run` to validate all paths without loading
the models.

## Acknowledgements

DynEoMT is derived from VidEoMT and retains its MIT license. The pseudo-label
pipeline uses [MemFlow](https://github.com/DQiaole/MemFlow). It also builds on
Detectron2, EoMT, Mask2Former, CAVIS, timm, and Hugging Face Transformers. Please
respect the licenses and citation requirements of the models and datasets used in
your experiments.

## Citation

The DynEoMT citation will be added with the public paper. Please also cite VidEoMT:

```bibtex
@inproceedings{norouzi2026videomt,
  author    = {Norouzi, Narges and Zulfikar, Idil and Cavagnero, Niccolo and
               Kerssies, Tommie and Leibe, Bastian and Dubbelman, Gijs and
               de Geus, Daan},
  title     = {VidEoMT: Your ViT is Secretly Also a Video Segmentation Model},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition},
  year      = {2026}
}
```
