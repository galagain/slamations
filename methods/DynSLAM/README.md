# DynSLAM

DynSLAM is the fast dynamic-SLAM preset built from the Nitro-SLAM binary ablation
study and the DynEOMT semantic mask proposal.

## Fast Preset

The default preset keeps only the two fastest retained FastTrack bits:

| Family | Mask | Meaning |
| --- | --- | --- |
| `FT` | `11000` | `FT1` ORB extraction GPU, `FT2` stereo matching GPU |
| `TM` | `0000` | TurboMap disabled |
| `FL` | `00000` | FastLoop disabled |
| `AD` | `00000` | adaptive controller disabled |

This is the requested fastest version: `FT1=1`, `FT2=1`, every other binary
parameter set to `0`.

Semantic masking is configured in `configs/tum_rgbd_dyneomt_stride16_exact` with
`Semantic.backend: "dyneomt"`. DynEOMT is expected to output a dynamic-pixel
mask; DynSLAM converts it into a feature keep mask before tracking.

The default wrapper uses the full VIPSeg ViT-L DynEOMT checkpoint because the
small runtime-table ViT-S checkpoint ran quickly but produced empty dynamic masks
on `rgbd_dataset_freiburg3_walking_xyz`.

## DynEOMT Source

The DynEOMT source recovered from the server is stored at:

```text
third_party/DynEOMT
```

DynSLAM also keeps a local link to it at:

```text
methods/DynSLAM/external/DynEOMT
```

The server tree was very large, so the recovered copy intentionally excludes the
heavy runtime artifacts: trained checkpoints, raw datasets, logs, and output
folders. Local Jetson checkpoints are available outside the thesis tree under
`/home/nvidia/dyneomt_jetson/checkpoints`.

## Run

```bash
./methods/DynSLAM/run_tum.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz
```

The wrapper uses the Nitro-SLAM Docker image with the DynSLAM configs mounted
as `/opt/dynslam_configs`, and forces:

```text
FT=11000 TM=0000 FL=00000 AD=00000
```

Results are written under:

```text
methods/DynSLAM/results/tum_rgbd/<sequence>/<version>/dynslam_ft11000_dyneomt_fast
```

Use `./methods/DynSLAM/run_tum.sh --show-config` to print the active preset.

## Measured Result

Validated run:

```text
rgbd_dataset_freiburg3_walking_xyz
dynslam_vitl_full_fr3_walking_xyz_fixed_20260831
```

Result directory:

```text
methods/DynSLAM/results/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz/dynslam_vitl_full_fr3_walking_xyz_fixed_20260831/dynslam_ft11000_dyneomt_fast
```

Performance:

| Metric | Value |
| --- | ---: |
| Estimated poses | 827 |
| Matched ATE pairs | 826 |
| ATE RMSE | 0.0208 m |
| RPE translation RMSE | 0.0172 m |
| RPE rotation RMSE | 0.4458 deg |
| Tracking FPS | 26.62 |
| Output-pose FPS | 16.07 |
| SLAM wall time | 51.47 s |

DynEOMT precompute for the same 827 associated RGB frames:

| Metric | Value |
| --- | ---: |
| DynEOMT throughput | 3.52 FPS |
| Mean DynEOMT latency | 268 ms |
| p95 DynEOMT latency | 336 ms |
| Dynamic frames | 796 / 827 |
| Mean dynamic pixels | 52,093 |

With the current external precompute wrapper, the conservative end-to-end time is
`234.93 s + 51.47 s = 286.40 s`, or about `2.89 FPS`. The SLAM integration itself
is stable; the remaining speed bottleneck is DynEOMT inference when run as a
separate precompute stage.

## Reduced DynEOMT Frequency

DynEOMT can be run on semantic keyframes instead of every RGB frame. The wrapper
uses a causal hold strategy: it infers one dynamic mask every `N` frames, then
reuses the latest mask for skipped frames. Set the rate with:

```bash
DYNSLAM_SEMANTIC_FRAME_STRIDE=32 ./methods/DynSLAM/run_tum.sh /home/nvidia/data/tum_rgbd/rgbd_dataset_freiburg3_walking_xyz dynslam_vitl_stride32_hold_20260903
```

Validated on `rgbd_dataset_freiburg3_walking_xyz` with the VIPSeg ViT-L DynEOMT
checkpoint:

| DynEOMT mode | DynEOMT calls | DynEOMT effective FPS | ATE RMSE | RPE trans. RMSE | RPE rot. RMSE | Tracking FPS | End-to-end FPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| every frame | 827 | 3.52 | 0.0208 m | 0.0172 m | 0.446 deg | 26.62 | 2.89 |
| stride 8 | 104 | 17.83 | 0.0195 m | 0.0155 m | 0.425 deg | 16.00 | 6.93 |
| stride 16 | 52 | 28.73 | 0.0300 m | 0.0198 m | 0.489 deg | 24.94 | 10.09 |
| stride 32 | 26 | 42.23 | 0.0241 m | 0.0174 m | 0.459 deg | 35.30 | 13.37 |
| stride 64 | 13 | 59.66 | 0.0255 m | 0.0180 m | 0.464 deg | 42.87 | 16.04 |

For this sequence, stride 32 is the best balanced setting: it keeps ATE close to
the dense DynEOMT run while making the semantic stage faster than the camera
rate. Stride 64 maximizes throughput but should be validated on more sequences
before being used as the default.
