# DynSLAM Fast FT11000 Preset

The active DynSLAM binary preset is:

```text
FT=11000
TM=0000
FL=00000
AD=00000
```

Only `FT1` and `FT2` are enabled.

| Bit | State | Role |
| --- | ---: | --- |
| `FT1` | 1 | ORB extraction GPU |
| `FT2` | 1 | stereo matching GPU |
| `FT3` | 0 | search local points GPU disabled |
| `FT4` | 0 | pose estimation GPU disabled |
| `FT5` | 0 | pose optimization GPU disabled |
| `TM1..TM4` | 0 | TurboMap disabled |
| `FL1..FL5` | 0 | FastLoop disabled |
| `AD1..AD5` | 0 | adaptive controller disabled |

This favors raw tracking speed and keeps the dynamic component isolated to the
DynEOMT feature mask. It deliberately avoids the heavier or less stable
combinations observed in the Nitro-SLAM sweep.
