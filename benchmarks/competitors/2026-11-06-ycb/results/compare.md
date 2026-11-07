# System performance comparison

- Protocol: `ycbineoat_bleach0_refmask_40f_v1`
- Session: `ycbineoat/bleach0` (40 frames)
- Input SHA-256: `39ae471e99f341be2296762e4224b9f12449ff49657c780781aceae5bd798b3b`
- Comparable inputs: **yes**

| System | Version | p50 (ms) | p95 (ms) | Peak RSS (MB) | Non-lost frames | Output digest |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| EvidenceScene native BundleTrack branch | `controlled-ycb-v1` | 30.235 | 33.049 | 2544.800 | 40 | `n/a` |
| Point2Pose | `controlled-ycb-v1` | 41.601 | 42.456 | 2219.900 | 40 | `n/a` |
