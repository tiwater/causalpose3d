# Recorded benchmark results

These files are the checked-in evidence for the `kernel-v1` benchmark.

## `04a1b6b` vs `6c0e8f4`

- Baseline: `04a1b6b` (v2.0.0 before the hot-path performance patch)
- Candidate: `6c0e8f4` (`perf: reduce per-voxel update overhead in surfel fusion`)
- Host: AMD Ryzen 7 9800X3D, Python 3.10.12, NumPy 1.21.5
- Command: `./benchmarks/performance/run_versions.sh 04a1b6b 6c0e8f4`
- Workloads: 7 kernel workloads
- Output digests: all identical
- Geometric-mean speed-up: 1.65x

The JSON files include the full environment metadata. The Markdown report is
the output of `compare.py`. These numbers are synthetic kernel microbenchmarks;
they are not recorded-session endpoint latency or accuracy evidence.
