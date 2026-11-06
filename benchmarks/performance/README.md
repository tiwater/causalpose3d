# Hot-path microbenchmarks

These benchmarks isolate output-preserving Python/NumPy paths in
`evidencescene`. They are **not** recorded-session endpoint latency and should
not be substituted for the Ticos Edge scorecards.

Run:

```sh
python3 benchmarks/performance/bench_hot_paths.py
# or
make perf
```

## Recorded development measurements

Same host, Python 3.10.12, NumPy 1.21.5, warm process. "Before" was measured
immediately before the corresponding patch; "after" is the median from the
checked-in benchmark.

| Case | Before | After | Speed-up |
| --- | ---: | ---: | ---: |
| `RevisableSurfelMap.snapshot`, 60k voxels, 1,500-point budget | 19.6 ms | 8.8 ms | 2.2x |
| `observation_voxel_centroids`, 60k points | 10.4 ms | 6.0 ms | 1.7x |
| `compare_circle_consensus_3d`, 400 rim points, 256 hypotheses | 4.59 ms | 1.80 ms | 2.5x |
| `fit_circular_aperture_3d`, 400 rim points | 4.38 ms | 1.51 ms | 2.9x |

The centroid change replaces `np.add.at` with three `np.bincount` accumulations
(4.7 ms → 0.34 ms for the accumulation step), preserving exact sums. The
snapshot change preserves the exact support/camera/voxel lexicographic
order and point normalization. The consensus change preserves the exact seeded
triplet stream, radius bounds, inlier threshold, strict tie-breaking and
per-hypothesis RMSE; a 500-case comparison against the original scalar loop
produced zero mismatches.

These numbers are development microbenchmarks only. They do not establish
end-to-end latency, live-site acceptance or a production cutover.
