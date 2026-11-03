# Comparative research short draft

This short draft is a compact alternative presentation of the existing
Ticos-Edge monograph, not a claim of a new state-of-the-art method or a
submission-ready paper. It prioritizes paired, failure-inclusive comparisons,
reports repeatability and output-representation limitations, and treats
Point2Pose as a possible future integration rather than an authorized owner.
The long-form monograph remains at
`../ticos_teleop_3d_reconstruction_monograph/main.tex`.
For a concise engineering decision and comparison limits, see
`../../research/2026-10-29-reconstruction-benchmark-decision-brief.md`.

## Manuscript organization

`main.tex` and `comparative-draft.pdf` now provide a concise comparative
manuscript organized around the public baselines, mask conditions, paired
recording, surfaces, and limits. They do not claim a new method or a complete
submission. The previous six-page evidence-dense draft is preserved as
`detailed-evidence.tex` and `detailed-evidence.pdf`; the evidence map below
covers that detailed record as well as the condensed main manuscript. A
deployment-relevant physical-reference, frozen held-out evaluation and
live-owner resolution remain outstanding; three Laval actions of two objects
provide motion-capture development evidence under stated input assumptions.

## Evidence map

- Point2Pose consistency-fork comparisons on three Laval actions, paired
  regression decomposition, native graph/fusion pose-version audit, exact
  volume-replay control, and global-graph switch ablation:
  `../laval-fork-development-20261030/README.md`. The graph switch helps
  cookiejar pose scores but harms both clock actions; final-pose rebuilding
  also fails to recover the fixed cookiejar surface. Figures retain these
  adverse outcomes rather than selecting only the favorable action. A later
  actual-factor audit identifies disconnected graph corrections; the optional
  admission fork improves ADD AUC on two actions and partly recovers the
  cookiejar surface, but stricter pose regressions remain. The methods section
  and `anchor-admission-table.tex` describe that distinct, default-off repair.
- Laval Vicon-referenced development comparison with shared SAM2 masks,
  registration assumption and offline BundleTrack depth-limit override:
  `../laval-mocap-input-audit-20261029/README.md` and
  `../laval-interaction-full-development-20261029/README.md` and
  `../laval-cookiejar-development-20261029/README.md`.
- Post-hoc cookiejar Vicon-aligned model and native surface comparison:
  `../laval-cookiejar-surface-development-20261029/README.md`.
- Unsealed process-local existing-owner replay and exact native input audit:
  `../existing-owner-offline-route-20261029/README.md`.
- Point2Pose shadow on captured existing-owner inputs, including the two
  explicitly absent observations: `../point2pose-live-owner-shadow-20261029/README.md`.
- Post-hoc Point2Pose replay using saved BundleTrack-filtered depth:
  `../point2pose-native-depth-ablation-20261029/README.md`.
- Instrumented Point2Pose correspondence and sampler-context diagnosis:
  `../point2pose-sampler-context-ablation-20261029/README.md`.
- Tracked-pixel distance to measured in-mask depth on the same recording:
  `../point2pose-depth-spatial-audit-20261029/README.md`.
- Cross-recording support fractions and same-frame SAM2 boundary audit:
  `../cross-recording-depth-support-20261029/README.md` and
  `../sam-depth-evidence-boundary-20261029/README.md`.

- DexYCB two-view grasped-bowl development comparison:
  `../dexycb-public-development-20261029/README.md` and
  `../dexycb-second-camera-development-20261029/README.md`. Both views
  derive from one shared multiview fit and an incomplete archive prefix;
  the released dataset is CC BY-NC 4.0. The tracking entries, reported loss
  and annotation-relative scores are reproducible with the saved receipts.

- Primary pose table: `../ycbineoat-public-recall-20261028/score.json`,
  `../ycbineoat-bleach0-development-20261028/results/recall.json`, and
  `../ycbineoat-two-object-extension-20261028/{sugar,tomato}-score.json`.
  The subsequently selected bleach-hard score is listed below. `generate_table.py`
  verifies the five source-file hashes and paired denominators before rendering all
  six rows into `pose-table.tex`; hash checking establishes file identity, not
  correctness or independent validation of the archived scores.
- Equal-sequence and frame-weighted post-hoc summaries of the five selected
  reference-mask recordings: `sequence-summary.json`, reproducible with
  `python3 sequence_summary.py`. Adjacent frames are correlated and these
  selected recordings do not justify population-level significance claims.
- Reported availability versus post-hoc reference agreement:
  `../ycbineoat-availability-vs-reference-20261029/results.json`.
- Bound archived-pose parity for the bleach reference score:
  `../registered-shadow-bound-bleach-score-20261029/results.json`.
- Additional bleach-hard comparison and reference-association limits:
  `../ycbineoat-bleach-hard-development-20261029/README.md`.
- Bleach-hard corresponding ADD, object-origin and rotation diagnostics:
  `../ycbineoat-bleach-hard-pose-metrics-20261029/results.json`.
- Bleach-hard selected-frame timestamp-gap diagnostic:
  `../ycbineoat-bleach-hard-gap-diagnostic-20261029/results.json`.
- Repeated bleach-hard BundleTrack depth-filter intervention:
  `../ycbineoat-bleach-hard-depth-filter-20261029/score.json`.
- Bleach-hard Point2Pose same-setting repeat:
  `../ycbineoat-bleach-hard-point2pose-repeat-20261029/summary.json`.
- Bleach-hard post-construction global NumPy-state intervention:
  `../ycbineoat-bleach-hard-effective-seed-20261029/summary.json`.
- Bleach-hard RANSAC sampling shadow and seed-zero control:
  `../ycbineoat-bleach-hard-ransac-shadow-20261029/summary.json`.
- Bleach-hard paired outcomes and descriptive frame-count groups:
  `../ycbineoat-bleach-hard-paired-blocks-20261029/results.json`.
- Bleach-hard reference-index offset projection sensitivity:
  `../ycbineoat-bleach-hard-reference-offset-20261029/results.json`.
- BundleSDF conditional bleach-hard pose diagnostic and processing caveats:
  `../bundlesdf-bleach-hard-pose-diagnostic-20261029/README.md`.
- Separate BundleSDF normalized input-boundary and trajectory repeat:
  `../bundlesdf-bleach-hard-boundary-replay-20261029/README.md`.
- Pose score for the normalized-boundary BundleSDF repeat:
  `../bundlesdf-bleach-hard-boundary-score-20261029/README.md`.
- BundleSDF global versus Point2Pose causal bleach-hard surfaces:
  `../bundlesdf-point2pose-bleach-hard-surface-20261029/README.md`.
- BundleSDF and Point2Pose frame-437 online surfaces:
  `../bundlesdf-point2pose-bleach-hard-online-20261029/README.md`.
- BundleSDF and Point2Pose frame-438 surfaces with normalized entry receipts:
  `../bundlesdf-point2pose-bleach-hard-boundary-surface-20261029/README.md`.
- Existing BundleTrack/causal surface replay and three-way 4 mm grid diagnostic:
  `../ticos-bleach-hard-causal-surface-20261029/README.md`.
- Existing causal surface with captured matched tracker-entry boundary:
  `../ticos-bleach-hard-causal-matched-20261029/README.md`.
- Existing 1,500-point selection and offline spatial alternative on bleach-hard:
  `../ticos-bleach-hard-snapshot-budget-20261029/README.md`.
- Support-constrained coarse point selection and cost:
  `../ticos-bleach-hard-coarse-selection-20261029/README.md`.
- Coarse selection over fourteen recorded snapshots and final adjacent transitions:
  `../ticos-bleach-hard-coarse-temporal-20261029/README.md`.
- Shadow-only surface-selection adapter and fallback validation:
  `../ticos-shadow-surface-selection-adapter-20261029/README.md`.
- Supported-map coverage by observation support on bleach-hard:
  `../ticos-shadow-surface-support-coverage-20261029/README.md`.
- Cross-case supported-map coverage and unavailable-pose distinction:
  `../ticos-shadow-surface-multicase-20261029/README.md`.
- Matched-entry YCB-M moving-camera development replay, depth support and
  post-hoc initialization sensitivity:
  `../ycbm-moving-camera-development-20261029/README.md`.
- Shared generated-mask comparisons: `../ycbineoat-generated-mask-recall-20261028/score.json`,
  `../ycbineoat-mustard-generated-mask-20261028/score.json`,
  `../ycbineoat-bleach-generated-mask-20261028/score.json`, and
  `../ycbineoat-two-object-extension-20261028/{sugar,tomato}-generated-score.json`.
  `generate_generated_mask_table.py` verifies their hashes and renders the table.
- Mustard event model-axis versus full orientation: `../ycbineoat-mustard-event-axis-20261029/results.json`.
- Mustard event neighborhood and ADD-S/rotation comparison: `../ycbineoat-mustard-rotation-window-20261029/results.json`.
- Mustard maximum rotation event and source-time audit: `../ycbineoat-mustard-rotation-event-20261029/results.json`.
- Post-hoc sugar/tomato threshold sensitivity:
  `../ycbineoat-threshold-sensitivity-20261029/result.json`.
- Seed-zero RANSAC wrapper control: `../ycbineoat-sugar-ransac-wrapper-control-20261029/score.json`.
- Point2Pose RANSAC-choice shadow intervention: `../ycbineoat-sugar-ransac-seed-shadow-20261029/score.json`.
- Point2Pose post-construction global NumPy-state intervention: `../ycbineoat-sugar-effective-numpy-seed-20261029/score.json`.
- Point2Pose registration constructor NumPy seed reset: `../ycbineoat-point2pose-seed-audit-20261029/results.json`.
- Point2Pose sugar alternate random seed: `../ycbineoat-sugar-point2pose-seed42-20261029/score.json`.
- Point2Pose sugar same-host repeat: `../ycbineoat-sugar-point2pose-repeat-20261029/score.json`.
- Corresponding-vertex ADD across BundleTrack repeat/filter runs: `../ycbineoat-sugar-add-repeat-20261029/results.json`.
- Corresponding-vertex ADD versus nearest-vertex ADD-S: `../ycbineoat-add-adds-20261029/results.json`.
- Local-axis versus full rotation: `../ycbineoat-axis-orientation-20261029/results.json`.
- Annotation-gauge orientation versus ADD-S: `../ycbineoat-orientation-diagnostic-20261029/results.json`.
- Paired outcome episodes on sugar and tomato: `../ycbineoat-paired-episodes-20261029/results.json`.
- Cracker surface and 1,500-point budget:
  `../ycbineoat-surface-development-20261029/README.md` and
  `wire-snapshot-budget.json`.
- Ticos filter repeats:
  `../ticos-0804-depth-percentile-20261029/full-filter-ab-score.json`.
- Sugar filter repeats and temporal localization:
  `../ycbineoat-sugar-depth-filter-20261029/score.json` and
  `repeat-windows.json`.
- Moving-wrist-camera cup baseline and inferred-depth shadow:
  `../point2pose-development-evidence-20261028/moving-cup-point2pose/summary.json`
  and `../point2pose-development-evidence-20261028/moving-cup-mask-fill-shadow/summary.json`.
  Their local scripts pin and check the historical input and output receipts.
- Point2Pose mask-dropout adapter audit:
  `../point2pose-dropout-adapter-audit-20261029/results.json`.
- Ticos recorded-output rotation shadow scorecard: `../ticos-0804-shadow-scorecard-rotation-20261029/results.json`.
- Repeated offline Point2Pose and BundleTrack surface worker timing: `../ticos-0804-worker-timing-repeat-20261029/summary.json`.
- Point2Pose two-object worker round-trip timing: `../ticos-0804-point2pose-worker-timing-20261029/summary.json`.
- Parallel versus serial supported-surface geometry: `../ticos-0804-parallel-bt-surface-geometry-20261029/summary.json`.
- Parallel versus serial existing-owner worker scheduling shadow: `../ticos-0804-parallel-bt-surface-workers-20261029/summary.json`.
- Live owner boundary and scheduling assessment: `../../plans/2026-10-29-existing-owner-parallel-scheduling-assessment.md`.
- Two BundleTrack workers including existing surface updates: `../ticos-0804-two-bundletrack-surface-workers-timing-20261029/summary.json`.
- Actual offline serial two-process BundleTrack schedule: `../ticos-0804-two-bundletrack-workers-timing-20261029/summary.json`.
- Sequential offline two-object component timing: `../ticos-0804-two-object-component-timing-20261029/summary.json`.
- Ticos object-1 BundleTrack repeat: `../ticos-0804-object1-bt-repeat-20261029/results.json`.
- Ticos object-1 BundleTrack / Point2Pose paired replay: `../ticos-0804-object1-bt-p2p-20261029/results.json`.
- Matched-source Ticos BundleTrack / Point2Pose trajectory agreement: `../ticos-0804-bundletrack-point2pose-paired-20261029/results.json`.
- Multi-object versus single-object Ticos Point2Pose replay:
  `../ticos-point2pose-multiobject-0804-20261029/summary.json`.
- Two-object frame-zero initialization and automatic-mask comparison:
  `../ticos-point2pose-multiobject-0804-20261029/single-seed/summary.json`.
- Post-hoc mask and pose temporal windows: `../ticos-point2pose-mask-pose-window-20261029/results.json`.
- Earlier Ticos surfaces projected to later recorded observations: `../ticos-0804-surface-future-projection-20261029/results.json`.
- Ticos surface reprojection to the input depth: `../ticos-0804-surface-reprojection-20261029/results.json`.
- Threshold variants projected to later RGB-D inputs: `../ticos-object1-threshold-future-projection-20261029/results.json`.
- Threshold variants compared to the same existing-owner surface: `../ticos-0804-object1-threshold-surface-reference-20261029/results.json`.
- Object-1 keyframe-270 actual TSDF update: `../ticos-object1-frame270-tsdf-update-20261029/summary.json`.
- Point2Pose object-1 keyframe threshold shadow: `../ticos-point2pose-keyframe-object1-3deg-20261029/results.json`.
- Point2Pose recorded keyframe criterion decisions: `../ticos-point2pose-keyframe-criterion-0804-20261029/results.json`.
- Ticos object-1 existing-owner surface versus Point2Pose mesh: `../ticos-0804-object1-surface-paired-20261029/results.json`.
- Ticos existing-owner surface versus Point2Pose mesh: `../ticos-0804-surface-paired-20261029/results.json`.
- Instrumented two-object TSDF integration and mesh comparison:
  `../ticos-point2pose-multisurface-0804-20261029/summary.json`.
- System checkpoint:
  `../../operations/recorded-system-acceptance-checkpoint-20261026.json`.
- Shadow gate policy and missing evidence:
  `../../plans/2026-09-20-shadow-pose-owner-ab-design.md`.
- Session-timing provenance guard and recorded-output audit:
  `../../operations/shadow-scorecard-timing-guard-20261029.md`.
- Physical-capture origin guard:
  `../../operations/configured-pouring-physical-source-guard-20261029.md`.

These source receipts contain their own provenance and caveats. Not all raw
inputs, trained weights or complete execution environments are distributed in
this folder. The six public sequences are development data with reference
masks and annotation-based scores; they do not establish blind physical
accuracy, independent segmentation, complete-scene quality or system latency.
The paper needs a focused hypothesis and independent evaluation
before submission.

The Point2Pose repository's `main` resolved to the evaluated commit
`51856226610df75e5c06e8de545bd27f7c4ba99c` on 2026-10-29, checked with
`git ls-remote https://github.com/tzuyuan/point-to-pose.git HEAD refs/heads/main`.
This is a time-specific source identity check, not proof that our replay
uses the authors' complete real-time pipeline or reproduces their timing.
The authors' [project page](https://point2pose.github.io/) separately reports
30 Hz on an RTX 4090 and marks YCBMultiTrack as coming soon as of that date;
neither statement is an independently measured result of our experiment.

From this directory:

```sh
python3 generate_table.py > /tmp/ticos-short-pose-table.tex
cmp /tmp/ticos-short-pose-table.tex pose-table.tex
python3 generate_generated_mask_table.py > /tmp/ticos-generated-mask-table.tex
cmp /tmp/ticos-generated-mask-table.tex generated-mask-table.tex
mkdir -p /tmp/ticos-comparative-short
tectonic -C --outdir /tmp/ticos-comparative-short main.tex
```

`comparative-draft.pdf` is the generated review copy. The primary citations are
Point2Pose (https://arxiv.org/abs/2604.10415), BundleTrack
(https://arxiv.org/abs/2108.00516), and FoundationPose
(https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf).

## Recent scope corrections

The repaired BundleSDF experiment is on **YCBInEOAT bleach-hard**, not the DexYCB grasp sequence discussed in the preceding paragraph. The manuscript now names that source explicitly. The artificial gap and head-view checks compare estimated trajectories using an existing BundleTrack-supported map; they do not establish physical accuracy or real occlusion recovery. The existing-session reentry failure and fail-closed change are documented in `../ticos-owner-session-gap-20261029/README.md` and do not establish successful recovery. These local artifacts include non-distributed inputs and runtime dependencies, so file checksums alone do not make the study independently reproducible.

As checked on 2026-10-29, the [Point2Pose project page](https://point2pose.github.io/) lists ECCV 2026 and describes a newer 30 Hz RTX 4090 demonstration; [arXiv v2](https://arxiv.org/abs/2604.10415v2) is dated 2026-08-25. The paper's local tests use the specifically pinned repository commit and recorded configurations; the project's demonstration is not a measurement from these experiments. The project page still lists YCBMultiTrack as coming soon.

## Cookiejar regression and surface update (2026-10-30)

The abstract and results now distinguish higher pose availability from the observed surface regression of the cumulative correctness fork. The fixed input-150 surface replay and paired pose decomposition are in `../laval-fork-development-20261030/README.md`. Figure 1 is generated from audited score files with:

```sh
.venv-bundlesdf/bin/python docs/papers/laval-fork-development-20261030/plot_cookiejar_tradeoff.py
```

Run that command from the repository root before compiling this directory's `main.tex`. The figure retains all 295 pose opportunities on the left and labels the different 150-frame-prefix surface window on the right. Rendering was checked; the build has no overfull boxes or undefined references. Remaining underfull-box warnings are typographic, not metric checks.

## Connected-graph rank and rejected intervention (2026-10-30)

The methods/results now include exact native optimizer replay and a local
constraint-rank diagnostic. All 71 captured optimizations reproduce exactly;
eight of 34 connected noninitial updates lack six constrained current-pose
dimensions. The optional rank-admission fork passes 25 related tests and all
704 actual tracker inputs match its comparators. It improves cookiejar pose
scores but harms both clock actions and the cookiejar snapshot's strict surface
fractions, so it is retained as a rejected development variant. This negative
result is included in the abstract and the expanded admission table.

Recompute `graph_observability_score.py` and `current_rank_score.py` in
`../laval-fork-development-20261030`, then run its `generate_anchor_table.py`
before compiling the paper. The evidence folder includes protocols, source
patches, mathematical/commit-boundary tests, complete failure-inclusive scores
and the next core-redesign direction. No production cutover or held-out claim
is introduced.

## Measured-observation ledger and negative result (2026-10-30)

The methods section now describes immutable feature-birth observations and
deferred factor insertion inside the native optimizer. A passive 704-frame
audit identifies 1,800 omitted eligible observation pairs. The implemented
fork passes 21 related tests and its complete three-action replay restores
current-pose connectivity, but reduces availability and failure-inclusive ADD
AUC on every action versus connectivity admission alone. The expanded table
and abstract retain this rejected intervention. The cookiejar surface result
is mixed rather than uniformly improved.

The scoring-only reference diagnostic compares measured points under their
Vicon poses at birth and later observation, including the exact first inserted
factor pairs on the common prefixes. It cannot identify the cause of subsequent
tracking loss. Reports, source patch, failed-startup receipt, test evidence and
reproduction scripts are in `../laval-fork-development-20261030/README.md`.
Regenerate `generate_anchor_table.py` there before building this manuscript.

## Causal surface transactions and factorial interaction (2026-10-30)

The new methods section describes an immutable native fusion ledger and causal
repair at admitted historical pose versions. Paired passive captures reproduce
1,408 original records, and six isolated endpoint replays reproduce native
volume arrays before testing corrected poses. The first failed grid-allocation
control is retained; exact dimensions now prevent float32 re-ceiling changes.

Two frozen forks pass 22/31 related tests and complete another 1,408 matched
tracker entries. The 2 x 2 table and interaction figure show opposite effects
of surface repair depending on birth-factor insertion. The combined arm has
cookiejar ADD AUC 0.7498 versus BundleTrack 0.7230, but lower availability and
strict translation counts, lower clock AUC than connectivity admission, and
worse cookiejar surface agreement. Both repair policies remain quarantined as
general replacements. This local gain is included alongside all regressions.

See `../laval-fork-development-20261030/README.md` for source patches, receipts,
complete per-frame scores, physical-reference qualifications and reproduction.
Run `generate_surface_transaction_table.py` and `plot_surface_transactions.py`
there before rebuilding the manuscript. The scorecard uses the original
physical-scoring environment; native graph diagnostics use pinned GTSAM.

## Method & mechanism restructuring and bit-exact runtime optimization (2026-11-03)

`main.tex` and `comparative-draft.pdf` have been structured following the
mechanism-focused paper foundation (`../../research/2026-10-30-bundletrack-point2pose-paper-foundation-zh.md`,
*Online Pose Estimation and Consistent Reconstruction of Unknown Objects under Degenerate Observations*):

1. **Failure Mechanisms and Factor-Graph Observability:** Formalizes temporal attribution / solver-return inconsistency ($F$), unanchored $\mathrm{SE}(3)$ gauge drift ($G$), nuisance-eliminated local rank deficiency $\bar{\mathbf{A}}_x = (\mathbf{I} - \mathbf{A}_y\mathbf{A}_y^+)\mathbf{A}_x$ ($R$), immutable feature-birth observation ledgers ($M$), and causal surface-history rebuild transactions ($S$).
2. **Native BundleTrack & Surface Selector Repairs (`2026-11-01`):** Summarizes CUDA RANSAC deterministic winner selection (`stable-winner` + count-direction), `CUDACache::storeFrame` $\mathrm{SE}(3)$ inverse-coefficient precision ($135/107 \to 191/136$ strict successes on Laval Clock/Jar), guarded contiguous-`NO_BA` frame retirement ($-928.4$ to $-1{,}994.0\,\mathrm{MiB}$ RSS), reference-pose and pixel/mesh-ray surface error attribution, and array-native `support10/coarse12mm` $1{,}500$-point selection ($C_{10}$ $40.57\%/52.34\% \to 85.10\%/85.05\%$, $-33\%$ selector latency).
3. **31-Framework Matcher Shadow & Zero-Loss Runtime Execution (`2026-11-02`):** Summarizes EfficientLoFTR (FP32/Full-MP), XFeat Semi-Dense, and ALIKED+LightGlue shadow matcher trials alongside `SAM2GpuWindowOverlapScope` (`../sam-gpu-window-cpu-overlap-20261102/README.md`) and `RequestEvidenceDeduplicationScope` (`../request-evidence-deduplication-20261102/README.md`), which cut exclusive `original_world` wrapper overhead by $72.9\%$ ($23.17 \to 6.28\,\mathrm{ms}$), eliminate $140.58\,\mathrm{ms}$ mid-request Gen-2 GC stalls, and reduce warm HTTP `endpoint_ms` $p_{50}/p_{95}$ from $155.33/219.06\,\mathrm{ms}$ ($176.40\,\mathrm{ms}$ initial baseline) to $127.80/153.40\,\mathrm{ms}$ with $100\%$ bit-exact parity across $499$ online equivalence checks.

