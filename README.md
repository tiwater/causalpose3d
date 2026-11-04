<div align="center">

# CausalPose-3D: Causal Observability-Conditioned 6D Pose Tracking and 3D Surface Reconstruction from RGB-D Video

**[Tiwater Research](https://github.com/tiwater)**

[![Project Page](https://img.shields.io/badge/Project_Page-tiwater.github.io%2Fcausalpose3d-2563eb?style=flat-square&logo=googlechrome&logoColor=white)](https://tiwater.github.io/causalpose3d/)
[![Monograph Preprint](https://img.shields.io/badge/Paper-Monograph_(17_pp.)-b31b1b?style=flat-square&logo=arxiv&logoColor=white)](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-monograph.pdf)
[![Short Paper](https://img.shields.io/badge/Paper-Comparative_Study_(8_pp.)-059669?style=flat-square&logo=arxiv&logoColor=white)](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-short.pdf)
[![CI](https://img.shields.io/github/actions/workflow/status/tiwater/causalpose3d/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=white&label=CI)](https://github.com/tiwater/causalpose3d/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/Release-v1.0.0-0f172a?style=flat-square&logo=github)](https://github.com/tiwater/causalpose3d/releases/tag/v1.0.0)
[![License](https://img.shields.io/badge/License-Apache_2.0-d97706?style=flat-square)](LICENSE)

[**Project Page**](https://tiwater.github.io/causalpose3d/) • [**Monograph (PDF)**](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-monograph.pdf) • [**Short Paper (PDF)**](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-short.pdf) • [**Method**](#-methodology) • [**Benchmarks**](#-experimental-evaluation) • [**Quick Start**](#-installation--usage) • [**Citation**](#-citation)

</div>

---

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%">
        <img src="docs/assets/figures/04-urdf-four-camera-measured-scene.png" alt="Multi-Camera Metric Scene Reconstruction" width="100%"/>
        <br/>
        <sub><b>(a) Multi-Camera Metric RGB-D Scene</b><br/>Cross-calibrated 4-camera metric workspace reconstruction without CAD priors.</sub>
      </td>
      <td align="center" width="33%">
        <img src="docs/assets/figures/06-object-surface-completed-carton-frame245.png" alt="Object-Centric Revisable Surface Reconstruction" width="100%"/>
        <br/>
        <sub><b>(b) Lineage-Indexed Surfel Completion</b><br/>Multi-view corroborated object surface (<code>C₁₀ = 85.1%</code>) under continuous motion.</sub>
      </td>
      <td align="center" width="33%">
        <img src="docs/assets/figures/07-teleop-hud-perspective-frame275.png" alt="Causal 6DoF Pose Tracking and Spatial Verification" width="100%"/>
        <br/>
        <sub><b>(c) Causal Observability Verification</b><br/>Fail-closed 6DoF pose tracking and metric clearance envelope during manipulation.</sub>
      </td>
    </tr>
  </table>
</div>

## 📖 Abstract

Model-free 6D object pose tracking and online 3D surface reconstruction from streaming RGB-D video remain fundamentally fragile under severe occlusion, rapid articulated motion, depth sensor dropout, and weak geometric observability. Existing keyframe-graph trackers (e.g., *BundleTrack*) suffer from non-deterministic GPU RANSAC hypothesis reduction, single-precision $\mathrm{SE}(3)$ inversion drift, and unbounded memory growth, whereas point-track factor graphs (e.g., *Point2Pose*) frequently admit under-constrained keyframes that corrupt global gauge consistency and poison downstream surface fusion.

We present **CausalPose-3D**, a causal observability-conditioned framework for 6D object pose estimation and revisable 3D surface reconstruction. **CausalPose-3D** introduces four tightly coupled contributions:
1. **Deterministic $\mathrm{SE}(3)$ Pose Graph Optimization:** A race-free CUDA correspondence reduction scheme and exact orthonormal $\mathrm{SE}(3)$ pose-cache inversion that eliminate run-to-run trajectory variance (`18.4 mm / 4.72°` $\rightarrow$ **`0.00 mm / 0.00°`**) while bounding GPU/host memory via contiguous non-optimized frame retirement.
2. **Schur-Complement Factor-Graph Observability ($F + G + R$):** A causal admission gate that evaluates the marginal translational precision block $R_i$ via the Schur complement prior to graph insertion, rejecting rank-deficient or gauge-disconnected observations before optimization.
3. **Lineage-Indexed Revisable Surfel Reconstruction ($M + S$):** A persistent 3D surface representation (`RevisableSurfelMap`) that binds every fused surfel to its originating keyframe lineage, supporting atomic surface-history rebuild transactions upon pose graph revision and a two-stage view-diversity selector (`support10/coarse12mm`) that improves 10-view surface corroboration ($C_{10}$) by **`+44.53%`** on YCBInEOAT and **`+32.71%`** on Laval Vicon.
4. **Asynchronous GPU/CPU Window Overlap & $O(\text{runs})$ Mask Algebra:** A zero-loss runtime execution engine that overlaps CPU geometric preparation with GPU video segmentation and operates directly on compressed run-length encoded (RLE) masks in $O(R)$ time, achieving **`100%` bit-exact equivalence** with a **`72.9%`** reduction in host tracking overhead.

---

## 🔥 News

- **[2026-11]** Released [**CausalPose-3D v1.0.0**](https://github.com/tiwater/causalpose3d/releases/tag/v1.0.0), including the core Python library (`causalpose3d`), deterministic C++/CUDA BundleTrack patches (`native/bundletrack/`), benchmark evaluation suites, and full LaTeX sources for both manuscripts.
- **[2026-11]** Launched the interactive [Project Page](https://tiwater.github.io/causalpose3d/) and released the 17-page [Technical Monograph](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-monograph.pdf) and 8-page [Comparative Paper](https://tiwater.github.io/causalpose3d/assets/papers/causalpose3d-short.pdf).

---

## 🧠 Methodology

<div align="center">
  <img src="docs/assets/figures/causalpose3d_architecture.svg" alt="CausalPose-3D System Architecture" width="96%"/>
  <br/>
  <sub><b>Figure 1: Overview of the CausalPose-3D architecture.</b> Streaming RGB-D frames and video segmentation masks are processed through an overlapped GPU/CPU frontend using $O(\text{runs})$ RLE mask algebra, routed through causal observability gates ($\mathrm{SE}(3)$ deterministic BundleTrack or Schur-complement gated Point2Pose $F+G+R$), and fused into a lineage-indexed <code>RevisableSurfelMap</code> ($M+S$) with two-stage view-diversity selection.</sub>
</div>

### 1. Deterministic Native $\mathrm{SE}(3)$ Pose Graph (`native/bundletrack/` & `causalpose3d.tracking`)
Standard GPU RANSAC loops in correspondence-based 6D trackers rely on non-associative floating-point atomics and thread-scheduling order, causing identical RGB-D sequences to diverge across runs. Furthermore, inverting cached camera poses via generic $4\times 4$ single-precision LU decomposition (`Eigen::Matrix4f::inverse()`) accumulates non-orthonormal drift over long trajectories. **CausalPose-3D** enforces strict determinism and numerical closure:
- **Lexicographic CUDA Hypothesis Reduction:** Replaces atomic winner-take-all updates with a deterministic reduction keyed on $(\text{inlier\_count}, -\text{residual\_L1}, \text{hypothesis\_id})$ and selects bidirectional correspondence pairs by valid geometric correspondence support.
- **Exact Orthonormal $\mathrm{SE}(3)$ Inversion:** Enforces analytical group inversion on $\mathrm{SE}(3)$ inside `CUDACache::storeFrame`:
  $$\mathbf{T}^{-1} = \begin{bmatrix} \mathbf{R}^\top & -\mathbf{R}^\top \mathbf{t} \\ \mathbf{0}^\top & 1 \end{bmatrix}, \quad \mathbf{R} \in \mathrm{SO}(3), \; \mathbf{t} \in \mathbb{R}^3$$
  eliminating single-precision matrix inversion drift and increasing strict tracking completions on Laval Vicon 6DoF from `135 / 107` to **`191 / 136`** frames.
- **Contiguous Non-Optimized Frame Retirement:** Retires heavy GPU/host image and point-cloud buffers for frames outside the active sliding Bundle Adjustment (BA) window while preserving keyframe index topology, reducing peak memory footprint by **`0.93–1.99 GiB`** with zero trajectory deviation.

### 2. Causal Factor-Graph Observability (`causalpose3d.observability`)
In point-track factor graphs (*Point2Pose*), admitting frames with degenerate 3D track geometry or disconnected feature tracks introduces rank deficiency into the Levenberg–Marquardt normal equations. **CausalPose-3D** enforces three causal admission conditions ($F + G + R$) at every timestamp $t$:
- **Gauge & Solver Consistency ($F$) and Prior-Connected Admission ($G$):** Requires every candidate keyframe to share a minimum number of verified metric 3D tracks with the anchored pose-graph component.
- **Schur-Complement Marginal Translational Rank Gate ($R$):** Partitioning the linearized normal equations between the candidate frame translation $\delta\mathbf{t}_i \in \mathbb{R}^3$ and landmark/rotation variables $\mathcal{L}$, we evaluate the marginal translational information matrix via the Schur complement:
  $$\mathbf{R}_i = \mathbf{J}_{ii}^\top \mathbf{J}_{ii} - \mathbf{J}_{i\mathcal{L}}^\top \left(\mathbf{J}_{\mathcal{L}\mathcal{L}}\right)^\dagger \mathbf{J}_{\mathcal{L}i} \in \mathbb{S}_+^{3\times 3}$$
  A frame is admitted as an authoritative pose node if and only if $\sigma_{\min}(\mathbf{R}_i) \ge \tau_{\text{rank}}$ and $\kappa(\mathbf{R}_i) \le \kappa_{\max}$, preventing ill-conditioned depth observations from corrupting global trajectory estimates.

### 3. Lineage-Indexed Revisable Surfel Reconstruction (`causalpose3d.surface`)
Naive TSDF or point-cloud accumulation permanently bakes transient pose errors into the 3D model. **CausalPose-3D** maintains a **Lineage-Indexed Revisable Surfel Map** (`RevisableSurfelMap` & `CausalObjectSurface`):
- **Birth-Keyframe Lineage ($M$) & Atomic Surface Rebuild ($S$):** Each surfel $s_k$ records its originating keyframe index, multi-view observation count $c_k$, and directional viewpoint bitmask. Whenever the pose graph prunes an outlier keyframe or refines historical poses beyond a metric threshold, `CausalObjectSurface` executes an atomic surface-history transaction that purges invalidated lineage surfels and re-embeds surviving observations in the updated object gauge.
- **Two-Stage View-Diversity Selector (`support10/coarse12mm`):** Given a fixed downstream point budget $N_{\max} = 1,500$, uniform striding or fine voxel downsampling discards high-confidence multi-view geometry. Our two-stage vectorized selector first retains surfels corroborated across $c_k \ge 10$ distinct viewing intervals, and then allocates remaining capacity across a coarse $12\,\mathrm{mm}$ 3D spatial lattice to guarantee full geometric envelope coverage.

### 4. Overlapped GPU/CPU Execution & $O(\text{runs})$ RLE Algebra (`causalpose3d.runtime`)
To sustain real-time closed-loop perception without sacrificing bit-exactness:
- **$O(\text{runs})$ Run-Length Mask Algebra:** Binary segmentation masks are represented as sorted 1D run intervals $\mathcal{R} = \{[s_m, e_m)\}_{m=1}^{R}$. Area, 2D centroid, bounding box, and pairwise mask IoU ($\mathcal{R}_A \cap \mathcal{R}_B$) are computed directly over run intervals in $O(R_A + R_B)$ time without materializing $H \times W$ boolean arrays.
- **GPU Window Overlap (`SAM2GpuWindowOverlapScope`):** Schedules deterministic CPU depth registration and RLE feature extraction concurrently within the asynchronous CUDA execution window of SAM 2 mask propagation.

### 5. Causal $\mathrm{SE}(3)$ Trajectory Prediction & L2 Teleoperation Safety (`causalpose3d.teleop`)
To support fail-closed **L2 Teleoperation Assistance** (e.g., precision pouring and container alignment) without acausal lookahead or unsafe false-positive guidance:
- **Multi-Horizon $\mathrm{SE}(3)$ Prediction & $C^1$-Continuous Re-Acquisition Blending (`CausalSe3TrajectoryPredictor`):** Extrapolates object poses across $100 / 250 / 500\,\mathrm{ms}$ horizons on $\mathrm{SE}(3)$ with strictly monotonic 95% confidence bounds ($\Delta t_1 < \Delta t_2 \implies w_{95}(\Delta t_1) < w_{95}(\Delta t_2)$). Upon re-acquisition after an occlusion gap, a cubic Hermite $C^1$ join blender transitions from the pre-join trajectory with zero position jump (`position_join_jump = 0.0 m`) and bounded velocity innovation.
- **3D Task-Feature Rim Consensus & First-Order $\mathrm{SE}(3)$ Covariance Propagation (`causalpose3d.teleop.task_features`):** Fits 3D container apertures (`MeasuredCircularAperture3D`) via iterative Huber SVD plane estimation, 8-sector boundary quantile selection, and 256-hypothesis 3-point RANSAC consensus (`compare_circle_consensus_3d`), and propagates object-local feature uncertainty (`MeasuredPointFeature3D`) through the right-trivialized $\mathrm{SE}(3)$ Jacobian $\boldsymbol{\Sigma}_{\mathbf{p},W} = \mathbf{J}_{\mathbf{p}}\boldsymbol{\Sigma}_{\xi}\mathbf{J}_{\mathbf{p}}^\top + \mathbf{R}\boldsymbol{\Sigma}_{O}\mathbf{R}^\top$.
- **Ballistic Liquid Stream Locus & Fail-Closed Rim-Clearance Gate (`causalpose3d.teleop.pouring_safety`):** Models the 3D parabolic liquid stream $\mathbf{x}(s) = \mathbf{p}_{\text{out}} + v_0 \mathbf{d}_{\text{out}} s + \frac{1}{2}\mathbf{g} s^2$, analytically propagates $(\boldsymbol{\Sigma}_{\mathbf{p}}, \sigma_\theta^2, \sigma_{v_0}^2)$ onto the target container's aperture plane to obtain the 95% landing uncertainty ellipse, and enforces a fail-closed rim-clearance envelope across all predicted horizons while outputting a minimum-norm shared-control guidance vector $\Delta\mathbf{p}_{\text{assist}}$.

---

## 📊 Experimental Evaluation

### 1. 6D Pose Tracking & Surface Reconstruction Benchmarks

We evaluate **CausalPose-3D** against upstream and deterministic variants of *BundleTrack* and *Point2Pose* on standard manipulation benchmarks (**YCBInEOAT**: `bleach0`, `sugar_box0`, `mustard0`; **Laval Vicon 6DoF**: `Clock`, `Jar`) under identical RGB-D and segmentation inputs.

| Method | Benchmark | Frames | Candidate Duty ($\uparrow$) | Mean ADD [mm] ($\downarrow$) | Mean ADD-S [mm] ($\downarrow$) | 10-View Corroboration $C_{10}$ ($\uparrow$) | Run-to-Run Spread ($\downarrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Upstream BundleTrack | YCBInEOAT | 430 | 84.2% | 34.9 | 14.8 | 40.57% | 18.4 mm / 4.72° |
| Deterministic BundleTrack | YCBInEOAT | 430 | 88.6% | 29.4 | 12.1 | 40.57% | **0.00 mm / 0.00°** |
| Point2Pose ($F+G+R+M+S$) | YCBInEOAT | 430 | 79.5% | **24.1** | **10.9** | 84.12% | **0.00 mm / 0.00°** |
| **CausalPose-3D (Ours)** | **YCBInEOAT** | **430** | **96.3%** | 26.8 | 11.4 | **85.10%** | **0.00 mm / 0.00°** |
| Deterministic BundleTrack | Laval Vicon 6DoF | 274 | 85.4% | 48.2 | 19.6 | 52.34% | **0.00 mm / 0.00°** |
| Point2Pose ($F+G+R+M+S$) | Laval Vicon 6DoF | 274 | 74.8% | **42.3** | **17.8** | 83.60% | **0.00 mm / 0.00°** |
| **CausalPose-3D (Ours)** | **Laval Vicon 6DoF** | **274** | **91.2%** | 45.7 | 18.9 | **85.05%** | **0.00 mm / 0.00°** |

<div align="center">
  <table>
    <tr>
      <td align="center" width="50%">
        <img src="docs/assets/figures/fig6_6dof_carton_pose_and_jump_gate.png" alt="6DoF Pose Trajectory and Jump Gate" width="100%"/>
        <br/>
        <sub><b>Figure 2:</b> Continuous 6DoF object trajectory and causal jump-gate verification under rapid articulated rotation.</sub>
      </td>
      <td align="center" width="50%">
        <img src="docs/assets/figures/fig3_runtime_latency_breakdown.png" alt="Per-Stage Runtime Latency Breakdown" width="100%"/>
        <br/>
        <sub><b>Figure 3:</b> Per-stage latency distribution before and after GPU/CPU window overlap and $O(\text{runs})$ RLE algebra.</sub>
      </td>
    </tr>
  </table>
</div>

### 2. Ablation: Surface Selection Policy (`RevisableSurfelMap`)

Comparison of surface selection policies under a fixed $N_{\max} = 1,500$ surfel budget across YCBInEOAT (`bleach0`) and Laval Vicon (`Clock`):

| Selection Policy | YCBInEOAT $C_{10}$ ($\uparrow$) | Laval Vicon $C_{10}$ ($\uparrow$) | Task Surface Coverage ($\uparrow$) | Selector Latency [ms] ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: |
| Uniform Index Stride (`stride`) | 40.57% | 52.34% | 71.78% | 2.21 ms |
| Uniform 3D Voxel Downsampling (`voxel5mm`) | 46.12% | 58.90% | 78.40% | 3.84 ms |
| Support-Only Top-$K$ (`support_topk`) | 86.40% | 86.10% | 64.15% | 1.19 ms |
| **Two-Stage `support10/coarse12mm` (Ours)** | **85.10%** | **85.05%** | **89.97%** | **1.48 ms** |

### 3. Ablation: Runtime Execution & Bit-Exact Verification

| Optimization Stage | Exclusive Host Overhead [ms] | End-to-End $p_{50}$ / $p_{95}$ [ms] | Bit-Exact Parity (499 Checks) |
| :--- | :---: | :---: | :---: |
| Baseline Sequential Execution | 23.17 ms | 155.33 / 219.06 ms | Reference |
| + `SAM2GpuWindowOverlapScope` | 14.82 ms | 141.10 / 184.50 ms | 499 / 499 (100%) |
| + `RequestEvidenceDeduplicationScope` & $O(\text{runs})$ RLE | 8.45 ms | 132.40 / 162.80 ms | 499 / 499 (100%) |
| + Vectorized Plane SVD & Boundary GC Control | **6.28 ms (-72.9%)** | **127.80 / 153.40 ms** | **499 / 499 (100%)** |

---

## 🛠️ Installation & Usage

### 1. Python Package Installation

**Requirements:** Python $\ge 3.10$, NumPy $\ge 1.24$, SciPy $\ge 1.11$, OpenCV $\ge 4.8$, PyTorch $\ge 2.1$ (optional for GPU acceleration).

```bash
# Option A: Install directly from the official GitHub Release wheel
pip install https://github.com/tiwater/causalpose3d/releases/download/v1.0.0/causalpose3d-1.0.0-py3-none-any.whl

# Option B: Clone and install in editable mode for research & development
git clone https://github.com/tiwater/causalpose3d.git
cd causalpose3d
pip install -e ".[dev]"
pytest
```

### 2. Quick Start: Python API

```python
import numpy as np
from causalpose3d.surface import RevisableSurfelMap, measured_free_space_margin
from causalpose3d.primitives.image_mask import encode_mask_rle, decode_mask_rle

# 1. Compact Row-Major Run-Length Mask Encoding (O(runs))
mask = np.zeros((480, 640), dtype=bool)
mask[120:260, 200:380] = True
rle = encode_mask_rle(mask)
restored = decode_mask_rle(rle, width=640, height=480)
assert np.array_equal(mask, restored)

# 2. Lineage-Indexed Revisable Surfel Reconstruction across Multi-View Observations
surfel_map = RevisableSurfelMap(
    voxel_size_m=0.005,
    maximum_voxels=50_000,
    maximum_live_bytes=16 * 1024 * 1024,
)
pts_view0 = np.array([[0.01, 0.02, 0.45], [0.012, 0.021, 0.451]], dtype=np.float64)
pts_view1 = np.array([[0.011, 0.019, 0.449], [0.04, -0.01, 0.46]], dtype=np.float64)
surfel_map.update({"kf_0": ("chest", pts_view0), "kf_1": ("head", pts_view1)})
points_m, obs_support, cam_support = surfel_map.snapshot(
    minimum_observation_support=1,
    maximum_point_count=1500,
)
print(f"Fused surfels: {len(points_m)}, multi-view support: {obs_support.tolist()}")

# 3. Fail-Closed L2 Teleoperation Pouring Safety Verification
from causalpose3d.teleop import (
    MeasuredPointFeature3D,
    fit_circular_aperture_3d,
    verify_l2_pouring_safety_envelope,
)

angles = np.linspace(-np.pi, np.pi, 120, endpoint=False)
rim_pts = np.column_stack([0.30 + 0.045 * np.cos(angles), 0.045 * np.sin(angles), np.full(120, 0.05)])
aperture = fit_circular_aperture_3d(rim_pts, coordinate_frame="table")
spout = MeasuredPointFeature3D(
    feature_id="pour_outlet",
    point_m=np.array([0.285, 0.0, 0.15]),
    direction=np.array([0.25, 0.0, -0.9682458]),
    covariance_3x3_m2=np.eye(3) * 4e-6,
    direction_std_rad=0.02,
    support_pixel_count=40,
    timestamp_ns=1_000_000_000,
    coordinate_frame="table",
)
verdict = verify_l2_pouring_safety_envelope(pour_outlet_world=spout, target_aperture_world=aperture)
print(f"L2 Status: {verdict.l2_pour_safety_status}, Rim Clearance (95%): {verdict.current_rim_clearance_95_m*1000:.1f} mm")
```

### 3. Native C++/CUDA `BundleTrack` Patches (`native/bundletrack/`)

To apply our deterministic CUDA hypothesis reduction, exact $\mathrm{SE}(3)$ pose-cache inversion, and contiguous frame retirement patches to a native `BundleSDF / BundleTrack` checkout:

```bash
cd native/bundletrack
# Prepare and verify the patched deterministic C++/CUDA BundleTrack source tree:
python3 prepare_source.py --help

# Or apply the patch series (0001-0004) directly to an upstream BundleSDF/BundleTrack checkout:
git apply patches/0001-cmake-sm120-compatibility.patch
git apply patches/0002-support-disabled-pose-optimization.patch
git apply patches/0003-release-ransac-confidence-buffer.patch
git apply patches/0004-select-dense-direction-by-count.patch
```

### 4. Reproducing Benchmark Tables & Paper Manuscripts

```bash
# Regenerate quantitative sequence summaries and LaTeX benchmark tables
python3 benchmarks/sequence_summary.py
python3 benchmarks/generate_table.py

# Compile the 8-page Comparative Study (papers/causalpose3d-short/main.tex)
tectonic papers/causalpose3d-short/main.tex

# Compile the 17-page Technical Monograph (papers/causalpose3d-monograph/main.tex)
tectonic papers/causalpose3d-monograph/main.tex
```

---

## 📂 Repository Organization

```text
causalpose3d/
├── causalpose3d/                  # Core Python library (100% standalone)
│   ├── surface/                   # RevisableSurfelMap, CausalObjectSurface (M, S), multi-camera surface fusion
│   ├── tracking/                  # PinnedBundleTrackRunner, 3-camera pose runtime, constrained Top XY refinement
│   ├── observability/             # Point2Pose F/G/R Schur-complement gating & pose-owner A/B evaluators
│   ├── runtime/                   # SAM2GpuWindowOverlapScope, O(runs) RLE algebra, vectorized plane RANSAC
│   ├── teleop/                    # L2 teleoperation: causal SE(3) trajectory, 3D task features & pouring safety
│   └── primitives/                # Registered RGB-D, mask-depth PCA pose, content-addressed hashing
├── native/bundletrack/            # Deterministic C++/CUDA BundleTrack sources & patch series (0001–0004)
├── benchmarks/                    # Reproducible evaluation receipts, ablation scripts, and table generators
├── papers/                        # LaTeX sources, figures, and compiled PDFs for both manuscripts
│   ├── causalpose3d-monograph/    # 17-page full technical monograph
│   └── causalpose3d-short/        # 8-page comparative paper
├── docs/                          # Academic project page (https://tiwater.github.io/causalpose3d/)
└── tests/                         # Unit and regression test suite
```

---

## 🔖 Citation

If you find **CausalPose-3D**, our deterministic C++/CUDA BundleTrack patches, or our factor-graph observability analysis useful in your research, please cite:

```bibtex
@article{tiwater2026causalpose3d,
  title   = {{CausalPose-3D}: Causal Observability-Conditioned 6D Pose Tracking and 3D Surface Reconstruction from {RGB-D} Video},
  author  = {{Tiwater Research}},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://tiwater.github.io/causalpose3d/}
}
```

---

## 🙏 Acknowledgements

This research builds upon foundational open-source contributions in 6D pose estimation, point tracking, and video segmentation, including [BundleSDF / BundleTrack](https://github.com/NVlabs/BundleSDF), [Point2Pose](https://github.com/point2pose/point2pose), [SAM 2](https://github.com/facebookresearch/sam2), [TAPIR / BootSTAPIR](https://github.com/google-deepmind/tapnet), and [Open3D](https://github.com/isl-org/Open3D). We thank the authors for making their code and datasets publicly available.
