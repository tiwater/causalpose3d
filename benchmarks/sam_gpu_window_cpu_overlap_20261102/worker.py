"""Same-thread CPU preparation inside SAM2's GPU execution window.

Preserves all existing perception owners (OriginalPersistentWorldRegistry,
ObjectCentricObservationSession, SAM2ObservationSession,
SAM2VideoMemoryTracker, PinnedBundleTrackRuntime, CausalObjectSurface),
exact state transition order, OpenMP thread restoration, and fail-closed
input validation. Spawns zero background worker threads.
"""

from __future__ import annotations

import contextvars
import ctypes
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time

import cv2
import numpy as np
import torch

from ticos_replay import bundletrack_live_input
from ticos_replay.bundletrack_live_input import RegisteredBundleTrackInput
from ticos_replay.pinned_bundletrack_runner import (
    PinnedBundleTrackRuntime,
    PinnedBundleTrackSession,
)
from ticos_replay.sam2_observation_session import SAM2ObservationSession
from ticos_replay import sam2_observation_session as sam2_session_mod
from ticos_vision.perception import (
    image_mask,
    original_persistent_world as world,
    sam2_cpu_preparation as preparation,
)
from ticos_vision.perception.sam2_gpu_window_overlap import (
    advance_registered_with_prepared_buffers,
    complete_evidence_hash_from_prefix,
    validate_and_create_pending_preparation,
)
from ticos_vision.perception.sam2_video_memory import SAM2VideoMemoryTracker

R = Path(__file__).resolve().parents[3]
output = Path(sys.argv[sys.argv.index("--output") + 1])
mode = os.environ["TICOS_SAM_GPU_WINDOW_OVERLAP_MODE"]
assert mode in ("control", "overlap", "validate")

current_ctx: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "sam_gpu_window_overlap_ctx", default=None
)

_lib, get_omp, _set_omp = preparation._openmp_functions()
native_get = [None]

orig_registry_advance = world.OriginalPersistentWorldRegistry.advance
orig_prepare_tracking = bundletrack_live_input.prepare_tracking_observation
orig_sam_session_advance = SAM2ObservationSession.advance
orig_tracker_track = SAM2VideoMemoryTracker.track
orig_encode_mask_rle = image_mask.encode_mask_rle
orig_mask_evidence_sha256 = sam2_session_mod._mask_evidence_sha256
orig_prefetch_hash = RegisteredBundleTrackInput.prefetch_evidence_hash
orig_session_advance_registered = PinnedBundleTrackSession.advance_registered
orig_runtime_track_registered = PinnedBundleTrackRuntime.track_registered
orig_thread_start = threading.Thread.start

wrapped_predictor_classes: dict[type, object] = {}
records: list[dict] = []
masks: list[dict] = []
native_inputs: list[dict] = []
inferences: list[dict] = []
native_calls: list[dict] = []
evidence_hash_threads_spawned = 0


def counters() -> dict[str, int | None]:
    return dict(
        torch_threads=torch.get_num_threads(),
        torch_omp=get_omp(),
        native_omp=native_get[0]() if native_get[0] else None,
    )


def memory_snapshot() -> dict[str, int]:
    return dict(
        rss_bytes=int(Path("/proc/self/statm").read_text().split()[1])
        * os.sysconf("SC_PAGE_SIZE"),
        cuda_allocated=int(torch.cuda.memory_allocated()),
        cuda_reserved=int(torch.cuda.memory_reserved()),
    )


memory_before = memory_snapshot()


def counted_thread_start(self, *a, **kw):
    global evidence_hash_threads_spawned
    if getattr(self, "name", "") == "ticos-edge-evidence-hash":
        evidence_hash_threads_spawned += 1
        ctx = current_ctx.get()
        if ctx is not None and ctx.get("record") is not None:
            ctx["record"]["evidence_hash_thread_spawned"] = True
    return orig_thread_start(self, *a, **kw)


def ensure_predictor_wrapped(predictor) -> None:
    cls = type(predictor)
    if cls in wrapped_predictor_classes or not hasattr(cls, "propagate_in_video"):
        return
    orig_propagate = cls.propagate_in_video
    wrapped_predictor_classes[cls] = orig_propagate

    def observed_propagate(self_pred, inference_state, *a, **kw):
        ctx = current_ctx.get()
        pending = ctx.get("pending") if ctx is not None else None
        is_active_state = (
            ctx is not None
            and ctx.get("active_state_id") == id(inference_state)
        )
        launch_ns = time.perf_counter_ns()
        for item in orig_propagate(self_pred, inference_state, *a, **kw):
            yield_ns = time.perf_counter_ns()
            if is_active_state and ctx.get("record") is not None:
                ctx["record"]["gpu_launch_to_yield_ms"] = round(
                    (yield_ns - launch_ns) / 1e6, 3
                )
            if (
                is_active_state
                and pending is not None
                and not pending.materialized
                and pending.state_obj_id == id(inference_state)
            ):
                pending.gpu_launch_to_yield_ms = round(
                    (yield_ns - launch_ns) / 1e6, 3
                )
                pending.materialize(in_gpu_window=True)
                if mode == "validate":
                    eager = orig_prepare_tracking(
                        pending.observation,
                        registered=pending.registered,
                        object_id=pending.object_id,
                        color_rgb=pending.color,
                        max_sync_skew_ms=ctx["max_sync_skew_ms"],
                    )
                    frame = pending.frame
                    exact = (
                        frame.stream_key == eager.stream_key
                        and frame.color_timestamp_ns == eager.color_timestamp_ns
                        and frame.depth_timestamp_ns == eager.depth_timestamp_ns
                        and frame.source_evidence_hash == eager.source_evidence_hash
                        and frame.feature_source_binding == eager.feature_source_binding
                        and frame.sensor_calibration == eager.sensor_calibration
                        and np.array_equal(frame.color_rgb, eager.color_rgb)
                        and np.array_equal(frame.depth_mm, eager.depth_mm)
                        and np.array_equal(frame.mask, eager.mask)
                        and np.array_equal(frame.intrinsics, eager.intrinsics)
                    )
                    ctx["record"]["prepared_frame_exact"] = bool(exact)
                    assert exact, "GPU-window prepared frame differed from eager preparation"
            if is_active_state:
                ctx["post_window_sync_start_ns"] = time.perf_counter_ns()
            yield item

    cls.propagate_in_video = observed_propagate


def observed_registry_advance(self, *a, **kw):
    ctx = dict(
        registry=self,
        pending=None,
        record=None,
        active_state_id=None,
        post_window_sync_start_ns=None,
        max_sync_skew_ms=self.config.maximum_rgbd_skew_ms,
        advance_start_ns=time.perf_counter_ns(),
    )
    token = current_ctx.set(ctx)
    try:
        return orig_registry_advance(self, *a, **kw)
    finally:
        pending = ctx["pending"]
        rec = ctx["record"]
        if pending is not None:
            assert pending.materialized, "Pending preparation was not materialized"
            if rec is not None:
                rec.update(
                    executed_in_gpu_window=bool(pending.executed_in_gpu_window),
                    gpu_window_cpu_prepare_ms=pending.gpu_window_cpu_prepare_ms,
                    evidence_hash_tail_ms=pending.evidence_hash_tail_ms,
                    rle_reused=bool(pending.rle_reused),
                    native_buffers_reused=bool(pending.native_buffers_reused),
                )
            pending.color_bgr = None
            pending.depth_m = None
            pending.evidence_hash_color_prefix = None
            pending.evidence_hash_prefix = None
            pending.visual_mask_rle_cache = None
            ctx["pending"] = None
        if rec is not None:
            rec["registry_advance_ms"] = round(
                (time.perf_counter_ns() - ctx["advance_start_ns"]) / 1e6, 3
            )
            records.append(rec)
        current_ctx.reset(token)


def observed_prepare_tracking(observation, *a, **kw):
    ctx = current_ctx.get()
    if ctx is None:
        return orig_prepare_tracking(observation, *a, **kw)
    active = ctx["registry"]._active
    tracker = getattr(
        getattr(getattr(active, "session", None), "masks", None), "tracker", None
    )
    if tracker is not None and getattr(tracker, "_predictor", None) is not None:
        ensure_predictor_wrapped(tracker._predictor)
        ctx["active_state_id"] = id(getattr(tracker, "_state", None))

    row = dict(
        warm_index=len(records) + 1,
        color_timestamp_ns=str(observation.color_timestamp_ns),
        depth_timestamp_ns=str(observation.depth_timestamp_ns),
        mode=mode,
        eligible=False,
        executed_in_gpu_window=False,
        rle_reused=False,
        native_buffers_reused=False,
        evidence_hash_thread_spawned=False,
    )
    ctx["record"] = row
    start_ns = time.perf_counter_ns()
    if mode != "control" and active is not None:
        pending = validate_and_create_pending_preparation(
            observation,
            registered=kw["registered"],
            object_id=kw["object_id"],
            color_rgb=kw.get("color_rgb"),
            max_sync_skew_ms=kw.get("max_sync_skew_ms", 20.0),
            active_session=active.session,
        )
        if pending is not None:
            row["eligible"] = True
            ctx["pending"] = pending
            row["input_prepare_ms"] = round(
                (time.perf_counter_ns() - start_ns) / 1e6, 3
            )
            return pending.frame
    result = orig_prepare_tracking(observation, *a, **kw)
    row["input_prepare_ms"] = round(
        (time.perf_counter_ns() - start_ns) / 1e6, 3
    )
    return result


def observed_sam_session_advance(self, frame, *a, **kw):
    ctx = current_ctx.get()
    start_ns = time.perf_counter_ns()
    if ctx is not None:
        ctx["in_sam_session_advance"] = True
    try:
        return orig_sam_session_advance(self, frame, *a, **kw)
    finally:
        if ctx is not None:
            ctx["in_sam_session_advance"] = False
            pending = ctx.get("pending")
            if pending is not None:
                pending.visual_mask_rle_cache = None
            if ctx.get("record") is not None:
                ctx["record"]["sam_session_advance_ms"] = round(
                    (time.perf_counter_ns() - start_ns) / 1e6, 3
                )


def observed_tracker_track(self, **kw):
    if getattr(self, "_predictor", None) is not None:
        ensure_predictor_wrapped(self._predictor)
    ctx = current_ctx.get()
    row = dict(
        before=counters(),
        configured_threads=self._cpu_prepare_threads,
        registry_context=ctx is not None,
    )
    start_ns = time.perf_counter_ns()
    try:
        result = orig_tracker_track(self, **kw)
        done_ns = time.perf_counter_ns()
        if ctx is not None and ctx.get("record") is not None:
            rec = ctx["record"]
            rec["sam_track_ms"] = round((done_ns - start_ns) / 1e6, 3)
            sync_start = ctx.get("post_window_sync_start_ns")
            if sync_start is not None:
                rec["post_cpu_gpu_sync_wait_ms"] = round(
                    (done_ns - sync_start) / 1e6, 3
                )
        masks.append(
            dict(
                registry_context=ctx is not None,
                mask_sha256=(
                    None
                    if result is None
                    else hashlib.sha256(result.mask.tobytes()).hexdigest()
                ),
            )
        )
        return result
    finally:
        pending = ctx.get("pending") if ctx is not None else None
        if pending is not None and not pending.materialized:
            pending.materialize(in_gpu_window=False)
        row["after"] = counters()
        inferences.append(row)


def observed_encode_mask_rle(mask):
    ctx = current_ctx.get()
    pending = ctx.get("pending") if ctx is not None else None
    if (
        mode != "control"
        and ctx is not None
        and ctx.get("in_sam_session_advance")
        and pending is not None
        and isinstance(mask, np.ndarray)
    ):
        cached = pending.visual_mask_rle_cache
        if cached is not None and cached[0] is mask:
            if mode == "validate":
                expected = orig_encode_mask_rle(mask)
                assert expected == cached[1], "Cached visual mask RLE mismatch"
                ctx["record"]["rle_exact"] = True
            pending.rle_reused = True
            return cached[1]
        rle = orig_encode_mask_rle(mask)
        pending.visual_mask_rle_cache = (mask, rle, "")
        return rle
    return orig_encode_mask_rle(mask)


def observed_mask_evidence_sha256(
    stream_key, color_timestamp_ns, depth_timestamp_ns, mask_rle
):
    ctx = current_ctx.get()
    pending = ctx.get("pending") if ctx is not None else None
    if mode != "control" and pending is not None:
        cached = pending.visual_mask_rle_cache
        if (
            cached is not None
            and mask_rle is cached[1]
            and stream_key == pending.stream_key
            and color_timestamp_ns == pending.color_timestamp_ns
            and depth_timestamp_ns == pending.depth_timestamp_ns
        ):
            if cached[2]:
                if mode == "validate":
                    expected = orig_mask_evidence_sha256(
                        stream_key,
                        color_timestamp_ns,
                        depth_timestamp_ns,
                        mask_rle,
                    )
                    assert expected == cached[2], "Cached mask_evidence_sha256 mismatch"
                return cached[2]
            digest = orig_mask_evidence_sha256(
                stream_key, color_timestamp_ns, depth_timestamp_ns, mask_rle
            )
            pending.visual_mask_rle_cache = (cached[0], cached[1], digest)
            return digest
    return orig_mask_evidence_sha256(
        stream_key, color_timestamp_ns, depth_timestamp_ns, mask_rle
    )


def observed_prefetch_hash(self):
    ctx = current_ctx.get()
    pending = ctx.get("pending") if ctx is not None else None
    if mode != "control" and pending is not None:
        if complete_evidence_hash_from_prefix(self, pending):
            if mode == "validate":
                expected = self._compute_evidence_hash()
                exact = self._evidence_hash == expected
                ctx["record"]["evidence_hash_exact"] = bool(exact)
                assert exact, "Prefix-completed evidence_hash mismatch"
            return
    return orig_prefetch_hash(self)


def observed_session_advance_registered(self, frame):
    ctx = current_ctx.get()
    pending = ctx.get("pending") if ctx is not None else None
    if mode != "control" and pending is not None:
        depth_unchanged = bool(
            np.shares_memory(frame.depth_mm, pending.frame.depth_mm)
        )
        ctx["record"]["depth_far_zeroed"] = not depth_unchanged
        if (
            mode == "validate"
            and pending.color_bgr is not None
            and pending.depth_m is not None
        ):
            expected_bgr = cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR)
            bgr_exact = np.array_equal(pending.color_bgr, expected_bgr)
            if depth_unchanged:
                expected_depth = np.ascontiguousarray(
                    frame.depth_mm, dtype=np.float32
                ) / np.float32(1000.0)
                depth_exact = np.array_equal(pending.depth_m, expected_depth)
            else:
                depth_exact = True
            exact = bool(bgr_exact and depth_exact)
            ctx["record"]["native_buffers_exact"] = exact
            assert exact, "GPU-window native buffers differed from eager conversion"
        return advance_registered_with_prepared_buffers(
            self, frame, pending, orig_session_advance_registered
        )
    return orig_session_advance_registered(self, frame)


def observed_runtime_track_registered(self, frame):
    if native_get[0] is None:
        fn = ctypes.CDLL(
            str(
                Path(
                    os.environ[
                        "TICOS_VISION_ORIGINAL_WORLD_BUNDLESDF_BUILD_ROOT"
                    ]
                )
                / "libBundleTrack.so"
            )
        ).omp_get_max_threads
        fn.restype = ctypes.c_int
        native_get[0] = fn
    ctx = current_ctx.get()
    row = dict(before=counters())
    native_inputs.append(
        dict(
            color_timestamp_ns=str(frame.color_timestamp_ns),
            input_evidence_hash=frame.evidence_hash,
        )
    )
    start_ns = time.perf_counter_ns()
    try:
        return orig_runtime_track_registered(self, frame)
    finally:
        if ctx is not None and ctx.get("record") is not None:
            ctx["record"]["native_track_registered_ms"] = round(
                (time.perf_counter_ns() - start_ns) / 1e6, 3
            )
        row["after"] = counters()
        native_calls.append(row)


world.OriginalPersistentWorldRegistry.advance = observed_registry_advance
bundletrack_live_input.prepare_tracking_observation = observed_prepare_tracking
SAM2ObservationSession.advance = observed_sam_session_advance
SAM2VideoMemoryTracker.track = observed_tracker_track
image_mask.encode_mask_rle = observed_encode_mask_rle
sam2_session_mod._mask_evidence_sha256 = observed_mask_evidence_sha256
RegisteredBundleTrackInput.prefetch_evidence_hash = observed_prefetch_hash
PinnedBundleTrackSession.advance_registered = observed_session_advance_registered
PinnedBundleTrackRuntime.track_registered = observed_runtime_track_registered
threading.Thread.start = counted_thread_start

try:
    runpy.run_path(
        str(
            R
            / "docs/papers/configured-replay-portal-lifetime-20261101/implemented_worker.py"
        ),
        run_name="__main__",
    )
finally:
    world.OriginalPersistentWorldRegistry.advance = orig_registry_advance
    bundletrack_live_input.prepare_tracking_observation = orig_prepare_tracking
    SAM2ObservationSession.advance = orig_sam_session_advance
    SAM2VideoMemoryTracker.track = orig_tracker_track
    image_mask.encode_mask_rle = orig_encode_mask_rle
    sam2_session_mod._mask_evidence_sha256 = orig_mask_evidence_sha256
    RegisteredBundleTrackInput.prefetch_evidence_hash = orig_prefetch_hash
    PinnedBundleTrackSession.advance_registered = orig_session_advance_registered
    PinnedBundleTrackRuntime.track_registered = orig_runtime_track_registered
    threading.Thread.start = orig_thread_start
    for cls, orig_fn in wrapped_predictor_classes.items():
        cls.propagate_in_video = orig_fn

    output.mkdir(parents=True, exist_ok=True)
    (output / "gpu-window-overlap.json").write_text(
        json.dumps(
            dict(
                mode=mode,
                warm_frames=records,
                masks=masks,
                native_inputs=native_inputs,
                inferences=inferences,
                native_calls=native_calls,
                evidence_hash_threads_spawned=evidence_hash_threads_spawned,
                background_worker_threads_created=0,
                memory_before=memory_before,
                memory_after=memory_snapshot(),
                worker_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                adapter_sha256=hashlib.sha256(
                    (
                        R
                        / "apps/scene-runtime/ticos_vision/perception/sam2_gpu_window_overlap.py"
                    ).read_bytes()
                ).hexdigest(),
                wire_authorized=False,
            ),
            indent=2,
        )
        + "\n"
    )
