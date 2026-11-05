"""Offline replay tap of the existing native owner, never a pose estimator.

Installed explicitly by configured_scene_replay_audit, not a live route.
Records native evidence before the session consumes it for surface fusion.
"""
from contextlib import contextmanager
import json
import uuid
import hashlib
import numpy as np


class _TracedRuntime:
    def __init__(self, runtime, stream, geometry_dir=None, capture_native_selection=False, capture_color=False):
        self._runtime = runtime
        self._stream = stream
        self._session_id = uuid.uuid4().hex
        self._geometry_dir = geometry_dir
        self._capture_native_selection = capture_native_selection
        self._capture_color = capture_color

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def track_registered(self, frame):
        selection = []
        if self._capture_native_selection:
            tracker = self._runtime.session.tracker
            original = tracker.make_frame
            def observe(color, depth, K, id_str, mask, *args, **kwargs):
                selection.append({"native_pre_frame_depth_m": np.array(depth, copy=True),
                                  "native_pre_frame_mask": np.array(mask, copy=True),
                                  "native_pre_frame_intrinsics": np.array(K, copy=True),
                                  "native_pre_frame_id": str(id_str)})
                return original(color, depth, K, id_str, mask, *args, **kwargs)
            tracker.make_frame = observe
            try:
                candidate = self._runtime.track_registered(frame)
            finally:
                tracker.make_frame = original
            if len(selection) != 1 or selection[0]["native_pre_frame_id"] != f"{candidate['output_index']:06d}":
                raise ValueError("native selection trace requires one exact current make_frame call")
        else:
            candidate = self._runtime.track_registered(frame)
        record = {
            "schema": "ticos.native-history-trace.v1",
            "session_id": self._session_id,
            "capture_boundary": "after_native_before_shape_and_surface",
            "publication_authorized": False,
            "frame": {
                "stream_key": list(frame.stream_key),
                "color_timestamp_ns": str(frame.color_timestamp_ns),
                "depth_timestamp_ns": str(frame.depth_timestamp_ns),
                "input_evidence_hash": frame.evidence_hash,
            },
            "candidate": candidate,
        }
        if self._geometry_dir is not None:
            artifact = self._geometry_dir / f"{self._session_id}-{candidate['output_index']:06d}.npz"
            with artifact.open("xb") as handle:
                np.savez_compressed(
                    handle, depth_mm=frame.depth_mm, mask=frame.mask,
                    intrinsics=frame.intrinsics,
                    color_timestamp_ns=str(frame.color_timestamp_ns),
                    depth_timestamp_ns=str(frame.depth_timestamp_ns),
                    input_evidence_hash=frame.evidence_hash,
                    stream_key_json=json.dumps(list(frame.stream_key)),
                    session_id=self._session_id, output_index=candidate["output_index"],
                    **({"color_rgb": frame.color_rgb,
                        "source_evidence_hash": frame.source_evidence_hash} if self._capture_color else {}),
                    **(selection[0] if selection else {}),
                )
            record["frame"]["geometry_artifact"] = {
                "path": f"{self._geometry_dir.name}/{artifact.name}",
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "scope": "exact_registered_native_input_not_independent_ground_truth",
            }
            if selection:
                record["frame"]["geometry_artifact"]["native_selection_boundary"] = (
                    "after_native_percentile_before_make_frame_and_hole_fill")
            if self._capture_color:
                record["frame"]["geometry_artifact"]["registered_color_captured"] = True
        self._stream.write(json.dumps(record, allow_nan=False) + "\n")
        self._stream.flush()
        # Identity-preserving return: the observer cannot replace native poses.
        return candidate


@contextmanager
def record_native_history(registry, path, *, capture_geometry=False, capture_native_selection=False, capture_color=False):
    """Temporarily wrap the replay process's existing native factory.

    The caller owns session shutdown; factory restoration occurs on all exits.
    Never attach to an already-running registry (including deferred SAM state).
    """
    if registry.diagnostics().get("connected"):
        raise ValueError("native history trace requires an inactive replay registry")
    if capture_native_selection and not capture_geometry:
        raise ValueError("native selection capture requires geometry capture")
    if capture_color and not capture_geometry:
        raise ValueError("registered color capture requires geometry capture")
    original = registry._runtime_factory
    with path.open("x", encoding="utf-8") as stream:
        geometry_dir = None
        if capture_geometry:
            geometry_dir = path.parent / "native-geometry"
            geometry_dir.mkdir(exist_ok=False)
        registry._runtime_factory = lambda config: _TracedRuntime(original(config), stream, geometry_dir, capture_native_selection, capture_color)
        try:
            yield
        finally:
            registry._runtime_factory = original
