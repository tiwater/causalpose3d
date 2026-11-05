"""Offline adapter only; never allocates or mutates a surface or estimates pose.

Producer/owner: docs/operations/supplemental-surface-packet-design-20261023.json.
"""
import numpy as np

from evidencescene.primitives.bundletrack_live_input import RegisteredBundleTrackInput
from evidencescene.primitives.geometry import (
    _backproject_masked_depth,
    _transform_points,
    _validate_rigid_transform,
)


def prepare_packet(primary, secondary, candidate, bridge, *, maximum_skew_ns):
    if not all(isinstance(f, RegisteredBundleTrackInput) for f in (primary, secondary)):
        raise ValueError('Registered inputs required')
    if type(maximum_skew_ns) is not int or maximum_skew_ns < 0:
        raise ValueError('Explicit nonnegative sensor skew bound required')
    if primary.stream_key[2] == secondary.stream_key[2]:
        raise ValueError('Supplemental camera must be distinct')
    if primary.stream_key[0] != secondary.stream_key[0]:
        raise ValueError('Supplemental object identity mismatch')
    if not isinstance(bridge, dict):
        raise ValueError('Measured bridge required')
    for key, expected in (
        ('stream_key', list(primary.stream_key)),
        ('input_evidence_hash', primary.evidence_hash),
        ('color_timestamp_ns', str(primary.color_timestamp_ns)),
        ('depth_timestamp_ns', str(primary.depth_timestamp_ns)),
    ):
        if candidate.get(key) != expected:
            raise ValueError('Primary pose evidence mismatch')
    if candidate.get('status') != 'candidate':
        raise ValueError('Current primary pose unavailable')
    if (bridge.get('primary_stream_key') != list(primary.stream_key)
            or bridge.get('secondary_stream_key') != list(secondary.stream_key)):
        raise ValueError('Measured bridge source mismatch')
    digest = bridge.get('calibration_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('Calibration provenance required')
    if max(abs(primary.color_timestamp_ns-secondary.color_timestamp_ns),
           abs(primary.depth_timestamp_ns-secondary.depth_timestamp_ns)) > maximum_skew_ns:
        raise ValueError('Paired sensor skew exceeds bound')
    primary_from_object = _validate_rigid_transform(candidate.get('camera_from_object'), name='existing pose')
    primary_from_secondary = _validate_rigid_transform(bridge.get('primary_from_secondary'), name='measured bridge')
    object_from_secondary = np.linalg.inv(primary_from_object) @ primary_from_secondary
    points = _backproject_masked_depth(secondary.depth_mm, secondary.mask,
        secondary.intrinsics, pixel_stride=3, depth_scale_m=.001)
    return dict(publication_authorized=False, production_gate_passed=False,
        scope='prepared_supplemental_observation_not_integrated',
        source_stream_key=list(secondary.stream_key),
        source_input_evidence_hash=secondary.evidence_hash,
        source_color_timestamp_ns=str(secondary.color_timestamp_ns),
        source_depth_timestamp_ns=str(secondary.depth_timestamp_ns),
        primary_stream_key=list(primary.stream_key),
        primary_input_evidence_hash=primary.evidence_hash,
        calibration_sha256=digest,
        object_from_secondary=object_from_secondary.tolist(),
        points_object_m=_transform_points(points, object_from_secondary))
