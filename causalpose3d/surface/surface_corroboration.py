"""Read original surface state; never publish, filter or estimate geometry.

Producer: CausalObjectSurface.update. State: its _records and RevisableSurfelMap.
The compressed inputs and classification cache below are disposable evidence,
not another pose/map owner. No live factory imports this development adapter.
"""
from time import perf_counter
import zlib

import numpy as np

from .causal_object_surface import _inverse_rigid_pose
from .object_surface_fusion import measured_free_space_margin


def classify(points, depth, mask, intrinsics, pose):
    """1 unknown, 2 agreement, 3 free-space, 4 behind, 5 ambiguous.

    Source exclusion (code 0) is applied separately from actual live ownership.
    These are conditional geometric relations, not material/reliability truth.
    """
    _, _, details = measured_free_space_margin(
        points, depth, intrinsics, pose, minimum_depth_m=.15,
        maximum_depth_m=3., depth_scale_m=.001, return_projection_details=True)
    ids = details['source_point_indices']
    x, y = details['pixels_floor_uv'].T
    samples = details['depth_samples_m']
    delta = samples - details['projected_depth_m'][:, None]
    inside = mask[y, x] & mask[y, x+1] & mask[y+1, x] & mask[y+1, x+1]
    agreement = inside & (np.ptp(samples, axis=1) <= .020) & (np.abs(delta) <= .020).all(axis=1)
    free = delta.min(axis=1) > .030
    behind = delta.max(axis=1) < -.030
    result = np.ones(len(points), np.uint8)
    result[ids] = 5
    result[ids[agreement]] = 2
    result[ids[free]] = 3
    result[ids[behind]] = 4
    return result


class SurfaceCorroborationShadow:
    """As-of adapter bound to one existing owner from its first update.

    Budgets cover compressed payload and retained cache arrays, not total RSS.
    Budget failure latches blocked until close; no stale result is returned.
    """
    def __init__(self, *, maximum_source_bytes=128*1024**2,
                 maximum_cache_bytes=32*1024**2):
        for value in (maximum_source_bytes, maximum_cache_bytes):
            if type(value) is not int or value < 1:
                raise ValueError('positive integer evidence budgets required')
        self.maximum_source_bytes = maximum_source_bytes
        self.maximum_cache_bytes = maximum_cache_bytes
        self._owner = None
        self._gauge = None
        self._index = -1
        self._sources = {}
        self._cache = {}
        self._keys = np.empty((0, 3), np.int64)
        self._points = np.empty((0, 3), np.float64)
        self._blocked = None
        self._closed = False

    def _clear_evidence(self):
        self._sources.clear()
        self._cache.clear()
        self._keys = np.empty((0, 3), np.int64)
        self._points = np.empty((0, 3), np.float64)

    def close(self):
        self._clear_evidence()
        self._closed = True

    def _budget_failure(self, reason):
        self._clear_evidence()
        self._blocked = reason
        return dict(status='blocked_shadow_budget', reason=reason,
                    publication_authorized=False, world_pose_authorized=False)

    def advance(self, owner, frame, *, audit_full=False):
        if self._closed or owner._closed:
            raise RuntimeError('closed shadow or surface')
        if type(audit_full) is not bool:
            raise ValueError('explicit audit boolean required')
        if self._owner is None:
            if owner._last_index != 0:
                raise ValueError('shadow must start at original first update')
        elif owner is not self._owner or owner._gauge_id != self._gauge:
            raise ValueError('shadow cannot cross owner/gauge identity')
        if (owner._last_index != self._index+1
                or owner._snapshot_metadata['input_evidence_hash'] != frame.evidence_hash
                or owner._key != frame.stream_key
                or owner._last_depth_ns != frame.depth_timestamp_ns):
            raise ValueError('current original owner/input chronology required')
        self._owner, self._gauge, self._index = owner, owner._gauge_id, owner._last_index
        if self._blocked:
            return dict(status='blocked_shadow_budget', reason=self._blocked,
                        publication_authorized=False, world_pose_authorized=False)
        started = perf_counter()
        records = owner._records
        retired = set(self._sources)-set(records)
        self._sources = {k: v for k, v in self._sources.items() if k in records}
        identity = f'{owner._last_index:06d}'
        if identity in records:
            depth = frame.depth_mm
            mask = np.asarray(frame.mask, bool)
            encoded_depth = zlib.compress(depth.tobytes(), 1)
            encoded_mask = zlib.compress(np.packbits(mask).tobytes(), 1)
            restored = np.frombuffer(zlib.decompress(encoded_depth), dtype=depth.dtype).reshape(depth.shape)
            restored_mask = np.unpackbits(np.frombuffer(zlib.decompress(encoded_mask), np.uint8), count=mask.size).reshape(mask.shape).astype(bool)
            if not np.array_equal(depth, restored) or not np.array_equal(mask, restored_mask):
                raise RuntimeError('lossless evidence roundtrip failed')
            self._sources[identity] = dict(depth=encoded_depth, mask=encoded_mask,
                shape=depth.shape, dtype=depth.dtype, K=frame.intrinsics.copy(),
                input_hash=frame.evidence_hash)
        source_bytes = sum(len(s['depth'])+len(s['mask'])+s['K'].nbytes for s in self._sources.values())
        if source_bytes > self.maximum_source_bytes:
            return self._budget_failure('compressed_source_payload_budget')
        points, support, cameras = owner._map.snapshot(
            minimum_observation_support=2, maximum_point_count=owner._map.maximum_voxels)
        keys = np.asarray(sorted(k for k, v in owner._map._totals.items() if v[1] >= 2), np.int64).reshape(-1, 3)
        if len(keys) != len(points):
            raise RuntimeError('original full snapshot/key population mismatch')
        views = sorted([k for k in records if records[k]['pose'] is not None
                 and k in owner._map._live and len(owner._map._live[k][1])], key=int)
        cache_bytes = keys.nbytes+points.nbytes+len(views)*(len(points)+128)
        if cache_bytes > self.maximum_cache_bytes:
            return self._budget_failure('classification_array_budget')
        previous = {tuple(k): i for i, k in enumerate(self._keys)}
        old_index = np.array([previous.get(tuple(k), -1) for k in keys], np.int64)
        stable = old_index >= 0
        ids = np.flatnonzero(stable)
        stable[ids] = (points[ids].view(np.uint64) == self._points[old_index[ids]].view(np.uint64)).all(axis=1)
        lookup = {tuple(k): i for i, k in enumerate(keys)}
        matrix = np.empty((len(views), len(points)), np.uint8)
        fresh_cache = {}
        first = np.full(len(points), -1, np.int32)
        direction0 = np.zeros_like(points)
        maximum = np.zeros(len(points))
        reused = recomputed = invalidations = 0
        cached_ms = full_ms = 0.
        for j, key in enumerate(views):
            record = records[key]
            source = self._sources[key]
            if source['input_hash'] != record['metadata']['input_evidence_hash']:
                raise ValueError('retained original source identity changed')
            depth = np.frombuffer(zlib.decompress(source['depth']), dtype=source['dtype']).reshape(source['shape'])
            mask = np.unpackbits(np.frombuffer(zlib.decompress(source['mask']), np.uint8), count=depth.size).reshape(depth.shape).astype(bool)
            version = record['pose_version']
            pose_bytes = record['pose'].tobytes()
            old = self._cache.get(key)
            can_reuse = (old is not None and old[0] == version
                         and old[1] == source['input_hash'] and old[2] == pose_bytes)
            valid = stable if can_reuse else np.zeros(len(points), bool)
            invalidations += int(old is not None and not can_reuse)
            begin = perf_counter()
            raw = np.ones(len(points), np.uint8)
            if valid.any():
                raw[valid] = old[3][old_index[valid]]
            raw[~valid] = classify(points[~valid], depth, mask, source['K'], record['pose'])
            cached_ms += (perf_counter()-begin)*1000
            reused += int(valid.sum())
            recomputed += int((~valid).sum())
            if audit_full:
                begin = perf_counter()
                full = classify(points, depth, mask, source['K'], record['pose'])
                full_ms += (perf_counter()-begin)*1000
                if not np.array_equal(raw, full):
                    raise RuntimeError('cached categories differ from uncached original projection')
            fresh_cache[key] = (version, source['input_hash'], pose_bytes, raw)
            category = raw.copy()
            source_ids = [lookup.get(tuple(k)) for k in owner._map._live[key][1]]
            excluded = [i for i in source_ids if i is not None]
            category[excluded] = 0
            matrix[j] = category
            agree = category == 2
            new = agree & (first < 0)
            direction = _inverse_rigid_pose(record['pose'])[:3, 3]-points
            length = np.linalg.norm(direction, axis=1)
            unit = np.divide(direction, length[:, None], out=np.zeros_like(direction), where=length[:, None] > 0)
            first[new] = j
            direction0[new] = unit[new]
            angle = np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i', unit, direction0), -1, 1)))
            angle[new] = 0.
            maximum[agree] = np.maximum(maximum[agree], angle[agree])
        maximum[first < 0] = np.nan
        count = (matrix == 2).sum(axis=0)
        keep = (count >= 2) & (maximum >= 10.)
        self._cache = fresh_cache
        self._keys, self._points = keys.copy(), points.copy()
        return dict(status='measured_shadow_only', publication_authorized=False,
            world_pose_authorized=False, index=self._index, points_object_m=points,
            support=support, camera_support=cameras, voxel_keys=keys,
            view_ids=views, view_pose_versions=[records[k]['pose_version'] for k in views],
            categories=matrix, agreement=count, angle_from_first_deg=maximum, keep=keep,
            source_payload_bytes=source_bytes, category_cache_bytes=cache_bytes,
            retired_source_count=len(retired), invalidated_view_versions=invalidations,
            reused_point_view_cells=reused, recomputed_point_view_cells=recomputed,
            cached_projection_ms=cached_ms, uncached_projection_ms=full_ms if audit_full else None,
            full_recompute_audited=audit_full, total_shadow_and_audit_ms=(perf_counter()-started)*1000)
