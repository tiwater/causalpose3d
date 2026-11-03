"""Opt-in frame-scoped reuse inside one existing shadow BootsTAPIR runner.

The original runner still owns preprocessing, query features and recurrent state.
This adapter shares only a pure image-feature call, for identical normalized
inputs on a frozen eval model. It has no publication or pose capability.
"""
from contextlib import contextmanager
from threading import get_ident


@contextmanager
def shared_bootstapir_frame_features(runner):
    """Reuse image grids within one serial frame; release them on every exit.

Caller must finish all providers for this frame inside the scope. Query batches
and states remain separate. Cache-hit call latency excludes the first call's
feature work; measure the entire scope for a frame latency comparison.
"""
    model, torch = runner._model, runner._torch
    sentinel = "_ticos_shared_frame_feature_scope"
    if sentinel in model.__dict__:
        raise RuntimeError("nested shared-frame feature scopes are unsupported")
    if model.training:
        raise ValueError("shared image features require an eval model")
    signature = tuple((id(p), p._version) for p in model.parameters())
    original = model.get_feature_grids
    had_override = "get_feature_grids" in model.__dict__
    prior_override = model.__dict__.get("get_feature_grids")
    thread = get_ident()
    cache = {}
    stats = dict(calls=0, computed=0, reused=0, closed=False)

    def feature_grids(frame, *, is_training=False, **kwargs):
        if get_ident() != thread:
            raise RuntimeError("shared-frame features require serial owner-thread calls")
        if model.training or is_training is not False or kwargs:
            raise ValueError("unsupported shared image-feature context")
        if tuple((id(p), p._version) for p in model.parameters()) != signature:
            raise RuntimeError("model weights changed inside the frame scope")
        stats["calls"] += 1
        if cache:
            saved = cache["frame"]
            if (frame.shape != saved.shape or frame.dtype != saved.dtype
                    or frame.device != saved.device or not torch.equal(frame, saved)):
                raise ValueError("shared image-feature scope received a different frame")
            stats["reused"] += 1
            return cache["grids"]
        grids = original(frame, is_training=False)
        # Snapshot the input so later caller mutation cannot validate a stale grid.
        with torch.inference_mode():
            cache.update(frame=frame.clone(), grids=grids)
        stats["computed"] += 1
        return grids

    setattr(model, sentinel, True)
    model.get_feature_grids = feature_grids
    try:
        yield stats
    finally:
        cache.clear()
        if had_override:
            model.get_feature_grids = prior_override
        else:
            delattr(model, "get_feature_grids")
        delattr(model, sentinel)
        stats["closed"] = True
