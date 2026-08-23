"""Small content-addressed caches for expensive pooled heatmaps."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


CACHE_VERSION = 1


def _event_signature(events) -> list[dict[str, object]]:
    signature = []
    for event in events:
        states = {}
        for name in ("state0_min_gamma", "state2_relaxed_gamma_plus"):
            path = Path(getattr(event.state_paths, name))
            stat = path.stat()
            states[name] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        signature.append(
            {
                "rank": int(event.rank),
                "load_step": int(event.load_step),
                "energy_drop": float(event.energy_drop),
                "states": states,
            }
        )
    return signature


def heatmap_cache_path(
    cache_root: Path,
    prefix: str,
    events,
    *,
    parameters: dict[str, object],
) -> Path:
    """Return a readable content-addressed cache filename."""

    payload = {
        "cache_version": CACHE_VERSION,
        "events": _event_signature(events),
        "parameters": parameters,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha1(encoded).hexdigest()[:16]
    return Path(cache_root) / f"{prefix}_heatmap_v{CACHE_VERSION}_{digest}.npz"


def load_heatmap_cache(path: Path) -> tuple[np.ndarray, ...] | None:
    """Load one cache, returning ``None`` only when it does not exist."""

    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        required = ("angle", "magnitude", "angle_std", "magnitude_std", "count", "edges")
        missing = [name for name in required if name not in cached]
        if missing:
            raise ValueError(f"Heatmap cache {path} is missing {missing}.")
        return tuple(np.asarray(cached[name]) for name in required)


def save_heatmap_cache(
    path: Path,
    metrics: tuple[np.ndarray, ...],
    count: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Write a cache atomically so interrupted plotting cannot poison it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    angle, magnitude, angle_std, magnitude_std = metrics
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        angle=angle,
        magnitude=magnitude,
        angle_std=angle_std,
        magnitude_std=magnitude_std,
        count=count,
        edges=edges,
    )
    os.replace(temporary, path)
