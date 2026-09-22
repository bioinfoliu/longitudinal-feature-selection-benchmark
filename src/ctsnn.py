"""Canonical implementations of SNN-FS and trajectory-aware SNN weights."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def _unit_interval(values: np.ndarray, floor: float = 1e-4) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    span = values.max() - values.min()
    if span <= 1e-12:
        return np.ones_like(values)
    return floor + (1.0 - floor) * (values - values.min()) / span


_LEGACY_METHOD_NAMES = {
    "SNN": "SNN-FS",
    "Hybrid": "TrajSNN",
    "w/o Robust": "TrajSNN",
    "STAR": "SNN-FS",
    "w/o Traj": "SNN-FS",
    "CTSNN-R": "TrajSNN",
    "CTSNN": "TrajSNN",
}


def canonical_method_name(method: str) -> str:
    """Map historical result-file keys to the current manuscript names."""
    return _LEGACY_METHOD_NAMES.get(method, method)


def prepare_distance_embedding(X: np.ndarray, robust: bool = False) -> np.ndarray:
    """Create the feature space used to build the neighbour graph.

    ``robust=False`` reproduces the original standardized feature space.
    ``robust=True`` adds a signed log transform and PCA to reduce distance
    concentration in high-dimensional biomarker matrices.
    """
    X = np.asarray(X, dtype=float)
    if robust:
        X = np.sign(X) * np.log1p(np.abs(X))
    X = StandardScaler().fit_transform(X)
    if robust and X.shape[1] > 20:
        n_components = min(30, X.shape[0] - 1, X.shape[1])
        X = PCA(n_components=n_components, random_state=2026).fit_transform(X)
    return X


def snn_weights(embedding: np.ndarray, k: int | None = None, mutual: bool = False) -> np.ndarray:
    """Shared-nearest-neighbour spatial centrality from the original CTSNN."""
    n = len(embedding)
    if n < 3:
        return np.ones(n)
    k = int(np.clip(int(0.3 * n) if k is None else k, 2, min(70, n - 1)))
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)
    adjacency = np.zeros((n, n), dtype=np.float32)
    for row, neighbours in enumerate(indices):
        neighbours = neighbours[neighbours != row][:k]
        adjacency[row, neighbours] = 1.0
    if mutual:
        adjacency *= adjacency.T
    shared = adjacency @ adjacency.T
    np.fill_diagonal(shared, 0.0)
    local_shared = shared * adjacency
    return _unit_interval(local_shared.sum(axis=1) / max(k, 1))


def temporal_proximity_weights(
    embedding: np.ndarray,
    times: np.ndarray,
    k: int | None = None,
    lambda_time: float = 0.25,
) -> np.ndarray:
    """Temporal proximity among molecular neighbours, without subject IDs.

    Visit times are normalized by the training-fold range, and the fixed
    ``lambda_time`` controls the decay of support from temporally distant
    neighbours. This is the trajectory component of TrajSNN.
    """
    times = np.asarray(times, dtype=float)
    n = len(embedding)
    if n < 3:
        return np.ones(n)
    if k is None:
        k = int(0.3 * n)
    k = int(np.clip(k, 2, n - 1))
    indices = NearestNeighbors(n_neighbors=k + 1).fit(embedding).kneighbors(return_distance=False)
    span = float(np.nanmax(times) - np.nanmin(times))
    if not np.isfinite(span) or span <= 1e-12:
        return np.ones(n)
    normalized = (times - np.nanmin(times)) / span
    scores = np.zeros(n, dtype=float)
    for row, neighbours in enumerate(indices):
        neighbours = neighbours[neighbours != row][:k]
        delta_t = np.abs(normalized[neighbours] - normalized[row])
        scores[row] = np.mean(np.exp(-delta_t / lambda_time))
    return _unit_interval(scores)


def ctsnn_weights(
    X: np.ndarray,
    ids: np.ndarray,
    method: str,
    times: np.ndarray | None = None,
    k_snn: int | None = None,
) -> np.ndarray:
    """Return manuscript-facing SNN-FS or trajectory-aware SNN weights.

    ``ids`` is retained for API compatibility with historical notebooks but is
    intentionally not used. Historical method keys are accepted as aliases.
    """
    method = canonical_method_name(method)
    if method == "Equal":
        return np.ones(len(X))
    embedding = prepare_distance_embedding(X, robust=False)
    spatial = snn_weights(embedding, k_snn, mutual=False)
    if method == "SNN-FS":
        return spatial
    if method == "TrajSNN":
        # Legacy direct calls may not provide times; in that case retain the
        # spatial SNN weight rather than reintroducing subject consistency.
        temporal = np.ones(len(X)) if times is None else temporal_proximity_weights(embedding, times)
        return _unit_interval(spatial * temporal)
    raise ValueError(f"Unknown CTSNN method: {method}")
