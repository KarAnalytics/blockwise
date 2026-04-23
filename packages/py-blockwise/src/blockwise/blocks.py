"""Block-count selection heuristic (elbow on the missing-pattern curve)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def _best_kmeans(M: np.ndarray, k: int, n_restarts: int = 10) -> KMeans:
    """Best of `n_restarts` k-means fits on the binary missingness matrix."""
    if k == 1:
        km = KMeans(n_clusters=1, n_init=1, random_state=0).fit(M)
        return km
    runs = [
        KMeans(n_clusters=k, n_init=1, random_state=s).fit(M)
        for s in range(n_restarts)
    ]
    # Paper's score: 100 - tot.withinss/totss * 100 — equivalent to smallest inertia.
    return min(runs, key=lambda r: r.inertia_)


def _subset_missing_frac(
    M: np.ndarray, k: int, low_threshold: float, n_restarts: int
) -> float:
    km = _best_kmeans(M, k, n_restarts)
    n, p = M.shape
    total_missing = 0
    for c in np.unique(km.labels_):
        idx = km.labels_ == c
        part = M[idx]
        completeness = part.sum(axis=0)
        keep = completeness > low_threshold * part.shape[0]
        if not keep.any():
            continue
        kept = part[:, keep]
        total_missing += (kept.shape[0] - kept.sum(axis=0)).sum()
    return total_missing / (n * p)


def _elbow_point(y: np.ndarray) -> int:
    """Return 1-indexed k at the elbow (max distance from first-last line)."""
    x = np.arange(1, len(y) + 1)
    if len(x) <= 2:
        return 1
    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    if den == 0:
        return 1
    return int(np.argmax(num / den)) + 1


def choose_num_blocks(
    X: pd.DataFrame,
    low_threshold: float = 0.05,
    n_restarts: int = 10,
    k_max: Optional[int] = None,
    random_state: int = 12345,
) -> dict:
    """Elbow heuristic for the number of blocks.

    Parameters
    ----------
    X : pandas.DataFrame
        Predictor frame; may contain ``NaN``.
    low_threshold : float
        Column-density threshold: in a candidate cluster, columns observed
        in fewer than ``low_threshold * n_rows`` rows are treated as absent.
    n_restarts : int
        k-means restarts per candidate k.
    k_max : int or None
        Upper bound on k (default ``min(n_features, 50)``).
    random_state : int
        Seed used when subsampling very large frames.

    Returns
    -------
    dict
        ``{"n_blocks": int, "missing_curve": np.ndarray}``.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if len(X) > 50_000:
        rng = np.random.default_rng(random_state)
        X = X.iloc[rng.choice(len(X), 50_000, replace=False)]
    M = (~X.isna()).to_numpy().astype(np.int8)
    if k_max is None:
        k_max = min(X.shape[1], 50)
    curve = np.array([
        _subset_missing_frac(M, k, low_threshold, n_restarts)
        for k in range(1, k_max + 1)
    ])
    return {"n_blocks": _elbow_point(curve), "missing_curve": curve}
