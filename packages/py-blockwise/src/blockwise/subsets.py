"""Overlapping-subsets construction used by BRM training."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .blocks import _best_kmeans
from .impute import hotdeck_impute


def build_subsets(
    X: pd.DataFrame,
    y: pd.Series,
    n_blocks: int,
    low_threshold: float = 0.05,
    n_restarts: int = 5,
) -> dict:
    """Cluster rows by missingness pattern, then build overlapping training subsets.

    Returns
    -------
    dict with keys:
        * ``overlapping``     - list of imputed DataFrames (one per block)
        * ``non_overlapping`` - same, without the set-theoretic merge step
        * ``columns``         - list of column-name lists (per block, incl. "__y__")
        * ``centers``         - np.ndarray of rounded 0/1 missingness centers
                                (shape: n_blocks x n_features)
    """
    M_df = (~X.isna()).astype(np.int8)
    M = M_df.to_numpy()
    km = _best_kmeans(M, n_blocks, n_restarts)
    centers = np.round(km.cluster_centers_).astype(int)

    data_xy = X.copy()
    data_xy["__y__"] = y.to_numpy()

    non_overlapping: list[pd.DataFrame] = []
    columns: list[list[str]] = []
    for c in range(n_blocks):
        idx = km.labels_ == c
        if not idx.any():
            non_overlapping.append(data_xy.iloc[[]][["__y__"]])
            columns.append(["__y__"])
            continue
        miss_c = M_df.iloc[idx]
        comp = miss_c.sum(axis=0)
        keep = comp[comp > low_threshold * miss_c.shape[0]].index.tolist()
        cols_c = keep + ["__y__"]
        sub = data_xy.iloc[idx][cols_c].copy()
        non_overlapping.append(sub)
        columns.append(cols_c)

    overlapping: list[pd.DataFrame] = []
    for i, sub_i in enumerate(non_overlapping):
        cols_i = set(columns[i])
        merged = sub_i.copy()
        for j, sub_j in enumerate(non_overlapping):
            if i == j:
                continue
            if cols_i.issubset(set(columns[j])):
                merged = pd.concat(
                    [merged, sub_j[list(columns[i])]], axis=0, ignore_index=True
                )
        overlapping.append(hotdeck_impute(merged))

    non_overlapping = [hotdeck_impute(s) for s in non_overlapping]

    return {
        "overlapping": overlapping,
        "non_overlapping": non_overlapping,
        "columns": columns,
        "centers": centers,
    }
