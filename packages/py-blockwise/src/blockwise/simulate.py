"""Simulate a blockwise missing pattern on complete data."""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import pandas as pd

PropMissing = Union[float, Sequence[float]]


def simulate_blockwise_missing(
    data: pd.DataFrame,
    blocks: Sequence[Sequence[str]],
    prop_missing: PropMissing,
    noise: float = 0.05,
    seed: int = 1234,
) -> pd.DataFrame:
    """Joint-mask column groups on random subsets of rows.

    Parameters
    ----------
    data : DataFrame
        The complete dataset to mask.
    blocks : sequence of sequences of str
        Each element is a set of column names to mask *jointly* on one
        independent random sample of rows.
    prop_missing : float or sequence of floats
        Row fraction masked per block. Scalar (applied to each block) or
        length-``len(blocks)`` vector.
    noise : float, default=0.05
        Extra per-column random-NA rate, applied only to columns named
        in any block. Set to 0 to disable.
    seed : int, default=1234
        Base seed for reproducibility.

    Returns
    -------
    DataFrame
        Copy of ``data`` with ``NaN`` introduced in the specified pattern.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("`data` must be a pandas DataFrame.")
    if not isinstance(blocks, (list, tuple)) or len(blocks) == 0:
        raise ValueError("`blocks` must be a non-empty sequence of column lists.")

    if np.isscalar(prop_missing):
        prop_missing = [float(prop_missing)] * len(blocks)
    if len(prop_missing) != len(blocks):
        raise ValueError("`prop_missing` must be scalar or length-`len(blocks)`.")

    out = data.copy()
    n = len(out)

    for i, cols in enumerate(blocks):
        cols_i = [c for c in cols if c in out.columns]
        if not cols_i:
            continue
        rng = np.random.default_rng(seed + i)
        k = int(prop_missing[i] * n)
        if k <= 0:
            continue
        rows = rng.choice(n, size=k, replace=False)
        out.loc[out.index[rows], cols_i] = np.nan

    if noise > 0:
        affected = sorted({c for b in blocks for c in b if c in out.columns})
        for k_idx, col in enumerate(affected):
            rng = np.random.default_rng(seed + len(blocks) + k_idx)
            k = int(noise * n)
            if k <= 0:
                continue
            rows = rng.choice(n, size=k, replace=False)
            out.loc[out.index[rows], col] = np.nan

    return out
