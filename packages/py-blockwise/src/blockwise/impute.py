"""Simple per-column imputation used inside BRM subsets.

At training time, each overlapping subset is filled with mean (numeric) or
mode (categorical / object) values. At predict time, any NA remaining in a
block's feature columns is filled against the training reference.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _is_categorical_like(s: pd.Series) -> bool:
    return s.dtype.kind in "bOSU" or isinstance(s.dtype, pd.CategoricalDtype)


def hotdeck_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise imputation: mode for categorical, mean for numeric.

    This is a lightweight stand-in for VIM::hotdeck from the R reference
    implementation. It matches its behavior closely enough on small,
    homogeneous per-subset training frames.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if not s.isna().any():
            continue
        if _is_categorical_like(s):
            mode = s.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
        else:
            fill = s.mean()
        out[col] = s.fillna(fill)
    return out


def impute_with_train(
    test: pd.DataFrame, train: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Fill NAs in `test` using column statistics from `train` (falls back to
    `test` itself if no training reference is supplied).
    """
    ref = train if train is not None else test
    out = test.copy()
    for col in out.columns:
        if not out[col].isna().any():
            continue
        if col in ref.columns:
            ref_col = ref[col]
        else:
            ref_col = out[col]
        if _is_categorical_like(ref_col):
            mode = ref_col.mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else ""
        else:
            fill = ref_col.mean()
        out[col] = out[col].fillna(fill)
    return out
