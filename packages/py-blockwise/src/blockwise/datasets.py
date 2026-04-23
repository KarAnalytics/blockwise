"""Lazy loaders for the three example datasets bundled with the package.

Datasets are shipped as CSVs in ``blockwise/data/``. Each loader returns a
``pandas.DataFrame`` with the same structure used in Srinivasan, Currim, and
Ram (2025).
"""
from __future__ import annotations

from importlib.resources import files
from typing import Optional

import pandas as pd


def _load(name: str) -> pd.DataFrame:
    path = files(__package__).joinpath("data").joinpath(f"{name}.csv")
    with path.open("rb") as f:
        return pd.read_csv(f)


def load_bike(preprocess: bool = True) -> pd.DataFrame:
    """Capital Bikeshare hourly demand (~17,380 rows).

    Parameters
    ----------
    preprocess : bool, default=True
        If True, drop bookkeeping columns (``instant``, ``dteday``, etc.) and
        cast calendar/weather codes to ``category`` to match the paper's setup.

    Response column: ``cnt``.
    """
    df = _load("bike")
    if preprocess:
        drop = [c for c in
                ["instant", "dteday", "atemp", "yr", "workingday", "holiday"]
                if c in df.columns]
        df = df.drop(columns=drop)
        if "weathersit" in df.columns:
            df.loc[df["weathersit"] == 4, "weathersit"] = 3
            df["weathersit"] = df["weathersit"].astype("category")
        for col in ("season", "weekday"):
            if col in df.columns:
                df[col] = df[col].astype("category")
    return df


def load_adult(preprocess: bool = True) -> pd.DataFrame:
    """UCI Adult income classification (~32,561 rows).

    Parameters
    ----------
    preprocess : bool, default=True
        If True, binary-encode ``salary`` to 0/1 and drop high-cardinality
        / low-signal columns (``fnlwgt``, ``capital-gain``, ``capital-loss``).

    Response column: ``salary``.
    """
    df = _load("adult")
    if preprocess:
        if "salary" in df.columns:
            df["salary"] = (
                df["salary"].astype(str).str.strip().isin([">50K", ">=50k", ">50K."])
            ).astype(int)
        for c in ("fnlwgt", "capital-gain", "capital-loss",
                  "capital.gain", "capital.loss"):
            if c in df.columns:
                df = df.drop(columns=c)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype("category")
    return df


def load_house(n_sample: Optional[int] = None, seed: int = 1234) -> pd.DataFrame:
    """King County house sales (~21,598 rows).

    Parameters
    ----------
    n_sample : int or None
        If given, return a random-sampled subset of this many rows
        (useful for quick demos).
    seed : int
        Seed for the optional sub-sample.

    Response column: ``price``.
    """
    df = _load("house")
    if n_sample is not None and n_sample < len(df):
        df = df.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    return df


__all__ = ["load_bike", "load_adult", "load_house"]
