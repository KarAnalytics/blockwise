"""Smoke tests for the BRM estimator and helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from blockwise import BRM, choose_num_blocks, simulate_blockwise_missing


def _toy_regression(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
    })
    y = (
        X["a"] + 2 * X["b"] - X["c"] + 0.5 * X["d"]
        + rng.normal(scale=0.1, size=n)
    ).to_numpy()
    return X, y


def _toy_classification(n: int = 500, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.normal(size=n),
        "d": rng.normal(size=n),
    })
    logits = X["a"] - X["b"] + 0.5 * X["c"]
    y = (logits + rng.normal(scale=0.3, size=n) > 0).astype(int).to_numpy()
    return X, y


def test_simulate_introduces_joint_masking():
    X, _ = _toy_regression(n=1000)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30, noise=0.0
    )
    assert Xm.isna().any().any()
    # Joint masking: many rows have both a and b NaN.
    both_ab = Xm["a"].isna() & Xm["b"].isna()
    assert both_ab.sum() > 0
    # c/d not in the first block, so single-rate NA marginals differ across groups
    assert Xm["a"].isna().sum() > 0
    assert Xm["c"].isna().sum() > 0


def test_simulate_validates_bad_prop_length():
    X, _ = _toy_regression(n=100)
    with pytest.raises(ValueError):
        simulate_blockwise_missing(X, blocks=[["a"], ["b"]], prop_missing=[0.3])


def test_choose_num_blocks_returns_valid_k():
    X, _ = _toy_regression(n=500)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30
    )
    out = choose_num_blocks(Xm, k_max=5)
    assert isinstance(out["n_blocks"], int)
    assert 1 <= out["n_blocks"] <= 5
    assert len(out["missing_curve"]) == 5


def test_brm_fit_predict_lm():
    X, y = _toy_regression(n=600)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30, noise=0.02
    )
    brm = BRM(estimator=LinearRegression(), n_blocks=2).fit(Xm, y)
    assert len(brm.models_) == 2

    preds = brm.predict(Xm.iloc[:100])
    assert preds.shape == (100,)
    assert np.isfinite(preds).all()
    # Correlation with ground truth should be high in a linear-with-noise problem.
    assert np.corrcoef(preds, y[:100])[0, 1] > 0.5


def test_brm_auto_n_blocks():
    X, y = _toy_regression(n=400)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30
    )
    brm = BRM(estimator=LinearRegression()).fit(Xm, y)
    assert brm.n_blocks_ >= 1


def test_brm_tree_learner():
    X, y = _toy_regression(n=400)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"]], prop_missing=0.30
    )
    brm = BRM(estimator=DecisionTreeRegressor(max_depth=5), n_blocks=2).fit(Xm, y)
    preds = brm.predict(Xm)
    assert preds.shape == (len(X),)


def test_brm_classification_predict_proba():
    X, y = _toy_classification(n=600)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30
    )
    brm = BRM(estimator=LogisticRegression(max_iter=500), n_blocks=2).fit(Xm, y)
    proba = brm.predict_proba(Xm.iloc[:50])
    assert proba.shape == (50, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_non_overlapping_variant_fits():
    X, y = _toy_regression(n=400)
    Xm = simulate_blockwise_missing(
        X, blocks=[["a", "b"], ["c", "d"]], prop_missing=0.30
    )
    brm = BRM(
        estimator=LinearRegression(), n_blocks=2, overlap=False
    ).fit(Xm, y)
    assert len(brm.models_) == 2
