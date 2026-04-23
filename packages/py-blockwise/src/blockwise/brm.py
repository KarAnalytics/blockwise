"""The BRM estimator — scikit-learn-compatible."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression

from .blocks import choose_num_blocks
from .impute import hotdeck_impute, impute_with_train
from .subsets import build_subsets


class BRM(BaseEstimator):
    """Blockwise Reduced Modeling estimator.

    BRM partitions the training data into ``n_blocks`` subsets based on
    per-row patterns of feature-missingness, fits one fresh clone of
    ``estimator`` per subset using only the features observed within that
    subset, and at prediction time routes each row of ``X`` to the subset
    model whose training-time missingness center is closest.

    Parameters
    ----------
    estimator : sklearn-style estimator, default=None
        Any object with ``fit(X, y)`` and ``predict(X)``. ``predict_proba(X)``
        is used when available for classification. Cloned per block.
        If None, :class:`sklearn.linear_model.LinearRegression` is used.
    n_blocks : int, optional
        Number of blocks. If ``None``, chosen by :func:`choose_num_blocks`.
    low_threshold : float, default=0.05
        Column-density threshold for including a predictor in a block's model.
    n_restarts : int, default=5
        k-means restarts for block assignment.
    overlap : bool, default=True
        Whether to enlarge each subset using the set-theoretic inclusion
        rule from the paper.

    Attributes
    ----------
    models_ : list
        One fitted estimator per block.
    centers_ : ndarray of shape (n_blocks, n_features)
        Rounded 0/1 missingness centers used to route test rows.
    columns_ : list of list of str
        Per-block feature-column names.
    feature_names_ : list of str
    n_blocks_ : int

    References
    ----------
    Srinivasan, K., Currim, F., Ram, S. (2025). A Reduced Modeling Approach
    for Making Predictions With Incomplete Data Having Blockwise Missing
    Patterns. *INFORMS Journal on Data Science*.
    """

    def __init__(
        self,
        estimator=None,
        n_blocks: Optional[int] = None,
        low_threshold: float = 0.05,
        n_restarts: int = 5,
        overlap: bool = True,
    ):
        self.estimator = estimator
        self.n_blocks = n_blocks
        self.low_threshold = low_threshold
        self.n_restarts = n_restarts
        self.overlap = overlap

    # ---------------------------------------------------------------- fit --
    def fit(self, X, y):
        X = self._as_frame(X)
        y = pd.Series(np.asarray(y), name="__y__")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        if self.n_blocks is None:
            self.n_blocks_ = choose_num_blocks(
                X, low_threshold=self.low_threshold
            )["n_blocks"]
        else:
            self.n_blocks_ = int(self.n_blocks)

        est = self.estimator if self.estimator is not None else LinearRegression()

        subs = build_subsets(
            X, y, self.n_blocks_,
            low_threshold=self.low_threshold,
            n_restarts=self.n_restarts,
        )
        parts = subs["overlapping"] if self.overlap else subs["non_overlapping"]

        self.models_: list = []
        self.columns_: list[list[str]] = []
        for part in parts:
            cols_x = [c for c in part.columns if c != "__y__"]
            self.columns_.append(cols_x)
            if len(part) == 0 or not cols_x:
                self.models_.append(None)
                continue
            m = clone(est)
            m.fit(part[cols_x], part["__y__"].to_numpy())
            self.models_.append(m)

        self.centers_ = subs["centers"]
        self.feature_names_ = list(X.columns)
        self.train_ref_ = X
        return self

    # ------------------------------------------------------------ predict --
    def predict(self, X) -> np.ndarray:
        X = self._align(X)
        nearest = self._route(X)

        preds = np.empty(len(X), dtype=float)
        preds[:] = np.nan
        for b, model in enumerate(self.models_):
            idx = np.flatnonzero(nearest == b)
            if idx.size == 0 or model is None:
                continue
            cols_b = self.columns_[b]
            X_b = X.iloc[idx][cols_b].copy()
            X_b = impute_with_train(X_b, self.train_ref_[cols_b])
            preds[idx] = np.asarray(model.predict(X_b)).ravel()
        return preds

    def predict_proba(self, X) -> np.ndarray:
        X = self._align(X)
        nearest = self._route(X)

        first = next((m for m in self.models_ if m is not None), None)
        if first is None or not hasattr(first, "predict_proba"):
            raise AttributeError(
                "Underlying estimator does not implement predict_proba."
            )

        # Determine n_classes from the first available model.
        sample_cols = self.columns_[self.models_.index(first)]
        sample_X = (
            self.train_ref_[sample_cols].iloc[:1].pipe(hotdeck_impute)
        )
        n_classes = first.predict_proba(sample_X).shape[1]

        proba = np.zeros((len(X), n_classes))
        for b, model in enumerate(self.models_):
            idx = np.flatnonzero(nearest == b)
            if idx.size == 0 or model is None:
                continue
            cols_b = self.columns_[b]
            X_b = X.iloc[idx][cols_b].copy()
            X_b = impute_with_train(X_b, self.train_ref_[cols_b])
            proba[idx] = model.predict_proba(X_b)
        return proba

    # ------------------------------------------------------------- helpers --
    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _align(self, X) -> pd.DataFrame:
        X = self._as_frame(X)
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = np.nan
        return X[self.feature_names_]

    def _route(self, X: pd.DataFrame) -> np.ndarray:
        M = (~X.isna()).astype(np.int8).to_numpy()
        # Squared Euclidean distance to each center.
        dists = ((self.centers_[None, :, :] - M[:, None, :]) ** 2).sum(axis=2)
        return dists.argmin(axis=1)
