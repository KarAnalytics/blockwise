"""BRM regression on the bundled housing dataset.

Run from the package directory:

    python examples/house_regression.py

Uses a Random Forest learner inside BRM and predicts log-price (a more
forgiving target than raw price for tree-based models on this data).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from blockwise import BRM, datasets, simulate_blockwise_missing


def main() -> None:
    # Subsample for speed; remove n_sample to use the full dataset.
    house = datasets.load_house(n_sample=5000, seed=0)
    feature_cols = [
        "bedrooms", "bathrooms", "sqft_living",
        "floors", "grade", "sqft_above", "yr_built",
    ]
    house = house[feature_cols + ["price"]].dropna().copy()
    house["log_price"] = np.log(house["price"])

    train, test = train_test_split(house, test_size=0.25, random_state=0)

    blocks = [
        ["bedrooms", "bathrooms", "sqft_living"],   # interior size
        ["floors", "grade", "sqft_above", "yr_built"],  # structure / quality
    ]
    train_miss = simulate_blockwise_missing(
        train, blocks=blocks, prop_missing=0.30, noise=0.05, seed=42
    )
    test_miss = simulate_blockwise_missing(
        test, blocks=blocks, prop_missing=0.30, noise=0.05, seed=43
    )

    X_train = train_miss[feature_cols]
    y_train = train_miss["log_price"].to_numpy()
    X_test = test_miss[feature_cols]
    y_test_log = test["log_price"].to_numpy()

    brm = BRM(
        estimator=RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    ).fit(X_train, y_train)
    y_pred_log = brm.predict(X_test)

    # Report on the original price scale for interpretability.
    y_test = np.exp(y_test_log)
    y_pred = np.exp(y_pred_log)

    print(f"n_blocks chosen by BRM : {brm.n_blocks_}")
    print(f"MAE (USD)              : {mean_absolute_error(y_test, y_pred):,.0f}")
    print(f"R^2 (log scale)        : {r2_score(y_test_log, y_pred_log):.3f}")


if __name__ == "__main__":
    main()
