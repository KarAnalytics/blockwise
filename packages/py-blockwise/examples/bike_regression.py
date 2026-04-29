"""End-to-end BRM regression on the bundled Capital Bikeshare dataset.

Run from the package directory:

    python examples/bike_regression.py

Workflow:
    1. Load the bundled `bike` dataset (numeric features only).
    2. Split into train and test BEFORE injecting missingness so the two
       sets have independent missing patterns (matches realistic deployment).
    3. Inject a blockwise missing pattern with `simulate_blockwise_missing`.
    4. Fit BRM with a GradientBoostingRegressor learner.
    5. Predict on the test set and report RMSE / R^2.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from blockwise import BRM, datasets, simulate_blockwise_missing


def main() -> None:
    bike = datasets.load_bike()
    bike = bike[["mnth", "hr", "temp", "hum", "windspeed", "cnt"]].dropna()

    train, test = train_test_split(bike, test_size=0.25, random_state=0)

    blocks = [["windspeed", "hum"], ["hr", "temp"]]
    train_miss = simulate_blockwise_missing(
        train, blocks=blocks, prop_missing=0.30, noise=0.05, seed=42
    )
    test_miss = simulate_blockwise_missing(
        test, blocks=blocks, prop_missing=0.30, noise=0.05, seed=43
    )

    X_train = train_miss.drop(columns="cnt")
    y_train = train_miss["cnt"].to_numpy()
    X_test = test_miss.drop(columns="cnt")
    y_test = test["cnt"].to_numpy()  # original (un-masked) y for evaluation

    brm = BRM(estimator=GradientBoostingRegressor(random_state=0)).fit(X_train, y_train)
    y_pred = brm.predict(X_test)

    print(f"n_blocks chosen by BRM : {brm.n_blocks_}")
    print(f"RMSE                   : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R^2                    : {r2_score(y_test, y_pred):.3f}")


if __name__ == "__main__":
    main()
