"""Binary classification with BRM on the bundled UCI Adult income dataset.

Run from the package directory:

    python examples/adult_classification.py

Demonstrates BRM's `predict_proba` with a logistic-regression learner on
data that has a blockwise missing pattern.
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from blockwise import BRM, datasets, simulate_blockwise_missing


def main() -> None:
    adult = datasets.load_adult()
    cols = ["age", "education-num", "hours-per-week"]
    adult = adult[cols + ["salary"]].dropna()

    train, test = train_test_split(
        adult, test_size=0.25, stratify=adult["salary"], random_state=0
    )

    blocks = [["age"], ["education-num", "hours-per-week"]]
    train_miss = simulate_blockwise_missing(
        train, blocks=blocks, prop_missing=0.25, noise=0.05, seed=42
    )
    test_miss = simulate_blockwise_missing(
        test, blocks=blocks, prop_missing=0.25, noise=0.05, seed=43
    )

    X_train = train_miss.drop(columns="salary")
    y_train = train_miss["salary"].to_numpy()
    X_test = test_miss.drop(columns="salary")
    y_test = test["salary"].to_numpy()

    brm = BRM(estimator=LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    y_pred = brm.predict(X_test).astype(int)
    proba = brm.predict_proba(X_test)[:, 1]

    print(f"n_blocks chosen by BRM : {brm.n_blocks_}")
    print(f"Accuracy               : {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC AUC                : {roc_auc_score(y_test, proba):.3f}")


if __name__ == "__main__":
    main()
