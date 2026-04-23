# blockwise

Blockwise Reduced Modeling (BRM) for tabular data with **blockwise missing
patterns** — a scikit-learn-compatible estimator.

## Install

```bash
pip install blockwise
```

## Quickstart

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from blockwise import BRM, simulate_blockwise_missing, datasets

bike = datasets.load_bike()
bike_miss = simulate_blockwise_missing(
    bike,
    blocks=[["windspeed", "hum", "weekday"],
            ["hr", "temp", "weathersit"]],
    prop_missing=0.30,
    noise=0.05,
)

X = bike_miss.drop(columns=["cnt"])
y = bike_miss["cnt"]

brm = BRM(estimator=GradientBoostingRegressor()).fit(X, y)
y_hat = brm.predict(X)
```

`BRM` is learner-agnostic: pass any estimator with `fit(X, y)` / `predict(X)`
(and optionally `predict_proba(X)` for classification). Each block's model is
a fresh clone of `estimator`.

## What BRM does

BRM partitions the training data into overlapping subsets based on per-row
feature-missing patterns, pre-trains one model per subset on only the
observed columns of that subset, and at prediction time routes each test row
to the subset model whose missingness pattern most closely matches.

See [`notebooks/`](notebooks/) for worked examples on the **bike** (regression),
**adult** (binary classification), and **house** (regression) datasets.

## Citation

If you use this package, please cite the paper that introduced the method:

> Srinivasan, K., Currim, F., and Ram, S. (2025). *A Reduced Modeling Approach
> for Making Predictions With Incomplete Data Having Blockwise Missing
> Patterns.* INFORMS Journal on Data Science.

A machine-readable `CITATION.cff` is included at the repo root.

## License

GPL-3.0-or-later. See `LICENSE`.
