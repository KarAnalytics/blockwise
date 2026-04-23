# blockwise — R and Python packages

Blockwise Reduced Modeling (BRM) for supervised learning on tabular data
with **blockwise missing patterns**, packaged as:

- [`r-blockwise/`](r-blockwise/) — R package (S3 class API)
- [`py-blockwise/`](py-blockwise/) — Python package (scikit-learn-compatible estimator)

Both packages implement the method introduced in:

> **Srinivasan, K., Currim, F., and Ram, S. (2025).** *A Reduced Modeling
> Approach for Making Predictions With Incomplete Data Having Blockwise
> Missing Patterns.* **INFORMS Journal on Data Science**.

The reproducible Code Ocean capsule for the paper lives at:
<https://codeocean.com/capsule/0274716/tree/v3>.

## What BRM does

BRM partitions the training data into **overlapping subsets** based on
per-row feature-missing patterns (via k-means on the binary missingness
matrix), pre-trains one model per subset on only the observed columns of
that subset, and at prediction time routes each test row to the subset
model whose missingness pattern most closely matches. This keeps all the
training rows, avoids aggressive imputation, and is superior to
listwise-deletion and single-model imputation baselines on typical
blockwise-missing workloads.

## Design decisions that apply to both packages

| Choice              | R (`blockwise`)                                        | Python (`blockwise`)                                            |
|---------------------|--------------------------------------------------------|------------------------------------------------------------------|
| API style           | S3: `brm()` returns `"brm"` object; `predict.brm()`     | sklearn `BaseEstimator`: `BRM().fit(X, y).predict(X)`            |
| Learner interface   | `learner(fit, predict, type)` — any fit/predict pair   | `estimator=` — any sklearn-compatible estimator                  |
| Convenience learners| `learner_lm`, `learner_rpart`, `learner_ranger`, `learner_gbm`, `learner_glm_binomial` | any sklearn estimator; examples use `LinearRegression`, `LogisticRegression`, `GradientBoostingRegressor`/`Classifier`, `DecisionTreeRegressor` |
| Block-count heuristic | `choose_num_blocks()` (elbow on missing-pattern curve) | `choose_num_blocks()`                                          |
| Missingness simulator | `simulate_blockwise_missing()`                        | `simulate_blockwise_missing()`                                  |
| Bundled datasets    | full `bike`, full `adult`, full `house` (xz-compressed .rda, ~0.5 MB total) | full `bike`, full `adult`, full `house`                          |

## Datasets and examples

Both packages ship the three datasets used as benchmarks in the paper, plus
worked examples using each:

| Dataset | Task                   | R vignette            | Python notebook       |
|---------|------------------------|-----------------------|-----------------------|
| bike    | regression (count)     | `vignette("bike")`     | `notebooks/bike.ipynb` |
| adult   | binary classification  | `vignette("adult")`    | `notebooks/adult.ipynb`|
| house   | regression             | `vignette("house")`    | `notebooks/house.ipynb`|

Both packages ship the full datasets. In the R package they are stored as
xz-compressed `.rda` files (~0.5 MB total — comfortably under CRAN's 5 MB
package-size limit); see
[`r-blockwise/data-raw/make_data.R`](r-blockwise/data-raw/make_data.R) for
the preparation script.

## Citation

`citation("blockwise")` in R, or `CITATION.cff` in the Python package root.
Please cite the INFORMS JDS paper if you use the method in published work.

## License

GPL-3.0-or-later (matches the Code Ocean capsule). See `LICENSE` in either
package root.
