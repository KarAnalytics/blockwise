# blockwise 0.1.0

First public release. Initial CRAN submission.

## Features

* `brm()` — fit a Blockwise Reduced Modeling ensemble (S3 class `"brm"`).
* `predict.brm()` — route test instances to their best-matching subset model.
* `choose_num_blocks()` — elbow heuristic for the number of blocks.
* `learner()` — learner-agnostic fit/predict specification; convenience
  builders for linear models (`learner_lm`, `learner_glm_binomial`),
  trees (`learner_rpart`), random forests (`learner_ranger`), and
  gradient boosting (`learner_gbm`).
* `simulate_blockwise_missing()` — mask complete data with a blockwise
  missing pattern for benchmarking.
* Bundled datasets: `bike`, `adult`, `house` — the three benchmark
  datasets used in Srinivasan, Currim, and Ram (2025)
  <doi:10.1287/ijds.2022.9016>.
