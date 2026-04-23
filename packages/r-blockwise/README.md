# blockwise (R)

Blockwise Reduced Modeling (BRM) for tabular data with **blockwise missing
patterns** — implements Srinivasan, Currim, and Ram (2025) as an R package
with a learner-agnostic, S3 class-based API.

## Install (development version)

```r
# install.packages("devtools")
devtools::install_github("KarAnalytics/blockwise", subdir = "packages/r-blockwise")
```

## Build the bundled datasets (one-time, for development)

The raw CSVs live in `../../data/`. Convert them to compressed `.rda` files
that ship with the package:

```r
setwd("packages/r-blockwise")
source("data-raw/make_data.R")
# Then regenerate docs + install:
devtools::document()
devtools::install()
```

## Quickstart

```r
library(blockwise)
data(bike)

# Induce a blockwise missing pattern
bike_miss <- simulate_blockwise_missing(
  bike,
  blocks       = list(c("windspeed", "hum", "weekday"),
                      c("hr", "temp", "weathersit")),
  prop_missing = 0.30,
  noise        = 0.05
)

X <- bike_miss[, setdiff(names(bike_miss), "cnt")]
y <- bike_miss$cnt

fit   <- brm(X, y, learner = learner_lm())       # or learner_gbm(), learner_rpart()
preds <- predict(fit, X)
```

Any model family plugs in via `learner(fit = <fn>, predict = <fn>, type = ...)`.
Convenience builders: `learner_lm`, `learner_glm_binomial`, `learner_rpart`,
`learner_ranger`, `learner_gbm`.

## Vignettes

- `vignette("bike")` — regression, full dataset.
- `vignette("adult")` — binary classification, full dataset.
- `vignette("house")` — regression, full dataset.

## Citation

Run `citation("blockwise")` for a ready-to-paste BibTeX entry, or see the
paper:

> Srinivasan, K., Currim, F., and Ram, S. (2025). *A Reduced Modeling
> Approach for Making Predictions With Incomplete Data Having Blockwise
> Missing Patterns.* INFORMS Journal on Data Science.

## License

GPL-3.
