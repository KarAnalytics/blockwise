test_that("simulate_blockwise_missing introduces joint masking", {
  set.seed(1)
  df <- data.frame(a = 1:500, b = 1:500, c = 1:500, d = 1:500)
  out <- simulate_blockwise_missing(
    df, blocks = list(c("a", "b"), c("c", "d")),
    prop_missing = 0.3, noise = 0
  )
  both_ab <- is.na(out$a) & is.na(out$b)
  expect_true(sum(both_ab) > 0)
  expect_true(any(is.na(out)))
})

test_that("brm fits and predicts on a toy regression problem", {
  set.seed(42)
  n <- 400
  X <- data.frame(
    a = rnorm(n), b = rnorm(n),
    c = rnorm(n), d = rnorm(n)
  )
  y <- X$a + 2 * X$b - X$c + 0.5 * X$d + rnorm(n, sd = 0.1)
  Xm <- simulate_blockwise_missing(
    X, blocks = list(c("a", "b"), c("c", "d")),
    prop_missing = 0.3, noise = 0.02
  )
  fit <- brm(Xm, y, learner = learner_lm(), n_blocks = 2L)
  expect_s3_class(fit, "brm")
  expect_length(fit$models, 2L)

  preds <- predict(fit, Xm[1:50, ])
  expect_length(preds, 50L)
  expect_true(all(is.finite(preds)))
  expect_gt(cor(preds, y[1:50]), 0.5)
})

test_that("choose_num_blocks returns a valid integer k", {
  set.seed(1)
  n <- 300
  X <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n), d = rnorm(n))
  Xm <- simulate_blockwise_missing(
    X, blocks = list(c("a", "b"), c("c", "d")), prop_missing = 0.3
  )
  out <- choose_num_blocks(Xm, k_max = 5L)
  expect_true(is.numeric(out$n_blocks))
  expect_gte(out$n_blocks, 1L)
  expect_lte(out$n_blocks, 5L)
  expect_length(out$missing_curve, 5L)
})

test_that("brm with auto n_blocks still fits", {
  set.seed(1)
  n <- 300
  X <- data.frame(a = rnorm(n), b = rnorm(n), c = rnorm(n), d = rnorm(n))
  y <- X$a + X$b - X$c + rnorm(n, sd = 0.1)
  Xm <- simulate_blockwise_missing(
    X, blocks = list(c("a", "b"), c("c", "d")), prop_missing = 0.3
  )
  fit <- brm(Xm, y, learner = learner_lm())
  expect_s3_class(fit, "brm")
  expect_gte(fit$n_blocks, 1L)
})
