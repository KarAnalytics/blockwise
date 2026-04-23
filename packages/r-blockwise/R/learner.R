#' Learner specification for BRM
#'
#' BRM trains one model per overlapping subset. The learner interface makes
#' that choice user-controlled: supply a \code{fit} function that takes
#' \code{(X, y)} and returns a fitted model, and a \code{predict} function
#' that takes \code{(model, X_new)} and returns a numeric prediction vector
#' (or a positive-class probability for binary classification).
#'
#' @param fit A function of the form \code{function(X, y) -> model}.
#' @param predict A function of the form \code{function(model, X_new) -> numeric}.
#' @param type Either \code{"regression"} or \code{"classification"}.
#' @return An object of class \code{"brm_learner"}.
#' @examples
#' \dontrun{
#'   my_learner <- learner(
#'     fit     = function(X, y) lm(y ~ ., data = cbind(X, y = y)),
#'     predict = function(m, X_new) predict(m, newdata = X_new),
#'     type    = "regression"
#'   )
#' }
#' @export
learner <- function(fit, predict, type = c("regression", "classification")) {
  type <- match.arg(type)
  stopifnot(is.function(fit), is.function(predict))
  structure(
    list(fit = fit, predict = predict, type = type),
    class = "brm_learner"
  )
}

#' @rdname learner
#' @export
learner_lm <- function() {
  learner(
    fit = function(X, y) stats::lm(y ~ ., data = cbind(X, y = y)),
    predict = function(m, X_new) as.numeric(stats::predict(m, newdata = X_new)),
    type = "regression"
  )
}

#' @rdname learner
#' @export
learner_glm_binomial <- function() {
  learner(
    fit = function(X, y) {
      df <- cbind(X, y = y)
      stats::glm(y ~ ., data = df, family = stats::binomial())
    },
    predict = function(m, X_new) {
      as.numeric(stats::predict(m, newdata = X_new, type = "response"))
    },
    type = "classification"
  )
}

#' @rdname learner
#' @param method rpart split method; one of \code{"anova"}, \code{"class"}, etc.
#' @param ... Additional arguments passed to the underlying fitter.
#' @export
learner_rpart <- function(method = "anova", ...) {
  if (!requireNamespace("rpart", quietly = TRUE)) {
    stop("Package 'rpart' required for learner_rpart().")
  }
  dots <- list(...)
  learner(
    fit = function(X, y) {
      df <- cbind(X, y = y)
      do.call(rpart::rpart,
              c(list(formula = y ~ ., data = df, method = method), dots))
    },
    predict = function(m, X_new) {
      p <- stats::predict(m, newdata = X_new)
      if (is.matrix(p)) as.numeric(p[, ncol(p)]) else as.numeric(p)
    },
    type = if (method == "anova") "regression" else "classification"
  )
}

#' @rdname learner
#' @export
learner_ranger <- function(...) {
  if (!requireNamespace("ranger", quietly = TRUE)) {
    stop("Package 'ranger' required for learner_ranger().")
  }
  dots <- list(...)
  learner(
    fit = function(X, y) do.call(ranger::ranger, c(list(y = y, x = X), dots)),
    predict = function(m, X_new) as.numeric(stats::predict(m, data = X_new)$predictions),
    type = "regression"
  )
}

#' @rdname learner
#' @param distribution gbm distribution (e.g. \code{"gaussian"}, \code{"bernoulli"}, \code{"poisson"}).
#' @param n.trees Number of trees.
#' @export
learner_gbm <- function(distribution = "gaussian", n.trees = 500, ...) {
  if (!requireNamespace("gbm", quietly = TRUE)) {
    stop("Package 'gbm' required for learner_gbm().")
  }
  dots <- list(...)
  learner(
    fit = function(X, y) {
      df <- cbind(X, y = y)
      do.call(gbm::gbm,
              c(list(formula = y ~ ., data = df,
                     distribution = distribution,
                     n.trees = n.trees, verbose = FALSE), dots))
    },
    predict = function(m, X_new) {
      as.numeric(gbm::predict.gbm(m, newdata = X_new,
                                  n.trees = m$n.trees, type = "response"))
    },
    type = if (distribution %in% c("bernoulli", "multinomial")) "classification" else "regression"
  )
}
