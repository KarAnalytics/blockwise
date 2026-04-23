#' Fit a Blockwise Reduced Modeling (BRM) ensemble
#'
#' BRM partitions the training data into \code{n_blocks} subsets based on
#' per-row patterns of feature-missingness, fits one instance of the supplied
#' \code{learner} per subset using only the features observed in that subset,
#' and at prediction time routes each test row to the subset model whose
#' training-time missingness pattern is closest.
#'
#' The learner interface is intentionally minimal: any \code{fit} /
#' \code{predict} pair can be plugged in via \code{\link{learner}()}, and
#' convenience specs are provided for common families
#' (\code{\link{learner_lm}}, \code{\link{learner_rpart}}, etc.).
#'
#' @param X A data.frame of predictors. May contain \code{NA}. Categorical
#'   predictors should be factors.
#' @param y A numeric vector (regression) or a 0/1 numeric vector
#'   (binary classification) of length \code{nrow(X)}.
#' @param learner A \code{\link{learner}} specification.
#'   Defaults to \code{\link{learner_lm}()}.
#' @param n_blocks Integer number of blocks; if \code{NULL}, chosen
#'   automatically by \code{\link{choose_num_blocks}}.
#' @param low_threshold Column-density threshold for including a predictor in
#'   a block's model. Default 0.05.
#' @param n_restarts k-means restarts for block assignment. Default 5.
#' @param overlap If \code{TRUE} (default) subsets are enlarged via the
#'   set-theoretic inclusion rule; if \code{FALSE}, the non-overlapping
#'   variant is used.
#' @return An object of class \code{"brm"}.
#' @references Srinivasan, K., Currim, F., Ram, S. (2025). A Reduced Modeling
#'   Approach for Making Predictions With Incomplete Data Having Blockwise
#'   Missing Patterns. \emph{INFORMS Journal on Data Science}.
#' @examples
#' \dontrun{
#'   data(bike, package = "blockwise")
#'   bike_miss <- simulate_blockwise_missing(
#'     bike,
#'     blocks       = list(c("hum", "windspeed", "weekday"),
#'                         c("hr", "temp", "weathersit")),
#'     prop_missing = 0.3
#'   )
#'   X <- bike_miss[, setdiff(names(bike_miss), "cnt")]
#'   y <- bike_miss$cnt
#'   fit <- brm(X, y, learner = learner_lm())
#'   preds <- predict(fit, X)
#' }
#' @export
brm <- function(X, y,
                learner = learner_lm(),
                n_blocks = NULL,
                low_threshold = 0.05,
                n_restarts = 5L,
                overlap = TRUE) {
  stopifnot(is.data.frame(X))
  if (nrow(X) != length(y)) {
    stop("X and y must have the same number of observations.")
  }
  if (!inherits(learner, "brm_learner")) {
    stop("`learner` must be a brm_learner object; see ?learner.")
  }

  y_df <- data.frame(y = y)
  names(y_df) <- "__y__"

  if (is.null(n_blocks)) {
    n_blocks <- choose_num_blocks(X, low_threshold = low_threshold)$n_blocks
  }
  n_blocks <- as.integer(n_blocks)

  subs <- build_subsets(X, y_df, n_blocks,
                        low_threshold = low_threshold,
                        n_restarts = n_restarts)
  parts <- if (overlap) subs$overlapping else subs$non_overlapping

  models <- lapply(parts, function(df) {
    X_i <- df[, setdiff(names(df), "__y__"), drop = FALSE]
    y_i <- df[["__y__"]]
    learner$fit(X_i, y_i)
  })

  block_columns <- lapply(subs$columns, function(cc) setdiff(cc, "__y__"))

  out <- list(
    models        = models,
    centers       = subs$centers,
    columns       = block_columns,
    learner       = learner,
    feature_names = names(X),
    n_blocks      = n_blocks,
    overlap       = overlap,
    train_ref     = X
  )
  class(out) <- "brm"
  out
}

#' @export
print.brm <- function(x, ...) {
  cat("Blockwise Reduced Model (BRM)\n")
  cat("  blocks        :", x$n_blocks, "\n")
  cat("  overlap       :", x$overlap, "\n")
  cat("  learner type  :", x$learner$type, "\n")
  cat("  features      :", length(x$feature_names), "\n")
  sz <- vapply(x$columns, length, integer(1L))
  cat("  cols / block  :", paste(sz, collapse = ", "), "\n")
  invisible(x)
}

#' @export
summary.brm <- function(object, ...) {
  structure(
    list(
      n_blocks      = object$n_blocks,
      overlap       = object$overlap,
      learner_type  = object$learner$type,
      feature_names = object$feature_names,
      block_sizes   = vapply(object$columns, length, integer(1L)),
      block_centers = object$centers
    ),
    class = "summary.brm"
  )
}

#' @export
print.summary.brm <- function(x, ...) {
  cat("Blockwise Reduced Model summary\n")
  cat("  n_blocks   :", x$n_blocks, "\n")
  cat("  overlap    :", x$overlap, "\n")
  cat("  learner    :", x$learner_type, "\n")
  cat("  n_features :", length(x$feature_names), "\n\n")
  cat("Feature set per block (1 = observed):\n")
  print(x$block_centers)
  invisible(x)
}
