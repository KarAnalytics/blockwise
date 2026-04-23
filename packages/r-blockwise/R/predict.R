#' Predict from a fitted BRM ensemble
#'
#' Each row of \code{newdata} is routed to the block whose training-time
#' missingness center is closest (Euclidean) to the row's missingness pattern.
#' The corresponding block model then predicts on that row, using only that
#' block's feature columns. Any \code{NA}s remaining in those columns are
#' filled by simple mean/mode imputation against the training reference.
#'
#' @param object A fitted \code{\link{brm}} object.
#' @param newdata A data.frame of predictors. May contain \code{NA}.
#' @param ... Unused.
#' @return A numeric vector of length \code{nrow(newdata)}.
#' @importFrom stats predict
#' @export
predict.brm <- function(object, newdata, ...) {
  stopifnot(is.data.frame(newdata))
  features <- object$feature_names

  missing_cols <- setdiff(features, names(newdata))
  for (col in missing_cols) newdata[[col]] <- NA
  newdata <- newdata[, features, drop = FALSE]

  t_miss <- matrix(as.integer(!is.na(newdata)),
                   nrow = nrow(newdata), ncol = ncol(newdata))
  colnames(t_miss) <- features

  centers <- as.matrix(object$centers)
  nearest <- integer(nrow(t_miss))
  for (i in seq_len(nrow(t_miss))) {
    d <- rowSums((centers - matrix(t_miss[i, ], nrow = nrow(centers),
                                   ncol = ncol(centers), byrow = TRUE))^2)
    nearest[i] <- which.min(d)
  }

  preds <- numeric(nrow(newdata))
  for (b in seq_along(object$models)) {
    idx <- which(nearest == b)
    if (!length(idx)) next
    cols_b <- object$columns[[b]]
    X_b <- newdata[idx, cols_b, drop = FALSE]
    X_b <- simple_impute(X_b, train_ref = object$train_ref[, cols_b, drop = FALSE])
    preds[idx] <- as.numeric(object$learner$predict(object$models[[b]], X_b))
  }
  preds
}
