#' Estimate the number of blocks via the elbow heuristic
#'
#' Applies k-means to the binary missingness-indicator matrix for
#' k = 1, ..., \code{k_max} and records, for each k, the fraction of
#' still-missing cells after dropping columns that are sparse within each
#' cluster. The curve is monotone-decreasing; BRM picks the k at its elbow.
#'
#' Typically called internally by \code{\link{brm}} when \code{n_blocks = NULL}.
#'
#' @param X A data.frame of predictors; may contain \code{NA}.
#' @param low_threshold Fraction below which a column is considered absent in
#'   a candidate subset. Default 0.05.
#' @param n_restarts Number of k-means restarts per k. Default 10.
#' @param k_max Upper bound on k. Default \code{min(ncol(X), 50)}.
#' @return A list with \code{n_blocks} (the chosen k) and \code{missing_curve}
#'   (numeric vector of length \code{k_max}).
#' @references Srinivasan, K., Currim, F., Ram, S. (2025). A Reduced Modeling
#'   Approach for Making Predictions With Incomplete Data Having Blockwise
#'   Missing Patterns. \emph{INFORMS Journal on Data Science}.
#' @importFrom stats kmeans
#' @export
choose_num_blocks <- function(X, low_threshold = 0.05, n_restarts = 10, k_max = NULL) {
  if (nrow(X) > 50000L) {
    set.seed(12345)
    X <- X[sample(nrow(X), 50000L, replace = FALSE), , drop = FALSE]
  }
  M <- as.data.frame(
    matrix(as.integer(!is.na(X)), nrow = nrow(X), ncol = ncol(X))
  )
  names(M) <- names(X)

  if (is.null(k_max)) k_max <- min(ncol(X), 50L)
  curve <- vapply(
    seq_len(k_max),
    function(k) subset_missing_frac(M, k, low_threshold, n_restarts),
    numeric(1)
  )
  list(n_blocks = elbow_point(curve), missing_curve = curve)
}

# Best-of-n k-means run for a given k (internal).
best_kmeans <- function(M, k, n_restarts = 10L) {
  if (k == 1L) {
    centers <- matrix(colMeans(M), nrow = 1L)
    colnames(centers) <- colnames(M)
    return(list(cluster = rep(1L, nrow(M)), centers = centers,
                totss = 0, tot.withinss = 0))
  }
  runs <- lapply(seq_len(n_restarts), function(s) {
    set.seed(s)
    stats::kmeans(M, centers = k)
  })
  scores <- vapply(runs,
                   function(r) 100 - r$tot.withinss / r$totss * 100,
                   numeric(1))
  runs[[which.max(scores)]]
}

# Missing-cell density after clustering rows by their missingness pattern.
subset_missing_frac <- function(M, k, low_threshold = 0.05, n_restarts = 10L) {
  km <- best_kmeans(M, k, n_restarts)
  parts <- split(M, km$cluster)
  total_missing <- 0
  for (p in parts) {
    p <- as.data.frame(p)
    completeness <- colSums(p)
    keep <- completeness > low_threshold * nrow(p)
    if (!any(keep)) next
    p_keep <- p[, keep, drop = FALSE]
    total_missing <- total_missing + sum(nrow(p_keep) - colSums(p_keep))
  }
  total_missing / (nrow(M) * ncol(M))
}

# Elbow = point most distant from the line (x1,y1)-(xN,yN).
elbow_point <- function(y) {
  x <- seq_along(y)
  if (length(x) <= 2L) return(1L)
  x1 <- x[1L]; y1 <- y[1L]
  x2 <- x[length(x)]; y2 <- y[length(y)]
  num <- abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
  den <- sqrt((y2 - y1)^2 + (x2 - x1)^2)
  if (den == 0) return(1L)
  which.max(num / den)
}
