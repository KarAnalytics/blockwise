#' Simulate a blockwise missing pattern on otherwise complete data
#'
#' Joint-masks groups of columns on randomly chosen rows, optionally adding
#' light column-wise random-NA noise on top. Useful for benchmarking BRM
#' on complete datasets; the default arguments reproduce the simulation
#' design used in Srinivasan, Currim, and Ram (2025).
#'
#' @param data A data.frame.
#' @param blocks A list of character vectors; each vector names the columns
#'   masked jointly for one block of rows.
#' @param prop_missing Proportion of rows affected per block. Either a scalar
#'   (applied to each block) or a numeric vector of length
#'   \code{length(blocks)}.
#' @param noise Extra per-column random-NA rate applied on top, restricted to
#'   columns named in any block. Default 0.05. Set to 0 to disable.
#' @param seed Optional integer base seed. Default 1234.
#' @return A data.frame of the same shape as \code{data}, with \code{NA}s
#'   introduced in the specified pattern.
#' @examples
#' df <- data.frame(a = 1:100, b = 1:100, c = 1:100, d = 1:100)
#' simulate_blockwise_missing(df,
#'   blocks       = list(c("a", "b"), c("c", "d")),
#'   prop_missing = 0.3)
#' @export
simulate_blockwise_missing <- function(data, blocks,
                                       prop_missing,
                                       noise = 0.05,
                                       seed = 1234L) {
  stopifnot(is.data.frame(data), is.list(blocks), length(blocks) >= 1L)
  if (length(prop_missing) == 1L) {
    prop_missing <- rep(prop_missing, length(blocks))
  }
  if (length(prop_missing) != length(blocks)) {
    stop("`prop_missing` must be a scalar or length-`length(blocks)`.")
  }

  out <- data
  n <- nrow(data)

  for (i in seq_along(blocks)) {
    cols_i <- intersect(blocks[[i]], names(out))
    if (!length(cols_i)) next
    set.seed(seed + i)
    rows_i <- sample(n, floor(prop_missing[i] * n), replace = FALSE)
    out[rows_i, cols_i] <- NA
  }

  if (noise > 0) {
    affected <- unique(unlist(blocks))
    affected <- intersect(affected, names(out))
    for (k in seq_along(affected)) {
      set.seed(seed + length(blocks) + k)
      rows_k <- sample(n, floor(noise * n), replace = FALSE)
      out[rows_k, affected[k]] <- NA
    }
  }
  out
}
