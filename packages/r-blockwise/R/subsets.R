# Build the overlapping (and non-overlapping) subsets used by BRM.
#
# Each subset corresponds to one cluster of rows with similar missingness
# pattern. Within each cluster, columns that are observed in fewer than
# low_threshold * nrow(cluster) rows are dropped. Overlapping subsets are then
# formed by merging in any other cluster whose observed-column set is a
# superset of the current cluster's observed-column set.
build_subsets <- function(X, y_df, n_blocks,
                          low_threshold = 0.05, n_restarts = 5L) {
  stopifnot(is.data.frame(X), is.data.frame(y_df), ncol(y_df) == 1L)
  y_name <- names(y_df)
  data_xy <- cbind(X, y_df)

  M <- as.data.frame(
    matrix(as.integer(!is.na(X)), nrow = nrow(X), ncol = ncol(X))
  )
  names(M) <- names(X)

  km <- best_kmeans(M, n_blocks, n_restarts)
  centers <- round(as.data.frame(km$centers), 0L)

  miss_parts <- split(M, km$cluster)
  data_parts <- split(data_xy, km$cluster)

  non_ov <- vector("list", length(miss_parts))
  cols_per <- vector("list", length(miss_parts))
  for (i in seq_along(miss_parts)) {
    completeness <- colSums(miss_parts[[i]])
    keep <- names(which(completeness > low_threshold * nrow(miss_parts[[i]])))
    non_ov[[i]] <- data_parts[[i]][, c(keep, y_name), drop = FALSE]
    cols_per[[i]] <- names(non_ov[[i]])
  }

  ov <- non_ov
  for (i in seq_along(non_ov)) {
    cols_i <- cols_per[[i]]
    for (j in seq_along(non_ov)) {
      if (i == j) next
      if (all(cols_i %in% cols_per[[j]])) {
        ov[[i]] <- rbind(ov[[i]], non_ov[[j]][, cols_i, drop = FALSE])
      }
    }
    ov[[i]] <- hotdeck_impute(ov[[i]])
    non_ov[[i]] <- hotdeck_impute(non_ov[[i]])
  }

  list(
    overlapping = ov,
    non_overlapping = non_ov,
    columns = cols_per,
    centers = centers,
    y_name = y_name
  )
}
