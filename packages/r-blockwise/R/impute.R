# Thin wrapper over VIM::hotdeck for per-subset imputation at training time.
hotdeck_impute <- function(df) {
  if (!any(is.na(df))) return(df)
  VIM::hotdeck(df, imp_var = FALSE)
}

# Fallback column-wise imputation used at predict time when a block's columns
# still contain NA in the test data: mean for numerics, mode for factors.
simple_impute <- function(df, train_ref = NULL) {
  for (j in names(df)) {
    na_idx <- is.na(df[[j]])
    if (!any(na_idx)) next
    ref_col <- if (!is.null(train_ref) && j %in% names(train_ref)) train_ref[[j]] else df[[j]]
    if (is.factor(df[[j]]) || is.character(df[[j]])) {
      tab <- sort(table(ref_col), decreasing = TRUE)
      if (length(tab) == 0L) next
      df[[j]][na_idx] <- names(tab)[1L]
    } else {
      df[[j]][na_idx] <- mean(as.numeric(ref_col), na.rm = TRUE)
    }
  }
  df
}
