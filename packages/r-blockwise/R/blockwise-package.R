#' blockwise: Reduced Modeling for Tabular Data with Blockwise Missingness
#'
#' Implements the Blockwise Reduced Modeling (BRM) method of Srinivasan,
#' Currim, and Ram (2025). The main entry points are:
#' \itemize{
#'   \item \code{\link{brm}} — fit a BRM ensemble.
#'   \item \code{\link{predict.brm}} — predict on new data.
#'   \item \code{\link{choose_num_blocks}} — elbow heuristic for choosing k.
#'   \item \code{\link{learner}} and siblings — learner-agnostic fit/predict spec.
#'   \item \code{\link{simulate_blockwise_missing}} — mask complete data with a
#'     blockwise missingness pattern.
#' }
#'
#' @keywords internal
"_PACKAGE"
