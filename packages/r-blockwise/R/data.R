#' Capital Bikeshare hourly demand data
#'
#' Hourly count of rental bikes between 2011 and 2012 in the Capital Bikeshare
#' system with the corresponding weather and seasonal information. Used as the
#' regression demonstration in Srinivasan, Currim, and Ram (2025).
#'
#' @format A data.frame with roughly 17,380 rows and the following columns:
#' \describe{
#'   \item{season, mnth, hr, weekday, weathersit}{Temporal and weather covariates.}
#'   \item{temp, hum, windspeed}{Numeric weather covariates.}
#'   \item{cnt}{Response: count of total rental bikes for that hour.}
#' }
#' @source UCI Machine Learning Repository:
#'   \url{https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset}
"bike"

#' UCI Adult income classification dataset
#'
#' Census-based binary classification dataset: predict whether a person's
#' annual income exceeds $50,000. Used as the classification demonstration
#' in Srinivasan, Currim, and Ram (2025).
#'
#' @format A data.frame with roughly 32,561 rows including \code{salary} (the
#'   0/1-valued response) and typical demographic/employment predictors.
#' @source UCI Machine Learning Repository:
#'   \url{https://archive.ics.uci.edu/dataset/2/adult}
"adult"

#' King County, WA house sales
#'
#' The King County house-sales dataset. Used as a regression demonstration
#' in Srinivasan, Currim, and Ram (2025).
#'
#' @format A data.frame with roughly 21,600 rows including \code{price}
#'   (the response) and typical property covariates such as
#'   \code{bedrooms}, \code{bathrooms}, \code{sqft_living},
#'   \code{sqft_lot}, \code{grade}, and \code{yr_built}.
#' @source Kaggle "House Sales in King County, USA" dataset.
"house"
