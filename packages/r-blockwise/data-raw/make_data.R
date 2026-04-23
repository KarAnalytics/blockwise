# Build the bundled data objects (bike, adult, house) from the raw CSVs in
# the parent project's `data/` directory. Run this once from the R package
# root, e.g.:
#
#   setwd("packages/r-blockwise")
#   source("data-raw/make_data.R")
#
# Outputs: data/bike.rda, data/adult.rda, data/house.rda (xz-compressed).

library(usethis)

raw_dir <- normalizePath(file.path("..", "..", "data"))
stopifnot(dir.exists(raw_dir))

# ---- bike: full dataset ----
bike_raw <- read.csv(file.path(raw_dir, "bike_all.csv"))
bike <- bike_raw[, !names(bike_raw) %in%
                   c("instant", "dteday", "atemp", "yr",
                     "workingday", "holiday")]
bike$weekday <- as.factor(bike$weekday)
bike$weathersit[bike$weathersit == 4L] <- 3L
bike$weathersit <- as.factor(bike$weathersit)
bike$season <- as.factor(bike$season)
usethis::use_data(bike, overwrite = TRUE, compress = "xz")

# ---- adult: full dataset ----
adult_raw <- read.csv(file.path(raw_dir, "adult.csv"))
adult <- as.data.frame(unclass(adult_raw), stringsAsFactors = TRUE)
# Binary-encode salary.
adult$salary <- ifelse(
  trimws(as.character(adult$salary)) %in% c(">50K", ">=50k", ">50K."),
  1L, 0L
)
adult <- adult[, !names(adult) %in%
                 c("fnlwgt", "capital.gain", "capital.loss",
                   "capital-gain", "capital-loss")]
usethis::use_data(adult, overwrite = TRUE, compress = "xz")

# ---- house: full dataset ----
house <- read.csv(file.path(raw_dir, "house.csv"))
usethis::use_data(house, overwrite = TRUE, compress = "xz")

message("Wrote data/bike.rda, data/adult.rda, data/house.rda.")
