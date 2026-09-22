#!/usr/bin/env Rscript

# Rank prefiltered features with the official PGEE SCAD-penalized GEE package.
# Input CSV columns: y, id, x001, x002, ...
# Output CSV columns: Feature, Score, Lambda, CV.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: rank_pgee.R INPUT.csv OUTPUT.csv TASK{classification|regression} SEED")
}

input_path <- args[[1]]
output_path <- args[[2]]
task <- args[[3]]
seed <- as.integer(args[[4]])

suppressPackageStartupMessages(library(PGEE))
set.seed(seed)
d <- read.csv(input_path, check.names = FALSE)
if (!all(c("y", "id") %in% names(d))) stop("Input must contain y and id columns.")
features <- setdiff(names(d), c("y", "id"))
if (!length(features)) stop("Input contains no feature columns.")

# PGEE 1.5's legacy formula evaluator requires these conventional field names.
formula_fixed <- as.formula("y ~ . - id")
family_value <- if (task == "classification") binomial(link = "logit") else gaussian()
# Package-native three-fold CV chooses a SCAD tuning parameter strictly within
# the supplied outer-training partition. All features were standardized by the
# Python adapter before this script is called.
lambda_grid <- c(0.05, 0.1, 0.3, 0.5)
fit_cv <- tryCatch(
  CVfit(
    formula = formula_fixed,
    id = d$id,
    data = d,
    family = family_value,
    scale.fix = TRUE,
    scale.value = 1,
    fold = 3,
    lambda.vec = lambda_grid,
    pindex = seq_along(features),
    eps = 1e-4,
    maxiter = 30,
    tol = 1e-3
  ),
  error = function(e) NULL
)
if (is.null(fit_cv) || !is.finite(fit_cv$lam.opt)) stop("PGEE CVfit failed for all candidate lambda values.")

fit_error <- NULL
fit <- tryCatch(
  PGEE(
    formula = formula_fixed,
    id = d$id,
    data = d,
    family = family_value,
    corstr = "independence",
    lambda = fit_cv$lam.opt,
    pindex = seq_along(features),
    eps = 1e-4,
    maxiter = 30,
    tol = 1e-3,
    silent = TRUE
  ),
  error = function(e) { fit_error <<- conditionMessage(e); NULL }
)
if (is.null(fit)) stop(sprintf("PGEE fit failed after CV tuning: %s", ifelse(is.null(fit_error), "unknown error", fit_error)))

coefficient <- coef(fit)
scores <- abs(as.numeric(coefficient[features]))
scores[!is.finite(scores)] <- 0
write.csv(
  data.frame(Feature = features, Score = scores, Lambda = fit_cv$lam.opt, CV = fit_cv$cv.min),
  output_path,
  row.names = FALSE
)
