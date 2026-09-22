#!/usr/bin/env Rscript

# Rank prefiltered continuous-outcome features with geeVerse's official qpgee
# implementation: an HBIC-tuned penalized quantile GEE for repeated measures.
# Input CSV columns: y, id, x001, x002, ...
# Output CSV columns: Feature, Score, Lambda, HBIC, Converged.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: rank_geeverse.R INPUT.csv OUTPUT.csv TASK{regression} SEED")
}
input_path <- args[[1]]
output_path <- args[[2]]
task <- args[[3]]
seed <- as.integer(args[[4]])
if (task != "regression") stop("geeVerse qpgee is registered only for continuous outcomes.")

suppressPackageStartupMessages(library(geeVerse))
set.seed(seed)
d <- read.csv(input_path, check.names = FALSE)
if (!all(c("y", "id") %in% names(d))) stop("Input must contain y and id columns.")
features <- setdiff(names(d), c("y", "id"))
if (!length(features)) stop("Input contains no feature columns.")

# qpgee.formula obtains cluster sizes from table(id), whereas the rows supplied
# to the estimating equation must be contiguous by participant.
d <- d[order(d$id), , drop = FALSE]
fit <- tryCatch(
  qpgee(
    y ~ . - id,
    id = d$id,
    data = d,
    tau = 0.5,
    corstr = "exchangeable",
    lambda = c(0.05, 0.1, 0.3, 0.5),
    method = "HBIC",
    ncore = 1,
    control = qpgeeControl(maxit = 75, standardize = FALSE)
  ),
  error = function(e) NULL
)
if (is.null(fit) || is.null(fit$coefficients) || !is.finite(fit$best_lambda)) {
  stop("geeVerse qpgee failed to return a finite fit.")
}

coefficient <- fit$coefficients
scores <- abs(as.numeric(coefficient[features]))
scores[!is.finite(scores)] <- 0
write.csv(
  data.frame(
    Feature = features,
    Score = scores,
    Lambda = fit$best_lambda,
    HBIC = fit$hbic,
    Converged = isTRUE(fit$converged)
  ),
  output_path,
  row.names = FALSE
)
