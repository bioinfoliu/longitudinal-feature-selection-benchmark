#!/usr/bin/env Rscript

# Rank prefiltered features with the official PGEE SCAD-penalized GEE
# estimator. Input columns: y, id, x001, x002, ...
# Output columns: Feature, Score, Lambda, CV.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) stop("Usage: rank_pgee.R INPUT.csv OUTPUT.csv TASK SEED")
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

if (task == "regression") {
  outcome_sd <- sd(d$y)
  if (is.finite(outcome_sd) && outcome_sd > 0) d$y <- (d$y - mean(d$y)) / outcome_sd
}

# PGEE 1.5's CVfit assumes equal cluster sizes. The benchmark contains
# irregular follow-up, so we tune the package's PGEE estimator with explicit
# participant-blocked folds rather than using CVfit.
formula_fixed <- as.formula(paste("y ~", paste(features, collapse = " + ")))
family_value <- if (task == "classification") binomial(link = "logit") else gaussian()
# A compact, pre-specified grid keeps nested evaluation feasible while still
# spanning weak, moderate, and strong SCAD penalization.
lambda_grid <- c(0.1, 0.3, 1.0)

fit_pgee <- function(frame, lambda) {
  working_data <- frame
  tryCatch(
    suppressWarnings(PGEE(
      formula = formula_fixed,
      id = id,
      data = working_data,
      family = family_value,
      corstr = "exchangeable",
      beta_int = rep(0, length(features) + 1),
      scale.fix = TRUE,
      scale.value = 1,
      lambda = lambda,
      pindex = seq_along(features),
      eps = 1e-4,
      maxiter = 20,
      tol = 1e-3,
      silent = TRUE
    )),
    error = function(e) { message("PGEE failure: ", conditionMessage(e)); NULL }
  )
}

unique_ids <- unique(d$id)
fold_count <- min(2L, length(unique_ids))
if (fold_count < 2L) stop("PGEE requires at least two participant folds.")
fold_map <- integer(0)
if (task == "classification") {
  subject_y <- tapply(d$y, d$id, function(z) z[[1]])
  negative <- sample(as.numeric(names(subject_y)[subject_y == 0]))
  positive <- sample(as.numeric(names(subject_y)[subject_y == 1]))
  fold_map[as.character(negative)] <- rep(seq_len(fold_count), length.out = length(negative))
  fold_map[as.character(positive)] <- rep(seq_len(fold_count), length.out = length(positive))
} else {
  shuffled <- sample(unique_ids)
  fold_map[as.character(shuffled)] <- rep(seq_len(fold_count), length.out = length(shuffled))
}

cv_values <- rep(Inf, length(lambda_grid))
for (lambda_index in seq_along(lambda_grid)) {
  fold_losses <- c()
  for (fold in seq_len(fold_count)) {
    validation_ids <- as.numeric(names(fold_map)[fold_map == fold])
    train_frame <- d[!(d$id %in% validation_ids), , drop = FALSE]
    validation_frame <- d[d$id %in% validation_ids, , drop = FALSE]
    fit_fold <- fit_pgee(train_frame, lambda_grid[[lambda_index]])
    if (is.null(fit_fold)) next
    coefficient <- coef(fit_fold)
    design <- model.matrix(formula_fixed, validation_frame)
    if (!all(colnames(design) %in% names(coefficient))) next
    eta <- as.numeric(design %*% coefficient[colnames(design)])
    prediction <- family_value$linkinv(eta)
    if (!all(is.finite(prediction))) next
    fold_losses <- c(
      fold_losses,
      sum(family_value$dev.resids(validation_frame$y, prediction, wt = 1)) / nrow(validation_frame)
    )
  }
  if (length(fold_losses) == fold_count) cv_values[[lambda_index]] <- mean(fold_losses)
}
if (!any(is.finite(cv_values))) stop("PGEE participant-blocked CV failed for all candidate lambda values.")

# Use the lowest-CV lambda; if its full-data fit is singular, move to the next
# CV-ranked lambda. This changes only the tuning candidate, never the selector.
fit <- NULL
chosen_index <- NA_integer_
for (candidate in order(cv_values)) {
  fit <- fit_pgee(d, lambda_grid[[candidate]])
  if (!is.null(fit) && all(is.finite(coef(fit)))) {
    chosen_index <- candidate
    break
  }
}
if (is.null(fit) || is.na(chosen_index)) stop("PGEE full-training fit failed for all CV-ranked lambda values.")

coefficient <- coef(fit)
scores <- abs(as.numeric(coefficient[features]))
scores[!is.finite(scores)] <- 0
write.csv(
  data.frame(
    Feature = features,
    Score = scores,
    Lambda = lambda_grid[[chosen_index]],
    CV = cv_values[[chosen_index]]
  ),
  output_path,
  row.names = FALSE
)
