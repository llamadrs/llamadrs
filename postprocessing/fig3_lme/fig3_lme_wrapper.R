suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(broom.mixed)
  library(stringr)
  library(readr)
  library(purrr)
})

# Source the fixed plotting functions (assumes fig3_lme_figures.R is in same directory)
if (file.exists("fig3_lme_figures.R")) {
  source("fig3_lme_figures.R")
} else if (file.exists("../fig3_lme_figures.R")) {
  source("../fig3_lme_figures.R")
} else {
  stop("Cannot find fig3_lme_figures.R. Please ensure it's in the working directory.")
}

# -------------------------------
# Wrapper for R6 Analyzer Object
# -------------------------------
make_all_figs_items <- function(analyzer, outdir = "../output/madrs_figs/items") {
  # Wrapper function that extracts models and data from the R6 analyzer object
  # and calls the plotting functions.
  # 
  # Args:
  #   analyzer: MADRSErrorAnalysis R6 object (or compatible)
  #   outdir: Output directory for figures
  
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  
  # Extract the interaction model
  fit_int <- analyzer$models$items$interaction
  if (is.null(fit_int)) {
    stop("No interaction model found. Run analyzer$fit_interaction_model_items() first.")
  }
  
  # Extract data
  df <- analyzer$items_data
  if (is.null(df)) {
    stop("No items_data found. Run analyzer$load_and_prepare_items_data() first.")
  }
  
  # Extract configuration
  use_within_between <- analyzer$use_within_between
  use_item_random_intercept <- analyzer$use_item_random_intercept
  random_slope_terms <- analyzer$random_slope_terms
  reml <- analyzer$reml
  maxiter <- analyzer$maxiter
  
  # Generate forest plots from interaction model
  print("Generating interaction model forest plots...")
  plot_forest_interaction_all(fit_int, outdir)
  plot_forest_predictors_model(fit_int, outdir)
  plot_forest_predictors_session(fit_int, outdir)
  plot_forest_predictors_interactions(fit_int, outdir)
  
  # Plot MADRS items (fixed effects or random effects depending on model)
  plot_forest_predictors_madrs(fit_int, outdir)  # Fixed effects version
  plot_forest_madrs_blups(fit_int, outdir)       # Random effects version (BLUPs)
  
  
  # Marginal effects
  plot_marginal_effects(fit_int, outdir)
  
  # Variance decomposition (if method exists)
  var_decomp <- tryCatch({
    analyzer$variance_decomposition_items()
  }, error = function(e) {
    warning("Could not compute variance decomposition: ", e$message)
    NULL
  })
  
  if (!is.null(var_decomp)) {
    plot_variance_decomposition(var_decomp, outdir)
  }
  
  # Paired t-test (if method exists)
  ttest_result <- tryCatch({
    analyzer$test_reasoning_effect_items()
  }, error = function(e) {
    warning("Could not compute t-test: ", e$message)
    NULL
  })
  
  if (!is.null(ttest_result)) {
    plot_paired_ttest(ttest_result, outdir)
  }
  
  message(sprintf("All figures saved to %s/", outdir))
  invisible(TRUE)
}
