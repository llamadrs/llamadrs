library(R6)
library(readxl)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(ggplot2)
library(svglite)
library(broom.mixed)
# NEW
library(modelbased)   # estimate_grouplevel()
library(performance)  # check_predictions()
library(see)
MADRSErrorAnalysis <- R6Class(
  "MADRSErrorAnalysis",

  public = list(
    excel_path = NULL,
    use_within_between = TRUE,
    use_item_random_intercept = TRUE,
    reml = TRUE,
    maxiter = 2e5,
    aggregate_seeds = TRUE,
    random_slope_terms = NULL,
    items_data = NULL,
    full_data = NULL,
    models = list(
      items = list(
        nonreason = NULL,
        reason    = NULL
      )
    ),

    initialize = function(excel_path,
                          use_within_between = TRUE,
                          use_item_random_intercept = TRUE,
                          reml = TRUE,
                          maxiter = 2e5,
                          random_slope_terms = NULL,
                          aggregate_seeds = TRUE) {
      self$excel_path <- excel_path
      self$use_within_between <- use_within_between
      self$use_item_random_intercept <- use_item_random_intercept
      self$reml <- reml
      self$maxiter <- maxiter
      self$aggregate_seeds <- aggregate_seeds

      # Random-slope candidates — include centered reasoning length Z
      if (is.null(random_slope_terms)) {
        self$random_slope_terms <- c(
          "session_item_severity_patitem_wc_z",
          "log_tokens_pat_wc_z"
          ,"log_reason_tokens_patitem_wc_z"
        )
      } else {
        self$random_slope_terms <- random_slope_terms
      }
    },

    # =========================
    # Loading / Preparation
    # =========================
    load_and_prepare_items_data = function(item_indices = 1:10, seeds = 3) {
      frames <- list()

      # Total score sheet
      tryCatch({
        df0 <- readxl::read_excel(self$excel_path, sheet = "00 - Total Score")
        df0$madrs_item <- 0
        frames[[1]] <- df0
      }, error = function(e) {
        warning(sprintf("Could not load '00 - Total Score': %s", e$message))
      })

      # Item sheets
      for (item_idx in item_indices) {
        sheet_name <- sprintf("%02d - %s", item_idx, private$get_item_name(item_idx))
        tryCatch({
          dfi <- readxl::read_excel(self$excel_path, sheet = sheet_name)
          dfi$madrs_item <- item_idx
          frames[[length(frames) + 1]] <- dfi
        }, error = function(e) {
          warning(sprintf("Could not load sheet '%s': %s", sheet_name, e$message))
        })
      }

      if (length(frames) == 0) stop("No sheets loaded. Check Excel path and sheet names.")

      wide_df <- dplyr::bind_rows(frames)
      long_df <- private$reshape_to_long(wide_df, seeds = seeds)

      # Average over seeds (also averages num_reason_tokens)
      if (self$aggregate_seeds && "seed" %in% names(long_df)) {
        long_df <- private$average_over_seeds(long_df)
      }

      long_df <- private$engineer_features(long_df, task = "items")
      # filter out rows where is_reasoning is 0
      print(long_df)
      # Factors
      factor_cols <- c("patient", "session", "model_name", "madrs_item", "architecture")
      for (col in factor_cols) if (col %in% names(long_df)) long_df[[col]] <- as.factor(long_df[[col]])
      if ("madrs_item" %in% names(long_df)) long_df$madrs_item <- stats::relevel(long_df$madrs_item, ref = "0")

      self$items_data <- long_df
      if (!"num_reason_tokens" %in% names(self$items_data)) {
        warning("num_reason_tokens not found; reasoning-length features will be NA.")
      }
      invisible(long_df)
    },

    # =========================
    # Build FE formula, save per-row design BEFORE fitting
    # =========================
    build_and_save_design_matrix = function(csv_path = "../output/model_design_items.csv") {
      private$require_data("items")
      df <- self$items_data

      # Fixed effects (no interactions for reasoning-length since 0 for non-reasoning)
      wc_terms   <- c("session_item_severity_patitem_wc_z",
                      "log_tokens_pat_wc_z"
                      ,"log_reason_tokens_patitem_wc_z"
                      
                    )
      mean_terms <- c("session_item_severity_patitem_mean_z",
                      "log_tokens_pat_mean_z"
                      ,"log_reason_tokens_patitem_mean_z"
                      )

      fe_terms <- c(mean_terms, wc_terms,
                    "log_params_z", "log_context_length_z",
                    "architecture"
                    #, "is_reasoning"
                    )
      fe_terms <- unique(fe_terms)

      re_parts <- c(
        "(1 | model_name)",
        "(1 | patient)",
        "(1 | session:patient)",
        "(1 | madrs_item)"
      )

      required_fe <- unique(fe_terms)
      missing <- setdiff(required_fe, names(df))
      if (length(missing)) {
        stop(sprintf("Missing columns for FE design: %s", paste(missing, collapse = ", ")))
      }

      fe_formula_str   <- paste("abs_error ~", paste(fe_terms, collapse = " + "))
      full_formula_str <- paste(fe_formula_str, "+", paste(re_parts, collapse = " + "))

      # Save design (FE only) for auditing
      mm <- stats::model.matrix(
        object = stats::as.formula(sub("^abs_error\\s*~", "~", fe_formula_str)),
        data   = df,
        contrasts.arg = NULL
      )

      export_df <- cbind(
        data.frame(
          abs_error = df$abs_error,
          patient   = df$patient,
          session   = df$session,
          model_name = df$model_name,
          madrs_item = df$madrs_item
        ),
        as.data.frame(mm)
      )

      dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)
      utils::write.csv(export_df, csv_path, row.names = FALSE)
      message(sprintf("Saved per-row FE design matrix & IDs to: %s", csv_path))

      list(
        fe_terms = fe_terms,
        fe_formula_string = fe_formula_str,
        full_formula_string = full_formula_str,
        design_matrix_path = csv_path
      )
    },

    # =========================
    # Modeling
    # =========================
    # =========================
# Stratified fits (non-reasoning vs reasoning), with FE pruning
# =========================
  fit_stratified_models_items = function() {
    private$require_data("items")
    df <- self$items_data

    # strata
    df_r <- df %>% dplyr::filter(.data$is_reasoning == 0)
    df_nr  <- df %>% dplyr::filter(.data$is_reasoning == 1)

    # candidate FE terms (match your design)
    wc_terms   <- c("session_item_severity_patitem_wc_z",
                    "log_tokens_pat_wc_z")
    mean_terms <- c("session_item_severity_patitem_mean_z",
                    "log_tokens_pat_mean_z")
    reason_terms <- c("log_reason_tokens_patitem_mean_z",
                    "log_reason_tokens_patitem_wc_z")
    fe_base    <- unique(c(mean_terms, wc_terms, reason_terms,
                          "log_params_z", "log_context_length_z",
                          "architecture"))

    # RE parts
    re_parts <- c("(1 | model_name)",
                  "(1 | patient)",
                  "(1 | session:patient)",
                  "(1 | madrs_item)")

    # prune FE per stratum
    keep_nr <- private$prune_fe_terms(df_nr, fe_base)

        fe_base    <- unique(c(mean_terms, wc_terms,
                          "log_params_z", "log_context_length_z",
                          "architecture"))

    # RE parts
    re_parts <- c("(1 | model_name)",
                  "(1 | patient)",
                  "(1 | session:patient)",
                  "(1 | madrs_item)")
    keep_r  <- private$prune_fe_terms(df_r,  fe_base)
    print(fe_base)
    cat("\n[NON-REASONING] FE terms:\n")
    print(keep_nr$fe_terms)
    cat("[REASONING] FE terms:\n")
    print(keep_r$fe_terms)

    f_nr <- private$build_formula("abs_error", keep_nr$fe_terms, re_parts)
    f_r  <- private$build_formula("abs_error", keep_r$fe_terms,  re_parts)

    fit_one <- function(formula, data) {
      fit0 <- lme4::lmer(
        formula = formula, data = data, REML = self$reml,
        control = lme4::lmerControl(optimizer = "Nelder_Mead",
                                    optCtrl = list(maxfun = max(5e4, self$maxiter/4)))
      )
      suppressWarnings(
        lme4::refit(fit0,
                    control = lme4::lmerControl(optimizer = "bobyqa",
                                                optCtrl   = list(maxfun = self$maxiter)))
      )
    }

    cat("Fitting NON-REASONING model...\n")
    fit_nr <- fit_one(f_nr, df_nr)
    cat("Fitting REASONING model...\n")
    fit_r  <- fit_one(f_r,  df_r)

    cat("isSingular(non-reason):", lme4::isSingular(fit_nr), "\n")
    cat("isSingular(reason):    ", lme4::isSingular(fit_r),  "\n\n")

    self$models$items$nonreason <- fit_nr
    self$models$items$reason    <- fit_r
    invisible(list(nonreason = fit_nr, reason = fit_r))
  },


    # =========================
    # Variance Decomposition
    # =========================
    variance_decomposition_items = function() {
  fits <- c("nonreason", "reason")
  rows <- list()

  # helper to safely read a group's variance (sum diag of VarCorr matrix)
  get_group_var <- function(vc, group_names) {
    # try provided names in order; return 0 if none exist
    for (gn in group_names) {
      if (gn %in% names(vc)) {
        return(sum(diag(as.matrix(vc[[gn]]))))
      }
    }
  }

  for (fit_name in fits) {
    fit <- self$models$items[[fit_name]]
    if (is.null(fit)) stop(sprintf("No fitted model for '%s'. Run fit_stratified_models_items() first.", fit_name))

    vc <- lme4::VarCorr(fit)
    var_resid <- stats::sigma(fit)^2

    # allow either "(1 | session:patient)" or "(1 | session)" schemas
    var_model   <- get_group_var(vc, c("model_name"))
    var_patient <- get_group_var(vc, c("patient"))
    var_session <- get_group_var(vc, c("session:patient", "session"))
    var_task    <- get_group_var(vc, c("madrs_item"))

    var_total <- var_model + var_patient + var_session + var_task + var_resid

    rows[[fit_name]] <- data.frame(
      fit         = fit_name,
      var_model   = var_model,
      var_patient = var_patient,
      var_session = var_session,
      var_task    = var_task,
      var_residual= var_resid,
      var_total   = var_total,
      pct_model   = 100 * var_model   / var_total,
      pct_patient = 100 * var_patient / var_total,
      pct_session = 100 * var_session / var_total,
      pct_task    = 100 * var_task    / var_total,
      pct_residual= 100 * var_resid   / var_total,
      row.names   = NULL
    )
  }

  out_df <- dplyr::bind_rows(rows)

  # also return as nested named lists to match your existing usage style
  to_named_list <- function(col) {
    x <- as.list(stats::setNames(out_df[[col]], out_df$fit))
    # ensure plain numerics (not data frames)
    lapply(x, as.numeric)
  }

  list(
    # per-component raw variances
    var_model    = to_named_list("var_model"),
    var_patient  = to_named_list("var_patient"),
    var_session  = to_named_list("var_session"),
    var_task     = to_named_list("var_task"),
    var_residual = to_named_list("var_residual"),
    var_total    = to_named_list("var_total"),

    # per-component percentages
    pct_model    = to_named_list("pct_model"),
    pct_patient  = to_named_list("pct_patient"),
    pct_session  = to_named_list("pct_session"),
    pct_task     = to_named_list("pct_task"),
    pct_residual = to_named_list("pct_residual"),

    # convenient tidy frame for plotting
    table        = out_df
  )
},

    # =========================
    # Reasoning vs Non-Reasoning (paired by patient)
    # =========================
    

    # =========================
    # Diagnostics (classic + prediction checks)
    # =========================
    plot_diagnostics_items = function(output_dir = "../output/diagnostics_items") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  save_png <- function(path, expr, width = 800, height = 600) {
    png(path, width = width, height = height)
    on.exit(dev.off(), add = TRUE)
    force(expr)
  }

  for (fit_name in c("nonreason", "reason")) {
    res <- self$models$items[[fit_name]]
    if (is.null(res)) {
      warning(sprintf("Skipping '%s': no fitted model found.", fit_name))
      next
    }

    # Use the model's own frame to avoid row-order or filtering mismatches
    mf   <- tryCatch(stats::model.frame(res), error = function(e) NULL)
    if (is.null(mf)) {
      warning(sprintf("Skipping '%s': could not retrieve model.frame().", fit_name))
      next
    }

    y     <- stats::model.response(mf)               # observed (abs_error)
    mu    <- stats::fitted(res)                      # fitted
    resid <- y - mu
    sig   <- stats::sigma(res)

    # 1) Residuals histogram
    save_png(file.path(output_dir, sprintf("residual_hist_%s.png", fit_name)), {
      hist(resid, breaks = 40, main = sprintf("Residuals Histogram (%s)", fit_name),
           xlab = "Residual", col = "lightblue", border = "white")
    })

    # 2) Residuals QQ
    save_png(file.path(output_dir, sprintf("residual_qq_%s.png", fit_name)), {
      qqnorm(resid, main = sprintf("Residuals Q-Q Plot (%s)", fit_name))
      qqline(resid, col = "red")
    })

    # 3) Posterior predictive–style check (parametric replicate)
    set.seed(123)
    y_rep <- mu + stats::rnorm(length(mu), 0, sig)

    save_png(file.path(output_dir, sprintf("ppc_hist_%s.png", fit_name)), {
      hist(y,     breaks = 40, col = rgb(0,0,1,0.5),  border = "white",
           main = sprintf("Posterior Predictive Check (%s)", fit_name),
           xlab = "Absolute Error", freq = FALSE)
      hist(y_rep, breaks = 40, col = rgb(1,0,0,0.3), border = "white",
           add = TRUE, freq = FALSE)
      legend("topright", c("Observed", "Replicated"),
             fill = c(rgb(0,0,1,0.5), rgb(1,0,0,0.3)), bty = "n")
    })

    cat(sprintf("Saved diagnostics for %s to %s/\n", fit_name, output_dir))
  }
},

run_prediction_checks = function(output_base = "../output/diagnostics_items/check_predictions") {
  dir.create(dirname(output_base), recursive = TRUE, showWarnings = FALSE)

  save_one <- function(plot_obj, path_base) {
    # handle single ggplot or patchwork
    if (inherits(plot_obj, "ggplot") || inherits(plot_obj, "patchwork")) {
      ggplot2::ggsave(filename = paste0(path_base, ".png"),
                      plot = plot_obj, width = 7.2, height = 5.0, dpi = 300)
      message(sprintf("Saved prediction checks to '%s.png'", path_base))
      return(invisible(TRUE))
    }

    # handle list of plots
    if (is.list(plot_obj) && length(plot_obj) > 0) {
      for (i in seq_along(plot_obj)) {
        if (inherits(plot_obj[[i]], "ggplot") || inherits(plot_obj[[i]], "patchwork")) {
          ggplot2::ggsave(filename = sprintf("%s_%02d.png", path_base, i),
                          plot = plot_obj[[i]], width = 7.2, height = 5.0, dpi = 300)
        }
      }
      message(sprintf("Saved multi-panel prediction checks to '%s_XX.png'", path_base))
      return(invisible(TRUE))
    }

    # fallback device
    png(filename = paste0(path_base, "_fallback.png"), width = 720, height = 500)
    on.exit(dev.off(), add = TRUE)
    print(plot_obj)
    message(sprintf("Saved prediction checks to '%s_fallback.png' (fallback renderer)", path_base))
    invisible(TRUE)
  }

  for (fit_name in c("nonreason", "reason")) {
    fit <- self$models$items[[fit_name]]
    if (is.null(fit)) {
      warning(sprintf("Skipping '%s': no fitted model found.", fit_name))
      next
    }

    # Compute diagnostics via performance::check_predictions()
    perf_obj <- performance::check_predictions(fit)

    # Turn into ggplot object(s) via {see}
    g <- tryCatch(see::plot(perf_obj), error = function(e) {
      warning(sprintf("plot(check_predictions()) failed for '%s': %s", fit_name, e$message))
      NULL
    })

    if (!is.null(g)) {
      save_one(g, sprintf("%s_%s", output_base, fit_name))
    }
  }

  invisible(output_base)
},

# BLUPs (items) via modelbased — stratified (nonreason / reason)
# =========================
plot_task_blups = function(output_base = "../output/blups_task", csv_base = "../output/blups_task") {
  if (!self$use_item_random_intercept) {
    message("No task random intercept in this model. Skipping BLUP plots.")
    return(invisible(NULL))
  }

  # --- helpers ---
  norm_cols <- function(x) {
    names(x) <- tolower(gsub("\\s+", "_", names(x)))
    x
  }
  pick_madrs <- function(d) {
    d <- norm_cols(d)
    req <- c("group", "level", "coefficient")
    if (!all(req %in% names(d))) {
      stop("Unexpected columns from estimate_grouplevel(): ",
           paste(names(d), collapse = ", "))
    }
    out <- dplyr::filter(d, .data$group == "madrs_item")
    tibble::tibble(
      madrs_item = as.character(out$level),
      estimate   = out$coefficient,
      ci_low     = if ("ci_low"  %in% names(out)) out$ci_low  else NA_real_,
      ci_high    = if ("ci_high" %in% names(out)) out$ci_high else NA_real_
    )
  }

  item_labels <- c(
    "0"="Item 0: Total Score","1"="Item 1: Apparent Sadness","2"="Item 2: Reported Sadness",
    "3"="Item 3: Inner Tension","4"="Item 4: Reduced Sleep","5"="Item 5: Reduced Appetite",
    "6"="Item 6: Concentration Difficulties","7"="Item 7: Lassitude","8"="Item 8: Inability to Feel",
    "9"="Item 9: Pessimistic Thoughts","10"="Item 10: Suicidal Thoughts"
  )

  for (fit_name in c("nonreason", "reason")) {
    fit <- self$models$items[[fit_name]]
    if (is.null(fit)) {
      warning(sprintf("Skipping '%s': no fitted model.", fit_name))
      next
    }

    # Ensure the model actually has random intercepts for madrs_item
    re_names <- names(lme4::VarCorr(fit))
    if (!("madrs_item" %in% re_names)) {
      warning(sprintf("Skipping '%s': model has no random intercept for madrs_item.", fit_name))
      next
    }

    # ---- compute totals and random deviations ----
    # NOTE: For lme4 models, type="total" usually has no CI (that's expected).
    blups_total  <- modelbased::estimate_grouplevel(fit, type = "total")
    blups_random <- modelbased::estimate_grouplevel(fit, type = "random")

    total_i  <- pick_madrs(blups_total)
    random_i <- pick_madrs(blups_random)
    names(random_i) <- c("madrs_item","dev","dev_low","dev_high")

    df <- dplyr::left_join(total_i, random_i, by = "madrs_item") %>%
      dplyr::arrange(.data$estimate)

    df$label <- factor(item_labels[df$madrs_item], levels = item_labels[df$madrs_item])

    # --- write CSV for auditing ---
    csv_path <- sprintf("%s_%s_total_and_random.csv", csv_base, fit_name)
    dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)
    utils::write.csv(df, csv_path, row.names = FALSE)
    message(sprintf("Wrote item BLUPs to '%s'", csv_path))

    # --- Plot 1: BLUP totals (CIs may be NA) ---
    p1 <- ggplot2::ggplot(df, ggplot2::aes(x = estimate, y = label)) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, alpha = 0.6) +
      { if (all(!is.na(df$ci_low))) ggplot2::geom_errorbar(
          ggplot2::aes(xmin = ci_low, xmax = ci_high), height = 0, alpha = 0.7
        ) else ggplot2::geom_blank() } +
      ggplot2::geom_point(size = 2.3, ggplot2::aes(color = estimate > 0)) +
      ggplot2::scale_color_manual(values = c(`TRUE` = "#d62728", `FALSE` = "#1f77b4"), guide = "none") +
      ggplot2::labs(
        x = "BLUP (group-level total)",
        y = NULL,
        title = sprintf("Item-level BLUPs — %s", fit_name),
        subtitle = "Totals (lme4 totals typically have no CI)"
      ) +
      ggplot2::theme_minimal(base_size = 11)

    out_tot <- sprintf("%s_%s.png", sub("\\.png$", "", output_base, ignore.case = TRUE), fit_name)
    dir.create(dirname(out_tot), recursive = TRUE, showWarnings = FALSE)
    ggplot2::ggsave(filename = out_tot, plot = p1, width = 7.0, height = 4.8, dpi = 300)
    message(sprintf("Saved BLUP totals plot to '%s'", out_tot))

    # --- Plot 2: random-effect deviations (has CI) ---
    p2 <- ggplot2::ggplot(df, ggplot2::aes(x = dev, y = label)) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, alpha = 0.6) +
      { if (all(!is.na(df$dev_low))) ggplot2::geom_errorbar(
          ggplot2::aes(xmin = dev_low, xmax = dev_high), height = 0, alpha = 0.7
        ) else ggplot2::geom_blank() } +
      ggplot2::geom_point(size = 2.3, ggplot2::aes(color = dev > 0)) +
      ggplot2::scale_color_manual(values = c(`TRUE` = "#d62728", `FALSE` = "#1f77b4"), guide = "none") +
      ggplot2::labs(
        x = "Random-effect deviation",
        y = NULL,
        title = sprintf("Item Random-Effect Deviations — %s", fit_name),
        subtitle = "Modelbased (random) with 95% CI"
      ) +
      ggplot2::theme_minimal(base_size = 11)

    out_dev <- sprintf("%s_%s_random_deviation.png",
                       sub("\\.png$", "", output_base, ignore.case = TRUE), fit_name)
    ggplot2::ggsave(filename = out_dev, plot = p2, width = 7.0, height = 4.8, dpi = 300)
    message(sprintf("Saved BLUP random-deviation plot to '%s'", out_dev))
  }

  invisible(TRUE)
},
    save_analyzer = function(path = "../output/analyzer.rds") {
      dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
      saveRDS(self, file = path)
      cat(sprintf("✓ Analyzer saved to: %s\n", path))
      cat("  Includes: data, models, and configuration\n")
      invisible(path)
    },

generate_report_items = function(output_path = "../output/error_analysis_items.txt") {
  # make sure the two stratified fits exist
  if (is.null(self$models$items$reason) || is.null(self$models$items$nonreason)) {
    if (!exists("self$fit_stratified_models_items") && !is.function(self$fit_stratified_models_items)) {
      stop("No fitted models found and fit_stratified_models_items() is not available.")
    }
    self$fit_stratified_models_items()
  }

  # open sink and guarantee closing even if an error occurs
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(output_path, open = "wt")
  sink(con); on.exit({ try(sink(), silent = TRUE); try(close(con), silent = TRUE) }, add = TRUE)

  cat(strrep("=", 80), "\n", sep = "")
  cat("MADRS ERROR ANALYSIS — TASK-LEVEL (SEED-AVERAGED)\n")
  cat(strrep("=", 80), "\n\n", sep = "")

  # basic counts
  cat(sprintf("Observations: %d\n", nrow(self$items_data)))
  cat(sprintf("Patients:     %d\n", dplyr::n_distinct(self$items_data$patient)))
  cat(sprintf("Models:       %d\n", dplyr::n_distinct(self$items_data$model_name)))
  cat(sprintf("Sessions:     %d\n\n", dplyr::n_distinct(self$items_data$session)))

  # ----------------------------------------
  # Per-model summaries (reason / nonreason)
  # ----------------------------------------
  for (model_name in c("nonreason", "reason")) {
    fit <- self$models$items[[model_name]]
    if (is.null(fit)) {
      cat(sprintf("[WARN] %s fit is missing. Skipping summary.\n\n", model_name))
      next
    }

    cat(strrep("-", 80), "\n", sep = "")
    cat(sprintf("MODEL SUMMARY — %s\n", toupper(model_name)))
    cat(strrep("-", 80), "\n", sep = "")
    print(summary(fit))
    cat("\n")

    # add a brief convergence/singularity note (useful for logs)
    conv <- tryCatch(summary(fit)$optinfo$conv$lme4$messages, error = function(e) NULL)
    cat("Convergence messages:\n")
    if (is.null(conv)) cat("  <none>\n") else print(conv)
    cat(sprintf("isSingular(fit): %s\n\n", lme4::isSingular(fit)))
  }

  # ----------------------------------------
  # Variance decomposition (stratified)
  # ----------------------------------------
  cat(strrep("-", 80), "\n", sep = "")
  cat("VARIANCE DECOMPOSITION (shares by source)\n")
  cat(strrep("-", 80), "\n", sep = "")

  var <- self$variance_decomposition_items()
  # var is a list of lists keyed by fit_name per your stratified function
  # assemble a neat table:
  try({
    fits <- intersect(names(var$pct_model), c("nonreason", "reason"))
    if (length(fits) > 0) {
      header <- sprintf("%-12s %-10s %-10s %-10s %-10s %-10s\n",
                        "Fit", "Model", "Patient", "Session", "Task", "Residual")
      cat(header)
      cat(strrep("-", nchar(header)), "\n", sep = "")

      for (fn in fits) {
        cat(sprintf("%-12s %-10.2f %-10.2f %-10.2f %-10.2f %-10.2f\n",
                    fn,
                    as.numeric(var$pct_model[[fn]]),
                    as.numeric(var$pct_patient[[fn]]),
                    as.numeric(var$pct_session[[fn]]),
                    as.numeric(var$pct_task[[fn]]),
                    as.numeric(var$pct_residual[[fn]])))
      }
      cat("\n")
    } else {
      cat("No variance decomposition available.\n\n")
    }
  }, silent = TRUE)

  # ----------------------------------------
  # Paired by-patient reasoning effect (ΔMAE)
  # ----------------------------------------
  cat(strrep("-", 80), "\n", sep = "")
  cat("REASONING EFFECT (PAIRED T-TEST BY PATIENT)\n")
  cat(strrep("-", 80), "\n", sep = "")

  ttest_res <- try(self$test_reasoning_effect_items(), silent = TRUE)
  if (inherits(ttest_res, "try-error") || is.null(ttest_res)) {
    cat("Insufficient overlapping patients for paired test or error computing test.\n\n")
  } else {
    pretty <- c(
      "n_patients", "reasoning_mae", "non_reasoning_mae",
      "mean_difference", "t_statistic", "p_value",
      "cohens_d", "ci_95_lower", "ci_95_upper"
    )
    for (k in pretty) if (!is.null(ttest_res[[k]])) {
      cat(sprintf("%-22s: %s\n", k, format(ttest_res[[k]])))
    }
    cat("\n")
  }

  cat(sprintf("Item-level report saved to %s\n", output_path))
  invisible(output_path)
}),

  # =========================
  # Private helpers
  # =========================
  private = list(
    # --- helpers for pruning & formula assembly ------------------------
has_variation_ = function(x) {
  ux <- unique(x[!is.na(x)])
  length(ux) >= 2
},

prune_fe_terms = function(data, fe_terms) {
  keep <- logical(length(fe_terms))
  dropped <- character(0)

  for (i in seq_along(fe_terms)) {
    trm <- fe_terms[i]
    if (!trm %in% names(data)) {
      dropped <- c(dropped, sprintf("%s (missing)", trm))
      next
    }
    v <- data[[trm]]

    if (is.numeric(v)) {
      if (private$has_variation_(v) && !isTRUE(all(v == 0 | is.na(v)))) {
        keep[i] <- TRUE
      } else dropped <- c(dropped, sprintf("%s (no variation / all 0)", trm))
    } else if (is.factor(v) || is.character(v)) {
      if (private$has_variation_(v)) keep[i] <- TRUE
      else dropped <- c(dropped, sprintf("%s (single level)", trm))
    } else {
      # last resort — try model.matrix()
      mm_ok <- FALSE
      try({
        mm <- stats::model.matrix(~ v)
        mm_ok <- ncol(mm) > 1
      }, silent = TRUE)
      if (mm_ok) keep[i] <- TRUE else dropped <- c(dropped, sprintf("%s (unusable)", trm))
    }
  }

  list(fe_terms = fe_terms[keep], dropped = dropped)
},

build_formula = function(response, fe_terms, re_terms) {
  fe <- if (length(fe_terms)) paste(fe_terms, collapse = " + ") else "1"
  stats::as.formula(paste(response, "~", fe, "+", paste(re_terms, collapse = " + ")))
},

    reshape_to_long = function(df, seeds = 3) {
      id_vars <- c("session", "patient", "visit_no", "visit_day", "edu", "age",
                   "gender", "rater", "diagnostic", "model_name", "model_family",
                   "architecture", "context_length", "is_reasoning_model",
                   "active_params", "total_params", "num_trans_tokens",
                   "ground_truth", "madrs_item")
    

      id_vars_present <- intersect(id_vars, names(df))
      long_parts <- list()

      for (s in 0:(seeds - 1)) {
        r_col <- paste0("rating_", s)
        e_col <- paste0("error_", s)
        if (!(r_col %in% names(df) && e_col %in% names(df))) next

        tmp <- df[, c(id_vars_present, r_col, e_col)]

        # per-seed reasoning length if available, else fallback
        nrt_col <- paste0("num_reason_tokens_", s)
        if (nrt_col %in% names(df)) {
          tmp$num_reason_tokens <- df[[nrt_col]]
        } else {
          tmp$num_reason_tokens <- 0
        }

        tmp$seed   <- s
        tmp$rating <- tmp[[r_col]]
        tmp$error  <- tmp[[e_col]]

        # Scale total score (item = 0) to 0-6
        idx_total <- tmp$madrs_item == 0
        tmp$rating[idx_total] <- tmp$rating[idx_total] / 10.0
        tmp$error[idx_total]  <- tmp$error[idx_total] / 10.0

        # missing → large error placeholder
        missing_idx <- is.na(tmp$rating) | is.na(tmp$error)
        tmp$error[missing_idx] <- 6

        tmp <- tmp[, !(names(tmp) %in% c(r_col, e_col))]
        long_parts[[length(long_parts) + 1]] <- tmp
      }

      if (length(long_parts) == 0) stop("No rating/error columns found.")

      out <- dplyr::bind_rows(long_parts)
      out$abs_error <- abs(out$error)

      # Scale ground_truth for total score
      idx_total <- out$madrs_item == 0
      out$ground_truth[idx_total] <- out$ground_truth[idx_total] / 10.0

      out
    },

    engineer_features = function(df, task) {
      out <- df

      # Model params
      if ("total_params" %in% names(out)) {
        out$params_numeric <- as.numeric(gsub("B", "", out$total_params))
        out$log_params <- log10(out$params_numeric * 1e9)
      }

      # Context length
      if ("context_length" %in% names(out)) {
        out$log_context_length <- log10(as.numeric(out$context_length) + 1)
      }

      # Transcript tokens
      if ("num_trans_tokens" %in% names(out)) {
        out$log_tokens <- log10(as.numeric(out$num_trans_tokens) + 1)
      }

      # Reasoning tokens (per-row)
      if ("num_reason_tokens" %in% names(out)) {
        out$log_reason_tokens <- log10(as.numeric(out$num_reason_tokens) + 1)

        if (all(c("patient","madrs_item") %in% names(out))) {
          out <- out %>%
            dplyr::group_by(patient, madrs_item) %>%
            dplyr::mutate(
              log_reason_tokens_patitem_mean = mean(log_reason_tokens, na.rm = TRUE),
              log_reason_tokens_patitem_wc   = log_reason_tokens - log_reason_tokens_patitem_mean
            ) %>% dplyr::ungroup()
        }
      }

      # Reasoning flag
      if ("is_reasoning_model" %in% names(out)) {
        out$is_reasoning <- as.integer(out$is_reasoning_model == "Yes")
      }

      # Session-item severity
      if (all(c("session", "madrs_item", "ground_truth") %in% names(out))) {
        out <- out %>%
          dplyr::group_by(session, madrs_item) %>%
          dplyr::mutate(session_item_severity = mean(ground_truth, na.rm = TRUE)) %>%
          dplyr::ungroup()
      }

      out <- private$add_standardizations(out, task = task)
      out
    },

    add_standardizations = function(df, task) {
      out <- df
      standardize_if_ok <- function(x) {
        mu <- mean(x, na.rm = TRUE)
        sd <- stats::sd(x, na.rm = TRUE)
        if (is.na(sd) || sd == 0) return(rep(0, length(x)))
        (x - mu) / sd
      }
      model_vars <- c("log_params", "log_context_length")

      for (var in c(model_vars)) {
        if (var %in% names(out)) out[[paste0(var, "_z")]] <- standardize_if_ok(out[[var]])
      }

      # Keep within/between for transcript tokens + item severity (and reasoning length)
      if (self$use_within_between) {
        if (all(c("patient", "log_tokens") %in% names(out))) {
          out <- out %>%
            dplyr::group_by(patient) %>%
            dplyr::mutate(
              log_tokens_pat_mean = mean(log_tokens, na.rm = TRUE),
              log_tokens_pat_wc   = log_tokens - log_tokens_pat_mean
            ) %>% dplyr::ungroup()
        }
        if (all(c("patient", "madrs_item", "session_item_severity") %in% names(out))) {
          out <- out %>%
            dplyr::group_by(patient, madrs_item) %>%
            dplyr::mutate(
              session_item_severity_patitem_mean = mean(session_item_severity, na.rm = TRUE),
              session_item_severity_patitem_wc   = session_item_severity - session_item_severity_patitem_mean
            ) %>% dplyr::ungroup()
        }

        for (nm in c("log_tokens_pat_wc", "log_tokens_pat_mean",
                     "session_item_severity_patitem_wc", "session_item_severity_patitem_mean",
                     "log_reason_tokens_patitem_wc", "log_reason_tokens_patitem_mean")) {
          if (nm %in% names(out)) out[[paste0(nm, "_z")]] <- standardize_if_ok(out[[nm]])
        }
      } else {
        session_vars <- c("log_tokens", "session_item_severity", "log_reason_tokens")
        for (var in session_vars) {
          if (var %in% names(out)) out[[paste0(var, "_z")]] <- standardize_if_ok(out[[var]])
        }
      }

      out
    },

    average_over_seeds = function(df) {
      id_cols    <- c("patient", "session", "model_name", "madrs_item")
      value_cols <- c("rating", "error", "abs_error", "num_reason_tokens")
      invariant_cols <- c(
        "visit_no", "visit_day", "edu", "age", "gender", "rater", "diagnostic",
        "model_family", "architecture", "context_length", "is_reasoning_model",
        "active_params", "total_params", "num_trans_tokens", "ground_truth",
        "params_numeric", "log_params", "log_context_length", "log_tokens",
        "is_reasoning", "session_item_severity"
      )

      id_cols        <- intersect(id_cols, names(df))
      value_cols     <- intersect(value_cols, names(df))
      invariant_cols <- setdiff(intersect(invariant_cols, names(df)), id_cols)

      collapsed <- df %>%
        dplyr::group_by(dplyr::across(all_of(id_cols))) %>%
        dplyr::summarise(
          dplyr::across(all_of(value_cols), ~mean(.x, na.rm = TRUE)),
          dplyr::across(all_of(invariant_cols), ~dplyr::first(.x)),
          n_seeds = dplyr::n(),
          .groups = "drop"
        )

      cat(sprintf("Collapsing seeds: median=%d, min=%d, max=%d\n",
                  stats::median(collapsed$n_seeds), min(collapsed$n_seeds), max(collapsed$n_seeds)))
      collapsed$n_seeds <- NULL

      cat(sprintf("After collapsing: %d patients, %d sessions, %d models, %d items.\n\n",
                  dplyr::n_distinct(collapsed$patient),
                  dplyr::n_distinct(collapsed$session),
                  dplyr::n_distinct(collapsed$model_name),
                  dplyr::n_distinct(collapsed$madrs_item)))
      print(utils::head(collapsed))
      collapsed
    },

    require_data = function(which) {
      if (which == "items") {
        if (is.null(self$items_data) || nrow(self$items_data) == 0)
          stop("items_data empty. Run load_and_prepare_items_data() first.")
      } else if (which == "full") {
        if (is.null(self$full_data) || nrow(self$full_data) == 0)
          stop("full_data empty. Run load_and_prepare_full_data() first.")
      }
    },

    get_item_name = function(idx) {
      names <- c(
        "Total Score", "Apparent Sadness", "Reported Sadness", "Inner Tension",
        "Reduced Sleep", "Reduced Appetite", "Concentration Difficulties",
        "Lassitude", "Inability to Feel", "Pessimistic Thoughts", "Suicidal Thoughts"
      )
      if (idx >= 0 && idx <= 10) names[idx + 1] else sprintf("Item %d", idx)
    }
  ))
# ==============================================================================
# Main Script with Smart Load/Save
# ==============================================================================
 if (interactive() || !exists("sourced")) {

  output_dir <- "../output"
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  analyzer_path <- file.path(output_dir, "analyzer.rds")

  if (file.exists(analyzer_path)) {
    cat("===========================================\n")
    cat("Found saved analyzer. Load it? (y/n): ")
    response <- "n"
    cat("===========================================\n\n")

    if (response == "y") {
      cat("Loading saved analyzer...\n")
      analyzer <- readRDS(analyzer_path)
      cat("✓ Loaded! Skipping data prep and model fitting.\n\n")
    } else {
      cat("Starting fresh analysis...\n")
      analyzer <- MADRSErrorAnalysis$new(
        file.path(output_dir, "llamadrs_predictions.xlsx"),
        use_within_between = TRUE,
        use_item_random_intercept = TRUE,
        reml = TRUE,
        maxiter = 2e5,
        random_slope_terms = c(
          "session_item_severity_patitem_wc_z",
          "log_tokens_pat_wc_z",
          "log_reason_tokens_patitem_wc_z"
        ),
        aggregate_seeds = TRUE
      )
      cat("Loading and preparing task-level data...\n")
      analyzer$load_and_prepare_items_data(item_indices = 1:10, seeds = 3)

      cat("Fitting stratified models (NON-REASONING / REASONING)...\n")
      analyzer$fit_stratified_models_items()  # << use your stratified fitter

      cat("\nSaving analyzer for future use...\n")
      analyzer$save_analyzer(analyzer_path)
    }
  } else {
    cat("Starting fresh analysis...\n")
    analyzer <- MADRSErrorAnalysis$new(
      file.path(output_dir, "llamadrs_predictions.xlsx"),
      use_within_between = TRUE,
      use_item_random_intercept = TRUE,
      reml = TRUE,
      maxiter = 2e5,
      random_slope_terms = c(
        "session_item_severity_patitem_wc_z",
        "log_tokens_pat_wc_z",
        "log_reason_tokens_patitem_wc_z"
      ),
      aggregate_seeds = TRUE
    )
    cat("Loading and preparing task-level data...\n")
    analyzer$load_and_prepare_items_data(item_indices = 1:10, seeds = 3)

    cat("Fitting stratified models (NON-REASONING / REASONING)...\n")
    analyzer$fit_stratified_models_items()  # << use your stratified fitter

    cat("\nSaving analyzer for future use...\n")
    analyzer$save_analyzer(analyzer_path)
  }

  # -----------------------------
  # Outputs (for both fits)
  # -----------------------------
  cat("\nSaving diagnostics (per fit)...\n")
  analyzer$plot_diagnostics_items(output_dir = file.path(output_dir, "diagnostics_items"))

  cat("Running prediction checks (per fit)...\n")
  # reuse your existing run_prediction_checks() by temporarily pointing
  # analyzer$models$items$interaction at each fit (nonreason, then reason)
  # so you don't have to write a new function.
  old_interaction <- analyzer$models$items$interaction

  analyzer$models$items$interaction <- analyzer$models$items$nonreason
  analyzer$run_prediction_checks(output_base = file.path(output_dir, "diagnostics_items/check_predictions_nonreason"))

  analyzer$models$items$interaction <- analyzer$models$items$reason
  analyzer$run_prediction_checks(output_base = file.path(output_dir, "diagnostics_items/check_predictions_reason"))

  analyzer$models$items$interaction <- old_interaction

  cat("Generating report (both fits + variance + paired test)...\n")
  analyzer$generate_report_items(file.path(output_dir, "error_analysis_items.txt"))

  cat("Plotting task BLUPs (per fit)...\n")
  # Same trick as above: reuse plot_task_blups() for each fit
  old_interaction <- analyzer$models$items$interaction

  analyzer$models$items$interaction <- analyzer$models$items$nonreason
  analyzer$plot_task_blups(output_base = file.path(output_dir, "blups_task_nonreason"),
                           csv_base    = file.path(output_dir, "blups_task_nonreason"))

  analyzer$models$items$interaction <- analyzer$models$items$reason
  analyzer$plot_task_blups(output_base = file.path(output_dir, "blups_task_reason"),
                           csv_base    = file.path(output_dir, "blups_task_reason"))

  analyzer$models$items$interaction <- old_interaction

  # Figures (overlay both fits)
  cat("Generating overlay figures...\n")
  source("mixed_effects_figures.R")
  make_full_plots_overlay(
    fit_nonreason = analyzer$models$items$nonreason,
    fit_reason    = analyzer$models$items$reason,
    outdir        = file.path(output_dir, "madrs_figs/items")
  )

  cat("\n✓ Done!\n")
}
