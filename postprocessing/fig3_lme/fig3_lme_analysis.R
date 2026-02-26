library(R6)
library(readxl)
library(dplyr)
library(lme4)
library(ggplot2)
# NEW
library(modelbased)   # estimate_grouplevel()
MADRSErrorAnalysis <- R6Class(
  "MADRSErrorAnalysis",

  public = list(
    excel_path = NULL,
    use_within_between = TRUE,
    reml = TRUE,
    maxiter = 2e5,
    aggregate_seeds = TRUE,
    items_data = NULL,
    models = list(
      items = list(
        nonreason = NULL,
        reason    = NULL
      )
    ),

    initialize = function(excel_path,
                          use_within_between = TRUE,
                          reml = TRUE,
                          maxiter = 2e5,
                          aggregate_seeds = TRUE) {
      self$excel_path <- excel_path
      self$use_within_between <- use_within_between
      self$reml <- reml
      self$maxiter <- maxiter
      self$aggregate_seeds <- aggregate_seeds
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

      long_df <- private$engineer_features(long_df)
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

# Stratified fits (non-reasoning vs reasoning), with FE pruning
# =========================
  fit_stratified_models_items = function() {
    private$require_data()
    df <- self$items_data

    # strata
    df_r <- df %>% dplyr::filter(.data$is_reasoning == 1)
    df_nr  <- df %>% dplyr::filter(.data$is_reasoning == 0)

    # candidate FE terms (match your design)
    wc_terms   <- c("session_item_severity_patitem_wc_z",
                    "log_tokens_pat_wc_z")
    mean_terms <- c("session_item_severity_patitem_mean_z",
                    "log_tokens_pat_mean_z")
    reason_terms <- c("log_reason_tokens_patitem_mean_z",
                    "log_reason_tokens_patitem_wc_z")
    fe_base_r    <- unique(c(mean_terms, wc_terms, reason_terms,
                          "log_params_z", "log_context_length_z",
                          "architecture"))

    # RE parts
    re_parts <- c("(1 | model_name)",
                  "(1 | patient)",
                  "(1 | session:patient)",
                  "(1 | madrs_item)")

    # prune FE per stratum
    keep_r <- private$prune_fe_terms(df_r, fe_base_r)

    fe_base_nr    <- unique(c(mean_terms, wc_terms,
                          "log_params_z", "log_context_length_z",
                          "architecture"))

    # RE parts
    re_parts <- c("(1 | model_name)",
                  "(1 | patient)",
                  "(1 | session:patient)",
                  "(1 | madrs_item)")
    keep_nr  <- private$prune_fe_terms(df_nr,  fe_base_nr)
    print(fe_base_nr)
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



# BLUPs (items) via modelbased — stratified (nonreason / reason)
# =========================
plot_task_blups = function(output_base = "../output/blups", csv_base = "../output/blups") {

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
    csv_path <- sprintf("%s_%s.csv", csv_base, fit_name)
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

    engineer_features = function(df) {
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

      out <- private$add_standardizations(out)
      out
    },

    add_standardizations = function(df) {
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

    require_data = function() {
        if (is.null(self$items_data) || nrow(self$items_data) == 0)
          stop("items_data empty. Run load_and_prepare_items_data() first.")
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
        reml = TRUE,
        maxiter = 2e5,
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
      reml = TRUE,
      maxiter = 2e5,
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

  cat("Generating report (both fits + variance + paired test)...\n")
  analyzer$generate_report_items(file.path(output_dir, "error_analysis_items.txt"))

  cat("Plotting task BLUPs (per fit)...\n")

  analyzer$plot_task_blups(output_base = file.path(output_dir, "blups"),
                           csv_base    = file.path(output_dir, "blups"))


  cat("\n✓ Done!\n")
}
