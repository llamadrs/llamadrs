# mixed_effects_figures.R
# Overlay fixed-effect and BLUP plots for stratified mixed-effects models
# Adapted from madrs_figs_fixed.R for the llamadrs postprocessing pipeline
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(broom.mixed)
  library(stringr)
  library(forcats)
  library(modelbased)
})

# ========= Palette / Theme (your settings) =========
BLUE_STRONG <- "#1f77b4"  # negative (benefit)
RED_STRONG  <- "#d62728"  # positive (harm)
GREY_STRONG <- "#000000"
GREY_WEAK   <- scales::alpha("black", 0.08)
HEADER_GRAY <- scales::alpha("black", 0.03)

SIZES_11PT <- list(
  tiny=6, scriptsize=8, footnotesize=9, small=10, normalsize=10.95,
  large=12, Large=14.4, LARGE=16, huge=20.74, Huge=24.88
)

maybe_enable_latex <- function() {
  use_tex <- nzchar(Sys.which("latex")) && nzchar(Sys.which("dvipng"))
  list(use_tex = use_tex, family = "serif")
}
theme_madrs_11pt <- function() {
  cfg <- maybe_enable_latex()
  base <- SIZES_11PT$LARGE
  theme_minimal(base_family = cfg$family, base_size = base) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = GREY_WEAK),
      panel.background = element_rect(fill = HEADER_GRAY, colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      text         = element_text(family = cfg$family, size = base, colour = GREY_STRONG),
      axis.text    = element_text(size = SIZES_11PT$LARGE, colour = GREY_STRONG),
      axis.title   = element_text(size = SIZES_11PT$LARGE, face = "bold", colour = GREY_STRONG),
      plot.title   = element_text(size = SIZES_11PT$LARGE, face = "bold", hjust = 0),
      legend.title = element_text(size = SIZES_11PT$footnotesize, face = "bold"),
      legend.text  = element_text(size = SIZES_11PT$footnotesize),
      strip.text   = element_text(size = SIZES_11PT$Large, face = "bold"),
      axis.title.y = element_text(margin = margin(r = 8)),
      axis.title.x = element_text(margin = margin(t = 6))
    )
}
theme_set(theme_madrs_11pt())

save_dual <- function(p, base, width = 7.2, height = 5.0) {
  dir.create(dirname(base), recursive = TRUE, showWarnings = FALSE)
  ggsave(paste0(base, ".png"), p, width = width, height = height, dpi = 300)
  ggsave(paste0(base, ".svg"), p, width = width, height = height)
}

item_labels <- c(
  "0" = "Total Score",
  "1" = "Item 1: Apparent Sadness",
  "2" = "Item 2: Reported Sadness",
  "3" = "Item 3: Inner Tension",
  "4" = "Item 4: Reduced Sleep",
  "5" = "Item 5: Reduced Appetite",
  "6" = "Item 6: Concentration Difficulties",
  "7" = "Item 7: Lassitude",
  "8" = "Item 8: Inability to Feel",
  "9" = "Item 9: Pessimistic Thoughts",
  "10" = "Item 10: Suicidal Thoughts"
)

# ========= Pretty term map =========
BASE_LABELS <- c(
  "architectureMoE"                      = "Mixture-of-Experts (vs Dense)",
  "log_params_z"                         = "Model size (log params, z)",
  "log_context_length_z"                 = "Context length (log, z)",

  # present in both fits:
  "session_item_severity_patitem_wc_z"   = "MADRS Item Score (within-patient, z)",
  "session_item_severity_patitem_mean_z" = "MADRS Item Score (mean, z)",
  "log_tokens_pat_wc_z"                  = "Transcript tokens (within-patient, z)",
  "log_tokens_pat_mean_z"                = "Transcript tokens (mean, z)",
  # present only in reasoning fit:
  "log_reason_tokens_patitem_wc_z"       = "Reasoning tokens (within-patient, z)",
  "log_reason_tokens_patitem_mean_z"     = "Reasoning tokens (mean, z)"
)
pretty_term <- function(x) ifelse(!is.na(BASE_LABELS[x]), BASE_LABELS[x], x)

# ========= Tidy helper (Wald CI) =========
tidy_fixed_wald <- function(fit, tag) {
  broom.mixed::tidy(fit, effects = "fixed", conf.int = FALSE) %>%
    mutate(
      conf.low  = estimate - 1.96 * std.error,
      conf.high = estimate + 1.96 * std.error,
      model_type = tag
    )
}
# ========= Build overlay data =========
prepare_overlay_df <- function(fit_nonreason, fit_reason) {
  df_nr <- tidy_fixed_wald(fit_nonreason, "Standard Models")
  df_r  <- tidy_fixed_wald(fit_reason,    "Reasoning Models")

  # keep only FE terms we want to compare across both fits
  keep_terms <- c(
    "session_item_severity_patitem_wc_z",
    "session_item_severity_patitem_mean_z",
    "log_tokens_pat_wc_z",
    "log_tokens_pat_mean_z",
    "log_params_z",
    "log_context_length_z",
    "architectureMoE",
    "log_reason_tokens_patitem_wc_z",
    "log_reason_tokens_patitem_mean_z"
  )

  # labels that must appear at the very bottom (in this order)
  reason_bottom <- c(

  "MADRS Item Score (within patient, z)",
  "MADRS Item Score (mean, z)",
    "Reasoning tokens (mean, z)",
    "Reasoning tokens (within-patient, z)"
  )

  # Build the shared dataframe and set factor levels for the final plot order
  shared <- bind_rows(df_nr, df_r) %>%
    dplyr::filter(term %in% keep_terms) %>%
    dplyr::mutate(term_label = pretty_term(term))

  # Use BASE_LABELS order for display, appending any extra terms not in BASE_LABELS
  base_order <- BASE_LABELS
  base_order <- base_order[base_order %in% shared$term_label]
  shared$term_label <- factor(shared$term_label, levels = base_order)

  shared <- shared %>%
    # Move reasoning-only terms to the bottom
    mutate(term_label = fct_relevel(term_label, reason_bottom, after = Inf)) %>%
    droplevels()
  # reasoning-only terms (exist only in the reasoning fit)
  reason_only <- df_r %>%
    dplyr::filter(term %in% c("log_reason_tokens_patitem_wc_z",
                              "log_reason_tokens_patitem_mean_z")) %>%
    dplyr::mutate(term_label = pretty_term(term)) %>%
    droplevels()
  print(shared, width = Inf)
  list(shared = shared, reason_only = reason_only)
}
plot_overlay_shared <- function(shared_df, outdir = "../output/madrs_figs/items") {
  # Labels to force to the bottom (this order)
  reason_bottom <- c(

  "MADRS Item Score (mean, z)",
  "MADRS Item Score (within patient, z)",
  "Reasoning tokens (mean, z)",
  "Reasoning tokens (within-patient, z)"
  )

  # Build final factor levels: alphabetical, then reasoning labels (if present)
  all_levels  <- sort(unique(as.character(shared_df$term_label)))
  final_levels <- c(setdiff(all_levels, reason_bottom),
                    reason_bottom[reason_bottom %in% all_levels])

  d <- shared_df %>%
    dplyr::mutate(term_label = factor(term_label, levels = final_levels)) %>%
    droplevels()

  print(d, width = Inf)

  p <- ggplot(d, aes(x = estimate, y = term_label, color = model_type)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high),
                  position = position_dodge(width = 0.5),
                  width = 0,                    # <- use width for horizontal bars
                  orientation = "y") +
    geom_point(position = position_dodge(width = 0.5), size = 2.6) +
    scale_color_manual(values = c("Standard Models" = BLUE_STRONG,
                                  "Reasoning Models" = RED_STRONG),
                       guide = "none") +
    labs(x = "Coefficients", y = NULL, color = NULL) +
    theme_madrs_11pt()

  save_dual(p, file.path(outdir, "coef_shared_overlay"),
            width = 7.2,
            height = max(3.8, 0.55 * dplyr::n_distinct(d$term_label) + 1.8))
  invisible(p)
}



# ========= Shared BLUP overlay (both fits) =========
plot_item_blups_shared <- function(fit_nonreason, fit_reason, outdir = "../output/madrs_figs/items") {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  pick_madrs <- function(d) {
    nm <- tolower(gsub("\\s+", "_", names(d)))
    names(d) <- nm
    stopifnot(all(c("group","level","coefficient") %in% names(d)))
    d %>% dplyr::filter(.data$group == "madrs_item")
  }

  # Helper to extract totals & random for a fit
  extract_blups <- function(fit, tag) {
    total  <- modelbased::estimate_grouplevel(fit, type = "total")   %>% pick_madrs()
    random <- modelbased::estimate_grouplevel(fit, type = "random")  %>% pick_madrs()

    total_df <- dplyr::transmute(total,  item = as.character(level), est = coefficient)
    random_df <- dplyr::transmute(
      random, item = as.character(level), dev = coefficient,
      lo = if ("ci_low"  %in% names(random))  ci_low  else NA_real_,
      hi = if ("ci_high" %in% names(random))  ci_high else NA_real_
    )

    dplyr::left_join(total_df, random_df, by = "item") %>%
      dplyr::mutate(model_type = tag)
  }

  df_nr <- extract_blups(fit_nonreason, "Standard Models")
  df_r  <- extract_blups(fit_reason,    "Reasoning Models")
  df    <- dplyr::bind_rows(df_nr, df_r)

  # Pretty item labels and ordering (by mean 'est' across models)
# robust numeric index for items (handles "10", "Item 10", etc.)
df <- df %>%
  dplyr::mutate(item_num = readr::parse_number(item),
                item_label = dplyr::recode(item, !!!item_labels, .default = as.character(item)))

# build the y-axis order by item number, then lock it in as factor levels
levels_by_item <- df %>%
  dplyr::distinct(item_num, item_label) %>%
  dplyr::arrange(item_num) %>%
  dplyr::pull(item_label)

df <- df %>%
  dplyr::mutate(item_label = factor(item_label, levels = levels_by_item)) %>%
  droplevels()

print(df, width = Inf)
  # --- Totals overlay ---
  p_tot <- ggplot(df, aes(x = est, y = item_label, color = model_type)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_point(position = position_dodge(width = 0.5), size = 2.4) +
    scale_color_manual(values = c("Standard Models" = BLUE_STRONG,
                                  "Reasoning Models" = RED_STRONG),
                       guide = "none") +
    labs(x = "BLUP (group-level total)", y = "MADRS item") +
    theme_madrs_11pt()
  save_dual(p_tot, file.path(outdir, "blups_items_totals_shared"), 7.2, 6.75)

  # --- Random deviations overlay (with CIs if present) ---
  p_rand <- ggplot(df, aes(x = dev, y = item_label, color = model_type)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(xmin = lo, xmax = hi),
                  position = position_dodge(width = 0.5),
                  width = 0, orientation = "y") +
    geom_point(position = position_dodge(width = 0.5), size = 2.4) +
    scale_color_manual(values = c("Standard Models" = BLUE_STRONG,
                                  "Reasoning Models" = RED_STRONG),
                       guide = "none") +
    labs(x = "Random-effect deviation", y = "MADRS item") +
    theme_madrs_11pt()
  save_dual(p_rand, file.path(outdir, "blups_items_random_shared"), 7.2, 6.75)

  invisible(list(totals = p_tot, random = p_rand))
}

# ========= Reasoning-only panel =========
plot_reasoning_only <- function(reason_only_df, outdir = "../output/madrs_figs/items") {
  if (!nrow(reason_only_df)) return(invisible(NULL))
  d <- reason_only_df %>% arrange(term_label) %>% mutate(term_label = factor(term_label, levels = term_label))
  p <- ggplot(d, aes(x = estimate, y = term_label)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(xmin = conf.low, xmax = conf.high), height = 0, colour = GREY_STRONG, orientation = "y") +
    geom_point(size = 2.6, colour = GREY_STRONG) +
    labs(x = "Coefficient (MAE; standardized predictors)", y = NULL,
         title = "Reasoning-only Predictors") +
    theme_madrs_11pt() +
    theme(panel.grid.major.y = element_blank())
  save_dual(p, file.path(outdir, "coef_reason_only"),
            width = 7.2, height = max(3.6, 0.55 * nrow(d) + 1.6))
  invisible(p)
}

# ========= Optional: BLUPs per stratum =========
plot_item_blups <- function(fit, tag = c("Standard Models","Reasoning Models"), outdir = "../output/madrs_figs/items") {


  tag <- match.arg(tag)
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

  pick_madrs <- function(d) {
    names(d) <- tolower(gsub("\\s+", "_", names(d)))
    stopifnot(all(c("group","level","coefficient") %in% names(d)))
    d %>% filter(.data$group == "madrs_item")
  }

  total  <- modelbased::estimate_grouplevel(fit, type = "total")  %>% pick_madrs()
  random <- modelbased::estimate_grouplevel(fit, type = "random") %>% pick_madrs()

  total_df  <- transmute(total,  item = as.character(level), est = coefficient)
  random_df <- transmute(random, item = as.character(level), dev = coefficient,
                         lo = if ("ci_low" %in% names(random)) ci_low else NA_real_,
                         hi = if ("ci_high"%in% names(random)) ci_high else NA_real_)
  df <- left_join(total_df, random_df, by = "item") %>% arrange(est)

  # totals
  p1 <- ggplot(df, aes(est, item)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_point(aes(color = est > 0), size = 2.2) +
    scale_color_manual(values = c(`TRUE`=RED_STRONG, `FALSE`=BLUE_STRONG), guide = "none") +
    labs(x = "BLUP (group-level total)", y = "MADRS item",
         title = paste("Item BLUPs —", tag)) +
    theme_madrs_11pt()
  save_dual(p1, file.path(outdir, paste0("blups_items_totals_", tag)), 7.0, 4.6)

  # random deviations
  p2 <- ggplot(df, aes(dev, item)) +
    geom_vline(xintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(xmin = lo, xmax = hi), height = 0, orientation = "y") +
    geom_point(aes(color = dev > 0), size = 2.2) +
    scale_color_manual(values = c(`TRUE`=RED_STRONG, `FALSE`=BLUE_STRONG), guide = "none") +
    labs(x = "Random-effect deviation", y = "MADRS item",
         title = paste("Item Random Deviations —", tag)) +
    theme_madrs_11pt()
  save_dual(p2, file.path(outdir, paste0("blups_items_random_", tag)), 7.0, 4.6)

  invisible(df)
}

# ========= Public entry: make both plots =========
make_full_plots_overlay <- function(fit_nonreason, fit_reason, outdir = "../output/madrs_figs/items") {
  dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
  cat("Preparing overlay plots...\n")
  cat("  Standard Models fit:\n");  print(fit_nonreason)
  cat("  Reasoning Models fit:\n"); print(fit_reason)

  df_list <- prepare_overlay_df(fit_nonreason, fit_reason)
  p1 <- plot_overlay_shared(df_list$shared, outdir)
  p2 <- plot_reasoning_only(df_list$reason_only, outdir)

  # Per-stratum BLUPs for each fit (existing)
  plot_item_blups(fit_nonreason, "Standard Models", outdir)
  plot_item_blups(fit_reason,    "Reasoning Models", outdir)

  # NEW: Shared overlays for BLUP totals & random deviations
  plot_item_blups_shared(fit_nonreason, fit_reason, outdir)

  invisible(list(shared = p1, reasoning_only = p2))
}
