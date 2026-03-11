"""
fig3_lme_plot.py — Publication-quality two-panel figure for LME analysis.

  (a) Fixed-effect coefficients   — parsed from error_analysis_items.txt
  (b) MADRS item random-intercept — loaded from blups_{nonreason,reason}.csv

All outputs are produced by fig3_lme_analysis.R.

Usage:
    python fig3_lme_plot.py                              # defaults
    python fig3_lme_plot.py --output-dir ../../output     # explicit dir
    python fig3_lme_plot.py -o fig3.png --dpi 600         # custom output
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================================
# TABLE-MATCHED PALETTE  (from LlaMADRS LaTeX / fig2_prompt_ablation.py)
# ============================================================================
# Neutrals
HDR_BG       = "#F4F6F8"   # header / panel background
TBL_BORDER   = "#D0D4D8"   # subtle border / spines
GRP_BAND     = "#C5C9CC"   # group-header band / grid

# Reasoning  (steel-blue ramp, same as fig2 box fills)
R_FILL       = "#7BA3D4"   # marker fill
R_EDGE       = "#2B4C7E"   # marker edge / CI line
R_SCATTER    = "#1E3A5F"   # (reserved for scatter)

# Non-Reasoning / Non-Reasoning  (amber ramp)
NR_FILL      = "#F0B060"   # marker fill
NR_EDGE      = "#B5651D"   # marker edge / CI line
NR_SCATTER   = "#8B4513"   # (reserved for scatter)


# ============================================================================
# Readable predictor labels
# ============================================================================
PREDICTOR_LABELS = {
    "session_item_severity_patitem_mean_z": "Severity\n(between-patient)",
    "log_tokens_pat_mean_z":               "Response Length\n(between-patient)",
    "session_item_severity_patitem_wc_z":  "Severity\n(within-patient)",
    "log_tokens_pat_wc_z":                 "Response Length\n(within-patient)",
    "log_reason_tokens_patitem_mean_z":    "Reasoning Length\n(between-patient)",
    "log_reason_tokens_patitem_wc_z":      "Reasoning Length\n(within-patient)",
    "log_params_z":                        "Model Size\n(log params)",
    "log_context_length_z":                "Context Length\n(log tokens)",
    "architectureMoE":                     "Architecture\n(MoE)",
}


# ============================================================================
# Parsing fixed effects from error_analysis_items.txt
# ============================================================================

def parse_fixed_effects(report_path: str):
    """
    Parse the two 'Fixed effects:' tables printed by lme4::summary()
    inside error_analysis_items.txt.

    Returns (nonreason_fe, reason_fe) where each is a dict:
        { predictor_name: {'est': float, 'se': float} }
    """
    if not os.path.isfile(report_path):
        sys.exit(f"ERROR: report not found: {report_path}\n"
                 f"  Run fig3_lme_analysis.R first.")

    with open(report_path) as f:
        text = f.read()

    # Split at the two MODEL SUMMARY headers
    blocks = {}
    for model_tag in ("NONREASON", "REASON"):
        header = f"MODEL SUMMARY — {model_tag}"
        idx = text.find(header)
        if idx < 0:
            sys.exit(f"ERROR: could not find '{header}' in {report_path}")
        block = text[idx:]
        # Trim at next model header or end
        next_hdr = block.find("MODEL SUMMARY —", len(header))
        if next_hdr > 0:
            block = block[:next_hdr]
        blocks[model_tag] = block

    def _extract_fe(block: str):
        # Find "Fixed effects:" section
        fe_start = block.find("Fixed effects:")
        if fe_start < 0:
            return {}
        fe_block = block[fe_start:]
        # It ends at a blank line followed by non-table content, or
        # "Correlation of Fixed" header
        fe_end = fe_block.find("Correlation of Fixed")
        if fe_end > 0:
            fe_block = fe_block[:fe_end]

        fe = {}
        for line in fe_block.splitlines():
            line = line.strip()
            if not line or line.startswith("Fixed effects") or line.startswith("Estimate"):
                continue
            # Lines look like:
            #   session_item_severity_patitem_mean_z  0.130806   0.006520  20.063
            #   (Intercept)                           0.902686   0.070471  12.809
            parts = line.split()
            if len(parts) < 3:
                continue
            name = parts[0]
            if name == "(Intercept)":
                continue  # skip intercept
            try:
                est = float(parts[1])
                se  = float(parts[2])
            except (ValueError, IndexError):
                continue
            fe[name] = {"est": est, "se": se}
        return fe

    nr_fe = _extract_fe(blocks["NONREASON"])
    r_fe  = _extract_fe(blocks["REASON"])
    return nr_fe, r_fe


# ============================================================================
# Data loading — BLUPs
# ============================================================================

def load_blups(csv_dir: str):
    """Load both BLUP CSVs and return (df_nonreason, df_reason)."""
    nr_path = os.path.join(csv_dir, "blups_nonreason.csv")
    r_path  = os.path.join(csv_dir, "blups_reason.csv")

    for p in (nr_path, r_path):
        if not os.path.isfile(p):
            sys.exit(f"ERROR: expected CSV not found: {p}\n"
                     f"  Run fig3_lme_analysis.R first to generate the BLUPs.")

    df_nr = pd.read_csv(nr_path)
    df_r  = pd.read_csv(r_path)

    for df in (df_nr, df_r):
        df["madrs_item"] = df["madrs_item"].astype(int)

    return df_nr, df_r


# ============================================================================
# Panel (a): Fixed-Effect Coefficients
# ============================================================================

def plot_fixed_effects(ax, nonreason_fe: dict, reason_fe: dict):
    """Plot fixed-effect coefficients with 95% CI on *ax*."""
    ax.set_facecolor(HDR_BG)

    # Union of all predictors, ordered as in reason_fe then any extra from nr
    all_preds = list(reason_fe.keys())
    for k in nonreason_fe:
        if k not in all_preds:
            all_preds.append(k)

    y_positions = np.arange(len(all_preds))
    offset = 0.15

    for i, pred in enumerate(all_preds):
        # Reasoning model
        if pred in reason_fe:
            est = reason_fe[pred]["est"]
            se  = reason_fe[pred]["se"]
            ci_lo, ci_hi = est - 1.96 * se, est + 1.96 * se
            ax.plot([ci_lo, ci_hi], [i - offset, i - offset],
                    color=R_EDGE, linewidth=2)
            ax.plot(est, i - offset, "o", color=R_EDGE,
                    markerfacecolor=R_FILL, markersize=8,
                    markeredgewidth=1)

        # Non-reasoning model
        if pred in nonreason_fe:
            est = nonreason_fe[pred]["est"]
            se  = nonreason_fe[pred]["se"]
            ci_lo, ci_hi = est - 1.96 * se, est + 1.96 * se
            ax.plot([ci_lo, ci_hi], [i + offset, i + offset],
                    color=NR_EDGE, linewidth=2)
            ax.plot(est, i + offset, "s", color=NR_EDGE,
                    markerfacecolor=NR_FILL, markersize=8,
                    markeredgewidth=1)

    ax.axvline(0, color=GRP_BAND, linestyle="--", linewidth=1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [PREDICTOR_LABELS.get(p, p) for p in all_preds],
        fontsize=9, color="#222222")
    ax.set_ylabel("Model/Session Predictor", fontweight="bold",
                  fontsize=10, color="#222222")
    ax.set_xlabel("Coefficient Estimate", fontweight="bold",
                  fontsize=10, color="#222222")
    ax.set_title("(a) Fixed-effect coefficients",
                 fontweight="bold", fontsize=11, pad=10, color="#222222")
    ax.grid(axis="x", linestyle="--", alpha=0.40, color=GRP_BAND, zorder=0)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_color(TBL_BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#555555")
    ax.tick_params(axis="y", colors="#555555")
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=R_FILL, markeredgecolor=R_EDGE,
               markeredgewidth=1.4, markersize=8, label="Reasoning"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=NR_FILL, markeredgecolor=NR_EDGE,
               markeredgewidth=1.4, markersize=8, label="Non-Reasoning"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
                    framealpha=0.97, edgecolor=TBL_BORDER, fancybox=True,
                    frameon=True, facecolor="white")
    leg.get_frame().set_linewidth(1.0)


# ============================================================================
# Panel (b): Item BLUPs (Random Intercepts)
# ============================================================================

def plot_item_blups(ax, df_nr: pd.DataFrame, df_r: pd.DataFrame):
    """Plot MADRS item random-intercept deviations on *ax*."""
    ax.set_facecolor(HDR_BG)

    # Canonical item ordering: 1–10, then 0 (Total Score at bottom)
    item_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    reason_sorted = df_r.set_index("madrs_item").loc[item_order].reset_index()

    # Build short labels
    item_short_labels = []
    for label in reason_sorted["label"]:
        if "Item 0:" in label:
            item_short_labels.append("Total Score")
        else:
            parts = label.split(": ", 1)
            item_short_labels.append(parts[1] if len(parts) > 1 else label)

    y_positions = np.arange(len(reason_sorted))
    offset = 0.15

    # Reasoning
    for i, row in reason_sorted.iterrows():
        ax.plot([row["dev_low"], row["dev_high"]], [i - offset, i - offset],
                color=R_EDGE, linewidth=2)
        ax.plot(row["dev"], i - offset, "o", color=R_EDGE,
                markerfacecolor=R_FILL, markersize=8, markeredgewidth=1)

    # Non-reasoning (match items)
    for i, row in reason_sorted.iterrows():
        item_id = row["madrs_item"]
        nr_row = df_nr[df_nr["madrs_item"] == item_id]
        if not nr_row.empty:
            nr = nr_row.iloc[0]
            ax.plot([nr["dev_low"], nr["dev_high"]], [i + offset, i + offset],
                    color=NR_EDGE, linewidth=2)
            ax.plot(nr["dev"], i + offset, "s", color=NR_EDGE,
                    markerfacecolor=NR_FILL, markersize=8, markeredgewidth=1)

    ax.axvline(0, color=GRP_BAND, linestyle="--", linewidth=1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(item_short_labels, fontsize=9, color="#222222")
    ax.set_ylabel("MADRS Item", fontweight="bold", fontsize=10, color="#222222")
    ax.set_xlabel("Deviation from Grand Mean", fontweight="bold",
                  fontsize=10, color="#222222")
    ax.set_title("(b) MADRS item random-intercept deviations",
                 fontweight="bold", fontsize=11, pad=10, color="#222222")
    ax.grid(axis="x", linestyle="--", alpha=0.40, color=GRP_BAND, zorder=0)
    ax.grid(axis="y", visible=False)
    for spine in ax.spines.values():
        spine.set_color(TBL_BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", colors="#555555")
    ax.tick_params(axis="y", colors="#555555")
    ax.invert_yaxis()

    legend_elements = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=R_FILL, markeredgecolor=R_EDGE,
               markeredgewidth=1.4, markersize=8, label="Reasoning"),
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=NR_FILL, markeredgecolor=NR_EDGE,
               markeredgewidth=1.4, markersize=8, label="Non-Reasoning"),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
                    framealpha=0.97, edgecolor=TBL_BORDER, fancybox=True,
                    frameon=True, facecolor="white")
    leg.get_frame().set_linewidth(1.0)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Fig 3 — LME fixed effects + item BLUPs.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../../output",
        help="Directory containing error_analysis_items.txt, "
             "blups_nonreason.csv, blups_reason.csv (default: ../../output)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output figure path (default: <output-dir>/fig3_mixed_effects.png)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Figure DPI (default: 300)",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display the figure interactively",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_path = args.output or os.path.join(out_dir, "fig3_mixed_effects.png")

    # ---- Load data ----
    report_path = os.path.join(out_dir, "error_analysis_items.txt")
    nonreason_fe, reason_fe = parse_fixed_effects(report_path)
    print(f"Parsed {len(nonreason_fe)} non-reasoning and "
          f"{len(reason_fe)} reasoning fixed effects.")

    df_nr, df_r = load_blups(out_dir)
    print(f"Loaded {len(df_nr)} non-reasoning and "
          f"{len(df_r)} reasoning BLUP rows.")

    # ---- Plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")
    plot_fixed_effects(ax1, nonreason_fe, reason_fe)
    plot_item_blups(ax2, df_nr, df_r)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
