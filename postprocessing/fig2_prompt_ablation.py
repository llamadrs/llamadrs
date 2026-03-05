import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ============================================================================
# TABLE‑MATCHED PALETTE  (from LlaMADRS LaTeX tables)
# ============================================================================
# Teal ramp  (source #5B99A8)
TBL_GOOD_LT  = "#D5E8ED"   # 20 %  — light teal
TBL_GOOD_DK  = "#A1CCD8"   # 45 %  — darker teal
HDR_ICON     = "#4A8896"    # saturated teal (accent / icon)

# Rose ramp  (source #C24168)
TBL_BAD_LT   = "#F0D3DD"   # 20 %  — light rose
TBL_BAD_DK   = "#DCA0B0"   # 45 %  — darker rose
ROSE_ACCENT  = "#C24168"    # saturated rose (accent)

# Neutrals
TBL_NEUTRAL  = "#ECEDEF"   # neutral gray cells
HDR_BG       = "#F4F6F8"   # header background
TBL_BORDER   = "#D0D4D8"   # subtle border
GRP_BAND     = "#C5C9CC"   # group‑header band
HDR_SUB      = "#7A8A90"   # muted gray for formulas / descriptors

# Clinical thresholds  (same as table helpers)
ITEM_MEANINGFUL_MAE  = 0.6
ITEM_SUBSTANTIAL_MAE = 1.2


def plot_ablation_readable(ablation_individual_results, ablation_mean_results, figsize=(10, 5.5)):
    DELTA = r'$\Delta$'

    # ---- Box fill colors — high‑contrast, distinct from teal/rose
    r_fill      = "#7BA3D4"       # Reasoning  boxes   (saturated steel blue)
    nr_fill     = "#F0B060"       # No‑Reasoning boxes (saturated amber)
    r_edge      = "#2B4C7E"       # Reasoning  edges   (dark navy)
    nr_edge     = "#B5651D"       # No‑Reasoning edges (dark burnt orange)
    r_scatter   = "#1E3A5F"       # scatter dots        (deep navy)
    nr_scatter  = "#8B4513"       # scatter dots        (saddle brown)

    # ---- Configs (each will render two boxes: R and NR)
    configs = [
        ("All Components",        "Qwen 3 Next (3B-80B)",               "Qwen 3 Next: NR (3B-80B)"),
        ("No Descriptions",       "Qwen 3 Next: No Desc (3B-80B)",       "Qwen 3 Next: NR, No Desc (3B-80B)"),
        ("No Demonstrations",     "Qwen 3 Next: No Dem (3B-80B)",        "Qwen 3 Next: NR, No Dem (3B-80B)"),
        ("Raw (No Desc, No Dem)", "Qwen 3 Next: Raw (3B-80B)",           "Qwen 3 Next: NR, Raw (3B-80B)"),
    ]

    # MADRS items 1..10
    item_indices = list(range(1, 11))

    # ---- Collect per‑item MAE distributions for each config
    dist_R, dist_NR, med_deltas = [], [], []
    for _, r_model, nr_model in configs:
        paired_r, paired_nr = [], []
        for i in item_indices:
            r = ablation_individual_results.get(r_model, {}).get(i, {}).get("mae_mean", np.nan)
            nr = ablation_individual_results.get(nr_model, {}).get(i, {}).get("mae_mean", np.nan)
            if np.isfinite(r) and np.isfinite(nr):
                paired_r.append(float(r))
                paired_nr.append(float(nr))
        dist_R.append(np.array(paired_r))
        dist_NR.append(np.array(paired_nr))
        if paired_r and paired_nr:
            med_deltas.append(float(np.median(np.array(paired_nr) - np.array(paired_r))))
        else:
            med_deltas.append(np.nan)

    # ---- Figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor(HDR_BG)

    # positions: for config i, center at i*3.0; R at center‑0.5, NR at center+0.5
    centers = np.arange(len(configs)) * 3.0
    pos_R  = centers - 0.5
    pos_NR = centers + 0.5

    def draw_boxes(data_list, positions, facecolor, edgecolor, scatter_color,
                   label, hatch=None, marker='o'):
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.85,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            medianprops=dict(linewidth=2.5, color='#111111'),
            boxprops=dict(linewidth=1.4, edgecolor=edgecolor, facecolor=facecolor),
            whiskerprops=dict(linewidth=1.2, color=edgecolor, linestyle='--'),
            capprops=dict(linewidth=1.2, color=edgecolor),
        )
        # Apply hatch for color-blind accessibility
        if hatch:
            for patch in bp['boxes']:
                patch.set_hatch(hatch)
                patch.set_edgecolor(edgecolor)
        # jittered item markers in the accent color (distinct shapes)
        rng = np.random.RandomState(42)
        for x, arr in zip(positions, data_list):
            if len(arr):
                jitter = (rng.rand(len(arr)) - 0.5) * 0.20
                ax.scatter(
                    np.full_like(arr, x) + jitter, arr,
                    s=36, alpha=0.80, color=scatter_color,
                    edgecolors='white', linewidths=0.6, zorder=5,
                    marker=marker,
                )
        return bp

    draw_boxes(dist_R,  pos_R,  r_fill,  r_edge,  r_scatter,
               "Reasoning",    hatch='///',  marker='o')
    draw_boxes(dist_NR, pos_NR, nr_fill, nr_edge, nr_scatter,
               "No-Reasoning", hatch='...',  marker='D')

    # ---- Clinical‑threshold reference lines (distinct dash patterns for B/W)
    ax.axhline(ITEM_MEANINGFUL_MAE, color=HDR_ICON, linewidth=1.4,
               linestyle=(0, (5, 3)), alpha=0.7, zorder=1)          # dashed
    ax.axhline(ITEM_SUBSTANTIAL_MAE, color=ROSE_ACCENT, linewidth=1.4,
               linestyle=(0, (1, 2)), alpha=0.7, zorder=1)          # dotted

    # ---- x‑axis labels
    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["All\nComponents", "No\nDescriptions", "No\nDemonstrations", "Raw\n(No Desc, No Dem)"],
        fontsize=11, color='#222222',
    )
    ax.tick_params(axis='x', length=0)  # hide tick marks

    ax.set_ylabel("Item Mean Absolute Error (lower is better)", color='#222222', fontsize=12)

    # ---- Median‑delta annotations above each pair
    ymax = 0.0
    for r_arr, nr_arr in zip(dist_R, dist_NR):
        if len(r_arr):  ymax = max(ymax, float(np.max(r_arr)))
        if len(nr_arr): ymax = max(ymax, float(np.max(nr_arr)))
    ymax = ymax if np.isfinite(ymax) and ymax > 0 else 2.0

    # Delta annotations — color‑coded by which model wins
    #   positive delta (NR > R) → Reasoning better → dark indigo box, white text
    #   negative delta (NR < R) → No‑Reasoning better → dark orange box, white text
    for i, d in enumerate(med_deltas):
        if np.isfinite(d):
            if d > 0:       # NR worse → Reasoning wins
                ec, fc, tc = r_edge, r_edge, 'white'
            elif d < 0:     # NR better → No-Reasoning wins
                ec, fc, tc = nr_edge, nr_edge, 'white'
            else:
                ec, fc, tc = HDR_SUB, HDR_SUB, 'white'
            sign = "+" if d > 0 else ""
            txt = ax.text(
                centers[i], ymax + 0.06,
                f"{DELTA}MAE (NR $-$ R) = {sign}{d:.2f}",
                ha='center', va='bottom', fontsize=7.5, fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec=ec,
                          linewidth=1.8, alpha=0.95),
                color=tc, zorder=10,
            )

    # ---- Threshold labels (right margin)
    trans = ax.get_yaxis_transform()
    ax.text(1.01, ITEM_MEANINGFUL_MAE, "Meaningful\nError", transform=trans,
            fontsize=8.5, color=HDR_ICON, va='center', fontweight='medium',
            linespacing=0.85, clip_on=False)
    ax.text(1.01, ITEM_SUBSTANTIAL_MAE, "Substantial\nError", transform=trans,
            fontsize=8.5, color=ROSE_ACCENT, va='center', fontweight='medium',
            linespacing=0.85, clip_on=False)

    # ---- Legend — includes hatch & marker shape for color-blind access
    from matplotlib.patches import Patch
    legend_entries = [
        Patch(facecolor=r_fill, edgecolor=r_edge, linewidth=1.4,
              hatch='///', label="Reasoning"),
        Patch(facecolor=nr_fill, edgecolor=nr_edge, linewidth=1.4,
              hatch='...', label="No-Reasoning"),
    ]
    leg = ax.legend(
        handles=legend_entries, framealpha=0.97, loc='lower right',
        edgecolor=TBL_BORDER, fancybox=True, fontsize=11,
        frameon=True, facecolor='white',
    )
    leg.get_frame().set_linewidth(1.0)

    # ---- Grid & spines — clean academic look
    ax.grid(axis='y', linestyle='--', alpha=0.40, color=GRP_BAND, zorder=0)
    ax.grid(axis='x', visible=False)
    for spine in ax.spines.values():
        spine.set_color(TBL_BORDER)
        spine.set_linewidth(0.8)
    ax.tick_params(axis='y', colors='#555555')

    ax.set_ylim(0.3, ymax + 0.25)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study figure from analysis results.")
    parser.add_argument("--results", type=str, nargs="?", default="../output/llamadrs_results.pkl",
                        help="Path to the results pickle file (default: ../output/llamadrs_results.pkl)")
    parser.add_argument("-o", "--output", type=str, default="../output/fig2_prompt_ablation.png",
                        help="Output figure path (default: ../output/fig2_prompt_ablation.png)")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI (default: 300)")
    parser.add_argument("--no-show", action="store_true", help="Don't display the figure interactively")
    args = parser.parse_args()

    with open(args.results, "rb") as f:
        data = pickle.load(f)

    mean_results = data["mean_results"]
    individual_results = data["individual_results"]

    fig = plot_ablation_readable(individual_results, mean_results)
    fig.savefig(args.output, dpi=args.dpi)
    print(f"Saved figure to {args.output}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()