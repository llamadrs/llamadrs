import argparse
import pickle
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import colors as mcolors

def _xcolor_mix_to_rgb(xc: str):
    """
    Convert xcolor-like 'blue!5' to an RGB tuple by mixing with white.
    'name!p' means p% of base color + (100-p)% white.
    """
    m = re.fullmatch(r"([A-Za-z]+)!(\d{1,3})", xc.strip())
    if not m:
        # fall back to Matplotlib's color parsing
        return mcolors.to_rgb(xc)
    name, pct = m.group(1), int(m.group(2))
    pct = max(0, min(100, pct))
    base = np.array(mcolors.to_rgb(name))
    white = np.array([1.0, 1.0, 1.0])
    mix = (pct / 100.0) * base + (1 - pct / 100.0) * white
    return tuple(mix.tolist())

def plot_ablation_readable(ablation_individual_results, ablation_mean_results, figsize=(10, 5.5)):
    use_tex = plt.rcParams.get('text.usetex', False)
    DELTA = r'$\Delta$'

    # ---- Box fill colors (blue & orange for high contrast)
    header_gray = _xcolor_mix_to_rgb("white!85")
    r_hdr_col   = _xcolor_mix_to_rgb("blue!30")
    nr_hdr_col  = _xcolor_mix_to_rgb("orangered!30")

    # Strong accent colors for annotations
    strong_blue   = mcolors.to_rgb("blue")
    strong_orange = mcolors.to_rgb("orangered")

    # ---- Configs (each will render two boxes: R and NR)
    configs = [
        ("All Components",        "Qwen 3 Next (80B)",                   "Qwen 3 Next (80B): No Reasoning"),
        ("No Descriptions",       "Qwen 3 Next (80B): No Desc",          "Qwen 3 Next (80B): No Reasoning, No Desc"),
        ("No Demonstrations",     "Qwen 3 Next (80B): No Dem",           "Qwen 3 Next (80B): No Reasoning, No Dem"),
        ("Raw (No Desc, No Dem)",     "Qwen 3 Next (80B): Raw",              "Qwen 3 Next (80B): No Reasoning, Raw"),
    ]

    # MADRS items 1..10
    item_indices = list(range(1, 11))

    # ---- Collect per-item MAE distributions for each config
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

    # ---- Single-axes figure with grouped boxplots
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(header_gray)

    # positions: for config i, center at i*3.0; R at center-0.5, NR at center+0.5
    centers = np.arange(len(configs)) * 3.0
    pos_R = centers - 0.5
    pos_NR = centers + 0.5

    def draw_boxes(data_list, positions, facecolor, label):
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=0.9,
            patch_artist=True,
            showfliers=True,
            whis=(5, 95),
            medianprops=dict(linewidth=1.5, color='black'),
            boxprops=dict(linewidth=1.0, edgecolor='black', facecolor=facecolor),
            whiskerprops=dict(linewidth=1.0, color='black'),
            capprops=dict(linewidth=1.0, color='black')
        )
        # light jittered item markers
        for x, arr in zip(positions, data_list):
            if len(arr):
                jitter = (np.random.rand(len(arr)) - 0.5) * 0.18
                ax.scatter(np.full_like(arr, x) + jitter, arr, s=18, alpha=0.6, edgecolors='none')
        return bp

    # Apply your colors
    draw_boxes(dist_R,  pos_R,  r_hdr_col,  "Reasoning")
    draw_boxes(dist_NR, pos_NR, nr_hdr_col, "No-Reasoning")

    # xticks under group centers with config labels
    ax.set_xticks(centers)
    ax.set_xticklabels(
        ["All\nComponents", "No\nDescriptions", "No\nDemonstrations", "Raw\n(No Desc, No Dem)"],
        fontsize=10, fontweight='bold'
    )

    ax.set_ylabel("Item MAE (lower is better)", fontweight='bold')
    # annotate median deltas above each pair
    ymax = 0.0
    for r_arr, nr_arr in zip(dist_R, dist_NR):
        if len(r_arr):  ymax = max(ymax, float(np.max(r_arr)))
        if len(nr_arr): ymax = max(ymax, float(np.max(nr_arr)))
    ymax = ymax if np.isfinite(ymax) and ymax > 0 else 2.0

    for i, d in enumerate(med_deltas):
        if np.isfinite(d):
            edge = strong_blue if d > 0 else strong_orange if d < 0 else (0.3, 0.3, 0.3)
            sign = "+" if d > 0 else ""
            ax.text(
                centers[i], ymax + 0.02, f"{sign}{d:.2f}",
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.28', fc='white', ec=edge, alpha=0.95),
                color=edge
            )


    # legend with your fills
    legend_entries = [
        Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=r_hdr_col,
               markeredgecolor='black', markeredgewidth=1.0, label="Reasoning"),
        Line2D([0], [0], marker='s', linestyle='none', markerfacecolor=nr_hdr_col,
               markeredgecolor='black', markeredgewidth=1.0, label="No-Reasoning"),
    ]
    ax.legend(handles=legend_entries, framealpha=0.95, loc='upper left')

    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.set_ylim(0.5, ymax + 0.12)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate ablation study figure from analysis results.")
    parser.add_argument("results", type=str, nargs="?", default="madrs_analysis_results.pkl",
                        help="Path to the results pickle file (default: madrs_analysis_results.pkl)")
    parser.add_argument("-o", "--output", type=str, default="ablation_study_readable.png",
                        help="Output figure path (default: ablation_study_readable.png)")
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