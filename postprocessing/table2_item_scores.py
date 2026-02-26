#!/usr/bin/env python3
"""
Enhanced LlaMADRS Academic Tables Suite with Reasoning Model Highlighting
Comprehensive LaTeX table generators with reasoning/non-reasoning visual distinction
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import re
import pickle
import argparse
from pathlib import Path


MODEL_DICT = {
    "segmented_Qw3_0.6b_gptq_4q": "Qwen 3 (0.6B)",
    "segmented_Qw3_1.7b_gptq_4q": "Qwen 3 (1.7B)",
    "segmented_Qw3_4b_awq_4q": "Qwen 3 (4B)",
    "segmented_Qw2.5_7b_1m_awq_4q": "Qwen 2.5 (7B): 1M",
    "segmented_DeepSeek_R1_Llama_8b_gptq_4q": "DeepSeek R1 Llama 3.1 (8B)",
    "segmented_Llama3.1_8b_gptq_4q": "Llama 3.1 (8B)",
    "segmented_Qw3_8b_awq_4q": "Qwen 3 (8B)",
    "segmented_Qw3_14b_awq_4q": "Qwen 3 (14B)",
    "segmented_Qw2.5_14b_1m_awq_4q": "Qwen 2.5 (14B): 1M",
    "segmented_Magistral_Small_2507_awq_4q": "Magistral Small 2507 (24B)",
    "segmented_PsyCare1.0_Llama3.1_8b": "PsyCare 1.0 Llama 3.1 (8B)",
    "segmented_Llama3.1_8b": "Llama 3.1 (8B): No Quant",
    "segmented_Qw3_30b_a3b_ar_4q": "Qwen 3 (3B-30B)",
    "segmented_Qw3_30b_a3b_ar_4q_NoR": "Qwen 3: No Reasoning (3B-30B)",
    "segmented_Gen3_27b_it_gptq_4q": "Gemma 3 IT (27B)",
    "segmented_Qw3_32b_awq_4q": "Qwen 3 (32B)",
    "segmented_QwQ_32b_awq_4q": "QwQ (32B)",
    "segmented_DeepSeek_R1_Qwen_32b_gptq_4q": "DeepSeek R1 Qwen 2.5 (32B)",
    "segmented_GPT_OSS_20b_mxfp4_4q": "GPT OSS 20B (3B-21B)",
    "segmented_DeepSeek_R1_Llama_70b_gptq_4q": "DeepSeek R1 Llama 3.3 (70B)",
    "segmented_Llama3.3_70b_gptq_4q": "Llama 3.3 (70B)",
    "segmented_Qw2.5_72b_gptq_4q": "Qwen 2.5 (72B)",
    "segmented_L4_Scout_17b_gptq_4q": "Llama 4 Scout (17B-109B)",
    "segmented_GPT_OSS_120b_mxfp4_4q": "GPT OSS 120B (5B-117B)",
    "segmented_Qw3_235b_a22b_ar_4q": "Qwen 3 (22B-235B)",
    "segmented_L4_Maverick_17b_gptq_4q": "Llama 4 Maverick (17B-400B)",
    "segmented_Qw3_Next_80b_a3b_ar_4q": "Qwen 3 Next (3B-80B)",
    "segmented_Qw3_Next_80b_a3b_ar_4q_NoR": "Qwen 3 Next: NR (3B-80B)",
}
# add ablations
for ablation in ["raw", "no_desc", "no_dem"]:
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_{ablation}"] = f"Qwen 3 Next: {ablation.replace('_', ' ').title()} (3B-80B)"
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_NoR_{ablation}"] = f"Qwen 3 Next: NR, {ablation.replace('_', ' ').title()} (3B-80B)"


MODEL_RANKS = {k: i for i, k in enumerate(MODEL_DICT.keys(), start=1)}
MODEL_REV_DICT = {v: k for k, v in MODEL_DICT.items()}

# ============================================================================
# VISUAL DESIGN + CLINICAL THRESHOLDS
# ============================================================================
ITEM_MEANINGFUL_MAE   = 0.6
ITEM_SUBSTANTIAL_MAE  = 1.2





def get_cell_color_item_mae(
    value: float,
    *,
    for_mean_row: bool = False,
    best: bool = False,
    worst: bool = False,
) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""

    highlight = best or worst

    if v < ITEM_MEANINGFUL_MAE:
        return r"\cellcolor{DarkGreen}" if highlight else r"\cellcolor{LightGreen}"
    if v >= ITEM_SUBSTANTIAL_MAE:
        return r"\cellcolor{DarkRed}" if highlight else r"\cellcolor{LightRed}"

    if for_mean_row:
        return ""
    return r"\cellcolor{DarkGray}" if highlight else r"\cellcolor{LightGray}"


def create_comprehensive_itemwise_table(
    individual_results: Dict,
    mean_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
) -> str:
    """
    Item-wise MAE table with reasoning / non-reasoning grouping.
    """

    def _abbrev_name(s: str) -> str:
        """Abbreviate model display name for compact display (keeps (..B) part as tiny)."""
        if custom_name_map and s in custom_name_map:
            s = custom_name_map[s]
        parts = s.split("(")
        name = parts[0].strip()
        size = "(" + parts[1].strip() if len(parts) > 1 else ""
        if len(name) > 28:
            name = name[:25] + "..."
        return f"{name} {{\\tiny {size}}}" if size else name

    item_indices = list(range(1, 11))
    rows = []

    for model in sorted(individual_results.keys()):
        model_id = MODEL_REV_DICT.get(model, model)
        if models_csv is None or model_id not in models_csv:
            continue
        if model_id in {
            "segmented_Llama3.1_8b",
            "segmented_PsyCare1.0_Llama3.1_8b",
            "segmented_Qw3_30b_a3b_ar_4q_NoR",
        }:
            continue

        model_row = models_csv[model_id]
        display_name = model

        entry = {"Model": _abbrev_name(display_name), "original_id": model_id}

        for i in item_indices:
            d = individual_results.get(model, {}).get(i, {})
            entry[f"I{i}_mean"] = d.get("mae_mean", np.nan)
            entry[f"I{i}_std"]  = d.get("mae_std", np.nan)

        d_overall = mean_results.get(model, {})
        entry["I11_mean"] = d_overall.get("mae_mean", np.nan)
        entry["I11_std"]  = d_overall.get("mae_std", np.nan)

        entry["is_reasoning"] = str(model_row.get("reasoning", "")).strip() == "Yes"
        rows.append(entry)

    df = pd.DataFrame(rows).sort_values(by="I11_mean").reset_index(drop=True)

    # Global best/worst by overall mean
    best_model_id = None
    worst_model_id = None
    if not df.empty:
        mean_col = pd.to_numeric(df["I11_mean"], errors="coerce")
        finite = mean_col[np.isfinite(mean_col)]
        if not finite.empty:
            best_model_id = df.loc[finite.idxmin(), "original_id"]
            worst_model_id = df.loc[finite.idxmax(), "original_id"]

    # Per-column min/max (computed once, not per row)
    col_stats = {}
    for i in item_indices + [11]:
        col = pd.to_numeric(df[f"I{i}_mean"], errors="coerce")
        good = col[np.isfinite(col)]
        col_stats[i] = {"min": float(good.min()), "max": float(good.max())} if not good.empty else None

    # Split by reasoning
    reasoning_df = df[df["is_reasoning"]].copy()
    non_reasoning_df = df[~df["is_reasoning"]].copy()

    # Group mean summary rows
    def _append_group_mean(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        if frame.empty:
            return frame
        mean_row = {"Model": label, "original_id": ""}
        for i in item_indices + [11]:
            mean_row[f"I{i}_mean"] = pd.to_numeric(frame[f"I{i}_mean"], errors="coerce").mean()
            mean_row[f"I{i}_std"]  = pd.to_numeric(frame[f"I{i}_std"],  errors="coerce").mean()
        return pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)

    reasoning_df = _append_group_mean(reasoning_df, r"\textbf{Mean (Reasoning)}")
    non_reasoning_df = _append_group_mean(non_reasoning_df, r"\textbf{Mean (Non-Reasoning)}")

    # --- Build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\renewcommand{\arraystretch}{1.05}")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    latex.append(r"{\footnotesize")

    meaningful_str = f"{ITEM_MEANINGFUL_MAE:g}"
    substantial_str = f"{ITEM_SUBSTANTIAL_MAE:g}"
    latex.append(
        r"\caption{Item-wise MAE$\pm$ std. dev (I1--I10) and mean across items. "
        r"For MoE models, size is $Active$--$Total$ parameters (e.g., 3B--30B = 3B active, 30B total). "
        r"\textcolor{black}{\fcolorbox{white}{LightGreen}{\strut\enspace}}\ $<\,0.6$ \emph{(Acceptable)}, "
        r"\textcolor{black}{\fcolorbox{white}{LightRed}{\strut\enspace}}\ $\geq\,1.2$ \emph{(Substantial)}.} "
    )
    latex.append(r"\label{tab:comprehensive_itemwise_reasoning}")
    latex.append(r"\vspace{1mm}")
    latex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l *{10}{c} c @{}}")
    latex.append(r"\toprule")
    latex.append(
        r"\rowcolor{gray!15} \textbf{Model} (\tiny{Size}) & "
        + " & ".join([f"\\textbf{{I{i}}}" for i in item_indices])
        + r" & \textbf{Mean} \\"
    )
    latex.append(r"\midrule")
    def _format_item_cell(m, s, col_idx, *, is_summary: bool) -> str:
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"
        if not (isinstance(s, (int, float)) and np.isfinite(s)):
            return r"\textemdash{}"

        value_main = f"{m:.2f}"

        # Best/worst markers only for real model rows (not group means)
        stat = col_stats.get(col_idx)
        best = worst = False
        if (not is_summary) and stat:
            mv = float(m)
            if abs(mv - stat["min"]) <= 1e-6:
                best = True
            elif abs(mv - stat["max"]) <= 1e-6:
                worst = True
        
        # Now compute color with correct best/worst
        cell_color = get_cell_color_item_mae(m, for_mean_row=is_summary, best=best, worst=worst)

        # Apply emphasis to main value only
        if best:
            value_main_fmt = rf"\textbf{{{value_main}}}"
        elif worst:
            value_main_fmt = rf"\textit{{{value_main}}}"
        else:
            value_main_fmt = value_main

        # Color appears once; std line is uncolored (cell background already set)
        value_str = rf"{cell_color}{value_main_fmt} \\ {cell_color}{{\tiny $\pm$ {s:.2f}}}"
        return rf"\makecell{{{value_str}}}"
    def _emit_rows(frame: pd.DataFrame):
        for _, rr in frame.iterrows():
            model_text = str(rr["Model"])
            is_summary = "Mean (" in model_text

            is_best_model = (not is_summary) and best_model_id and (rr.get("original_id", "") == best_model_id)
            is_worst_model = (not is_summary) and worst_model_id and (rr.get("original_id", "") == worst_model_id)

            if is_summary:
                # Thin rule before summary row
                model_cell = rf"\midrule[\lightrulewidth] \rowcolor{{gray!15}} {model_text}"
            elif is_best_model:
                model_cell = rf"\cellcolor{{DarkGray}} \textbf{{{model_text}}}"
            elif is_worst_model:
                model_cell = rf"\cellcolor{{DarkGray}} \textit{{{model_text}}}"
            else:
                model_cell = model_text

            cells = [model_cell]
            for i in item_indices + [11]:
                cells.append(_format_item_cell(
                    rr.get(f"I{i}_mean", np.nan),
                    rr.get(f"I{i}_std", np.nan),
                    i,
                    is_summary=is_summary,
                ))

            latex.append(" & ".join(cells) + r" \\")

    # Reasoning section
    latex.append(
        rf"\multicolumn{{12}}{{c}}"
        rf"{{\cellcolor{{ReasonBand}}\textsc{{Reasoning Models}}}} \\"
    )
    latex.append(r"\midrule[\lightrulewidth]")
    _emit_rows(reasoning_df)

    # Non-reasoning section
    latex.append(r"\midrule")
    latex.append(
        rf"\multicolumn{{12}}{{c}}"
        rf"{{\cellcolor{{NonRBand}}\textsc{{Non-Reasoning Models}}}} \\"
    )
    latex.append(r"\midrule[\lightrulewidth]")
    _emit_rows(non_reasoning_df)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular*}")
    latex.append(r"}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


# ============================================================================
# MASTER GENERATOR
# ============================================================================
def generate_all_academic_tables(
    individual_results: Dict,
    mean_results: Dict,
    binary_results: Optional[Dict] = None,
    sum_results: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
    output_file: str = "academic_tables_reasoning.tex",
    custom_name_map: Optional[Dict] = None
) -> str:
    """
    Generate comprehensive suite of academic tables with reasoning model highlighting.
    """
    all_tables = []

    print("  ✓ Item-wise performance with reasoning grouping")
    all_tables.append(create_comprehensive_itemwise_table(
        individual_results,
        mean_results=mean_results,
        custom_name_map=custom_name_map,
        models_csv=models_csv
    ))
    all_tables.append("\n% " + "="*70 + "\n")

    full_latex = "\n\n".join(all_tables)


    with open(output_file, "w", encoding="utf-8") as f:
        f.write(full_latex)

    print("\n" + "=" * 70)
    print("✅ TABLES GENERATED")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print("=" * 70)

    return full_latex


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate item-wise LaTeX tables from get_results.py outputs"
    )
    parser.add_argument(
        "--results", type=str, default="../output/llamadrs_results.pkl",
        help="Path to results pickle produced by get_results.py",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="../output/table2_item_scores.tex",
        help="Output .tex file path",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with results_path.open("rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Run get_results.py first to create ../output/llamadrs_results.pkl")
        raise SystemExit(2)

    individual_results = data.get("individual_results", {})
    mean_results = data.get("mean_results", {})
    binary_results = data.get("binary_results", None)
    sum_results = data.get("sum_results", None)
    models_csv = data.get("models_csv", None)

    _ = generate_all_academic_tables(
        individual_results,
        mean_results,
        binary_results,
        sum_results,
        models_csv,
        output_file=str(out_path),
    )

    print("\n" + "=" * 70)
    print("Table generation complete!")
    print(f"Output file: {out_path}")
    print("=" * 70)