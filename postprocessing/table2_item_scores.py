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
    "segmented_Qw3_30b_a3b_ar_4q_NoR": "Qwen 3 (3B-30B): No Reasoning",
    "segmented_Gen3_27b_it_gptq_4q": "Gemma 3 (27B) IT",
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
    "segmented_Qw3_Next_80b_a3b_ar_4q": "Qwen 3 Next (80B)",
    "segmented_Qw3_Next_80b_a3b_ar_4q_NoR": "Qwen 3 Next (80B): No Reasoning",
}
# add ablations
for ablation in ["raw", "no_desc", "no_dem"]:
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_{ablation}"] = f"Qwen 3 Next (80B): {ablation.replace('_', ' ').title()}"
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_NoR_{ablation}"] = f"Qwen 3 Next (80B): No Reasoning, {ablation.replace('_', ' ').title()}"

MODEL_RANKS = {k: i for i, k in enumerate(MODEL_DICT.keys(), start=1)}
MODEL_REV_DICT = {v: k for k, v in MODEL_DICT.items()}

# ============================================================================
# VISUAL DESIGN + CLINICAL THRESHOLDS
# ============================================================================
# ---- Thresholds (unchanged) ------------------------------------------------
ITEM_MEANINGFUL_MAE   = 0.6   # "Acceptable" in your example legend
ITEM_SUBSTANTIAL_MAE  = 1.2   # "Substantial" in your example legend


def get_cell_color_item_mae(value: float, *, for_mean_row: bool = False) -> str:
    """
    Match the example output:
      - < 0.6   -> green!15
      - >= 1.2  -> red!15
      - else    -> white!15 for normal model rows, and no fill for mean summary rows
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""

    if v < ITEM_MEANINGFUL_MAE:
        return r"\cellcolor{green!15}"
    if v >= ITEM_SUBSTANTIAL_MAE:
        return r"\cellcolor{red!15}"

    return "" if for_mean_row else r"\cellcolor{white!15}"


def create_comprehensive_itemwise_table(
    individual_results: Dict,
    mean_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
) -> str:
    """
    Item-wise MAE table formatted to match the provided example output.
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

    # Build rows from model ids in results
    for model in sorted(individual_results.keys()):
        
        model_id = MODEL_REV_DICT.get(model, model)
        # keep filtering logic based on model_id (internal id)
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

        # Overall (mean across items)
        d_overall = mean_results.get(model, {})
        entry["I11_mean"] = d_overall.get("mae_mean", np.nan)
        entry["I11_std"]  = d_overall.get("mae_std", np.nan)

        entry["is_reasoning"] = str(model_row.get("reasoning", "")).strip() == "Yes"
        rows.append(entry)

    df = pd.DataFrame(rows).sort_values(by="I11_mean").reset_index(drop=True)

    # Best/worst markers per column computed BEFORE adding mean summary rows
    col_stats = {}
    for i in item_indices + [11]:
        col = pd.to_numeric(df[f"I{i}_mean"], errors="coerce")
        good = col[np.isfinite(col)]
        col_stats[i] = {"min": float(good.min()), "max": float(good.max())} if not good.empty else None

    # Split by reasoning
    reasoning_df = df[df["is_reasoning"]].copy()
    non_reasoning_df = df[~df["is_reasoning"]].copy()

    # Add group mean rows (placed at bottom of each group)
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

    # Build LaTeX
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Comprehensive Item-wise Performance: "
        r"\textcolor{black}{\fcolorbox{white}{green!15}{\strut\enspace}}\ $<\,0.6$ \emph{(Acceptable)},"
        r"\textcolor{black}{\fcolorbox{white}{red!15}{\strut\enspace}}\ $\geq\,1.2$ \emph{(Substantial)}.}"
    )
    latex.append(r"\label{tab:comprehensive_itemwise_reasoning}")
    latex.append(r"\renewcommand{\arraystretch}{1.05}")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    latex.append(r"{\footnotesize")
    latex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccccc}")
    latex.append(r"\toprule")
    latex.append(
        r"\rowcolor{gray!15} \textbf{Model} & "
        + " & ".join([f"\\textbf{{I{i}}}" for i in item_indices])
        + r" & \textbf{Mean} \\"
    )
    latex.append(r"\midrule")

    def _emit_rows(frame: pd.DataFrame):
        for _, rr in frame.iterrows():
            model_text = str(rr["Model"])
            is_summary = ("Mean (" in model_text)

            # Model column forced white background (matches example)
            cells = [r"\cellcolor{white} " + model_text]

            for i in item_indices + [11]:
                m = rr.get(f"I{i}_mean", np.nan)
                s = rr.get(f"I{i}_std", np.nan)

                if not (isinstance(m, (int, float)) and np.isfinite(m)) or not (isinstance(s, (int, float)) and np.isfinite(s)):
                    cells.append(r"\textemdash{}")
                    continue

                # Match example formatting
                value_main = f"{m:.2f}"
                value_str = f"{value_main} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"

                # Best/worst markers only for real model rows (not group means)
                stat = col_stats.get(i)
                if (not is_summary) and stat:
                    if abs(float(m) - stat["min"]) <= 1e-6:
                        value_str = f"\\textbf{{{value_main}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
                    elif abs(float(m) - stat["max"]) <= 1e-6:
                        value_str = f"\\textit{{{value_main}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"

                cell_color = get_cell_color_item_mae(m, for_mean_row=is_summary)

                if cell_color:
                    cells.append(f"\\makecell{{{cell_color}{value_str}}}")
                else:
                    cells.append(f"\\makecell{{{value_str}}}")

            latex.append(" & ".join(cells) + r" \\")

    # Reasoning section
    latex.append(r"\multicolumn{12}{c}{\cellcolor{gray!10}\textbf{Reasoning Models}} \\")
    latex.append(r"\midrule[0.5pt]")
    _emit_rows(reasoning_df)

    # Non-reasoning section
    latex.append(r"")
    latex.append(r"\midrule[1pt]")
    latex.append(r"\multicolumn{12}{c}{\cellcolor{gray!10}\textbf{Non-Reasoning Models}} \\")
    latex.append(r"\midrule[0.5pt]")
    _emit_rows(non_reasoning_df)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular*}")
    latex.append(r"}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


# ============================================================================
# 4. MASTER GENERATOR (fix missing all_tables init)
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
    all_tables = []  # <-- FIX: initialize

    print("  ✓ Item-wise performance with reasoning grouping")
    all_tables.append(create_comprehensive_itemwise_table(
        individual_results,
        mean_results=mean_results,
        custom_name_map=custom_name_map,
        models_csv=models_csv
    ))
    all_tables.append("\n% " + "="*70 + "\n")

    full_latex = "\n\n".join(all_tables)

    header = r"""% ============================================================================
"""
    full_latex = header + full_latex

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
    # Example usage with your data
    try:
        with open("madrs_analysis_results.pkl", "rb") as f:
            data = pickle.load(f)
            individual_results = data["individual_results"]
            mean_results = data["mean_results"]
            binary_results = data.get("binary_results", None)
            sum_results = data.get("sum_results", None)
            models_csv = data.get("models_csv", None)

        # Generate tables (toggle ranking table above if desired)
        latex_output = generate_all_academic_tables(
            individual_results,
            mean_results,
            binary_results,
            sum_results,
            models_csv,
            output_file="table_model_items.tex",
        )

        print("\n" + "="*70)
        print("Table generation complete!")
        print("="*70)

    except FileNotFoundError:
        print("Data file not found. Please run analysis first.")
        print("Example data structure needed:")
        print("  - individual_results: Dict[model_name, Dict[item, metrics]]")
        print("  - mean_results: Dict[model_name, metrics]")
        print("  - binary_results: Dict[model_name, classification_metrics]")
