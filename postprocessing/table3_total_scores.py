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
    "segmented_Qw3_Next_80b_a3b_ar_4q_NoR": "Qwen 3 Next: No Reasoning (3B-80B)",
}
# add ablations
for ablation in ["raw", "no_desc", "no_dem"]:
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_{ablation}"] = f"Qwen 3 Next: {ablation.replace('_', ' ').title()} (3B-80B)"
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_NoR_{ablation}"] = f"Qwen 3 Next: No Reasoning, {ablation.replace('_', ' ').title()} (3B-80B)"

MODEL_RANKS = {k: i for i, k in enumerate(MODEL_DICT.keys(), start=1)}
MODEL_REV_DICT = {v: k for k, v in MODEL_DICT.items()}

# ============================================================================
# VISUAL DESIGN + CLINICAL THRESHOLDS
# ============================================================================
# ---- Total-score thresholds (as you already have) --------------------------
TOTAL_MAE_IS_NORMALIZED = False
TOTAL_MEANINGFUL_MAE    = 0.6 if TOTAL_MAE_IS_NORMALIZED else 6.0
TOTAL_SUBSTANTIAL_MAE   = 1.2 if TOTAL_MAE_IS_NORMALIZED else 12.0


def get_cell_color_total_mae(value: float, *, for_summary: bool = False) -> str:
    """
    Match example for TOTAL MAE coloring:
      - < 6   -> green!15
      - >=12  -> red!15
      - else  -> white!15 (and keep white!15 also for summary rows; example does that)
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""

    if v < TOTAL_MEANINGFUL_MAE:
        return r"\cellcolor{green!15}"
    if v >= TOTAL_SUBSTANTIAL_MAE:
        return r"\cellcolor{red!15}"
    return r"\cellcolor{white!15}"


def create_comprehensive_ranking_table(
    individual_results: Dict,
    binary_results: Dict,
    sum_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
    include_all: bool = False,
) -> str:
    """
    Total-score + F1 table formatted to match your example output.
    """

    def _format_size(total_params, active_params) -> str:
        """Format size like table2: main value + tiny parenthetical."""
        tp = "" if total_params is None else str(total_params).strip()
        ap = "" if active_params is None else str(active_params).strip()
        if not tp and not ap:
            return ""
        if tp and ap and tp != ap:
            return f"{ap}-{tp}"
        return tp or ap

    # --- build rows ---------------------------------------------------------
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

        model_meta = models_csv[model_id]

        # Display name: from MODEL_DICT or fallback to id
        display = MODEL_DICT.get(model_id, model_id)
        if custom_name_map and display in custom_name_map:
            display = custom_name_map[display]

        # Match example: show base name only in first column (no (..B) part there)
        model_base = display.split("(")[0].strip()

        # Context length formatting like your code, but example uses k / m
        ctx = int(model_meta["context_length"])
        context_length = f"{int(ctx/1_000_000)}m" if ctx >= 1_000_000 else f"{int(ctx/1_000)}k"

        # Size column: match table2 formatting (tiny parenthetical when total!=active)
        total_params = model_meta.get("total_params", "")
        active_params = model_meta.get("active_params", "")
        size = _format_size(total_params, active_params)

        # MOE column: example shows "No"/"MoE"/"Dense" depending on your metadata.
        # Your metadata has "architecture" (e.g., "MOE" or "Dense"). We'll map to example style.
        arch = str(model_meta.get("architecture", "")).strip()
        if arch.upper() == "MOE":
            moe_cell = "MoE"
        elif arch:
            moe_cell = "Dense"
        else:
            moe_cell = "No"

        # Reasoning flag for grouping
        reasoning = str(model_meta.get("reasoning", "")).strip() == "Yes"

        # Direct total MAE (item 0 convention)
        direct_mae = np.nan
        direct_mae_std = np.nan
        if model in individual_results and 0 in individual_results[model]:
            direct_mae = individual_results[model][0].get("mae_mean", np.nan)
            direct_mae_std = individual_results[model][0].get("mae_std", np.nan)

        # Sum MAE
        sum_mae = np.nan
        sum_mae_std = np.nan
        if model in sum_results:
            sum_mae = sum_results[model].get("mae_mean", np.nan)
            sum_mae_std = sum_results[model].get("mae_std", np.nan)

        # F1s
        direct_f1 = np.nan
        direct_f1_std = np.nan
        sum_f1 = np.nan
        sum_f1_std = np.nan
        if model in binary_results:
            td = binary_results[model].get("total_direct", {})
            ts = binary_results[model].get("total_sum", {})
            direct_f1 = td.get("f1_mean", np.nan)
            direct_f1_std = td.get("f1_std", np.nan)
            sum_f1 = ts.get("f1_mean", np.nan)
            sum_f1_std = ts.get("f1_std", np.nan)

        rows.append({
            "model_id": model_id,
            "model_base": model_base,
            "size": size,
            "moe": moe_cell,
            "context_length": context_length,
            "direct_mae": direct_mae,
            "direct_mae_std": direct_mae_std,
            "sum_mae": sum_mae,
            "sum_mae_std": sum_mae_std,
            "direct_f1": direct_f1,
            "direct_f1_std": direct_f1_std,
            "sum_f1": sum_f1,
            "sum_f1_std": sum_f1_std,
            "reasoning": reasoning,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("direct_mae", na_position="last").reset_index(drop=True)

    # Stats for bold/italic (best/worst) per metric column
    def _finite_minmax(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None
        return {"min": float(s.min()), "max": float(s.max())}

    col_stats = {
        "direct_mae": _finite_minmax(df["direct_mae"]),
        "sum_mae": _finite_minmax(df["sum_mae"]),
        "direct_f1": _finite_minmax(df["direct_f1"]),
        "sum_f1": _finite_minmax(df["sum_f1"]),
    }

    def format_cell(m, s, col_name, *, is_lower_better: bool, tie_tol: float = 1e-6) -> str:
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"

        main = f"{m:.2f}"
        stat = col_stats.get(col_name)

        # bold/italic best/worst like your example (they bold some, italic some)
        if stat:
            if is_lower_better:
                if abs(m - stat["min"]) <= tie_tol:
                    main = rf"\textbf{{{main}}}"
                elif abs(m - stat["max"]) <= tie_tol:
                    main = rf"\textit{{{main}}}"
            else:
                if abs(m - stat["max"]) <= tie_tol:
                    main = rf"\textbf{{{main}}}"
                elif abs(m - stat["min"]) <= tie_tol:
                    main = rf"\textit{{{main}}}"

        # color rules:
        if "mae" in col_name:
            color = get_cell_color_total_mae(m)
        else:
            color = r"\cellcolor{white!15}"

        # example uses: \makecell{\cellcolor{X}5.90 {\tiny $\pm$ 0.17}}
        if not (isinstance(s, (int, float)) and np.isfinite(s)):
            return rf"\makecell{{{color}{main}}}"
        return rf"\makecell{{{color}{main} {{\tiny $\pm$ {s:.2f}}}}}"

    # Split into groups
    reasoning_df = df[df["reasoning"]].reset_index(drop=True)
    non_reasoning_df = df[~df["reasoning"]].reset_index(drop=True)

    # --- build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(
        r"\caption{Comprehensive Model Performance: "
        r"\textcolor{black}{\fcolorbox{white}{green!15}{\strut\enspace}}\ $<\,"
        + f"{int(TOTAL_MEANINGFUL_MAE) if float(TOTAL_MEANINGFUL_MAE).is_integer() else TOTAL_MEANINGFUL_MAE:g}"
        + r"$ \emph{(Acceptable)}, "
        r"\textcolor{black}{\fcolorbox{white}{red!15}{\strut\enspace}}\ $\geq\,"
        + f"{int(TOTAL_SUBSTANTIAL_MAE) if float(TOTAL_SUBSTANTIAL_MAE).is_integer() else TOTAL_SUBSTANTIAL_MAE:g}"
        + r"$ \emph{(Substantial)}.}"
    )
    latex.append(r"\label{tab:comprehensive_ranking}")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    latex.append(r"\small")
    latex.append(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}X c c c | c | c | c |c}")
    latex.append(r"\toprule")

    # two-row header (exact structure from your example)
    latex.append(
        r"% --- two-row header ---"
    )
    latex.append(
        r"\cellcolor{white!7} & \cellcolor{white!7} & \cellcolor{white!7} & \cellcolor{white!7} &"
        r"\multicolumn{2}{c}{\cellcolor{white!7}\textbf{MAE} $\downarrow$} &"
        r"\multicolumn{2}{c}{\cellcolor{white!7}\textbf{F1} $\uparrow$ (Total $\geq$ 20)} \\"
    )
    latex.append(
        r"\multirow{-2}{*}{\cellcolor{white!7}\textbf{Model}} &"
        r"\multirow{-2}{*}{\cellcolor{white!7}\textbf{Size}} &"
        r"\multirow{-2}{*}{\cellcolor{white!7}\textbf{MOE}} &"
        r"\multirow{-2}{*}{\cellcolor{white!7}\textbf{Context Length}} &"
        r"\cellcolor{white!7}\textbf{Direct} & \cellcolor{white!7}\textbf{Sum} &"
        r"\cellcolor{white!7}\textbf{Direct} & \cellcolor{white!7}\textbf{Sum} \\"
    )
    latex.append(r"\midrule")
    latex.append(r"")
    latex.append(r"\midrule")

    def _emit_group(title: str, frame: pd.DataFrame):
        latex.append(rf"\multicolumn{{8}}{{c}}{{\cellcolor{{white!10}}\textbf{{{title}}}}} \\")
        latex.append(r"\midrule[0.5pt]")
        for _, row in frame.iterrows():
            model_cell = r"\cellcolor{white} " + row["model_base"]
            line = (
                f"{model_cell} & {row['size']} & {row['moe']} & {row['context_length']} & "
                f"{format_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)} & "
                f"{format_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)} & "
                f"{format_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)} & "
                f"{format_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)} \\\\"
            )
            latex.append(line)

    _emit_group("Reasoning Models", reasoning_df)
    latex.append(r"\midrule[1pt]")
    _emit_group("Non-Reasoning Models", non_reasoning_df)

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")
    latex.append(r"\vspace{2mm}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


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

    print("=" * 70)
    print("🎨 GENERATING ACADEMIC TABLES WITH REASONING HIGHLIGHTS")
    print("=" * 70)


    print("  ✓ Comprehensive ranking with reasoning highlights")
    all_tables.append(create_comprehensive_ranking_table(
        individual_results, binary_results, sum_results, custom_name_map, models_csv
    ))
    all_tables.append("\n% " + "="*70 + "\n")

    full_latex = "\n\n".join(all_tables)

    header = r"""% ============================================================================
"""
    full_latex = header + full_latex

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)

    print("\n" + "=" * 70)
    print("✅ TABLES WITH REASONING HIGHLIGHTS GENERATED")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print(f"🧠 Reasoning models clearly distinguished with violet highlighting")
    print("=" * 70)

    return full_latex

# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate total-score LaTeX tables from get_results.py outputs"
    )
    parser.add_argument(
        "--results",
        type=str,
        default="../output/llamadrs_results.pkl",
        help="Path to results pickle produced by get_results.py",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../output/table3_total_scores.tex",
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
    binary_results = data.get("binary_results", {}) or {}
    sum_results = data.get("sum_results", {}) or {}
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
