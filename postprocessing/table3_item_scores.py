#!/usr/bin/env python3
"""
Enhanced LlaMADRS Academic Tables Suite with Reasoning Model Highlighting
Comprehensive LaTeX table generators with reasoning/non-reasoning visual distinction
ACL/AI2 clean style: bold best, teal/rose/gray cell tints.
Best/worst cells get darker shades of teal/rose.
Rounded-corner (pill-style) backgrounds on metric cells via TikZ.
Entire table wrapped in tcolorbox with rounded border.
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
# CLINICAL THRESHOLDS
# ============================================================================
ITEM_MEANINGFUL_MAE   = 0.6
ITEM_SUBSTANTIAL_MAE  = 1.2


# ============================================================================
# CELL COLOR HELPER
# ============================================================================

def _color_item_mae(
    value: float,
    *,
    is_best: bool = False,
    is_worst: bool = False,
) -> str:
    """Return color name for item-level MAE cell tint."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if v < ITEM_MEANINGFUL_MAE:
        return "tblGoodDk" if is_best else "tblGoodLt"
    if v >= ITEM_SUBSTANTIAL_MAE:
        return "tblBadDk" if is_worst else "tblBadLt"
    return "tblNeutral"


def create_comprehensive_itemwise_table(
    individual_results: Dict,
    mean_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
) -> str:
    """
    Item-wise MAE table with reasoning / non-reasoning grouping.
    ACL/AI2 style with tcolorbox, rcell pills, teal/rose palette.
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

    # Per-column best/worst
    col_bw = {}
    for i in item_indices + [11]:
        col = pd.to_numeric(df[f"I{i}_mean"], errors="coerce")
        good = col[np.isfinite(col)]
        if not good.empty:
            col_bw[i] = (float(good.min()), float(good.max()))
        else:
            col_bw[i] = (None, None)

    # Split by reasoning
    reasoning_df = df[df["is_reasoning"]].copy()
    non_reasoning_df = df[~df["is_reasoning"]].copy()

    # --- Format cell --------------------------------------------------------
    def _format_item_cell(m, s, col_idx, *, is_summary: bool) -> str:
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"
        if not (isinstance(s, (int, float)) and np.isfinite(s)):
            return r"\textemdash{}"

        main = f"{m:.2f}"

        bw_best, bw_worst = col_bw.get(col_idx, (None, None))
        is_best = (not is_summary) and bw_best is not None and abs(m - bw_best) <= 1e-6
        is_worst = (not is_summary) and bw_worst is not None and abs(m - bw_worst) <= 1e-6

        if is_best:
            main = rf"\textbf{{{main}}}"

        color = _color_item_mae(m, is_best=is_best, is_worst=is_worst)

        # Two-line cell: mean on top, ± std below
        top_line = main
        bot_line = rf"{{\tiny $\pm$ {s:.2f}}}"

        if color:
            return rf"\rcell{{{color}}}{{\makecell{{{top_line} \\ {bot_line}}}}}"
        return rf"\makecell{{{top_line} \\ {bot_line}}}"

    # --- Build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\footnotesize")
    latex.append(
        r"\caption{Item-wise MAE\,$\pm$\,std for I1--I10 with mean. "
        r"Formatted as Table~\ref{tab:table2_total_scores}.}"
    )
    latex.append(r"\label{tab:table3_item_scores}")

    # --- Rounded table frame ------------------------------------------------
    latex.append(r"\begin{tcolorbox}[")
    latex.append(r"  enhanced,")
    latex.append(r"  boxrule=0.5pt,")
    latex.append(r"  colframe=tblBorder,")
    latex.append(r"  colback=white,")
    latex.append(r"  arc=8pt,")
    latex.append(r"  outer arc=8pt,")
    latex.append(r"  left=3pt, right=3pt, top=3pt, bottom=3pt,")
    latex.append(r"  boxsep=0pt,")
    latex.append(r"  before upper={\arrayrulecolor{tblBorder}\renewcommand{\arraystretch}{1.35}\setlength{\tabcolsep}{2pt}},")
    latex.append(r"]")

    # 12 columns: Model + I1..I10 + Mean
    latex.append(
        r"\begin{tabularx}{\linewidth}{@{} "
        r">{\hsize=0.30\hsize\raggedright\arraybackslash}X "
        + r" ".join([r">{\hsize=0.07\hsize\centering\arraybackslash}X"] * 10)
        + r" !{\color{tblBorder}\vrule width 0.5pt}"
        + r" >{\hsize=0.07\hsize\centering\arraybackslash}X"
        r" @{}}"
    )
    latex.append(r"\arrayrulecolor{tblBorder}")

    # Header row
    item_headers = " & ".join([rf"\textsf{{I{i}}}" for i in item_indices])
    latex.append(
        r"\rcell{tblNeutral}{\textsf{Model}\,{\scriptsize\textcolor{hdrSub}{(Size)}}}"
        r" & " + " & ".join([rf"\rcell{{tblNeutral}}{{\textsf{{I{i}}}}}" for i in item_indices]) +
        r" & \rcell{tblNeutral}{\textsf{Mean}} \\"
    )
    latex.append(r"\addlinespace[3pt]")

    # --- Emit group ---------------------------------------------------------
    def _emit_group(title: str, frame: pd.DataFrame):
        latex.append(
            r"\rowcolor{tblNeutral} \multicolumn{12}{@{}l@{}}"
            rf"{{\small\textsc{{{title}}}}} \\"
        )
        latex.append(r"\addlinespace[3pt]")
        for _, rr in frame.iterrows():
            model_text = str(rr["Model"])
            cells = [model_text]
            for i in item_indices + [11]:
                cells.append(_format_item_cell(
                    rr.get(f"I{i}_mean", np.nan),
                    rr.get(f"I{i}_std", np.nan),
                    i,
                    is_summary=False,
                ))
            latex.append(" & ".join(cells) + r" \\")
        latex.append(r"\addlinespace[3pt]")

    _emit_group("Reasoning Models", reasoning_df)
    _emit_group("Non-Reasoning Models", non_reasoning_df)

    latex.append(r"\end{tabularx}")
    latex.append(r"\end{tcolorbox}")
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

    print("=" * 70)
    print("🎨 GENERATING ACADEMIC TABLES (ACL/AI2 STYLE)")
    print("=" * 70)

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
        description="Generate Table 3 (item-wise) LaTeX from get_results.py outputs"
    )
    parser.add_argument(
        "--results", type=str, default="../output/llamadrs_results.pkl",
        help="Path to results pickle produced by get_results.py",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="../output/table3_item_scores.tex",
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