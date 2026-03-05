#!/usr/bin/env python3
"""
Enhanced LlaMADRS Academic Tables Suite with Reasoning Model Highlighting
Comprehensive LaTeX table generators with reasoning/non-reasoning visual distinction
ACL/AI2 clean style: bold best, teal/rose/gray cell tints.
Best/worst cells get darker shades of teal/rose.
Rounded-corner (pill-style) backgrounds on metric cells via TikZ.
Entire table wrapped in tcolorbox with rounded border.
Creative header with icons, formulas, sans-serif typography.
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
TOTAL_MAE_IS_NORMALIZED = False
TOTAL_MEANINGFUL_MAE    = 0.6 if TOTAL_MAE_IS_NORMALIZED else 6.0
TOTAL_SUBSTANTIAL_MAE   = 1.2 if TOTAL_MAE_IS_NORMALIZED else 12.0

# ============================================================================
# COLOR PREAMBLE — paste into your .tex before \begin{document}
# ============================================================================
COLOR_PREAMBLE = r"""
% --- LlaMADRS table colors (BIG-bench palette) -----------------------------
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{tikz}
\usepackage[most]{tcolorbox}
\usepackage{fontawesome5}
\usepackage{multirow}
% Teal ramp  (source #5B99A8)
\definecolor{tblGoodLt}{HTML}{D5E8ED}   % 20% — normal good cell
\definecolor{tblGoodDk}{HTML}{A1CCD8}   % 45% — best-in-column good cell
% Rose ramp  (source #C24168)
\definecolor{tblBadLt}{HTML}{F0D3DD}    % 20% — normal bad cell
\definecolor{tblBadDk}{HTML}{DCA0B0}    % 45% — worst-in-column bad cell
% Neutral
\definecolor{tblNeutral}{HTML}{ECEDEF}  % gray 20%
% Group header band — unified gray
\definecolor{grpBand}{HTML}{C5C9CC}
% Table border
\definecolor{tblBorder}{HTML}{D0D4D8}   % subtle gray border
% Header background
\definecolor{hdrBg}{HTML}{F4F6F8}
% Header accents
\definecolor{hdrIcon}{HTML}{4A8896}     % saturated teal for icons
\definecolor{hdrSub}{HTML}{7A8A90}      % muted gray for formulas / descriptors
% --- Rounded-corner cell background (pill style) ---------------------------
\newcommand{\rcell}[2]{%
  \tikz[baseline=(X.base)]{%
    \node[fill=#1, rounded corners=3pt,
          inner xsep=4pt, inner ysep=1.5pt,
          minimum height=3.2ex] (X) {#2};}%X
}
% --- Safe FontAwesome icon fallback ---------------------------------------
% (If an icon name ever changes, compilation will still succeed.)
\providecommand{\faBolt}{\faIcon{bolt}}
\providecommand{\faCalculator}{\faIcon{calculator}}
"""

# ============================================================================
# CELL COLOR HELPERS
# ============================================================================

def _color_mae(value: float, *, is_best: bool = False, is_worst: bool = False) -> str:
    """Return color name for MAE cell tint."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if v < TOTAL_MEANINGFUL_MAE:
        return "tblGoodDk" if is_best else "tblGoodLt"
    if v >= TOTAL_SUBSTANTIAL_MAE:
        return "tblBadDk" if is_worst else "tblBadLt"
    return "tblNeutral"


def _color_f1(value: float, *, q75: float, q25: float,
              is_best: bool = False, is_worst: bool = False) -> str:
    """Return color name for F1 cell tint."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if v >= q75:
        return "tblGoodDk" if is_best else "tblGoodLt"
    if v < q25:
        return "tblBadDk" if is_worst else "tblBadLt"
    return "tblNeutral"


def create_comprehensive_ranking_table(
    individual_results: Dict,
    binary_results: Dict,
    sum_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
    include_all: bool = False,
) -> str:
    """
    Total-score + F1 table with reasoning / non-reasoning grouping.
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


    def _format_size(total_params, active_params) -> str:
        tp = "" if total_params is None else str(total_params).strip()
        ap = "" if active_params is None else str(active_params).strip()
        if not tp and not ap:
            return ""
        if tp and ap and tp != ap:
            return f"{ap}--{tp}"
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
        display = _abbrev_name(model)
        model_base = display

        ctx = int(model_meta["context_length"])
        context_length = f"{int(ctx/1_000_000)}m" if ctx >= 1_000_000 else f"{int(ctx/1_000)}k"

        arch = str(model_meta.get("architecture", "")).strip()
        if arch.upper() == "MOE":
            moe_cell = "MoE"
        elif arch:
            moe_cell = "Dense"
        else:
            moe_cell = "---"

        reasoning = str(model_meta.get("reasoning", "")).strip() == "Yes"

        DTS_mae = np.nan
        DTS_mae_std = np.nan
        if model in individual_results and 0 in individual_results[model]:
            DTS_mae = individual_results[model][0].get("mae_mean", np.nan)
            DTS_mae_std = individual_results[model][0].get("mae_std", np.nan)

        sum_mae = np.nan
        sum_mae_std = np.nan
        if model in sum_results:
            sum_mae = sum_results[model].get("mae_mean", np.nan)
            sum_mae_std = sum_results[model].get("mae_std", np.nan)

        DTS_f1 = np.nan
        DTS_f1_std = np.nan
        sum_f1 = np.nan
        sum_f1_std = np.nan
        if model in binary_results:
            td = binary_results[model].get("total_direct", {})
            ts = binary_results[model].get("total_sum", {})
            DTS_f1 = td.get("f1_mean", np.nan)
            DTS_f1_std = td.get("f1_std", np.nan)
            sum_f1 = ts.get("f1_mean", np.nan)
            sum_f1_std = ts.get("f1_std", np.nan)

        rows.append({
            "model_id": model_id,
            "model_base": model_base,
            "moe": moe_cell,
            "context_length": context_length,
            "DTS_mae": DTS_mae,
            "DTS_mae_std": DTS_mae_std,
            "sum_mae": sum_mae,
            "sum_mae_std": sum_mae_std,
            "DTS_f1": DTS_f1,
            "DTS_f1_std": DTS_f1_std,
            "sum_f1": sum_f1,
            "sum_f1_std": sum_f1_std,
            "reasoning": reasoning,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("DTS_mae", na_position="last").reset_index(drop=True)

    # --- Per-column: best and worst values ---------------------------------
    def _best_worst(series: pd.Series, lower_is_better: bool):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None, None
        if lower_is_better:
            return float(s.min()), float(s.max())
        return float(s.max()), float(s.min())

    col_bw = {}
    for col, lower in [("DTS_mae", True), ("sum_mae", True),
                       ("DTS_f1", False), ("sum_f1", False)]:
        col_bw[col] = _best_worst(df[col], lower)

    # F1 quartiles for color thresholds
    def _finite_quartiles(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None
        return {"q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75))}

    f1_quartiles = {
        "DTS_f1": _finite_quartiles(df["DTS_f1"]),
        "sum_f1": _finite_quartiles(df["sum_f1"]),
    }

    def format_cell(m, s, col_name, *, is_lower_better: bool, tie_tol: float = 1e-6) -> str:
        """Bold best, rounded teal/rose/gray pill (no second-best styling)."""
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"

        main = f"{m:.2f}"
        bw_best, bw_worst = col_bw.get(col_name, (None, None))

        is_best = bw_best is not None and abs(m - bw_best) <= tie_tol
        is_worst = bw_worst is not None and abs(m - bw_worst) <= tie_tol

        if is_best:
            main = rf"\textbf{{{main}}}"

        # Resolve color name
        color = ""
        if "mae" in col_name:
            color = _color_mae(m, is_best=is_best, is_worst=is_worst)
        elif "f1" in col_name:
            fq = f1_quartiles.get(col_name)
            if fq:
                color = _color_f1(m, q75=fq["q75"], q25=fq["q25"],
                                  is_best=is_best, is_worst=is_worst)

        std_part = ""
        if isinstance(s, (int, float)) and np.isfinite(s):
            std_part = rf" {{\scriptsize $\pm${s:.2f}}}"

        inner = f"{main}{std_part}"

        if color:
            return rf"\rcell{{{color}}}{{{inner}}}"
        return inner

    # Split into groups
    reasoning_df = df[df["reasoning"]].reset_index(drop=True)
    non_reasoning_df = df[~df["reasoning"]].reset_index(drop=True)

    # --- build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\footnotesize")

    # Caption — separated MAE and F1 legends
    meaningful_str = f"{int(TOTAL_MEANINGFUL_MAE) if float(TOTAL_MEANINGFUL_MAE).is_integer() else TOTAL_MEANINGFUL_MAE:g}"
    substantial_str = f"{int(TOTAL_SUBSTANTIAL_MAE) if float(TOTAL_SUBSTANTIAL_MAE).is_integer() else TOTAL_SUBSTANTIAL_MAE:g}"
    all_f1 = pd.to_numeric(pd.concat([df["DTS_f1"], df["sum_f1"]], ignore_index=True), errors="coerce")
    all_f1 = all_f1[np.isfinite(all_f1)]
    f1_q25_str = f"{float(all_f1.quantile(0.25)):.2f}" if not all_f1.empty else "Q1"
    f1_q75_str = f"{float(all_f1.quantile(0.75)):.2f}" if not all_f1.empty else "Q3"

    latex.append(
        r"\caption{Total-score evaluation: reasoning vs.\ non-reasoning. "
        r"\textbf{DTS}: $\hat{T}{=}f_{\theta}(x)$; "
        r"\textbf{ItS}: $\hat{T}{=}\sum\hat{y}_i$. "
        r"MoE sizes: Active--Total params. "
        r"\textbf{Bold}\,=\,best; darker\,=\,best/worst. "
        r"\textbf{MAE}: "
        r"\protect\colorbox{tblGoodLt}{\strut Acceptable} (${<}$\," + meaningful_str + r"), "
        r"\protect\colorbox{tblBadLt}{\strut Substantial} (${\geq}$\," + substantial_str + r"). "
        r"\textbf{F1} ($T{\geq}20$): "
        r"\protect\colorbox{tblGoodLt}{\strut ${\geq}$\,Q3} (" + f1_q75_str + r"), "
        r"\protect\colorbox{tblBadLt}{\strut ${<}$\,Q1} (" + f1_q25_str + r").}"
    )
    latex.append(r"\label{tab:table2_total_scores}")

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
    latex.append(r"  before upper={\arrayrulecolor{tblBorder}\renewcommand{\arraystretch}{1.35}\setlength{\tabcolsep}{3pt}},")
    latex.append(r"]")

    # Tabularx inside the rounded box
    latex.append(
        r"\begin{tabularx}{\linewidth}{@{} "
        r">{\hsize=0.46\hsize\raggedright\arraybackslash}X "
        r">{\hsize=0.12\hsize\centering\arraybackslash}X "
        r">{\hsize=0.12\hsize\centering\arraybackslash}X "
        r">{\hsize=0.17\hsize\centering\arraybackslash}X "
        r">{\hsize=0.17\hsize\centering\arraybackslash}X "
        r">{\hsize=0.17\hsize\centering\arraybackslash}X "
        r">{\hsize=0.17\hsize\centering\arraybackslash}X "
        r"@{}}"
    )
    latex.append(r"\arrayrulecolor{tblBorder}")

    # Row 1: sub-column headers
    latex.append(
        r"\rcell{tblNeutral}{\textsf{Model}\,{\scriptsize\textcolor{hdrSub}{(Size)}}}"
        r" & \rcell{tblNeutral}{\textsf{Arch.}}"
        r" & \rcell{tblNeutral}{\textsf{Ctx.}}"
        r" & \multicolumn{2}{c}{"
            r"\rcell{tblNeutral}{\textsf{MAE}\,{\scriptsize$\downarrow$}}"
        r"}"
        r" & \multicolumn{2}{c}{"
            r"\rcell{tblNeutral}{\textsf{F1}\,{\scriptsize$\uparrow$}}"
        r"} \\"
    )
    latex.append(r"\addlinespace[3pt]")

    def _emit_group(title: str, frame: pd.DataFrame):
        latex.append(
            r"\rowcolor{tblNeutral} \multicolumn{3}{@{}l@{}}"
            rf"{{\small\textsc{{{title}}}}} & \textsc{{DTS}} &  \textsc{{ItS}} & \textsc{{DTS}} &  \textsc{{ItS}}\\"
        )
        latex.append(r"\addlinespace[1.5pt]")
        for _, row in frame.iterrows():
            name = row["model_base"]
            line = (
                rf"{name} & {row['moe']} & {row['context_length']} & "
                f"{format_cell(row['DTS_mae'], row['DTS_mae_std'], 'DTS_mae', is_lower_better=True)} & "
                f"{format_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)} & "
                f"{format_cell(row['DTS_f1'], row['DTS_f1_std'], 'DTS_f1', is_lower_better=False)} & "
                f"{format_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)} \\\\"
            )
            latex.append(line)
        latex.append(r"\addlinespace[1.5pt]")

    _emit_group("Reasoning Models", reasoning_df)
    _emit_group("Non-Reasoning Models", non_reasoning_df)

    latex.append(r"\end{tabularx}")
    latex.append(r"\end{tcolorbox}")
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
    print("🎨 GENERATING ACADEMIC TABLES (ACL/AI2 STYLE)")
    print("=" * 70)

    print("  ✓ Comprehensive ranking with reasoning highlights")
    all_tables.append(create_comprehensive_ranking_table(
        individual_results, binary_results, sum_results, custom_name_map, models_csv
    ))
    all_tables.append("\n% " + "="*70 + "\n")

    full_latex = "\n\n".join(all_tables)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)

    print("\n" + "=" * 70)
    print("✅ TABLES GENERATED")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
    print(f"📋 Add this to your preamble:\n{COLOR_PREAMBLE}")
    print("=" * 70)

    return full_latex

# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Table 2 (total-score) LaTeX from get_results.py outputs"
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
        default="../output/table2_total_scores.tex",
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