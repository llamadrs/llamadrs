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
TOTAL_MAE_IS_NORMALIZED = False
TOTAL_MEANINGFUL_MAE    = 0.6 if TOTAL_MAE_IS_NORMALIZED else 6.0
TOTAL_SUBSTANTIAL_MAE   = 1.2 if TOTAL_MAE_IS_NORMALIZED else 12.0

# F1 thresholds are computed from data quartiles (Q3 = good, Q1 = poor)

# ---- Color palette (HTML hex) ---------------------------------------------


def get_cell_color_total_mae(
    value: float,
    *,
    for_summary: bool = False,
    best: bool = False,
    worst: bool = False,
) -> str:
    """
    Total-score MAE coloring:
      - < meaningful  -> LightGreen / DarkGreen (if best/worst)
      - >= substantial -> LightRed / DarkRed
      - else -> LightGray / DarkGray
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""

    highlight = best or worst

    if v < TOTAL_MEANINGFUL_MAE:
        return r"\cellcolor{DarkGreen}" if highlight else r"\cellcolor{LightGreen}"
    if v >= TOTAL_SUBSTANTIAL_MAE:
        return r"\cellcolor{DarkRed}" if highlight else r"\cellcolor{LightRed}"

    return r"\cellcolor{DarkGray}" if highlight else r"\cellcolor{LightGray}"


def get_cell_color_f1(
    value: float,
    *,
    q75: float,
    q25: float,
    best: bool = False,
    worst: bool = False,
) -> str:
    """
    F1 coloring (higher is better) based on data quartiles:
      - >= Q3 (75th pct) -> LightGreen / DarkGreen
      - <  Q1 (25th pct) -> LightRed / DarkRed
      - else -> LightGray / DarkGray
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""

    highlight = best or worst

    if v >= q75:
        return r"\cellcolor{DarkGreen}" if highlight else r"\cellcolor{LightGreen}"
    if v < q25:
        return r"\cellcolor{DarkRed}" if highlight else r"\cellcolor{LightRed}"

    return r"\cellcolor{DarkGray}" if highlight else r"\cellcolor{LightGray}"


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

        display = model
        display = _abbrev_name(display)

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

    best_model_id = None
    worst_model_id = None
    mae_series = pd.to_numeric(df["direct_mae"], errors="coerce")
    finite = mae_series[np.isfinite(mae_series)]
    if not finite.empty:
        best_model_id = df.loc[finite.idxmin(), "model_id"]
        worst_model_id = df.loc[finite.idxmax(), "model_id"]

    # Per-column min/max for bold/italic
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

    # Compute F1 quartiles from actual data for color thresholds
    def _finite_quartiles(series: pd.Series):
        s = pd.to_numeric(series, errors="coerce")
        s = s[np.isfinite(s)]
        if s.empty:
            return None
        return {"q25": float(s.quantile(0.25)), "q75": float(s.quantile(0.75))}

    f1_quartiles = {
        "direct_f1": _finite_quartiles(df["direct_f1"]),
        "sum_f1": _finite_quartiles(df["sum_f1"]),
    }

    def format_cell(m, s, col_name, *, is_lower_better: bool, tie_tol: float = 1e-6) -> str:
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"

        main = f"{m:.2f}"
        stat = col_stats.get(col_name)

        is_best = False
        is_worst = False

        if stat:
            if is_lower_better:
                if abs(m - stat["min"]) <= tie_tol:
                    is_best = True
                    main = rf"\textbf{{{main}}}"
                elif abs(m - stat["max"]) <= tie_tol:
                    is_worst = True
                    main = rf"\textit{{{main}}}"
            else:
                if abs(m - stat["max"]) <= tie_tol:
                    is_best = True
                    main = rf"\textbf{{{main}}}"
                elif abs(m - stat["min"]) <= tie_tol:
                    is_worst = True
                    main = rf"\textit{{{main}}}"

        if "mae" in col_name:
            color = get_cell_color_total_mae(m, best=is_best, worst=is_worst)
        elif "f1" in col_name:
            fq = f1_quartiles.get(col_name)
            if fq:
                color = get_cell_color_f1(m, q75=fq["q75"], q25=fq["q25"],
                                          best=is_best, worst=is_worst)
            else:
                color = r"\cellcolor{DarkGray}" if (is_best or is_worst) else r"\cellcolor{LightGray}"
        else:
            color = r"\cellcolor{DarkGray}" if (is_best or is_worst) else r"\cellcolor{LightGray}"

        std_part = ""
        if isinstance(s, (int, float)) and np.isfinite(s):
            std_part = rf" \scriptsize{{$\pm${s:.2f}}}"

        return rf"{color} {main}{std_part}"

    # Split into groups
    reasoning_df = df[df["reasoning"]].reset_index(drop=True)
    non_reasoning_df = df[~df["reasoning"]].reset_index(drop=True)

    # --- build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\renewcommand{\arraystretch}{1.15}")
    latex.append(r"\setlength{\tabcolsep}{3pt}")
    latex.append(r"\small")

    # Caption — use combined F1 quartiles across DTS and ItS for legend
    meaningful_str = f"{int(TOTAL_MEANINGFUL_MAE) if float(TOTAL_MEANINGFUL_MAE).is_integer() else TOTAL_MEANINGFUL_MAE:g}"
    substantial_str = f"{int(TOTAL_SUBSTANTIAL_MAE) if float(TOTAL_SUBSTANTIAL_MAE).is_integer() else TOTAL_SUBSTANTIAL_MAE:g}"
    all_f1 = pd.to_numeric(pd.concat([df["direct_f1"], df["sum_f1"]], ignore_index=True), errors="coerce")
    all_f1 = all_f1[np.isfinite(all_f1)]
    f1_q25_val = f"{float(all_f1.quantile(0.25)):.2f}" if not all_f1.empty else "Q1"
    f1_q75_val = f"{float(all_f1.quantile(0.75)):.2f}" if not all_f1.empty else "Q3"
    latex.append(
        r"\caption{Reasoning vs.\ non-reasoning models on MADRS total scoring. "
        r"\textbf{DTS}: Direct Total Scoring (single LLM call). "
        r"\textbf{ItS}: Item-then-Sum (10 item predictions summed post-hoc). "
        r"For MoE models, size is Active--Total parameters. "
        r"\textbf{Bold} = best; \textit{italic} = worst. \\"
        r"\textcolor{black}{\fcolorbox{white}{LightGreen}{\strut\enspace}}~MAE~$<$\," + meaningful_str + r" (acceptable) / F1~$\geq$\," + f1_q75_val + r" (Q3), "
        r"\textcolor{black}{\fcolorbox{white}{LightRed}{\strut\enspace}}~MAE~$\geq$\," + substantial_str + r" (substantial) / F1~$<$\," + f1_q25_val + r" (Q1).}"
    )
    latex.append(r"\label{tab:comprehensive_ranking}")
    latex.append(r"\vspace{1mm}")

    # Pure booktabs — no vertical rules
    latex.append(r"\begin{tabularx}{\textwidth}{@{} >{\raggedright\arraybackslash}X c c c c c c @{}}")
    latex.append(r"\toprule")

    # Clean two-row header — no vertical pipe
    latex.append(
        r"\rowcolor{gray!15} & & "
        r"& \multicolumn{2}{c}{\textbf{MAE}~$\downarrow$}"
        r"& \multicolumn{2}{c}{\textbf{F1}~$\uparrow$} \\"
    )
    latex.append(r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    latex.append(
        r"\rowcolor{gray!15} \textbf{Model} (\tiny{Size}) & \textbf{Architecture} & \textbf{Ctx. Len.}"
        r" & \textbf{DTS} & \textbf{ItS}"
        r" & \textbf{DTS} & \textbf{ItS} \\"
    )
    latex.append(r"\midrule")

    def _emit_group(title: str, frame: pd.DataFrame, band_color: str):
        latex.append(
            rf"\multicolumn{{7}}{{c}}"
            rf"{{\cellcolor{{{band_color}}}\textsc{{{title}}}}} \\"
        )
        latex.append(r"\midrule[\lightrulewidth]")
        for _, row in frame.iterrows():
            name = row["model_base"]
            is_best_model = best_model_id is not None and row["model_id"] == best_model_id
            is_worst_model = worst_model_id is not None and row["model_id"] == worst_model_id
            model_bold_op = "\\textbf{" if (is_best_model  or is_worst_model) else ""
            model_bold_cl = "}" if (is_best_model or is_worst_model) else ""
            if is_best_model or is_worst_model:
                model_color = "DarkGray"
            else:
                model_color = "white"
    
            line = (
                rf"\cellcolor{{{model_color}}}{model_bold_op}{name}{model_bold_cl} & \cellcolor{{{model_color}}}{model_bold_op}{row['moe']}{model_bold_cl}  & \cellcolor{{{model_color}}}{model_bold_op}{row['context_length']}{model_bold_cl} & "
                f"{format_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)} & "
                f"{format_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)} & "
                f"{format_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)} & "
                f"{format_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)} \\\\"
            )
            latex.append(line)

    _emit_group("Reasoning Models", reasoning_df, "ReasonBand")
    latex.append(r"\midrule")
    _emit_group("Non-Reasoning Models", non_reasoning_df, "NonRBand")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabularx}")
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

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)

    print("\n" + "=" * 70)
    print("✅ TABLES WITH REASONING HIGHLIGHTS GENERATED")
    print("=" * 70)
    print(f"📄 Output file: {output_file}")
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