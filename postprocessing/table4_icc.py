#!/usr/bin/env python3
"""
Table 4: Item-wise ICC (Intraclass Correlation Coefficient) with comparison
to Iannuzzo et al. (2024) human inter-rater reliability on MADRS items.

Computes ICC(3,1) for each model × each MADRS item (I1–I10) using raw
per-session predictions from the pickled results, then produces a LaTeX table
comparing LLM–clinician agreement to human–human agreement.

ACL/AI2 style: tcolorbox, rcell pills, teal/rose palette, reasoning grouping.
"""

import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pickle
import argparse
from pathlib import Path


# ============================================================================
# MODEL DICTIONARY  (kept in sync with other table scripts)
# ============================================================================
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
for ablation in ["raw", "no_desc", "no_dem"]:
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_{ablation}"] = (
        f"Qwen 3 Next: {ablation.replace('_', ' ').title()} (3B-80B)"
    )
    MODEL_DICT[f"segmented_Qw3_Next_80b_a3b_ar_4q_NoR_{ablation}"] = (
        f"Qwen 3 Next: NR, {ablation.replace('_', ' ').title()} (3B-80B)"
    )

MODEL_REV_DICT = {v: k for k, v in MODEL_DICT.items()}

# ============================================================================
# Item names
# ============================================================================
MADRS_ITEM_NAMES = {
    1: "Apparent Sadness",
    2: "Reported Sadness",
    3: "Inner Tension",
    4: "Reduced Sleep",
    5: "Reduced Appetite",
    6: "Concentration",
    7: "Lassitude",
    8: "Inability to Feel",
    9: "Pessimistic Thoughts",
    10: "Suicidal Thoughts",
}

# ============================================================================
# Iannuzzo et al. human inter-rater ICC per MADRS item
# ============================================================================
IANNUZZO_ICC = {
    1: 0.92,   # Apparent Sadness
    2: 0.94,   # Reported Sadness
    3: 0.92,   # Inner Tension
    4: 0.86,   # Reduced Sleep
    5: 0.94,   # Reduced Appetite
    6: 0.90,   # Concentration
    7: 0.90,   # Lassitude
    8: 0.94,   # Inability to Feel
    9: 0.93,   # Pessimistic Thoughts
    10: 0.97,  # Suicidal Thoughts
}

# ============================================================================
# ICC COMPUTATION
# ============================================================================

def compute_icc_for_model_item(
    excel_cells: Dict,
    excel_ground_truth: Dict,
    model_display_name: str,
    item: int,
    icc_type: str = "icc3",
) -> Dict[str, float]:
    """
    Compute ICC between a model's predictions and ground truth for one item.

    For each of the 3 runs, computes ICC(3,1) between model predictions and
    ground truth across sessions. Returns mean and std across runs.

    Parameters
    ----------
    excel_cells : dict
        excel_cells[item][model_name][session_id] = {rating_0, rating_1, ...}
    excel_ground_truth : dict
        excel_ground_truth[item][session_id] = int
    model_display_name : str
        Display name of the model.
    item : int
        MADRS item index (1–10).
    icc_type : str
        ICC type index name in pingouin output. Default "icc3".

    Returns
    -------
    dict with keys "icc_mean", "icc_std", "n_sessions"
    """
    icc_type_index = {
        "icc1": 0, "icc2": 1, "icc3": 2,
        "icc1k": 3, "icc2k": 4, "icc3k": 5,
    }
    idx = icc_type_index.get(icc_type, 2)

    gt_dict = excel_ground_truth.get(item, {})
    model_cells = excel_cells.get(item, {}).get(model_display_name, {})

    if not gt_dict or not model_cells:
        return {"icc_mean": np.nan, "icc_std": np.nan, "n_sessions": 0}

    icc_values = []
    for run in range(3):
        ratings_model = []
        ratings_gt = []
        for session_id, gt_val in gt_dict.items():
            cell = model_cells.get(session_id, {})
            pred = cell.get(f"rating_{run}", np.nan)
            if (
                isinstance(pred, (int, float))
                and np.isfinite(pred)
                and isinstance(gt_val, (int, float))
                and np.isfinite(gt_val)
            ):
                ratings_model.append(float(pred))
                ratings_gt.append(float(gt_val))

        if len(ratings_model) < 10:
            continue

        # Build long-format DataFrame for pingouin
        n = len(ratings_model)
        df = pd.DataFrame({
            "targets": list(range(n)) + list(range(n)),
            "raters": [0] * n + [1] * n,
            "ratings": ratings_gt + ratings_model,
        })

        try:
            icc_result = pg.intraclass_corr(
                data=df, targets="targets", raters="raters", ratings="ratings"
            )
            icc_val = icc_result["ICC"].values[idx]
            if np.isfinite(icc_val):
                icc_values.append(icc_val)
        except Exception:
            continue

    if not icc_values:
        return {"icc_mean": np.nan, "icc_std": np.nan, "n_sessions": 0}

    return {
        "icc_mean": float(np.mean(icc_values)),
        "icc_std": float(np.std(icc_values)),
        "n_sessions": len(ratings_model) if ratings_model else 0,
    }


def compute_all_icc(
    excel_cells: Dict,
    excel_ground_truth: Dict,
    models_csv: Optional[Dict] = None,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute ICC for every model and every item (1–10).

    Returns
    -------
    dict[model_display_name][item] = {"icc_mean": ..., "icc_std": ..., "n_sessions": ...}
    """
    # Collect all model display names from excel_cells (using item 1 as reference)
    all_models = set()
    for item in range(1, 11):
        if item in excel_cells:
            all_models.update(excel_cells[item].keys())

    results = {}
    for model_name in sorted(all_models):
        model_id = MODEL_REV_DICT.get(model_name, model_name)
        # Skip models not in models_csv if provided
        if models_csv is not None and model_id not in models_csv:
            continue
        # Skip specific models
        if model_id in {
            "segmented_Llama3.1_8b",
            "segmented_PsyCare1.0_Llama3.1_8b",
            "segmented_Qw3_30b_a3b_ar_4q_NoR",
        }:
            continue

        results[model_name] = {}
        for item in range(1, 11):
            results[model_name][item] = compute_icc_for_model_item(
                excel_cells, excel_ground_truth, model_name, item
            )
        # Also compute mean across items
        item_iccs = [
            results[model_name][i]["icc_mean"]
            for i in range(1, 11)
            if np.isfinite(results[model_name][i]["icc_mean"])
        ]
        results[model_name]["mean"] = {
            "icc_mean": float(np.mean(item_iccs)) if item_iccs else np.nan,
            "icc_std": float(np.std(item_iccs)) if item_iccs else np.nan,
        }

    return results


# ============================================================================
# COLOR HELPER
# ============================================================================

def _color_icc(
    value: float,
    *,
    q75: Optional[float] = None,
    q25: Optional[float] = None,
    is_best: bool = False,
    is_worst: bool = False,
) -> str:
    """Return color name for ICC cell tint (higher is better)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if q75 is not None and v >= q75:
        return "tblGoodDk" if is_best else "tblGoodLt"
    if q25 is not None and v < q25:
        return "tblBadDk" if is_worst else "tblBadLt"
    return "tblNeutral"


# ============================================================================
# LaTeX TABLE GENERATOR
# ============================================================================

def create_icc_table(
    icc_results: Dict,
    models_csv: Optional[Dict] = None,
    custom_name_map: Optional[Dict] = None,
) -> str:
    """
    Transposed ICC table.

    Rows   = MADRS items (sorted by Iannuzzo ICC descending) + Mean + Spearman ρ
    Cols   = Item name | Iannuzzo (H-H) | Worst Model | Best Model
             | Mean Reasoning | Mean Non-Reasoning

    Last row: Spearman correlation between each column's per-item ICC vector
    and the Iannuzzo human–human ICC vector.
    """

    item_indices = list(range(1, 11))

    # --- Sort items by Iannuzzo ICC (descending) ----------------------------
    sorted_items = sorted(item_indices, key=lambda i: IANNUZZO_ICC.get(i, 0), reverse=True)

    # --- Classify models into reasoning / non-reasoning ---------------------
    reasoning_models, non_reasoning_models = [], []
    for model_name in icc_results:
        model_id = MODEL_REV_DICT.get(model_name, model_name)
        is_reasoning = False
        if models_csv is not None and model_id in models_csv:
            is_reasoning = str(models_csv[model_id].get("reasoning", "")).strip() == "Yes"
        if is_reasoning:
            reasoning_models.append(model_name)
        else:
            non_reasoning_models.append(model_name)

    # --- Helper: get ICC mean/std for a model/item -------------------------
    def _get_icc(model_name: str, item: int) -> float:
        return icc_results.get(model_name, {}).get(item, {}).get("icc_mean", np.nan)

    def _get_icc_std(model_name: str, item: int) -> float:
        return icc_results.get(model_name, {}).get(item, {}).get("icc_std", np.nan)

    # --- Identify best / worst models by mean ICC (tiebreak: lower std) -----
    model_mean_icc = {}
    model_std_icc = {}
    for model_name in icc_results:
        vals = [_get_icc(model_name, i) for i in item_indices]
        finite = [v for v in vals if np.isfinite(v)]
        model_mean_icc[model_name] = float(np.mean(finite)) if finite else np.nan
        model_std_icc[model_name] = float(np.std(finite)) if finite else np.nan



    best_model = MODEL_DICT.get("segmented_Qw2.5_72b_gptq_4q")
    worst_model = MODEL_DICT.get("segmented_Qw3_0.6b_gptq_4q")

    def _abbrev_name(s: str) -> str:
        if custom_name_map and s in custom_name_map:
            s = custom_name_map[s]
        parts = s.split("(")
        name = parts[0].strip()
        size = "(" + parts[1].strip() if len(parts) > 1 else ""
        if len(name) > 22:
            name = name[:19] + "..."
        return f"{name} {{\\tiny {size}}}" if size else name

    best_label = _abbrev_name(best_model)
    worst_label = _abbrev_name(worst_model)

    # --- Build per-item vectors for each summary column ---------------------
    iannuzzo_vec = []       # Iannuzzo human–human ICC
    best_vec = []           # Best model ICC per item
    best_std_vec = []       # Best model ICC std per item
    worst_vec = []          # Worst model ICC per item
    worst_std_vec = []      # Worst model ICC std per item
    mean_reason_vec = []    # Mean reasoning models ICC per item
    mean_reason_std_vec = [] # Std across reasoning models per item
    mean_nonreason_vec = [] # Mean non-reasoning models ICC per item
    mean_nonreason_std_vec = [] # Std across non-reasoning models per item

    for i in sorted_items:
        iannuzzo_vec.append(IANNUZZO_ICC.get(i, np.nan))
        best_vec.append(_get_icc(best_model, i))
        best_std_vec.append(_get_icc_std(best_model, i))
        worst_vec.append(_get_icc(worst_model, i))
        worst_std_vec.append(_get_icc_std(worst_model, i))
        # Mean over reasoning models
        r_vals = [_get_icc(m, i) for m in reasoning_models]
        r_finite = [v for v in r_vals if np.isfinite(v)]
        mean_reason_vec.append(float(np.mean(r_finite)) if r_finite else np.nan)
        mean_reason_std_vec.append(float(np.std(r_finite)) if r_finite else np.nan)
        # Mean over non-reasoning models
        nr_vals = [_get_icc(m, i) for m in non_reasoning_models]
        nr_finite = [v for v in nr_vals if np.isfinite(v)]
        mean_nonreason_vec.append(float(np.mean(nr_finite)) if nr_finite else np.nan)
        mean_nonreason_std_vec.append(float(np.std(nr_finite)) if nr_finite else np.nan)

    # --- Column-wise quartiles and extremes for model columns ---------------
    _model_col_vecs = [worst_vec, best_vec, mean_reason_vec, mean_nonreason_vec]
    _col_q = []   # (q75, q25) per column
    _col_bw = []  # (best, worst) per column
    for _vec in _model_col_vecs:
        _finite = [v for v in _vec if np.isfinite(v)]
        if _finite:
            _col_q.append((float(np.quantile(_finite, 0.75)),
                           float(np.quantile(_finite, 0.25))))
            _col_bw.append((max(_finite), min(_finite)))
        else:
            _col_q.append((None, None))
            _col_bw.append((None, None))

    # --- Spearman correlations versus Iannuzzo ------------------------------
    def _spearman_vs_iannuzzo(vec: List[float]) -> Tuple[float, float]:
        """Return (rho, p-value) comparing vec to iannuzzo_vec."""
        pairs = [
            (ia, v)
            for ia, v in zip(iannuzzo_vec, vec)
            if np.isfinite(ia) and np.isfinite(v)
        ]
        if len(pairs) < 4:
            return (np.nan, np.nan)
        ia_arr, v_arr = zip(*pairs)
        rho, p = stats.spearmanr(ia_arr, v_arr)
        return (rho, p)

    spearman_best = _spearman_vs_iannuzzo(best_vec)
    spearman_worst = _spearman_vs_iannuzzo(worst_vec)
    spearman_reason = _spearman_vs_iannuzzo(mean_reason_vec)
    spearman_nonreason = _spearman_vs_iannuzzo(mean_nonreason_vec)

    # --- Compute column-wise means (over items) for a "Mean" row -----------
    def _col_mean(vec: List[float]) -> float:
        finite = [v for v in vec if np.isfinite(v)]
        return float(np.mean(finite)) if finite else np.nan

    def _col_std(vec: List[float]) -> float:
        finite = [v for v in vec if np.isfinite(v)]
        return float(np.std(finite)) if finite else np.nan

    def _fmt(v: float, *, color: str = "", bold: bool = False,
             std: float = np.nan) -> str:
        if not (isinstance(v, (int, float)) and np.isfinite(v)):
            return r"\textemdash{}"
        main = f"{v:.2f}"
        if bold:
            main = rf"\textbf{{{main}}}"
        if isinstance(std, (int, float)) and np.isfinite(std):
            if color:
                return rf"\rcell{{{color}}}{{\makecell{{{main} {{\scriptsize $\pm$ {std:.2f}}}}}}}"
            return rf"\makecell{{{main} {{\scriptsize $\pm$ {std:.2f}}}}}"
        if color:
            return rf"\rcell{{{color}}}{{{main}}}"
        return main

    # --- Build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(
        r"\caption{ICC(3,1) by MADRS item: human inter-rater agreement "
        r"(Iannuzzo et al.) versus LLM--clinician agreement. "
        r"Columns show the best and worst performing models, "
        r"and the mean across reasoning / non-reasoning models. "
        r"The last row reports Spearman $\rho$ between each column and the "
        r"human reference. Items sorted by Iannuzzo ICC (descending).}"
    )
    latex.append(r"\label{tab:table4_icc}")

    # tcolorbox
    latex.append(r"\begin{tcolorbox}[")
    latex.append(r"  enhanced,")
    latex.append(r"  boxrule=0.5pt,")
    latex.append(r"  colframe=tblBorder,")
    latex.append(r"  colback=white,")
    latex.append(r"  arc=8pt,")
    latex.append(r"  outer arc=8pt,")
    latex.append(r"  left=3pt, right=3pt, top=3pt, bottom=3pt,")
    latex.append(r"  boxsep=0pt,")
    latex.append(
        r"  before upper={\arrayrulecolor{tblBorder}"
        r"\renewcommand{\arraystretch}{1.35}"
        r"\setlength{\tabcolsep}{4pt}},"
    )
    latex.append(r"]")

    # 6 columns: Item | Iannuzzo | Worst | Best | Mean Reasoning | Mean Non-Reasoning
    latex.append(
        r"\begin{tabular}{@{} l "
        r"c "                                      # Iannuzzo
        r"!{\color{tblBorder}\vrule width 0.5pt} "
        r"c c "                                    # Worst, Best
        r"c c "                                    # Mean R, Mean NR
        r"@{}}"
    )
    latex.append(r"\arrayrulecolor{tblBorder}")

    # --- Header -------------------------------------------------------------
    latex.append(
        r"\rcell{tblNeutral}{\textsf{MADRS Item}}"
        r" & \rcell{tblNeutral}{\makecell{\textsf{Iannuzzo (H--H)}}}"
        r" & \rcell{tblNeutral}{\makecell{\textsf{Worst Model}}}"
        r" & \rcell{tblNeutral}{\makecell{\textsf{Best Model}}}"
        r" & \rcell{tblNeutral}{\makecell{\textsf{Mean Reasoning}}}"
        r" & \rcell{tblNeutral}{\makecell{\textsf{Mean Non-Reas.}}}"
        r" \\"
    )
    latex.append(r"\addlinespace[2pt]")

    # --- Sub-header: model names for worst/best -----------------------------
    latex.append(
        r" & "
        r" & {\scriptsize " + worst_label + r"}"
        r" & {\scriptsize " + best_label + r"}"
        r" & "
        r" & "
        r" \\"
    )
    latex.append(r"\addlinespace[2pt]")
    latex.append(r"\midrule")
    latex.append(r"\addlinespace[3pt]")

    # --- Item rows (sorted by Iannuzzo descending) --------------------------
    for idx, i in enumerate(sorted_items):
        item_name = MADRS_ITEM_NAMES.get(i, f"Item {i}")
        ia_val = iannuzzo_vec[idx]
        b_val = best_vec[idx]
        b_std = best_std_vec[idx]
        w_val = worst_vec[idx]
        w_std = worst_std_vec[idx]
        mr_val = mean_reason_vec[idx]
        mr_std = mean_reason_std_vec[idx]
        mnr_val = mean_nonreason_vec[idx]
        mnr_std = mean_nonreason_std_vec[idx]

        cells = [rf"\textsf{{I{i}}} {item_name}"]
        cells.append(_fmt(ia_val, color="tblGoodLt"))
        for ci, (v, s) in enumerate([(w_val, w_std), (b_val, b_std), (mr_val, mr_std), (mnr_val, mnr_std)]):
            q75, q25 = _col_q[ci]
            cbest, cworst = _col_bw[ci]
            is_col_best = cbest is not None and np.isfinite(v) and abs(v - cbest) <= 1e-6
            is_col_worst = cworst is not None and np.isfinite(v) and abs(v - cworst) <= 1e-6
            color = _color_icc(v, q75=q75, q25=q25, is_best=is_col_best, is_worst=is_col_worst)
            cells.append(_fmt(v, color=color, bold=is_col_best, std=s))

        latex.append(" & ".join(cells) + r" \\")

    # --- Mean row -----------------------------------------------------------
    latex.append(r"\addlinespace[2pt]")
    latex.append(r"\midrule")
    latex.append(r"\addlinespace[3pt]")

    mean_cells = [r"\textit{Mean}"]
    mean_cells.append(_fmt(_col_mean(iannuzzo_vec), color="tblGoodLt"))
    for ci, (vec, std_vec) in enumerate([
        (worst_vec, worst_std_vec),
        (best_vec, best_std_vec),
        (mean_reason_vec, mean_reason_std_vec),
        (mean_nonreason_vec, mean_nonreason_std_vec),
    ]):
        m = _col_mean(vec)
        s = _col_std(std_vec) if std_vec else np.nan
        q75, q25 = _col_q[ci]
        color = _color_icc(m, q75=q75, q25=q25)
        mean_cells.append(_fmt(m, color=color, std=s))
    latex.append(" & ".join(mean_cells) + r" \\")

    # --- Spearman ρ row -----------------------------------------------------
    latex.append(r"\addlinespace[2pt]")
    latex.append(r"\midrule")
    latex.append(r"\addlinespace[3pt]")

    def _fmt_spearman(rho: float, p: float) -> str:
        if not np.isfinite(rho):
            return r"\textemdash{}"
        stars = ""
        if p < 0.001:
            stars = r"^{***}"
        elif p < 0.01:
            stars = r"^{**}"
        elif p < 0.05:
            stars = r"^{*}"
        return rf"${rho:.2f}{stars}$"

    spearman_cells = [r"\textit{Spearman $\rho$}"]
    spearman_cells.append("")  # Iannuzzo column (self-correlation = 1, skip)
    spearman_cells.append(_fmt_spearman(*spearman_worst))
    spearman_cells.append(_fmt_spearman(*spearman_best))
    spearman_cells.append(_fmt_spearman(*spearman_reason))
    spearman_cells.append(_fmt_spearman(*spearman_nonreason))
    latex.append(" & ".join(spearman_cells) + r" \\")

    latex.append(r"\addlinespace[2pt]")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{tcolorbox}")
    latex.append(r"\end{table*}")

    return "\n".join(latex)


# ============================================================================
# FULL MODEL TABLE  (all models × items, reasoning/non-reasoning grouping)
# ============================================================================

def create_full_model_icc_table(
    icc_results: Dict,
    models_csv: Optional[Dict] = None,
    custom_name_map: Optional[Dict] = None,
) -> str:
    """
    Full model ICC table: rows = every model, columns = I1..I10 + Mean.
    Reasoning / non-reasoning grouping.  Style mirrors table3_item_scores.
    """

    def _abbrev_name(s: str) -> str:
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

    for model_name, item_dict in icc_results.items():
        model_id = MODEL_REV_DICT.get(model_name, model_name)

        entry = {
            "Model": _abbrev_name(model_name),
            "original_id": model_id,
        }

        for i in item_indices:
            d = item_dict.get(i, {})
            entry[f"I{i}_mean"] = d.get("icc_mean", np.nan)
            entry[f"I{i}_std"] = d.get("icc_std", np.nan)

        d_mean = item_dict.get("mean", {})
        entry["I11_mean"] = d_mean.get("icc_mean", np.nan)
        entry["I11_std"] = d_mean.get("icc_std", np.nan)

        if models_csv is not None and model_id in models_csv:
            entry["is_reasoning"] = (
                str(models_csv[model_id].get("reasoning", "")).strip() == "Yes"
            )
        else:
            entry["is_reasoning"] = False

        rows.append(entry)

    if not rows:
        return ""

    df = pd.DataFrame(rows).sort_values(by="I11_mean", ascending=False).reset_index(drop=True)

    # Per-column best/worst and quartiles
    col_bw = {}
    col_q = {}
    for i in item_indices + [11]:
        col = pd.to_numeric(df[f"I{i}_mean"], errors="coerce")
        good = col[np.isfinite(col)]
        if not good.empty:
            col_bw[i] = (float(good.max()), float(good.min()))
            col_q[i] = (float(good.quantile(0.75)), float(good.quantile(0.25)))
        else:
            col_bw[i] = (None, None)
            col_q[i] = (None, None)

    reasoning_df = df[df["is_reasoning"]].copy()
    non_reasoning_df = df[~df["is_reasoning"]].copy()

    # --- Format cell --------------------------------------------------------
    def _format_icc_cell(m, s, col_idx, *, is_reference: bool = False) -> str:
        if not (isinstance(m, (int, float)) and np.isfinite(m)):
            return r"\textemdash{}"

        main = f"{m:.2f}"

        bw_best, bw_worst = col_bw.get(col_idx, (None, None))
        q75, q25 = col_q.get(col_idx, (None, None))

        is_best = (not is_reference) and bw_best is not None and abs(m - bw_best) <= 1e-6
        is_worst = (not is_reference) and bw_worst is not None and abs(m - bw_worst) <= 1e-6

        if is_best:
            main = rf"\textbf{{{main}}}"

        color = _color_icc(m, q75=q75, q25=q25, is_best=is_best, is_worst=is_worst)

        if is_reference:
            return rf"\rcell{{tblGoodLt}}{{{main}}}"

        if isinstance(s, (int, float)) and np.isfinite(s):
            top_line = main
            bot_line = rf"{{\tiny $\pm$ {s:.2f}}}"
            if color:
                return rf"\rcell{{{color}}}{{\makecell{{{top_line} \\ {bot_line}}}}}"
            return rf"\makecell{{{top_line} \\ {bot_line}}}"
        else:
            if color:
                return rf"\rcell{{{color}}}{{{main}}}"
            return main

    # --- Build LaTeX --------------------------------------------------------
    latex = []
    latex.append(r"\begin{table*}[!tb]")
    latex.append(r"\centering")
    latex.append(r"\footnotesize")
    latex.append(
        r"\caption{Item-wise ICC(3,1)\,$\pm$\,std for I1--I10 with mean across items "
        r"for all models. Iannuzzo et al.\ human inter-rater ICC shown as shaded "
        r"reference row. Bold denotes best LLM value per column; "
        r"darker shades indicate best/worst cells.}"
    )
    latex.append(r"\label{tab:table4_icc_full}")

    # tcolorbox
    latex.append(r"\begin{tcolorbox}[")
    latex.append(r"  enhanced,")
    latex.append(r"  boxrule=0.5pt,")
    latex.append(r"  colframe=tblBorder,")
    latex.append(r"  colback=white,")
    latex.append(r"  arc=8pt,")
    latex.append(r"  outer arc=8pt,")
    latex.append(r"  left=3pt, right=3pt, top=3pt, bottom=3pt,")
    latex.append(r"  boxsep=0pt,")
    latex.append(
        r"  before upper={\arrayrulecolor{tblBorder}"
        r"\renewcommand{\arraystretch}{1.25}"
        r"\setlength{\tabcolsep}{2pt}},"
    )
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
    latex.append(
        r"\rcell{tblNeutral}{\textsf{Model}\,{\scriptsize\textcolor{hdrSub}{(Size)}}}"
        r" & "
        + " & ".join(
            [rf"\rcell{{tblNeutral}}{{\textsf{{I{i}}}}}" for i in item_indices]
        )
        + r" & \rcell{tblNeutral}{\textsf{Mean}} \\"
    )
    latex.append(r"\addlinespace[3pt]")

    # --- Iannuzzo et al. reference row --------------------------------------
    ref_cells = [r"\textit{Iannuzzo et al.\ (Human)}"]
    iannuzzo_vals = []
    for i in item_indices:
        val = IANNUZZO_ICC.get(i, np.nan)
        iannuzzo_vals.append(val)
        ref_cells.append(_format_icc_cell(val, np.nan, i, is_reference=True))
    iannuzzo_mean = float(np.mean(iannuzzo_vals))
    ref_cells.append(_format_icc_cell(iannuzzo_mean, np.nan, 11, is_reference=True))
    latex.append(" & ".join(ref_cells) + r" \\")
    latex.append(r"\addlinespace[3pt]")
    latex.append(r"\arrayrulecolor{tblBorder}\midrule")
    latex.append(r"\addlinespace[3pt]")

    # --- Emit group ---------------------------------------------------------
    def _emit_group(title: str, frame: pd.DataFrame):
        if frame.empty:
            return
        latex.append(
            r"\multicolumn{12}{@{}l@{}}"
            rf"{{\rcell{{tblNeutral}}{{\small\textsc{{{title}}}}}}} \\"
        )
        latex.append(r"\addlinespace[3pt]")
        for _, rr in frame.iterrows():
            cells = [str(rr["Model"])]
            for i in item_indices + [11]:
                cells.append(
                    _format_icc_cell(
                        rr.get(f"I{i}_mean", np.nan),
                        rr.get(f"I{i}_std", np.nan),
                        i,
                    )
                )
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
    excel_cells: Dict,
    excel_ground_truth: Dict,
    models_csv: Optional[Dict] = None,
    output_file: str = "table4_icc.tex",
    custom_name_map: Optional[Dict] = None,
) -> str:
    all_tables = []

    print("=" * 70)
    print("  GENERATING ICC TABLE (ACL/AI2 STYLE)")
    print("=" * 70)

    print("  Computing ICC(3,1) for all models × items ...")
    icc_results = compute_all_icc(excel_cells, excel_ground_truth, models_csv)

    # Print summary
    print(f"  Computed ICC for {len(icc_results)} models")
    for model_name in sorted(
        icc_results, key=lambda m: icc_results[m].get("sum", {}).get("icc_mean", -1), reverse=True
    ):
        mean_val = icc_results[model_name].get("mean", {}).get("icc_mean", np.nan)
        sum_val = icc_results[model_name].get("sum", {}).get("icc_mean", np.nan)
        print(f"    {model_name:45s}  mean ICC = {mean_val:.3f}  sum ICC = {sum_val:.3f}")

    print("\n  Building LaTeX tables ...")

    # Summary table (transposed: items as rows, summary columns)
    all_tables.append(
        create_icc_table(
            icc_results,
            models_csv=models_csv,
            custom_name_map=custom_name_map,
        )
    )
    all_tables.append("\n% " + "=" * 70 + "\n")
    summary_latex = "\n\n".join(all_tables)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary_latex)

    print("\n" + "=" * 70)
    print("  TABLES GENERATED")
    print("=" * 70)
    print(f"  Output file: {output_file}")
    print("=" * 70)
    
    all_tables = []
    
    # Full model table (all models × items)
    print("  ✓ Full model ICC table")
    all_tables.append(
        create_full_model_icc_table(
            icc_results,
            models_csv=models_csv,
            custom_name_map=custom_name_map,
        )
    )
    all_tables.append("\n% " + "=" * 70 + "\n")

    full_latex = "\n\n".join(all_tables)

    with open(f"{output_file}_full.tex", "w", encoding="utf-8") as f:
        f.write(full_latex)

    print("\n" + "=" * 70)
    print("  TABLES GENERATED")
    print("=" * 70)
    print(f"  Output file: {output_file}_full.tex")
    print("=" * 70)

    return summary_latex


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 4 (item-wise ICC with Iannuzzo et al. comparison) "
            "from get_results.py outputs"
        )
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
        default="../output/table4_icc.tex",
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

    excel_cells = data.get("excel_cells", {})
    excel_ground_truth = data.get("excel_ground_truth", {})
    models_csv = data.get("models_csv", None)

    _ = generate_all_academic_tables(
        excel_cells,
        excel_ground_truth,
        models_csv=models_csv,
        output_file=str(out_path),
    )

    print("\n" + "=" * 70)
    print("Table generation complete!")
    print(f"Output file: {out_path}")
    print("=" * 70)
