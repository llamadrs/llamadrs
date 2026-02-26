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
# ENHANCED VISUAL DESIGN UTILITIES
# ==========


def get_reasoning_row_color(is_reasoning: bool, is_top_performer: bool = False) -> str:
    """Get row color based on reasoning capability and performance."""
    if is_reasoning:
        if is_top_performer:
            return r"\rowcolor{blue!8}"  # Reasoning + top performer
        else:
            return r"\rowcolor{violet!5}"  # Reasoning models get violet tint
    elif is_top_performer:
        return r"\rowcolor{green!5}"  # Non-reasoning top performer
    return ""

def calculate_color_intensity(value: float, min_val: float, max_val: float, 
                              is_lower_better: bool = True) -> int:
    """Calculate color intensity (0-40) based on value position."""
    if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val):
        return 0
    
    if max_val == min_val:
        return 0
    
    # Normalize to 0-1
    normalized = (value - min_val) / (max_val - min_val)
    
    if not is_lower_better:
        normalized = 1 - normalized  # Invert for higher-is-better
    
    # Map to intensity levels with non-linear scaling for emphasis
    if normalized < 0.05:
        return 40  # Darkest for best
    elif normalized < 0.10:
        return 32
    elif normalized < 0.20:
        return 25
    elif normalized < 0.35:
        return 18
    elif normalized < 0.50:
        return 12
    elif normalized < 0.70:
        return 8
    elif normalized < 0.85:
        return 5
    else:
        return 0  # No color for worst

def get_cell_color(value: float, min_val: float, max_val: float, 
                   is_lower_better: bool = True, tie_tol: float = 1e-6) -> str:
    """Generate LaTeX cell color command based on value."""
    if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val):
        return ""
    
    # Check for ties at extrema
    if abs(value - min_val) <= tie_tol:
        return r"\cellcolor{blue!40}" if is_lower_better else r"\cellcolor{red!40}"
    if abs(value - max_val) <= tie_tol:
        return r"\cellcolor{red!40}" if is_lower_better else r"\cellcolor{blue!40}"
    
    # Calculate position-based intensity
    normalized = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
    
    if not is_lower_better:
        normalized = 1 - normalized
    
    # Determine which gradient to use
    if normalized < 0.5:  # Closer to best
        intensity = calculate_color_intensity(value, min_val, max_val, is_lower_better)
        if intensity > 0:
            return f"\\cellcolor{{blue!{intensity}}}"
    else:  # Closer to worst
        dist_from_worst = abs(value - max_val) / (max_val - min_val) if max_val != min_val else 1
        if is_lower_better:
            dist_from_worst = abs(value - max_val) / (max_val - min_val) if max_val != min_val else 1
        else:
            dist_from_worst = abs(value - min_val) / (max_val - min_val) if max_val != min_val else 1
        
        if dist_from_worst < 0.05:
            intensity = 40
        elif dist_from_worst < 0.10:
            intensity = 32
        elif dist_from_worst < 0.20:
            intensity = 25
        elif dist_from_worst < 0.35:
            intensity = 18
        else:
            intensity = 0
        
        if intensity > 0:
            return f"\\cellcolor{{red!{intensity}}}"
    
    return ""

def format_value_with_std(mean: float, std: float, decimals: int = 3) -> str:
    """Format value as 'mean ± std' with proper LaTeX spacing."""
    if np.isnan(mean) or np.isnan(std):
        return r"\textemdash{}"
    
    format_str = f"{{:.{decimals}f}}"
    return f"{format_str.format(mean)} $\\pm$ {format_str.format(std)}"
def table1(
    individual_results: Dict,
    binary_results: Dict,
    sum_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
    include_all: bool = False,
) -> str:
    """
    Create comprehensive performance ranking table with reasoning model highlighting.
    Includes direct MAE, sum MAE, and binary classification F1 scores.
    """    
    def _abbrev_name(s: str) -> str:
        """Abbreviate model name for compact display."""
        if custom_name_map and s in custom_name_map:
            s = custom_name_map[s]
        parts = s.split("(")
        name = parts[0].strip()
        size = "(" + parts[1].strip() if len(parts) > 1 else ""
        
        if len(name) > 22:
            name = name[:19] + "..."
        
        return f"{name} {{\\tiny {size}}}" if size else name
    
    rows = []
    for model in individual_results.keys():
        # Skip if model not in CSV or in exclusion list
        model_name = MODEL_REV_DICT.get(model, model)
        if model_name not in [
            "segmented_Llama3.1_8b_gptq_4q", 
            "segmented_PsyCare1.0_Llama3.1_8b", 
        ]:
            continue
        
        model_row = models_csv[model_name]
        
        # Get direct MAE from individual_results (item 0)
        direct_mae = np.nan
        direct_mae_std = np.nan
        if model in individual_results and 0 in individual_results[model]:
            direct_mae = individual_results[model][0].get('mae_mean', np.nan)
            direct_mae_std = individual_results[model][0].get('mae_std', np.nan)
        
        # Get sum MAE from sum_results
        sum_mae = np.nan
        sum_mae_std = np.nan
        if model in sum_results:
            sum_mae = sum_results[model].get('mae_mean', np.nan)
            sum_mae_std = sum_results[model].get('mae_std', np.nan)
        
        # Get binary F1 scores
        direct_f1 = np.nan
        direct_f1_std = np.nan
        sum_f1 = np.nan
        sum_f1_std = np.nan
        
        if model in binary_results:
            if 'total_direct' in binary_results[model]:
                direct_f1 = binary_results[model]['total_direct'].get('f1_mean', np.nan)
                direct_f1_std = binary_results[model]['total_direct'].get('f1_std', np.nan)
            
            if 'total_sum' in binary_results[model]:
                sum_f1 = binary_results[model]['total_sum'].get('f1_mean', np.nan)
                sum_f1_std = binary_results[model]['total_sum'].get('f1_std', np.nan)
        model_row['context_length'] = int(model_row['context_length'])
        row = {
            'model': model,
            'context_length': f"{int(model_row['context_length'] / 1_000_000)}m" if model_row['context_length'] >= 1_000_000 else f"{int(model_row['context_length'] / 1_000)}k",
            "size": f"{model_row['total_params']}({model_row['active_params']})" if model_row['total_params'] != model_row['active_params'] else f"{model_row['total_params']}",
            'direct_mae': direct_mae,
            'direct_mae_std': direct_mae_std,
            'sum_mae': sum_mae,
            'sum_mae_std': sum_mae_std,
            'direct_f1': direct_f1,
            'direct_f1_std': direct_f1_std,
            'sum_f1': sum_f1,
            'sum_f1_std': sum_f1_std,
        }
        
        
        # Determine architecture family
        if 'Qwen' in model or 'QwQ' in model:
            row['family'] = 'Qwen'
        elif 'Llama' in model:
            row['family'] = 'Llama'
        elif 'DeepSeek' in model:
            row['family'] = 'DeepSeek'
        elif 'Gemma' in model:
            row['family'] = 'Gemma'
        elif 'GPT' in model:
            row['family'] = 'GPT'
        elif 'Magistral' in model:
            row['family'] = 'Magistral'
        else:
            row['family'] = 'Other'
        
        # Check for special features
        row['reasoning'] = model_row['reasoning'] == "Yes"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort by direct MAE (primary metric)
    df = df.sort_values('direct_mae').reset_index(drop=True)
    
    # Calculate statistics for coloring
    col_stats = {
        'direct_mae': {'min': df['direct_mae'].min(), 'max': df['direct_mae'].max()},
        'sum_mae': {'min': df['sum_mae'].min(), 'max': df['sum_mae'].max()},
        'direct_f1': {'min': df['direct_f1'].min(), 'max': df['direct_f1'].max()},
        'sum_f1': {'min': df['sum_f1'].min(), 'max': df['sum_f1'].max()},
    }
    
    def format_metric_cell(m, s, col_name, is_lower_better=True, tie_tol=1e-6):
        """Format a metric cell with color, bold/italic styling."""
        if np.isnan(m):
            return "—"
        
        value_str = f"{m:.2f} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
        
        stat = col_stats.get(col_name)
        if stat:
            cell_color = get_cell_color(m, stat["min"], stat["max"], is_lower_better=is_lower_better, tie_tol=tie_tol)
            
            # Bold for best, italic for worst
            if is_lower_better:
                if abs(m - stat["min"]) <= tie_tol:
                    value_str = f"\\textbf{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
                elif abs(m - stat["max"]) <= tie_tol:
                    value_str = f"\\textit{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
            else:  # higher is better
                if abs(m - stat["max"]) <= tie_tol:
                    value_str = f"\\textbf{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
                elif abs(m - stat["min"]) <= tie_tol:
                    value_str = f"\\textit{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
            
            cell = f"\\makecell{{{cell_color}{value_str}}}"
        else:
            cell = f"\\makecell{{{value_str}}}"
        
        return cell
    
    # Group by reasoning/non-reasoning
    reasoning_df = df[df['reasoning']].reset_index(drop=True)
    non_reasoning_df = df[~df['reasoning']].reset_index(drop=True)
    
    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Base vs Domain-finetuned Model Performance: Llama 3.1 (8B) Base and PsyCare 1.0}")
    latex.append(r"\label{tab:base_vs_finetuned}")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    latex.append(r"\small")
    
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\rowcolor{gray!15} \textbf{Model} & \textbf{Size} &  \textbf{Variant}  &")
    latex.append(r"\textbf{Direct MAE} $\downarrow$ & \textbf{Sum MAE} $\downarrow$ & ")
    latex.append(r"\textbf{Direct F1} $\uparrow$ & \textbf{Sum F1} $\uparrow$ \\")
    latex.append(r"\midrule")
    
    # First show reasoning models
    if not reasoning_df.empty:
        latex.append(r"\multicolumn{7}{c}{\cellcolor{violet!10}\textbf{Reasoning Models}} \\")
        latex.append(r"\midrule[0.5pt]")
        
        for idx, row in reasoning_df.iterrows():
            rank = idx + 1
            model_name = row['model'].split('(')[0].strip()
            
            
            context_length = row['context_length']
            size = row['size']
            variant = "Fine-tuned" if "PsyCare" in model_name else "Base"
            
            row_color = r"\rowcolor{violet!5}"
            
            # Format metric cells
            direct_mae_cell = format_metric_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)
            sum_mae_cell = format_metric_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)
            direct_f1_cell = format_metric_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)
            sum_f1_cell = format_metric_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)
            
            # Bold top 3 overall (based on original df ranking)
            original_rank = df[df['model'] == row['model']].index[0] + 1
            if original_rank <= 3:
                model_name = f"\\textbf{{{model_name}}}"
            

            line = (f"{model_name} & {size} & {variant} &"
                f"{direct_mae_cell} & {sum_mae_cell} & "
                f"{direct_f1_cell} & {sum_f1_cell} \\\\")
            
            latex.append(f"{row_color} {line}")
    
    # Separator between groups
    if not reasoning_df.empty and not non_reasoning_df.empty:
        latex.append(r"\midrule[1pt]")
        latex.append(r"\multicolumn{7}{c}{\cellcolor{gray!10}\textbf{Non-Reasoning Models}} \\")
        latex.append(r"\midrule[0.5pt]")
    
    # Then show non-reasoning models
    for idx, row in non_reasoning_df.iterrows():
        rank = idx + 1
        
        model_name = row['model'].split('(')[0].strip()
        
        context_length = row['context_length']
        size = row['size']
        
        # Format metric cells
        direct_mae_cell = format_metric_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)
        sum_mae_cell = format_metric_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)
        direct_f1_cell = format_metric_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)
        sum_f1_cell = format_metric_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)
        
        # Bold top 3 overall (based on original df ranking)
        original_rank = df[df['model'] == row['model']].index[0] + 1
        if original_rank <= 3:
            model_name = f"\\textbf{{{model_name}}}"
        
        line = (f"{model_name} & {size} & {context_length} &"
                f"{direct_mae_cell} & {sum_mae_cell} & "
                f"{direct_f1_cell} & {sum_f1_cell} \\\\")
        
        latex.append(f"{line}")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\vspace{2mm}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\small")
    latex.append(r"\item \textbf{Comparison:} Base Llama 3.1 (8B) vs fine-tuned PsyCare 1.0 variant")
    latex.append(r"\item Direct: Full transcript analysis. Sum: Sum of items 1-10.")
    latex.append(r"\item F1: Binary classification (threshold $\geq$ 20 for screening).")
    latex.append(r"\item \textbf{Bold/Italic:} Column-best/worst performance across both models")
    latex.append(r"\end{tablenotes}")
    latex.append(r"\end{table*}")
    
    return "\n".join(latex)
def table2(
    individual_results: Dict,
    binary_results: Dict,
    sum_results: Dict,
    custom_name_map: Optional[Dict] = None,
    models_csv: Optional[pd.DataFrame] = None,
    include_all: bool = False,
) -> str:
    """
    Create comprehensive performance ranking table with reasoning model highlighting.
    Includes direct MAE, sum MAE, and binary classification F1 scores.
    """    
    def _abbrev_name(s: str) -> str:
        """Abbreviate model name for compact display."""
        if custom_name_map and s in custom_name_map:
            s = custom_name_map[s]
        parts = s.split("(")
        name = parts[0].strip()
        size = "(" + parts[1].strip() if len(parts) > 1 else ""
        
        if len(name) > 22:
            name = name[:19] + "..."
        
        return f"{name} {{\\tiny {size}}}" if size else name
    
    rows = []
    for model in individual_results.keys():
        # Skip if model not in CSV or in exclusion list
        model_name = MODEL_REV_DICT.get(model, model)
        if model_name not in [
            "segmented_Llama3.1_8b_gptq_4q", 
            "segmented_Llama3.1_8b", 
        ]:
            continue
        
        model_row = models_csv[model_name]
        
        # Get direct MAE from individual_results (item 0)
        direct_mae = np.nan
        direct_mae_std = np.nan
        if model in individual_results and 0 in individual_results[model]:
            direct_mae = individual_results[model][0].get('mae_mean', np.nan)
            direct_mae_std = individual_results[model][0].get('mae_std', np.nan)
        
        # Get sum MAE from sum_results
        sum_mae = np.nan
        sum_mae_std = np.nan
        if model in sum_results:
            sum_mae = sum_results[model].get('mae_mean', np.nan)
            sum_mae_std = sum_results[model].get('mae_std', np.nan)
        
        # Get binary F1 scores
        direct_f1 = np.nan
        direct_f1_std = np.nan
        sum_f1 = np.nan
        sum_f1_std = np.nan
        
        if model in binary_results:
            if 'total_direct' in binary_results[model]:
                direct_f1 = binary_results[model]['total_direct'].get('f1_mean', np.nan)
                direct_f1_std = binary_results[model]['total_direct'].get('f1_std', np.nan)
            
            if 'total_sum' in binary_results[model]:
                sum_f1 = binary_results[model]['total_sum'].get('f1_mean', np.nan)
                sum_f1_std = binary_results[model]['total_sum'].get('f1_std', np.nan)
        model_row['context_length'] = int(model_row['context_length'])
        row = {
            'model': model,
            'context_length': f"{int(model_row['context_length'] / 1_000_000)}m" if model_row['context_length'] >= 1_000_000 else f"{int(model_row['context_length'] / 1_000)}k",
            "quantization": model_row['quantization'] if model_row['quantization'] != "No" else "No Quant",
            "size": f"{model_row['total_params']}({model_row['active_params']})" if model_row['total_params'] != model_row['active_params'] else f"{model_row['total_params']}",
            'direct_mae': direct_mae,
            'direct_mae_std': direct_mae_std,
            'sum_mae': sum_mae,
            'sum_mae_std': sum_mae_std,
            'direct_f1': direct_f1,
            'direct_f1_std': direct_f1_std,
            'sum_f1': sum_f1,
            'sum_f1_std': sum_f1_std,
        }
        
        
        # Determine architecture family
        if 'Qwen' in model or 'QwQ' in model:
            row['family'] = 'Qwen'
        elif 'Llama' in model:
            row['family'] = 'Llama'
        elif 'DeepSeek' in model:
            row['family'] = 'DeepSeek'
        elif 'Gemma' in model:
            row['family'] = 'Gemma'
        elif 'GPT' in model:
            row['family'] = 'GPT'
        elif 'Magistral' in model:
            row['family'] = 'Magistral'
        else:
            row['family'] = 'Other'
        
        # Check for special features
        row['reasoning'] = model_row['reasoning'] == "Yes"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    # Sort by direct MAE (primary metric)
    df = df.sort_values('direct_mae').reset_index(drop=True)
    
    # Calculate statistics for coloring
    col_stats = {
        'direct_mae': {'min': df['direct_mae'].min(), 'max': df['direct_mae'].max()},
        'sum_mae': {'min': df['sum_mae'].min(), 'max': df['sum_mae'].max()},
        'direct_f1': {'min': df['direct_f1'].min(), 'max': df['direct_f1'].max()},
        'sum_f1': {'min': df['sum_f1'].min(), 'max': df['sum_f1'].max()},
    }
    
    def format_metric_cell(m, s, col_name, is_lower_better=True, tie_tol=1e-6):
        """Format a metric cell with color, bold/italic styling."""
        if np.isnan(m):
            return "—"
        
        value_str = f"{m:.2f} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
        
        stat = col_stats.get(col_name)
        if stat:
            cell_color = get_cell_color(m, stat["min"], stat["max"], is_lower_better=is_lower_better, tie_tol=tie_tol)
            
            # Bold for best, italic for worst
            if is_lower_better:
                if abs(m - stat["min"]) <= tie_tol:
                    value_str = f"\\textbf{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
                elif abs(m - stat["max"]) <= tie_tol:
                    value_str = f"\\textit{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
            else:  # higher is better
                if abs(m - stat["max"]) <= tie_tol:
                    value_str = f"\\textbf{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
                elif abs(m - stat["min"]) <= tie_tol:
                    value_str = f"\\textit{{{m:.2f}}} \\\\ {{\\tiny $\\pm$ {s:.2f}}}"
            
            cell = f"\\makecell{{{cell_color}{value_str}}}"
        else:
            cell = f"\\makecell{{{value_str}}}"
        
        return cell
    
    # Group by reasoning/non-reasoning
    reasoning_df = df[df['reasoning']].reset_index(drop=True)
    non_reasoning_df = df[~df['reasoning']].reset_index(drop=True)
    
    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\caption{Quantization Impact on Model Performance: Quantized vs Non-Quantized Llama 3.1 (8B)}")
    latex.append(r"\label{tab:quantization_impact}")
    latex.append(r"\renewcommand{\arraystretch}{1.1}")
    latex.append(r"\setlength{\tabcolsep}{2.5pt}")
    latex.append(r"\small")
    latex.append(r"\small")
    
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\rowcolor{gray!15} \textbf{Model} & \textbf{Size} &  \textbf{Quantization}  &")
    latex.append(r"\textbf{Direct MAE} $\downarrow$ & \textbf{Sum MAE} $\downarrow$ & ")
    latex.append(r"\textbf{Direct F1} $\uparrow$ & \textbf{Sum F1} $\uparrow$ \\")
    latex.append(r"\midrule")
    
    # First show reasoning models
    if not reasoning_df.empty:
        latex.append(r"\multicolumn{7}{c}{\cellcolor{violet!10}\textbf{Reasoning Models}} \\")
        latex.append(r"\midrule[0.5pt]")
        
        for idx, row in reasoning_df.iterrows():
            rank = idx + 1
            model_name = row['model'].split('(')[0].strip()
            
            
            quantized = row["quantization"]
            size = row['size']
            
            row_color = r"\rowcolor{violet!5}"
            
            # Format metric cells
            direct_mae_cell = format_metric_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)
            sum_mae_cell = format_metric_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)
            direct_f1_cell = format_metric_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)
            sum_f1_cell = format_metric_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)
            
            # Bold top 3 overall (based on original df ranking)
            original_rank = df[df['model'] == row['model']].index[0] + 1
            if original_rank <= 3:
                model_name = f"\\textbf{{{model_name}}}"
            

            line = (f"{model_name} & {size} & {quantized} &"
                f"{direct_mae_cell} & {sum_mae_cell} & "
                f"{direct_f1_cell} & {sum_f1_cell} \\\\")
            
            latex.append(f"{row_color} {line}")
    
    # Separator between groups
    if not reasoning_df.empty and not non_reasoning_df.empty:
        latex.append(r"\midrule[1pt]")
        latex.append(r"\multicolumn{7}{c}{\cellcolor{gray!10}\textbf{Non-Reasoning Models}} \\")
        latex.append(r"\midrule[0.5pt]")
    
    # Then show non-reasoning models
    for idx, row in non_reasoning_df.iterrows():
        rank = idx + 1
        
        model_name = row['model'].split('(')[0].strip()
        
        context_length = row['context_length']
        size = row['size']
        quantized = row["quantization"]
        
        # Format metric cells
        direct_mae_cell = format_metric_cell(row['direct_mae'], row['direct_mae_std'], 'direct_mae', is_lower_better=True)
        sum_mae_cell = format_metric_cell(row['sum_mae'], row['sum_mae_std'], 'sum_mae', is_lower_better=True)
        direct_f1_cell = format_metric_cell(row['direct_f1'], row['direct_f1_std'], 'direct_f1', is_lower_better=False)
        sum_f1_cell = format_metric_cell(row['sum_f1'], row['sum_f1_std'], 'sum_f1', is_lower_better=False)
        
        # Bold top 3 overall (based on original df ranking)
        original_rank = df[df['model'] == row['model']].index[0] + 1
        if original_rank <= 3:
            model_name = f"\\textbf{{{model_name}}}"
        
        line = (f"{model_name} & {size} & {quantized} &"
                f"{direct_mae_cell} & {sum_mae_cell} & "
                f"{direct_f1_cell} & {sum_f1_cell} \\\\")
        
        latex.append(f"{line}")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\vspace{2mm}")
    latex.append(r"\begin{tablenotes}")
    latex.append(r"\small")
    latex.append(r"\item \textbf{Comparison:} 4-bit GPTQ quantized vs full precision Llama 3.1 (8B)")
    latex.append(r"\item Direct: Full transcript analysis. Sum: Sum of items 1-10.")
    latex.append(r"\item F1: Binary classification (threshold $\geq$ 20 for screening).")
    latex.append(r"\item \textbf{Bold/Italic:} Column-best/worst performance across both variants")
    latex.append(r"\end{tablenotes}")
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
    
    all_tables.append(table1(
        individual_results, binary_results, sum_results, custom_name_map, models_csv
    ))
    all_tables.append("\n% " + "="*70 + "\n")
    
    # Enhanced comprehensive ranking with reasoning highlights
    print("  ✓ Comprehensive ranking with reasoning highlights")
    all_tables.append(table2(
        individual_results, binary_results, sum_results, custom_name_map, models_csv
    ))
    
  
    full_latex = "\n\n".join(all_tables)
   
    
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)
    

    
    return full_latex

# Example usage

if __name__ == "__main__":
    # Example usage with your data
    import pickle
    
    try:
        with open("madrs_analysis_results.pkl", "rb") as f:
            data = pickle.load(f)
            individual_results = data["individual_results"]
            mean_results = data["mean_results"]
            binary_results = data.get("binary_results", None)
            sum_results=data.get("sum_results", None)
            models_csv = data.get("models_csv", None)
        
        # Generate all tables
        latex_output = generate_all_academic_tables(
            individual_results,
            mean_results,
            binary_results,
            sum_results,
            models_csv,
            output_file="llamadrs_academic_tables.tex"
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