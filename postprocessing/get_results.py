"""
get_results.py – Full MADRS analysis pipeline.

Reads prediction JSON files, computes per-item and mean MAE / R² metrics,
calculates sum-of-items predictions and binary classification F1 scores,
and writes:
  • llamadrs_results.pkl   (all results in one pickle)
  • llamadrs_predictions.xlsx  (per-video predictions)

Usage:
    python get_results.py                           # defaults
    python get_results.py --base-dir /path/to/sessions --output-dir ../output
"""

import argparse
import atexit
import csv
import getpass
import json
import os
import pickle
import re
import shutil
import time
from dataclasses import dataclass
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

madrs_no_dict = {
    0: "madrs",
    1: "madrs_01_apparent_sadness",
    2: "madrs_02_reported_sadness",
    3: "madrs_03_inner_tension",
    4: "madrs_04_reduced_sleep",
    5: "madrs_05_reduced_appetite",
    6: "madrs_06_concentration_difficulties",
    7: "madrs_07_lassitude",
    8: "madrs_08_inability_to_feel",
    9: "madrs_09_pessimistic_thoughts",
    10: "madrs_10_suicidal_thoughts",
}

MADRS_DICT = {
    0: "madrs_totalscore",
    1: "madrs1_apparentsadness",
    2: "madrs2_reportedsadness",
    3: "madrs3_tension",
    4: "madrs4_sleep",
    5: "madrs5_appetite",
    6: "madrs6_concentration",
    7: "madrs7_lassitude",
    8: "madrs8_inabilitytofeel",
    9: "madrs9_pessimisticthoughts",
    10: "madrs10_suicidalthoughts",
}

MADRS_ITEM_NAMES = {
    0: "Total Score",
    1: "Apparent Sadness",
    2: "Reported Sadness",
    3: "Inner Tension",
    4: "Reduced Sleep",
    5: "Reduced Appetite",
    6: "Concentration Difficulties",
    7: "Lassitude",
    8: "Inability to Feel",
    9: "Pessimistic Thoughts",
    10: "Suicidal Thoughts",
}


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

    MODEL_DICT[f"segmented_Qw3_235b_a22b_ar_4q_{ablation}"] = (
        f"Qwen 3: {ablation.replace('_', ' ').title()} (22B-235B)"
    )

MODEL_RANKS = {k: i for i, k in enumerate(MODEL_DICT.keys(), start=1)}
MODEL_REV_DICT = {v: k for k, v in MODEL_DICT.items()}

RATING_PATTERN = re.compile(r'"rating":\s*(\d+(?:\.\d+)?)', re.IGNORECASE)
EXPLANATION_PATTERN = re.compile(r"explanation:\s*([^\n]+)", re.IGNORECASE)

GT_CAP_TOTAL = 60
GT_CAP_ITEM = 6

# Pre-compute threshold epoch (09/07/2025 is d/m/Y)
THRESHOLD_EPOCH = time.mktime(time.strptime("09/07/2025", "%d/%m/%Y"))

# ---------------------------------------------------------------------------
# Globals set by CLI (populated in main)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path("../madrs_sessions")
ORIGINAL_BASE_DIR: Path = Path("../madrs_sessions")
CSV_MODELS: Path = Path("../models.csv")
PATIENTS_CSV: Path = Path("../CAMI/cami_patients.csv")
SESSIONS_CSV: Path = Path("../CAMI/cami_ra_sessions_.csv")
REPORTS_CSV: Path = Path("../CAMI/CAMI_reports.csv")
SCRATCH_DIR: Path | None = None

base_models: list[str] = []


# ---------------------------------------------------------------------------
# Scratch workspace (optional acceleration)
# ---------------------------------------------------------------------------
def setup_scratch_workspace() -> Path:
    """Copy data to scratch for faster I/O, return new paths."""
    username = getpass.getuser()
    scratch_dir = Path(f"/scratch/madrs_analysis_{username}_{os.getpid()}")

    # Clean up old scratch dirs from this user
    for old_dir in Path("/scratch").glob(f"madrs_analysis_{username}_*"):
        if old_dir != scratch_dir and not any(old_dir.rglob("*.lock")):
            try:
                shutil.rmtree(old_dir)
            except Exception:
                pass

    if scratch_dir.exists():
        print(f"Using existing scratch workspace: {scratch_dir}")
        return scratch_dir

    print(f"Setting up scratch workspace: {scratch_dir}")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    original_base = ORIGINAL_BASE_DIR
    scratch_base = scratch_dir / "madrs_sessions"

    print("Copying session structure to scratch...")
    for item in original_base.rglob("*"):
        if item.is_file() and item.suffix in [".txt", ".json"]:
            if "llamadrs_pred_json" in item.parts:
                continue
            rel_path = item.relative_to(original_base)
            dest = scratch_base / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)

    print("Copying CSV files to scratch...")
    for csv_file in [PATIENTS_CSV, SESSIONS_CSV, REPORTS_CSV, CSV_MODELS]:
        if csv_file.exists():
            shutil.copy2(csv_file, scratch_dir / csv_file.name)

    print(f"Scratch workspace ready: {scratch_dir}")
    atexit.register(lambda: shutil.rmtree(scratch_dir, ignore_errors=True))
    return scratch_dir


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8192)
def _load_json_cached(path_str: str) -> dict:
    p = Path(path_str)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8192)
def _read_text_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8", errors="ignore")


def _safe_extract_rating_from_text(text: str):
    m = RATING_PATTERN.search(text)
    if not m:
        return None
    try:
        return int(float(m.group(1)))
    except Exception:
        return None


def _reasoning_path_for(pred_path: Path, madrs_item: int, run: int) -> Path:
    return pred_path.parent / f"madrs{madrs_item}_output_{run}_reasoning.txt"


def _read_reasoning_and_count_tokens(pred_path: Path, madrs_item: int, run: int):
    try:
        p = _reasoning_path_for(pred_path, madrs_item, run)
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="ignore").strip()
            if not txt:
                return ("", 0)
            enc = tiktoken.get_encoding("cl100k_base")
            return (txt, len(enc.encode(txt)))
    except Exception:
        pass
    return ("", 0)


# ---------------------------------------------------------------------------
# Session indexing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionIndex:
    files: dict  # {(item, run): (pred_path, cfg_path)}


def _index_prediction_files_static(session_dir: Path, output_dir: str) -> dict:
    session_str = str(session_dir)
    if SCRATCH_DIR and "/scratch/" in session_str:
        session_str = session_str.replace(
            str(SCRATCH_DIR / "madrs_sessions"), str(ORIGINAL_BASE_DIR)
        )
        session_dir = Path(session_str)

    base = session_dir / "llamadrs_pred_json" / output_dir
    if not base.exists():
        return {}

    out = {}
    for entry in os.scandir(base):
        if not entry.is_file():
            continue
        m = re.match(r"madrs(\d+)_output_(\d+)\.json$", entry.name)
        if not m:
            continue
        item = int(m.group(1))
        run = int(m.group(2))
        p = Path(entry.path)
        cfg = p.with_name(p.stem + "_config.json")
        out[(item, run)] = (p, cfg)
    return out


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------
def _process_session_worker(args):
    (
        sid, output_dir, run, session_info, session_lookup_entry,
        by_item, _MADRS_DICT, _madrs_no_dict,
        _THRESHOLD_EPOCH, _GT_CAP_TOTAL, _GT_CAP_ITEM,
    ) = args
    try:
        session_dir = Path(session_info["session_dir"])
        file_index = _index_prediction_files_static(session_dir, output_dir)
        sess_data = session_lookup_entry
        individual_results = {}
        creation_time = None

        gt_total = int(sum(int(sess_data[_MADRS_DICT[i]]) for i in range(1, 11)))

        for madrs_item in range(0, 11):
            if sid not in by_item.get(madrs_item, set()):
                continue

            if madrs_item == 0:
                ground_truth = gt_total
                gt_cap = _GT_CAP_TOTAL
            else:
                ground_truth = int(sess_data[_MADRS_DICT[madrs_item]])
                gt_cap = _GT_CAP_ITEM
                
            
            backoff = ground_truth - gt_cap

            pred_tuple = file_index.get((madrs_item, run))
            failure = 0
            failure_reason = ""
            rating = None
            explanation = ""
            reasoning_text = ""
            num_reason_tokens = 0

            if not pred_tuple:
                rating = backoff
                failure = 1
                failure_reason = "missing file"
                continue
            else:
                pred_path, cfg_path = pred_tuple
                try:
                    data = _load_json_cached(str(pred_path))
                    rating = data.get("rating")
                except Exception:
                    data = None
                    try:
                        content = _read_text_cached(str(pred_path))
                        rating = _safe_extract_rating_from_text(content)
                    except Exception:
                        rating = None

                if data is not None:
                    explanation = data.get("explanation", "")
                else:
                    try:
                        explanation = (
                            EXPLANATION_PATTERN.search(content).group(1).strip()
                            if EXPLANATION_PATTERN.search(content)
                            else ""
                        )
                    except Exception:
                        explanation = None

                reasoning_text, num_reason_tokens = _read_reasoning_and_count_tokens(
                    pred_path, madrs_item, run
                )

            if rating is None:
                rating = backoff
                failure = 1
                failure_reason = "missing rating"

            if rating is not None and rating > gt_cap:
                continue

            creation_time_str = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(creation_time))
                if creation_time
                else ""
            )

            individual_results[madrs_item] = {
                "rating": rating,
                "explanation": explanation,
                "ground_truth": ground_truth,
                "failure": failure,
                "failure_reason": failure_reason,
                "creation_time": creation_time_str,
                "num_trans_tokens": session_info[madrs_item]["num_trans_tokens"],
                "transcription": session_info[madrs_item]["transcription"],
                "reasoning_text": reasoning_text,
                "num_reason_tokens": num_reason_tokens,
            }

        return {
            "patient": session_info["patient_dir"],
            "session": sid,
            "visit_day": sess_data.get("day", ""),
            "visit_no": sess_data.get("visit_no", ""),
            "edu": sess_data.get("edu", ""),
            "age": sess_data.get("age", ""),
            "gender": sess_data.get("gender", ""),
            "individual_items": individual_results,
        }
    except Exception as e:
        print(f"Error processing session {sid} run {run}: {e}")
        return None


# ---------------------------------------------------------------------------
# MADRSProcessor
# ---------------------------------------------------------------------------
class MADRSProcessor:
    def __init__(self):
        self.session_lookup = None

    def load_data(self):
        patients_df = pd.read_csv(PATIENTS_CSV)
        sessions_df = pd.read_csv(SESSIONS_CSV)
        reports_df = pd.read_csv(REPORTS_CSV)
        reports_df = reports_df[reports_df["interview_type"] == "RA"]
        reports_df["patient"] = reports_df["subject_id"]
        reports_df["session"] = (
            reports_df["osir_audio_video_file"].str.split("/").str[-2]
        )
        reports_df = reports_df.drop_duplicates(subset=["patient", "session"])
        df_total = patients_df.merge(sessions_df, on="patient", how="left")
        df_total = df_total.merge(reports_df, on=["patient", "session"], how="left")
        df_total["visit_no"] = df_total.groupby(["patient"])["date"].rank(
            "dense", ascending=True
        )
        df_total = df_total[df_total["session"].notna()].copy()
        df_total["madrs8_inabilitytofeel"] = df_total["madrs8_inabilitytofeel_x"]
        for col in df_total.columns:
            if col.endswith("_x") and col[:-2] not in df_total.columns:
                df_total[col[:-2]] = df_total[col]
        self.session_lookup = df_total.set_index("session").to_dict("index")
        return self.session_lookup

    def get_valid_sessions(self):
        by_item = {i: set() for i in range(0, 11)}
        info_by_session = {}

        for madrs_no in range(1, 11):
            madrs_label = MADRS_DICT[madrs_no]
            key = madrs_no_dict[madrs_no]
            madrs_file = BASE_DIR / f"{madrs_label}.txt"
            if not madrs_file.is_file():
                continue

            for session_path in madrs_file.read_text(encoding="utf-8").splitlines():
                transcriptions_dir = Path(session_path)
                session_dir = transcriptions_dir.parent
                patient_dir = session_dir.parent.parent
                sid = session_dir.name

                transcription_path = transcriptions_dir / f"{key}_merged_cleaned.txt"
                with open(transcription_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                encoding = tiktoken.get_encoding("cl100k_base")
                num_trans_tokens = len(encoding.encode(content))

                if sid not in info_by_session:
                    info_by_session[sid] = {
                        "patient_dir": patient_dir.name,
                        "session_dir": str(session_dir),
                        "session_id": sid,
                        "transcriptions_dir": str(transcriptions_dir),
                    }
                    full_transcription_path = (
                        transcriptions_dir / f"{madrs_no_dict[0]}_merged_cleaned.txt"
                    )
                    with open(full_transcription_path, "r", encoding="utf-8") as f:
                        full_content = f.read().strip()
                    full_num_trans_tokens = len(encoding.encode(full_content))
                    info_by_session[sid][0] = {
                        "transcription": full_content,
                        "num_trans_tokens": full_num_trans_tokens,
                    }

                info_by_session[sid][madrs_no] = {
                    "transcription": content,
                    "num_trans_tokens": num_trans_tokens,
                }
                by_item[madrs_no].add(sid)

        all_sessions = set().union(*by_item.values()) if by_item else set()
        by_item[0] = set(all_sessions)

        return {
            "by_item": by_item,
            "info_by_session": info_by_session,
            "all_sessions": all_sessions,
        }

    def process_model_run_parallel(
        self, output_dir, run, valid_sessions_struct, max_workers=None
    ):
        all_sids = sorted(valid_sessions_struct["by_item"].get(0, set()))
        print(f"Processing run {run} in {output_dir} with {len(all_sids)} sessions.")

        if max_workers is None:
            max_workers = 32

        worker_args = []
        for sid in all_sids:
            session_info = valid_sessions_struct["info_by_session"][sid]
            session_lookup_entry = self.session_lookup[sid]
            args = (
                sid, output_dir, run, session_info, session_lookup_entry,
                valid_sessions_struct["by_item"], MADRS_DICT, madrs_no_dict,
                THRESHOLD_EPOCH, GT_CAP_TOTAL, GT_CAP_ITEM,
            )
            worker_args.append(args)

        with Pool(processes=max_workers) as pool:
            results_list = list(
                tqdm(
                    pool.imap(_process_session_worker, worker_args),
                    total=len(worker_args),
                    desc=f"{output_dir} - Run {run}",
                    unit="session",
                )
            )

        individual_metrics = {}
        for madrs_item in range(0, 11):
            vals = [
                r["individual_items"].get(madrs_item)
                for r in results_list
                if r is not None
            ]
            vals = [v for v in vals if v is not None]
            expected = len(valid_sessions_struct["by_item"].get(madrs_item, set()))
            if len(vals) != expected:
                print(
                    f"  Warning: item {madrs_item} has {len(vals)} ratings "
                    f"but expected {expected}"
                )
                continue

            item_ratings = np.array([v["rating"] for v in vals], dtype=float)
            item_ground_truths = np.array([v["ground_truth"] for v in vals], dtype=float)
            item_failures = np.array([v["failure"] for v in vals], dtype=int)

            print(
                f"MADRS item {madrs_item}: {item_ratings.size} ratings, "
                f"{int(item_failures.sum())} failures"
            )

            if item_ratings.size == 0:
                continue
            if int(item_failures.sum()) >= item_ratings.size - 10:
                continue


            individual_metrics[madrs_item] = {
                "mae": float(mean_absolute_error(item_ground_truths, item_ratings)),
                "r2": float(r2_score(item_ground_truths, item_ratings)),
                "qwk": float(cohen_kappa_score(item_ground_truths, item_ratings, weights='quadratic')),
                "failures": int(item_failures.sum()),
            }

        return results_list, individual_metrics

    def process_model_runs_batch(
        self, output_dir, model_name, valid_sessions, runs=range(0, 5)
    ):
        run_results = []
        run_individual_metrics = []
        run_mean_maes, run_mean_r2s, run_count_failures, run_mean_qwks = [], [], [], []

        for run in tqdm(runs, desc=f"Processing runs for {model_name}"):
            results_list, individual_metrics = self.process_model_run_parallel(
                output_dir, run, valid_sessions
            )
            if not results_list:
                print(f"  No valid predictions found for run {run}")
                continue

            run_results.append(results_list)
            run_individual_metrics.append(individual_metrics)

            usable = [
                m
                for k, m in individual_metrics.items()
                if k != 0 and not np.isnan(m.get("mae", np.nan))
            ]
            if usable:
                run_mean_maes.append(np.mean([m["mae"] for m in usable]))
                run_mean_r2s.append(
                    np.mean(
                        [m["r2"] for m in usable if not np.isnan(m.get("r2", np.nan))]
                    )
                )
                
                run_mean_qwks.append(
                    float(np.mean(
                    [m["qwk"] for m in usable if np.isfinite(m.get("qwk", np.nan))]
                    ))
                )
                run_count_failures.append(
                    np.sum([m.get("failures", 0) for m in usable])
                )

        return (
            run_results,
            run_individual_metrics,
            run_mean_maes,
            run_mean_r2s,
            run_mean_qwks,
            run_count_failures,
        )

    # ------------------------------------------------------------------ Excel
    def _write_predictions_excel(
        self, excel_cells, excel_ground_truth, session_meta, models_csv_dict,
        out_path="llamadrs_predictions.xlsx",
    ):
        def _sheet_name(i):
            base = f"{i:02d} - {MADRS_ITEM_NAMES[i]}"
            return (base[:28] + "...") if len(base) > 31 else base

        with pd.ExcelWriter(
            out_path,
            engine="xlsxwriter",
            engine_kwargs={"options": {"use_zip64": True}},
        ) as writer:
            for i in range(0, 11):
                sessions_gt = set(excel_ground_truth[i].keys())
                sessions_pred = (
                    set().union(*(colmap.keys() for colmap in excel_cells[i].values()))
                    if excel_cells[i]
                    else set()
                )
                all_sessions = sorted(sessions_gt.union(sessions_pred))

                rows = []
                for sid in all_sessions:
                    meta = self.session_lookup.get(sid, {})
                    gt = excel_ground_truth[i].get(sid, np.nan)

                    for model_name, sid_map in sorted(excel_cells[i].items()):
                        if sid not in sid_map:
                            continue
                        if MODEL_REV_DICT.get(model_name, "") not in base_models:
                            continue
                        model_row = models_csv_dict.get(
                            MODEL_REV_DICT.get(model_name, ""), {}
                        )
                        value = sid_map[sid]

                        row = {
                            "session": sid,
                            "patient": meta.get("patient", ""),
                            "visit_no": meta.get("visit_no", ""),
                            "visit_day": meta.get("visit_day", ""),
                            "edu": meta.get("edu", ""),
                            "age": meta.get("age", ""),
                            "gender": meta.get("gender", ""),
                            "rater": meta.get("redcap_user", ""),
                            "diagnostic": meta.get("diagnostic", ""),
                            "model_name": model_name,
                            "model_family": model_row.get("model_family", "Unknown"),
                            "architecture": model_row.get("architecture", "Unknown"),
                            "context_length": model_row.get(
                                "context_length", "Unknown"
                            ),
                            "is_reasoning_model": model_row.get(
                                "reasoning", "Unknown"
                            ),
                            "active_params": model_row.get(
                                "active_params", "Unknown"
                            ),
                            "total_params": model_row.get("total_params", "Unknown"),
                            "transcription": value.get("transcription", ""),
                            "num_trans_tokens": value.get("num_trans_tokens", 0),
                            "ground_truth": gt,
                        }

                        for run_idx in range(3):
                            row.update(
                                {
                                    f"explanation_{run_idx}": value.get(
                                        f"explanation_{run_idx}", ""
                                    ),
                                    f"reasoning_{run_idx}": value.get(
                                        f"reasoning_{run_idx}", ""
                                    ),
                                    f"rating_{run_idx}": value.get(
                                        f"rating_{run_idx}", np.nan
                                    ),
                                    f"error_{run_idx}": value.get(
                                        f"error_{run_idx}", np.nan
                                    ),
                                    f"creation_time_{run_idx}": value.get(
                                        f"creation_time_{run_idx}", ""
                                    ),
                                    f"failure_reason_{run_idx}": value.get(
                                        f"failure_reason_{run_idx}", ""
                                    ),
                                    f"num_reason_tokens_{run_idx}": value.get(
                                        f"num_reason_tokens_{run_idx}", 0
                                    ),
                                }
                            )
                        rows.append(row)

                if not rows:
                    continue

                df = pd.DataFrame(rows)
                df.sort_values(by=["model_name", "session"], inplace=True)
                df.reset_index(drop=True, inplace=True)

                col_order = [
                    "session", "patient", "visit_no", "visit_day", "edu", "age",
                    "gender", "rater", "diagnostic", "model_name", "model_family",
                    "context_length", "is_reasoning_model", "architecture",
                    "active_params", "total_params", "transcription",
                    "num_trans_tokens", "ground_truth",
                ]
                for run_idx in range(3):
                    col_order.extend(
                        [
                            f"explanation_{run_idx}",
                            f"reasoning_{run_idx}",
                            f"rating_{run_idx}",
                            f"error_{run_idx}",
                            f"creation_time_{run_idx}",
                            f"failure_reason_{run_idx}",
                            f"num_reason_tokens_{run_idx}",
                        ]
                    )
                df = df[col_order]

                sheet_name = _sheet_name(i)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                ws.freeze_panes(1, 0)
                ws.autofilter(0, 0, df.shape[0], df.shape[1] - 1)

        print(f"Excel written to: {out_path}")

    # --------------------------------------------------------------- Pipeline
    def run_analysis(self, excel_path="llamadrs_predictions.xlsx"):
        self.load_data()

        mean_results: dict[str, dict] = {}
        individual_results: dict[str, dict] = {}

        excel_cells = {i: {} for i in range(0, 11)}
        excel_ground_truth = {i: {} for i in range(0, 11)}
        session_meta: dict[str, dict] = {}

        # Load models_.csv once
        if CSV_MODELS.exists():
            with CSV_MODELS.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                models_csv_dict = {
                    f"segmented_{row['short_name']}": row for row in reader
                }
        else:
            models_csv_dict = {}

        valid = self.get_valid_sessions()
        by_item = valid["by_item"]

        for madrs_no in range(0, 11):
            print(
                f"MADRS item {madrs_no}: "
                f"{len(by_item.get(madrs_no, set()))} valid sessions found."
            )

        if not by_item or not by_item.get(0):
            return {}, {}, {}, {}, {}, {}, {}

        for output_dir, model_name in tqdm(
            sorted(list(MODEL_DICT.items())), desc="Models"
        ):
            (
                run_results,
                run_individual_metrics,
                run_mean_maes,
                run_mean_r2s,
                run_mean_qwks,
                run_count_failures,
            ) = self.process_model_runs_batch(
                output_dir, model_name, valid, runs=range(0, 3)
            )

            col_key = model_name

            for run_idx, results_list in enumerate(run_results):
                for r in results_list:
                    sid = r["session"]
                    if sid not in session_meta:
                        session_meta[sid] = {
                            "patient": r.get("patient", ""),
                            "visit_day": r.get("visit_day", ""),
                            "visit_no": r.get("visit_no", ""),
                            "edu": r.get("edu", ""),
                            "age": r.get("age", ""),
                            "gender": r.get("gender", ""),
                        }

                    for i in range(0, 11):
                        cell = r["individual_items"].get(i)
                        if not cell:
                            continue

                        excel_cells[i].setdefault(col_key, {}).setdefault(sid, {})
                        excel_cells[i][col_key][sid][f"creation_time_{run_idx}"] = (
                            cell.get("creation_time", np.nan)
                        )
                        excel_cells[i][col_key][sid]["transcription"] = cell.get(
                            "transcription", ""
                        )
                        excel_cells[i][col_key][sid]["num_trans_tokens"] = cell.get(
                            "num_trans_tokens", 0
                        )
                        excel_cells[i][col_key][sid][f"reasoning_{run_idx}"] = (
                            cell.get("reasoning_text", "")
                        )
                        excel_cells[i][col_key][sid][
                            f"num_reason_tokens_{run_idx}"
                        ] = cell.get("num_reason_tokens", 0)

                        if cell.get("failure", 0) == 1:
                            excel_cells[i][col_key][sid][f"rating_{run_idx}"] = np.nan
                            excel_cells[i][col_key][sid][f"explanation_{run_idx}"] = ""
                            excel_cells[i][col_key][sid][f"error_{run_idx}"] = np.nan
                            excel_cells[i][col_key][sid][
                                f"failure_reason_{run_idx}"
                            ] = cell.get("failure_reason", "unknown")
                        else:
                            r_val = cell.get("rating", np.nan)
                            g_val = cell.get("ground_truth", np.nan)
                            excel_cells[i][col_key][sid][
                                f"explanation_{run_idx}"
                            ] = cell.get("explanation", "")
                            excel_cells[i][col_key][sid][
                                f"rating_{run_idx}"
                            ] = r_val
                            excel_cells[i][col_key][sid][f"error_{run_idx}"] = (
                                float(abs(r_val - g_val))
                                if pd.notna(r_val) and pd.notna(g_val)
                                else np.nan
                            )
                            excel_cells[i][col_key][sid][
                                f"failure_reason_{run_idx}"
                            ] = ""

                        excel_ground_truth[i].setdefault(
                            sid, cell.get("ground_truth", np.nan)
                        )

            if run_mean_maes:
                mean_results[model_name] = {
                    "mae_mean": float(np.mean(run_mean_maes)),
                    "mae_std": (
                        float(np.std(run_mean_maes))
                        if len(run_mean_maes) > 1
                        else 0.0
                    ),
                    "r2_mean": (
                        float(np.nanmean(run_mean_r2s))
                        if len(run_mean_r2s) > 0
                        else float("nan")
                    ),
                    "r2_std": (
                        float(np.nanstd(run_mean_r2s))
                        if len(run_mean_r2s) > 1
                        else 0.0
                    ),
                    "failures_mean": (
                        float(np.mean(run_count_failures))
                        if len(run_count_failures) > 0
                        else float("nan")
                    ),
                    "failures_std": (
                        float(np.std(run_count_failures))
                        if len(run_count_failures) > 1
                        else 0.0
                    ),
                    "is_reasoning_model": models_csv_dict.get(output_dir, {}).get(
                        "reasoning", ""
                    ),
                    "context_length": models_csv_dict.get(output_dir, {}).get(
                        "context_length", "Unknown"
                    ),
                }

            individual_results[model_name] = {}
            for madrs_item in range(0, 11):
                item_maes = [
                    m[madrs_item]["mae"]
                    for m in run_individual_metrics
                    if madrs_item in m
                ]
                item_r2s = [
                    m[madrs_item]["r2"]
                    for m in run_individual_metrics
                    if madrs_item in m
                ]
                item_qwks = [
                    m[madrs_item].get("qwk", float("nan"))
                    for m in run_individual_metrics
                    if madrs_item in m
                ]
                item_failures = [
                    m[madrs_item]["failures"]
                    for m in run_individual_metrics
                    if madrs_item in m
                ]

                if item_maes:
                    max_num_samples = (
                        max(
                            sum(
                                1
                                for r in run_result
                                if madrs_item in r["individual_items"]
                            )
                            for run_result in run_results
                        )
                        if run_results
                        else 0
                    )
                    individual_results[model_name][madrs_item] = {
                        "mae_mean": float(np.mean(item_maes)),
                        "mae_std": (
                            float(np.std(item_maes))
                            if len(item_maes) > 1
                            else 0.0
                        ),
                        "r2_mean": (
                            float(np.mean(item_r2s)) if item_r2s else float("nan")
                        ),
                        "r2_std": (
                            float(np.std(item_r2s)) if len(item_r2s) > 1 else 0.0
                        ),
                        "qwk_mean": (
                            float(np.mean(item_qwks))
                            if len(item_qwks) > 0
                            else float("nan")
                        ),
                        "qwk_std": (
                            float(np.std(item_qwks))
                            if len(item_qwks) > 1
                            else 0.0
                        ),
                        "num_samples": int(max_num_samples),
                        "failures_mean": (
                            float(np.mean(item_failures))
                            if item_failures
                            else float("nan")
                        ),
                        "failures_std": (
                            float(np.std(item_failures))
                            if len(item_failures) > 1
                            else 0.0
                        ),
                        "is_reasoning_model": models_csv_dict.get(
                            output_dir, {}
                        ).get("reasoning", ""),
                        "context_length": models_csv_dict.get(output_dir, {}).get(
                            "context_length", "Unknown"
                        ),
                    }

        self._write_predictions_excel(
            excel_cells, excel_ground_truth, session_meta, models_csv_dict,
            out_path=excel_path,
        )

        return (
            mean_results, individual_results, excel_cells,
            excel_ground_truth, session_meta, valid, models_csv_dict,
        )


# ---------------------------------------------------------------------------
# calculate_sum_predictions
# ---------------------------------------------------------------------------
def calculate_sum_predictions(processor, valid_sessions, models_csv_dict):
    """Calculate sum of items 1-10 predictions for each model."""
    processor.load_data()
    sum_results = {}

    for output_dir, model_name in MODEL_DICT.items():
        print(f"Calculating sum predictions for {model_name}...")
        run_results, _, _, _, _, _ = processor.process_model_runs_batch(
            output_dir, model_name, valid_sessions, runs=range(0, 3)
        )

        run_metrics = []
        for run_idx, results_list in enumerate(run_results):
            predictions, ground_truths = [], []
            failures = 0

            for session_result in results_list:
                sum_pred = 0
                sum_gt = 0
                has_all_items = True

                for item in range(1, 11):
                    if item in session_result["individual_items"]:
                        item_data = session_result["individual_items"][item]
                        if item_data["failure"] == 1:
                            has_all_items = False
                            failures += 1
                            break
                        sum_pred += item_data["rating"]
                        sum_gt += item_data["ground_truth"]
                    else:
                        has_all_items = False
                        break

                if has_all_items:
                    predictions.append(sum_pred)
                    ground_truths.append(sum_gt)

            if predictions:
                predictions = np.array(predictions)
                ground_truths = np.array(ground_truths)
                run_metrics.append(
                    {
                        "mae": float(mean_absolute_error(ground_truths, predictions)),
                        "r2": float(r2_score(ground_truths, predictions)),
                        "qwk": float(cohen_kappa_score(ground_truths, predictions, weights='quadratic')),
                        "failures": failures,
                        "n_samples": len(predictions),
                    }
                )

        if run_metrics:
            sum_results[model_name] = {
                "mae_mean": float(np.mean([m["mae"] for m in run_metrics])),
                "mae_std": float(np.std([m["mae"] for m in run_metrics])),
                "r2_mean": float(np.mean([m["r2"] for m in run_metrics])),
                "r2_std": float(np.std([m["r2"] for m in run_metrics])),
                "qwk_mean": float(np.mean([m["qwk"] for m in run_metrics])),
                "qwk_std": float(np.std([m["qwk"] for m in run_metrics])),
                "failures_mean": float(
                    np.mean([m["failures"] for m in run_metrics])
                ),
                "n_samples": int(np.mean([m["n_samples"] for m in run_metrics])),
                "context_length": models_csv_dict.get(output_dir, {}).get(
                    "context_length", "Unknown"
                ),
                "reasoning": models_csv_dict.get(output_dir, {}).get(
                    "reasoning", "No"
                ),
            }

    return sum_results


# ---------------------------------------------------------------------------
# calculate_binary_classification_metrics
# ---------------------------------------------------------------------------
def calculate_binary_classification_metrics(
    processor, valid_sessions, models_csv_dict
):
    """Calculate F1 scores for binary classification tasks."""
    binary_results = {}

    for output_dir, model_name in MODEL_DICT.items():
        print(f"Calculating binary F1 for {model_name}...")
        run_results, _, _, _, _, _ = processor.process_model_runs_batch(
            output_dir, model_name, valid_sessions, runs=range(0, 3)
        )

        model_metrics = {
            "items": {i: [] for i in range(1, 11)},
            "total_direct": [],
            "total_sum": [],
        }

        for run_idx, results_list in enumerate(run_results):
            # Individual items (threshold >= 3)
            for item in range(1, 11):
                y_true, y_pred = [], []
                for session_result in results_list:
                    if item in session_result["individual_items"]:
                        item_data = session_result["individual_items"][item]
                        if item_data["failure"] == 0:
                            gt = item_data["ground_truth"]
                            pred = item_data["rating"]
                            y_true.append(1 if gt >= 3 else 0)
                            y_pred.append(1 if pred >= 3 else 0)
                if y_true:
                    model_metrics["items"][item].append(
                        {
                            "f1": f1_score(y_true, y_pred, zero_division=0),
                            "precision": precision_score(
                                y_true, y_pred, zero_division=0
                            ),
                            "recall": recall_score(y_true, y_pred, zero_division=0),
                            "n": len(y_true),
                        }
                    )

            # Total direct (Item 0, threshold >= 20)
            y_true_direct, y_pred_direct = [], []
            for session_result in results_list:
                if 0 in session_result["individual_items"]:
                    item_data = session_result["individual_items"][0]
                    if item_data["failure"] == 0:
                        gt = item_data["ground_truth"]
                        pred = item_data["rating"]
                        y_true_direct.append(1 if gt >= 20 else 0)
                        y_pred_direct.append(1 if pred >= 20 else 0)
            if y_true_direct:
                model_metrics["total_direct"].append(
                    {
                        "f1": f1_score(y_true_direct, y_pred_direct, zero_division=0),
                        "precision": precision_score(
                            y_true_direct, y_pred_direct, zero_division=0
                        ),
                        "recall": recall_score(
                            y_true_direct, y_pred_direct, zero_division=0
                        ),
                        "n": len(y_true_direct),
                    }
                )

            # Total sum (threshold >= 20)
            y_true_sum, y_pred_sum = [], []
            for session_result in results_list:
                sum_pred, sum_gt = 0, 0
                has_all = True
                for item in range(1, 11):
                    if item in session_result["individual_items"]:
                        item_data = session_result["individual_items"][item]
                        if item_data["failure"] == 1:
                            has_all = False
                            break
                        sum_pred += item_data["rating"]
                        sum_gt += item_data["ground_truth"]
                    else:
                        has_all = False
                        break
                if has_all:
                    y_true_sum.append(1 if sum_gt >= 20 else 0)
                    y_pred_sum.append(1 if sum_pred >= 20 else 0)
            if y_true_sum:
                model_metrics["total_sum"].append(
                    {
                        "f1": f1_score(y_true_sum, y_pred_sum, zero_division=0),
                        "precision": precision_score(
                            y_true_sum, y_pred_sum, zero_division=0
                        ),
                        "recall": recall_score(
                            y_true_sum, y_pred_sum, zero_division=0
                        ),
                        "n": len(y_true_sum),
                    }
                )

        # Aggregate across runs
        binary_results[model_name] = {
            "items": {},
            "total_direct": {},
            "total_sum": {},
            "context_length": models_csv_dict.get(output_dir, {}).get(
                "context_length", "Unknown"
            ),
            "reasoning": models_csv_dict.get(output_dir, {}).get("reasoning", "No"),
        }

        for item in range(1, 11):
            if model_metrics["items"][item]:
                metrics_list = model_metrics["items"][item]
                binary_results[model_name]["items"][item] = {
                    "f1_mean": float(np.mean([m["f1"] for m in metrics_list])),
                    "f1_std": float(np.std([m["f1"] for m in metrics_list])),
                    "precision_mean": float(
                        np.mean([m["precision"] for m in metrics_list])
                    ),
                    "recall_mean": float(
                        np.mean([m["recall"] for m in metrics_list])
                    ),
                    "n_samples": int(np.mean([m["n"] for m in metrics_list])),
                }

        if model_metrics["total_direct"]:
            ml = model_metrics["total_direct"]
            binary_results[model_name]["total_direct"] = {
                "f1_mean": float(np.mean([m["f1"] for m in ml])),
                "f1_std": float(np.std([m["f1"] for m in ml])),
                "precision_mean": float(np.mean([m["precision"] for m in ml])),
                "recall_mean": float(np.mean([m["recall"] for m in ml])),
                "n_samples": int(np.mean([m["n"] for m in ml])),
            }

        if model_metrics["total_sum"]:
            ml = model_metrics["total_sum"]
            binary_results[model_name]["total_sum"] = {
                "f1_mean": float(np.mean([m["f1"] for m in ml])),
                "f1_std": float(np.std([m["f1"] for m in ml])),
                "precision_mean": float(np.mean([m["precision"] for m in ml])),
                "recall_mean": float(np.mean([m["recall"] for m in ml])),
                "n_samples": int(np.mean([m["n"] for m in ml])),
            }

    return binary_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    global BASE_DIR, ORIGINAL_BASE_DIR, CSV_MODELS, PATIENTS_CSV, SESSIONS_CSV
    global REPORTS_CSV, SCRATCH_DIR, base_models

    parser = argparse.ArgumentParser(
        description="Run full MADRS analysis pipeline and export results."
    )
    parser.add_argument(
        "--base-dir", type=str,
        default="../madrs_sessions",
        help="Base directory containing MADRS session data",
    )
    parser.add_argument(
        "--models-csv", type=str, default="../models.csv",
        help="Path to models.csv",
    )
    parser.add_argument(
        "--patients-csv", type=str,
        default="../CAMI/cami_patients.csv",
        help="Path to cami_patients.csv",
    )
    parser.add_argument(
        "--sessions-csv", type=str,
        default="../CAMI/cami_ra_sessions_.csv",
        help="Path to cami_ra_sessions_.csv",
    )
    parser.add_argument(
        "--reports-csv", type=str,
        default="../CAMI/CAMI_reports.csv",
        help="Path to CAMI_reports.csv",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="../output",
        help="Directory for output files (default: ../output)",
    )
    parser.add_argument(
        "--no-scratch", action="store_true",
        help="Disable scratch workspace acceleration",
    )
    parser.add_argument(
        "--skip-sum", action="store_true",
        help="Skip sum-of-items prediction calculation",
    )
    parser.add_argument(
        "--skip-binary", action="store_true",
        help="Skip binary classification F1 calculation",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Force re-running the full analysis even if results pickle exists",
    )
    args = parser.parse_args()

    # Set global paths
    ORIGINAL_BASE_DIR = Path(args.base_dir)
    BASE_DIR = ORIGINAL_BASE_DIR
    CSV_MODELS = Path(args.models_csv)
    PATIENTS_CSV = Path(args.patients_csv)
    SESSIONS_CSV = Path(args.sessions_csv)
    REPORTS_CSV = Path(args.reports_csv)

    # Optional scratch acceleration
    if not args.no_scratch and Path("/scratch").is_dir():
        SCRATCH_DIR = setup_scratch_workspace()
        BASE_DIR = SCRATCH_DIR / "madrs_sessions"
        if (SCRATCH_DIR / CSV_MODELS.name).exists():
            CSV_MODELS = SCRATCH_DIR / CSV_MODELS.name
        if (SCRATCH_DIR / PATIENTS_CSV.name).exists():
            PATIENTS_CSV = SCRATCH_DIR / PATIENTS_CSV.name
        if (SCRATCH_DIR / SESSIONS_CSV.name).exists():
            SESSIONS_CSV = SCRATCH_DIR / SESSIONS_CSV.name
        if (SCRATCH_DIR / REPORTS_CSV.name).exists():
            REPORTS_CSV = SCRATCH_DIR / REPORTS_CSV.name

    # Read base_models from CSV
    if CSV_MODELS.exists():
        models_df = pd.read_csv(CSV_MODELS)
        base_models = [f"segmented_{row['short_name']}" for _, row in models_df.iterrows()]
    else:
        print(f"Warning: {CSV_MODELS} not found, base_models will be empty")
        base_models = []

    # Ensure output dir exists
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / "llamadrs_results.pkl"
    excel_path = out_dir / "llamadrs_predictions.xlsx"

    # ---- Run main analysis ----
    print("=" * 80)
    print("RUNNING FULL MADRS ANALYSIS PIPELINE")
    print("=" * 80)

    processor = MADRSProcessor()

    # If a previous results pickle exists and reprocess not requested, load it
    results_bundle = None
    mean_results = {}
    individual_results = {}
    excel_cells = {}
    excel_ground_truth = {}
    session_meta = {}
    valid = {}
    models_csv_dict = {}
    sum_results = {}
    binary_results = {}

    if pkl_path.exists() and not args.reprocess:
        print(f"Found existing results at {pkl_path}, loading and skipping full run_analysis.")
        try:
            with open(pkl_path, "rb") as f:
                results_bundle = pickle.load(f)
        except Exception as e:
            print(f"Warning: failed to load existing pickle ({e}), will re-run analysis.")
            results_bundle = None

    if results_bundle is not None:
        mean_results = results_bundle.get("mean_results", {})
        individual_results = results_bundle.get("individual_results", {})
        excel_cells = results_bundle.get("excel_cells", {})
        excel_ground_truth = results_bundle.get("excel_ground_truth", {})
        session_meta = results_bundle.get("session_meta", {})
        valid = results_bundle.get("valid", {})
        models_csv_dict = results_bundle.get("models_csv", {})
        sum_results = results_bundle.get("sum_results", {}) or {}
        binary_results = results_bundle.get("binary_results", {}) or {}
    else:
        # Run full analysis and get base results
        (
            mean_results, individual_results, excel_cells,
            excel_ground_truth, session_meta, valid, models_csv_dict,
        ) = processor.run_analysis(excel_path=str(excel_path))

        # Save initial bundle (sum/binary left empty until computed)
        results_bundle = {
            "mean_results": mean_results,
            "individual_results": individual_results,
            "excel_cells": excel_cells,
            "excel_ground_truth": excel_ground_truth,
            "session_meta": session_meta,
            "valid": valid,
            "models_csv": models_csv_dict,
            "sum_results": {},
            "binary_results": {},
        }
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(results_bundle, f)
            print(f"\nResults saved to {pkl_path}")
        except Exception as e:
            print(f"Warning: failed to write initial results pickle: {e}")

    # ---- Sum predictions ----
    if not args.skip_sum:
        if sum_results:
            print("Sum predictions already present in loaded results; skipping calculation.")
        else:
            print("\n" + "=" * 80)
            print("CALCULATING SUM PREDICTIONS (Items 1-10)")
            print("=" * 80)
            sum_results = calculate_sum_predictions(processor, valid, models_csv_dict)
            print(f"Completed sum calculations for {len(sum_results)} models")
            # update and save
            results_bundle["sum_results"] = sum_results
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(results_bundle, f)
                print(f"Updated results saved to {pkl_path}")
            except Exception as e:
                print(f"Warning: failed to update pickle with sum_results: {e}")
    else:
        print("Skipping sum-of-items prediction calculation per CLI flag.")

    # ---- Binary classification ----
    if not args.skip_binary:
        if binary_results:
            print("Binary classification results already present in loaded results; skipping calculation.")
        else:
            print("\n" + "=" * 80)
            print("CALCULATING BINARY CLASSIFICATION F1 SCORES")
            print("=" * 80)
            binary_results = calculate_binary_classification_metrics(
                processor, valid, models_csv_dict
            )
            print(f"Completed F1 calculations for {len(binary_results)} models")
            # update and save
            results_bundle["binary_results"] = binary_results
            try:
                with open(pkl_path, "wb") as f:
                    pickle.dump(results_bundle, f)
                print(f"Updated results saved to {pkl_path}")
            except Exception as e:
                print(f"Warning: failed to update pickle with binary_results: {e}")
    else:
        print("Skipping binary classification F1 calculation per CLI flag.")

    # ---- Final bundle (ensure variables in scope) ----
    # results_bundle should already be up to date; expose variables locally
    sum_results = results_bundle.get("sum_results", {})
    binary_results = results_bundle.get("binary_results", {})

    # ---- Summary ----
    if mean_results:
        sorted_results = sorted(mean_results.items(), key=lambda x: x[1]["mae_mean"])
        print(f"\nTop 5 models by MAE (mean of items):")
        for model_name, metrics in sorted_results[:5]:
            print(
                f"  {model_name}: "
                f"MAE={metrics['mae_mean']:.3f}±{metrics['mae_std']:.3f}  "
                f"R²={metrics['r2_mean']:.3f}±{metrics['r2_std']:.3f}  "
                f"QWK={metrics.get('qwk_mean', float('nan')):.3f}±{metrics.get('qwk_std', float('nan')):.3f}  "
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
