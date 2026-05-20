import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP14_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step14_pf_zt_prediction"
DEFAULT_STEP13_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step13_sigma_validation"
DEFAULT_STEP12_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"
DEFAULT_STEP11_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step11_unit_normalized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step15_pf_zt_error_analysis"

EXCEL_PREVIEW_ROWS = 100_000
ERROR_LEVELS = ["excellent", "good", "moderate", "poor", "not_available"]
ZT_THRESHOLDS = [0.5, 1.0, 1.5]

REQUIRED_STEP14_FILES = {
    "thermoelectric_predictions": "thermoelectric_predictions_step14.csv",
    "sample_results": "pf_zt_sample_results_step14.csv",
    "classification": "zt_high_performance_classification_step14.csv",
    "problem_rows": "pf_zt_problem_rows_step14.csv",
    "problem_samples": "pf_zt_problem_samples_step14.csv",
}
OPTIONAL_STEP14_FILES = {
    "validation_predictions": "thermoelectric_validation_predictions_step14.csv",
    "validation_sample_results": "pf_zt_validation_sample_results_step14.csv",
    "material_summary": "pf_zt_material_summary_step14.csv",
    "error_distribution": "zt_error_distribution_step14.csv",
}

ROW_REQUIRED = [
    "sample_key",
    "temperature_K",
    "sigma_pred_for_pf_zt_S_per_m_step14",
    "power_factor_pred_W_per_mK2_step14",
    "zt_pred_from_sigma_step14",
]
SAMPLE_REQUIRED = ["sample_key"]

META_COLUMNS = [
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "tau_eff_step12",
    "tau_eff_unit_step12",
    "tau_eff_mode_step12",
    "fitting_source_actual_step10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Step14 PF/ZT prediction errors.")
    parser.add_argument("--step14_dir", type=Path, default=DEFAULT_STEP14_DIR)
    parser.add_argument("--step13_dir", type=Path, default=DEFAULT_STEP13_DIR)
    parser.add_argument("--step12_dir", type=Path, default=DEFAULT_STEP12_DIR)
    parser.add_argument("--step11_dir", type=Path, default=DEFAULT_STEP11_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zt_threshold", type=float, default=1.0)
    parser.add_argument("--top_n_manual_review", type=int, default=500)
    parser.add_argument("--top_n_best_candidates", type=int, default=300)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_bool(value: Any) -> bool:
    return normalize_text(value).casefold() in {"true", "1", "yes", "y"}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def header_columns(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_csv_selected(path: Path, desired: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if desired is None:
        return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)
    columns = header_columns(path)
    usecols = [column for column in desired if column in columns]
    return pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def validate_required(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def input_paths(step14_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, filename in REQUIRED_STEP14_FILES.items():
        path = step14_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required Step14 input file not found: {path}")
        paths[label] = path
    for label, filename in OPTIONAL_STEP14_FILES.items():
        path = step14_dir / filename
        if path.exists():
            paths[label] = path
    return paths


def error_level(relative_error: pd.Series) -> pd.Series:
    values = pd.to_numeric(relative_error, errors="coerce")
    level = pd.Series("not_available", index=values.index, dtype="object")
    level.loc[np.isfinite(values) & (values <= 0.20)] = "excellent"
    level.loc[np.isfinite(values) & (values > 0.20) & (values <= 0.50)] = "good"
    level.loc[np.isfinite(values) & (values > 0.50) & (values <= 1.00)] = "moderate"
    level.loc[np.isfinite(values) & (values > 1.00)] = "poor"
    return level


def direction(pred: pd.Series, obs: pd.Series, relative: pd.Series) -> pd.Series:
    pred_num = pd.to_numeric(pred, errors="coerce")
    obs_num = pd.to_numeric(obs, errors="coerce")
    rel_num = pd.to_numeric(relative, errors="coerce")
    result = pd.Series("not_available", index=pred.index, dtype="object")
    available = np.isfinite(pred_num) & np.isfinite(obs_num)
    result.loc[available & np.isfinite(rel_num) & (rel_num <= 0.20)] = "near"
    result.loc[available & ~(np.isfinite(rel_num) & (rel_num <= 0.20)) & (pred_num > obs_num)] = "over_predicted"
    result.loc[available & ~(np.isfinite(rel_num) & (rel_num <= 0.20)) & (pred_num < obs_num)] = "under_predicted"
    result.loc[available & ~(np.isfinite(rel_num) & (rel_num <= 0.20)) & (pred_num == obs_num)] = "near"
    return result


def row_category(row: pd.Series) -> str:
    if row["zt_obs_error_level_step15"] == "poor" or row["zt_calc_error_level_step15"] == "poor":
        return "large_zt_error"
    if row["pf_error_level_step15"] == "poor":
        return "large_pf_error"
    if row["zt_obs_error_level_step15"] == "not_available" and row["zt_calc_error_level_step15"] == "not_available":
        return "missing_zt_eval"
    if row["pf_error_level_step15"] == "not_available":
        return "missing_pf_eval"
    if row["zt_obs_error_level_step15"] in {"excellent", "good"}:
        return "good_prediction"
    return "review"


def row_note(row: pd.Series) -> str:
    notes: list[str] = []
    if row["pf_error_level_step15"] == "poor":
        notes.append("large PF error")
    if row["zt_obs_error_level_step15"] == "poor":
        notes.append("large ZT error vs observed")
    if row["zt_calc_error_level_step15"] == "poor":
        notes.append("large ZT error vs calculated")
    if row["zt_obs_error_level_step15"] == "not_available":
        notes.append("ZT observed comparison unavailable")
    if row["zt_calc_error_level_step15"] == "not_available":
        notes.append("ZT calculated comparison unavailable")
    return "; ".join(notes) if notes else "ok"


def build_error_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows.copy()
    ensure_columns(
        output,
        [
            "pf_relative_error_step14",
            "pf_log_error_step14",
            "zt_pred_vs_obs_relative_error_step14",
            "zt_pred_vs_obs_log_error_step14",
            "zt_pred_vs_calc_relative_error_step14",
            "zt_pred_vs_calc_log_error_step14",
            "zt_pred_from_sigma_step14",
            "zt_obs_dimensionless_step11",
            "zt_calc_from_obs_step11",
        ],
    )
    output["pf_error_level_step15"] = error_level(output["pf_relative_error_step14"])
    output["zt_obs_error_level_step15"] = error_level(output["zt_pred_vs_obs_relative_error_step14"])
    output["zt_calc_error_level_step15"] = error_level(output["zt_pred_vs_calc_relative_error_step14"])
    output["zt_error_direction_vs_obs_step15"] = direction(
        output["zt_pred_from_sigma_step14"],
        output["zt_obs_dimensionless_step11"],
        output["zt_pred_vs_obs_relative_error_step14"],
    )
    output["zt_error_direction_vs_calc_step15"] = direction(
        output["zt_pred_from_sigma_step14"],
        output["zt_calc_from_obs_step11"],
        output["zt_pred_vs_calc_relative_error_step14"],
    )
    output["row_error_category_step15"] = output.apply(row_category, axis=1)
    output["row_error_note_step15"] = output.apply(row_note, axis=1)
    return output


def first_nonempty(series: pd.Series) -> Any:
    for value in series:
        if normalize_text(value):
            return value
    return series.iloc[0] if len(series) else ""


def enrich_sample_metadata(samples: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = [column for column in META_COLUMNS if column in rows.columns]
    metadata = rows.groupby("sample_key", sort=False).agg({column: first_nonempty for column in metadata_cols}).reset_index()
    output = samples.copy()
    for column in metadata_cols:
        if column in output.columns:
            output = output.drop(columns=[column])
    output = output.merge(metadata, on="sample_key", how="left")
    ensure_columns(output, META_COLUMNS)
    return output


def sample_direction(row: pd.Series) -> str:
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    rel = pd.to_numeric(pd.Series([row.get("zt_pred_vs_obs_mape_step14")]), errors="coerce").iloc[0]
    if not math.isfinite(pred) or not math.isfinite(obs):
        return "not_available"
    if math.isfinite(rel) and rel <= 0.20:
        return "near"
    if pred > obs:
        return "over_predicted"
    if pred < obs:
        return "under_predicted"
    return "near"


def classify_sample(row: pd.Series, threshold: float) -> str:
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    obs_mape = pd.to_numeric(pd.Series([row.get("zt_pred_vs_obs_mape_step14")]), errors="coerce").iloc[0]
    calc_mape = pd.to_numeric(pd.Series([row.get("zt_pred_vs_calc_mape_step14")]), errors="coerce").iloc[0]
    obs_quality = normalize_text(row.get("zt_pred_vs_obs_quality_step14"))
    calc_quality = normalize_text(row.get("zt_pred_vs_calc_quality_step14"))
    n_zt_obs = pd.to_numeric(pd.Series([row.get("n_zt_obs_eval_rows_step14")]), errors="coerce").iloc[0]
    n_zt_calc = pd.to_numeric(pd.Series([row.get("n_zt_calc_eval_rows_step14")]), errors="coerce").iloc[0]

    if math.isfinite(obs) and math.isfinite(pred):
        if obs >= threshold and pred < threshold:
            return "high_zt_false_negative"
        if obs < threshold and pred >= threshold:
            return "high_zt_false_positive"
    if (not math.isfinite(n_zt_obs) or n_zt_obs <= 0) and (not math.isfinite(n_zt_calc) or n_zt_calc <= 0):
        return "missing_eval_data"
    if obs_quality in {"excellent", "good"} or (math.isfinite(obs_mape) and obs_mape <= 0.5):
        return "good_prediction"
    if math.isfinite(obs_mape) and obs_mape > 1.0 or math.isfinite(calc_mape) and calc_mape > 1.0:
        if math.isfinite(calc_mape) and calc_mape > 1.0:
            return "sigma_related_error_likely"
        return "large_error_needs_review"
    if math.isfinite(calc_mape) and calc_mape <= 0.5 and math.isfinite(obs_mape) and obs_mape > 1.0:
        return "zt_obs_inconsistency_likely"
    if calc_quality == "poor":
        return "sigma_related_error_likely"
    if obs_quality == "poor":
        return "large_error_needs_review"
    return "not_evaluated"


def priority_score(row: pd.Series, threshold: float) -> int:
    score = 0
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    obs_mape = pd.to_numeric(pd.Series([row.get("zt_pred_vs_obs_mape_step14")]), errors="coerce").iloc[0]
    calc_mape = pd.to_numeric(pd.Series([row.get("zt_pred_vs_calc_mape_step14")]), errors="coerce").iloc[0]
    n_zt_obs = pd.to_numeric(pd.Series([row.get("n_zt_obs_eval_rows_step14")]), errors="coerce").iloc[0]
    category = normalize_text(row.get("zt_error_analysis_category_step15"))
    if math.isfinite(obs) and obs >= threshold:
        score += 40
    if math.isfinite(pred) and pred >= threshold:
        score += 25
    if category == "high_zt_false_negative":
        score += 40
    if category == "high_zt_false_positive":
        score += 30
    if category in {"large_error_needs_review", "sigma_related_error_likely"}:
        score += 30
    if math.isfinite(obs_mape) and obs_mape > 1.0:
        score += 25
    if math.isfinite(calc_mape) and calc_mape > 1.0:
        score += 20
    if normalize_bool(row.get("nanocarbon_keyword_detected_step9")):
        score += 20
    if normalize_text(row.get("n_or_p")).casefold() == "mixed":
        score += 10
    if not math.isfinite(n_zt_obs) or n_zt_obs <= 2:
        score += 10
    if normalize_bool(row.get("rare_metal_flag_auto_step9")):
        score += 5
    if normalize_bool(row.get("toxicity_flag_auto_step9")):
        score += 5
    if category == "good_prediction":
        score -= 10
    if category in {"not_evaluated", "missing_eval_data"}:
        score -= 10
    return int(max(score, 0))


def priority_tier(score: int) -> str:
    if score >= 80:
        return "A"
    if score >= 50:
        return "B"
    if score >= 25:
        return "C"
    return "low"


def sintering_reason(row: pd.Series, threshold: float) -> str:
    reasons: list[str] = []
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    category = normalize_text(row.get("zt_error_analysis_category_step15"))
    if math.isfinite(obs) and obs >= threshold:
        reasons.append("high observed ZT")
    if category in {"large_error_needs_review", "sigma_related_error_likely"}:
        reasons.append("large ZT prediction error")
    if category == "high_zt_false_negative":
        reasons.append("high-ZT false negative")
    if category == "high_zt_false_positive":
        reasons.append("high-ZT false positive")
    if normalize_text(row.get("needs_manual_review_step15")) == "yes":
        reasons.append("important candidate for paper")
    if math.isfinite(pred) and pred >= threshold:
        reasons.append("high predicted ZT")
    if normalize_text(row.get("needs_sintering_check_later_step14")) == "yes":
        reasons.append("carried from Step14")
    return "; ".join(dict.fromkeys(reasons))


def sample_problem_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if pd.to_numeric(pd.Series([row.get("pf_mape_step14")]), errors="coerce").iloc[0] > 1:
        reasons.append("large PF error")
    if pd.to_numeric(pd.Series([row.get("zt_pred_vs_obs_mape_step14")]), errors="coerce").iloc[0] > 1:
        reasons.append("large ZT error vs observed ZT")
    if pd.to_numeric(pd.Series([row.get("zt_pred_vs_calc_mape_step14")]), errors="coerce").iloc[0] > 1:
        reasons.append("large ZT error vs calculated ZT")
    category = normalize_text(row.get("zt_error_analysis_category_step15"))
    if category == "high_zt_false_negative":
        reasons.append("observed high ZT missed")
    if category == "high_zt_false_positive":
        reasons.append("predicted high ZT false positive")
    if category == "zt_obs_inconsistency_likely":
        reasons.append("ZT observed/calculated inconsistency")
    if pd.to_numeric(pd.Series([row.get("n_zt_obs_eval_rows_step14")]), errors="coerce").iloc[0] <= 0:
        reasons.append("insufficient ZT evaluation rows")
    if category in {"zt_obs_inconsistency_likely", "sigma_related_error_likely"}:
        reasons.append("possible unit or temperature alignment issue")
    if normalize_text(row.get("needs_manual_review_step15")) == "yes":
        reasons.append("manual review required")
    if normalize_text(row.get("needs_sintering_check_later_step15")) == "yes":
        reasons.append("sintering check required later")
    return "; ".join(dict.fromkeys(reasons)) if reasons else "no major issue"


def manual_note(row: pd.Series) -> str:
    category = normalize_text(row.get("zt_error_analysis_category_step15"))
    notes: list[str] = []
    if category in {"large_error_needs_review", "sigma_related_error_likely", "zt_obs_inconsistency_likely"}:
        notes.append("check ZT mismatch and reported ZT curve")
    if normalize_bool(row.get("nanocarbon_keyword_detected_step9")):
        notes.append("check additive/structure information")
    if normalize_text(row.get("n_or_p")).casefold() in {"unknown", "mixed"}:
        notes.append("check n/p statement in paper")
    if normalize_text(row.get("needs_sintering_check_later_step15")) == "yes":
        notes.append("check sintering method later after error review")
    if category == "high_zt_false_negative":
        notes.append("check high-ZT false negative")
    if category == "high_zt_false_positive":
        notes.append("check high-ZT false positive")
    return "; ".join(dict.fromkeys(notes)) if notes else "check key reported values"


def build_error_samples(samples: pd.DataFrame, rows: pd.DataFrame, problem_samples_step14: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = enrich_sample_metadata(samples, rows)
    if "needs_sintering_check_later_step14" not in output.columns and "needs_sintering_check_later_step14" in problem_samples_step14.columns:
        output = output.merge(
            problem_samples_step14[["sample_key", "needs_sintering_check_later_step14"]].drop_duplicates("sample_key"),
            on="sample_key",
            how="left",
        )
    ensure_columns(output, ["needs_sintering_check_later_step14"])
    output["pf_error_level_sample_step15"] = error_level(output["pf_mape_step14"])
    output["zt_obs_error_level_sample_step15"] = error_level(output["zt_pred_vs_obs_mape_step14"])
    output["zt_calc_error_level_sample_step15"] = error_level(output["zt_pred_vs_calc_mape_step14"])
    output["zt_error_direction_sample_step15"] = output.apply(sample_direction, axis=1)
    output["zt_error_analysis_category_step15"] = output.apply(lambda row: classify_sample(row, threshold), axis=1)
    output["zt_error_analysis_note_step15"] = output["zt_error_analysis_category_step15"]
    output["manual_review_priority_score_step15"] = output.apply(lambda row: priority_score(row, threshold), axis=1)
    output["manual_review_priority_tier_step15"] = output["manual_review_priority_score_step15"].map(priority_tier)
    must_review = output["zt_error_analysis_category_step15"].isin(
        ["high_zt_false_negative", "high_zt_false_positive", "large_error_needs_review", "sigma_related_error_likely"]
    )
    output["needs_manual_review_step15"] = np.where(
        output["manual_review_priority_tier_step15"].isin(["A", "B"]) | must_review,
        "yes",
        "no",
    )
    output["sintering_check_reason_step15"] = output.apply(lambda row: sintering_reason(row, threshold), axis=1)
    output["needs_sintering_check_later_step15"] = np.where(
        output["sintering_check_reason_step15"].astype(str).str.len() > 0,
        "yes",
        "no",
    )
    output["pf_zt_problem_reason_step15"] = output.apply(sample_problem_reason, axis=1)
    return output


def classify_high_zt_case(row: pd.Series, threshold: float) -> str:
    obs = pd.to_numeric(pd.Series([row.get("zt_obs_max_step14")]), errors="coerce").iloc[0]
    calc = pd.to_numeric(pd.Series([row.get("zt_calc_from_obs_max_step14")]), errors="coerce").iloc[0]
    pred = pd.to_numeric(pd.Series([row.get("zt_pred_max_step14")]), errors="coerce").iloc[0]
    observed = obs if math.isfinite(obs) else calc
    if not math.isfinite(observed) or not math.isfinite(pred):
        return ""
    if observed >= threshold and pred >= threshold:
        return "true_positive"
    if observed >= threshold and pred < threshold:
        return "false_negative"
    if observed < threshold and pred >= threshold:
        return "false_positive"
    return ""


def compute_classification(samples: pd.DataFrame, threshold: float, source: str) -> dict[str, Any]:
    obs = pd.to_numeric(samples["zt_obs_max_step14"], errors="coerce")
    calc = pd.to_numeric(samples["zt_calc_from_obs_max_step14"], errors="coerce")
    pred = pd.to_numeric(samples["zt_pred_max_step14"], errors="coerce")
    observed = obs.where(np.isfinite(obs), calc)
    available = np.isfinite(observed) & np.isfinite(pred)
    obs_pos = observed >= threshold
    pred_pos = pred >= threshold
    tp = int((available & obs_pos & pred_pos).sum())
    fp = int((available & ~obs_pos & pred_pos).sum())
    fn = int((available & obs_pos & ~pred_pos).sum())
    tn = int((available & ~obs_pos & ~pred_pos).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else math.nan
    recall = tp / (tp + fn) if tp + fn > 0 else math.nan
    f1 = 2 * precision * recall / (precision + recall) if math.isfinite(precision) and math.isfinite(recall) and precision + recall > 0 else math.nan
    accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else math.nan
    specificity = tn / (tn + fp) if tn + fp > 0 else math.nan
    return {
        "evaluation_source_step14": source,
        "threshold": threshold,
        "n_samples": int(available.sum()),
        "n_observed_positive": int((available & obs_pos).sum()),
        "n_predicted_positive": int((available & pred_pos).sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2 if math.isfinite(recall) and math.isfinite(specificity) else math.nan,
        "false_negative_rate": fn / (tp + fn) if tp + fn > 0 else math.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn > 0 else math.nan,
    }


def build_high_zt_classification(samples: pd.DataFrame, step14_classification: pd.DataFrame) -> pd.DataFrame:
    rows = [compute_classification(samples, threshold, "step12_all_fit") for threshold in ZT_THRESHOLDS]
    if not step14_classification.empty:
        for _, row in step14_classification.iterrows():
            if normalize_text(row.get("evaluation_source_step14")) == "step12_all_fit":
                continue
            threshold = pd.to_numeric(pd.Series([row.get("threshold")]), errors="coerce").iloc[0]
            if math.isfinite(threshold):
                rows.append(
                    {
                        **{column: row.get(column, "") for column in step14_classification.columns},
                        "balanced_accuracy": "",
                        "false_negative_rate": "",
                        "false_positive_rate": "",
                    }
                )
    output = pd.DataFrame(rows)
    for column in ["true_positive", "false_positive", "false_negative", "true_negative"]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def build_missed_false_positive(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = samples.copy()
    output["classification_case_step15"] = output.apply(lambda row: classify_high_zt_case(row, threshold), axis=1)
    output = output[output["classification_case_step15"].isin(["true_positive", "false_negative", "false_positive"])].copy()
    output["threshold_step15"] = threshold
    columns = [
        "sample_key",
        "classification_case_step15",
        "threshold_step15",
        "material_system",
        "n_or_p",
        "composition",
        "DOI",
        "doi_url",
        "paper_title",
        "zt_obs_max_step14",
        "zt_pred_max_step14",
        "zt_calc_from_obs_max_step14",
        "zt_pred_vs_obs_mape_step14",
        "zt_pred_vs_calc_mape_step14",
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "manual_review_priority_score_step15",
        "needs_manual_review_step15",
        "needs_sintering_check_later_step15",
        "sintering_check_reason_step15",
    ]
    return select_columns(output, columns)


def select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    ensure_columns(df, columns)
    return df.loc[:, columns].copy()


def build_best_candidates(samples: pd.DataFrame, top_n: int, threshold: float) -> pd.DataFrame:
    obs = pd.to_numeric(samples["zt_obs_max_step14"], errors="coerce")
    pred = pd.to_numeric(samples["zt_pred_max_step14"], errors="coerce")
    obs_mape = pd.to_numeric(samples["zt_pred_vs_obs_mape_step14"], errors="coerce")
    enough_eval = pd.to_numeric(samples["n_zt_obs_eval_rows_step14"], errors="coerce").fillna(0) > 0
    candidates = samples[((obs >= threshold) | (pred >= threshold)) & enough_eval].copy()
    candidates["_sort_obs"] = pd.to_numeric(candidates["zt_obs_max_step14"], errors="coerce").fillna(-np.inf)
    candidates["_sort_pred"] = pd.to_numeric(candidates["zt_pred_max_step14"], errors="coerce").fillna(-np.inf)
    candidates["_sort_mape"] = pd.to_numeric(candidates["zt_pred_vs_obs_mape_step14"], errors="coerce").fillna(np.inf)
    candidates = candidates.sort_values(
        ["_sort_obs", "_sort_pred", "_sort_mape", "manual_review_priority_score_step15"],
        ascending=[False, False, True, False],
    ).head(top_n)
    columns = [
        "sample_key",
        "material_system",
        "n_or_p",
        "composition",
        "DOI",
        "doi_url",
        "paper_title",
        "zt_obs_max_step14",
        "zt_pred_max_step14",
        "zt_calc_from_obs_max_step14",
        "pf_mape_step14",
        "zt_pred_vs_obs_mape_step14",
        "zt_pred_vs_calc_mape_step14",
        "tau_eff_step12",
        "nanocarbon_keyword_detected_step9",
        "nanocarbon_type_auto_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "additive_auto_step9",
        "additive_manual_step9",
        "structure_auto_step9",
        "structure_manual_step9",
        "needs_manual_review_step15",
        "needs_sintering_check_later_step15",
    ]
    return select_columns(candidates, columns)


def build_manual_review(samples: pd.DataFrame, high_zt_cases: pd.DataFrame, top_n: int) -> pd.DataFrame:
    case_map = high_zt_cases.set_index("sample_key")["classification_case_step15"].to_dict() if not high_zt_cases.empty else {}
    output = samples[
        samples["needs_manual_review_step15"].eq("yes")
        | samples["manual_review_priority_tier_step15"].isin(["A", "B"])
    ].copy()
    output["classification_case_step15"] = output["sample_key"].map(case_map).fillna("")
    output["manual_review_note_step15"] = output.apply(manual_note, axis=1)
    output = output.sort_values("manual_review_priority_score_step15", ascending=False).head(top_n)
    columns = [
        "sample_key",
        "manual_review_priority_score_step15",
        "manual_review_priority_tier_step15",
        "zt_error_analysis_category_step15",
        "material_system",
        "n_or_p",
        "composition",
        "DOI",
        "doi_url",
        "paper_title",
        "zt_obs_max_step14",
        "zt_pred_max_step14",
        "zt_pred_vs_obs_mape_step14",
        "zt_pred_vs_calc_mape_step14",
        "classification_case_step15",
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "additive_auto_step9",
        "additive_manual_step9",
        "structure_auto_step9",
        "structure_manual_step9",
        "needs_sintering_check_later_step15",
        "sintering_check_reason_step15",
        "manual_review_note_step15",
    ]
    return select_columns(output, columns)


def build_validation_errors(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame(
            columns=[
                "sample_key",
                "material_system",
                "n_or_p",
                "composition",
                "validation_method_step13",
                "n_validation_rows_step14",
                "validation_pf_mape_step14",
                "validation_zt_pred_vs_obs_mape_step14",
                "validation_zt_pred_vs_calc_mape_step14",
                "validation_zt_quality_step14",
                "validation_error_level_step15",
                "validation_error_category_step15",
                "needs_manual_review_step15",
            ]
        )
    output = validation.copy()
    output["validation_error_level_step15"] = error_level(output["validation_zt_pred_vs_obs_mape_step14"])
    output["validation_error_category_step15"] = np.where(
        output["validation_error_level_step15"].isin(["excellent", "good"]),
        "good_prediction",
        np.where(output["validation_error_level_step15"].eq("poor"), "large_error_needs_review", "review"),
    )
    output["needs_manual_review_step15"] = np.where(
        output["validation_error_category_step15"].eq("large_error_needs_review"), "yes", "no"
    )
    columns = [
        "sample_key",
        "material_system",
        "n_or_p",
        "composition",
        "validation_method_step13",
        "n_validation_rows_step14",
        "validation_pf_mape_step14",
        "validation_zt_pred_vs_obs_mape_step14",
        "validation_zt_pred_vs_calc_mape_step14",
        "validation_zt_quality_step14",
        "validation_error_level_step15",
        "validation_error_category_step15",
        "needs_manual_review_step15",
    ]
    return select_columns(output, columns)


def median_numeric(values: pd.Series) -> float:
    return pd.to_numeric(values, errors="coerce").median()


def build_summary_by_material(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return build_group_summary(samples, ["material_system", "n_or_p"], threshold)


def build_summary_by_np(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return build_group_summary(samples, ["n_or_p"], threshold)


def build_group_summary(samples: pd.DataFrame, group_cols: list[str], threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_values, group in samples.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = {column: value for column, value in zip(group_cols, group_values)}
        obs = pd.to_numeric(group["zt_obs_max_step14"], errors="coerce")
        pred = pd.to_numeric(group["zt_pred_max_step14"], errors="coerce")
        row.update(
            {
                "sample_count": len(group),
                "zt_eval_sample_count": int((pd.to_numeric(group["n_zt_obs_eval_rows_step14"], errors="coerce") > 0).sum()),
                "median_pf_mape_step15": median_numeric(group["pf_mape_step14"]),
                "median_zt_pred_vs_obs_mape_step15": median_numeric(group["zt_pred_vs_obs_mape_step14"]),
                "median_zt_pred_vs_calc_mape_step15": median_numeric(group["zt_pred_vs_calc_mape_step14"]),
                "median_zt_pred_vs_obs_log_rmse_step15": median_numeric(group["zt_pred_vs_obs_log_rmse_step14"]),
                "median_zt_pred_vs_calc_log_rmse_step15": median_numeric(group["zt_pred_vs_calc_log_rmse_step14"]),
                "good_prediction_sample_count": int(group["zt_error_analysis_category_step15"].eq("good_prediction").sum()),
                "large_error_sample_count": int(group["zt_error_analysis_category_step15"].isin(["large_error_needs_review", "sigma_related_error_likely"]).sum()),
                "high_zt_observed_sample_count": int((obs >= threshold).sum()),
                "high_zt_predicted_sample_count": int((pred >= threshold).sum()),
                "high_zt_true_positive_count": int(((obs >= threshold) & (pred >= threshold)).sum()),
                "high_zt_false_negative_count": int(((obs >= threshold) & (pred < threshold)).sum()),
                "high_zt_false_positive_count": int(((obs < threshold) & (pred >= threshold)).sum()),
                "manual_review_sample_count": int(group["needs_manual_review_step15"].eq("yes").sum()),
                "sintering_check_sample_count": int(group["needs_sintering_check_later_step15"].eq("yes").sum()),
            }
        )
        if "material_system" in group_cols:
            row.update(
                {
                    "nanocarbon_sample_count": int(group["nanocarbon_keyword_detected_step9"].map(normalize_bool).sum()),
                    "rare_metal_flag_sample_count": int(group["rare_metal_flag_auto_step9"].map(normalize_bool).sum()),
                    "toxicity_flag_sample_count": int(group["toxicity_flag_auto_step9"].map(normalize_bool).sum()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_summary(samples: pd.DataFrame, threshold: float) -> pd.DataFrame:
    feature_columns = [
        "nanocarbon_keyword_detected_step9",
        "rare_metal_flag_auto_step9",
        "toxicity_flag_auto_step9",
        "n_or_p",
        "fitting_source_actual_step10",
        "tau_eff_mode_step12",
    ]
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        ensure_columns(samples, [feature])
        for value, group in samples.groupby(feature, dropna=False):
            obs = pd.to_numeric(group["zt_obs_max_step14"], errors="coerce")
            pred = pd.to_numeric(group["zt_pred_max_step14"], errors="coerce")
            rows.append(
                {
                    "feature_name_step15": feature,
                    "feature_value_step15": value,
                    "sample_count": len(group),
                    "zt_eval_sample_count": int((pd.to_numeric(group["n_zt_obs_eval_rows_step14"], errors="coerce") > 0).sum()),
                    "median_zt_pred_vs_obs_mape_step15": median_numeric(group["zt_pred_vs_obs_mape_step14"]),
                    "median_zt_pred_vs_calc_mape_step15": median_numeric(group["zt_pred_vs_calc_mape_step14"]),
                    "median_pf_mape_step15": median_numeric(group["pf_mape_step14"]),
                    "high_zt_observed_sample_count": int((obs >= threshold).sum()),
                    "high_zt_predicted_sample_count": int((pred >= threshold).sum()),
                    "large_error_sample_count": int(group["zt_error_analysis_category_step15"].isin(["large_error_needs_review", "sigma_related_error_likely"]).sum()),
                    "manual_review_sample_count": int(group["needs_manual_review_step15"].eq("yes").sum()),
                }
            )
    return pd.DataFrame(rows)


def value_counts_rows(prefix: str, series: pd.Series) -> list[tuple[str, str]]:
    return [(f"{prefix}_{key}_count", str(int(value))) for key, value in series.fillna("").astype(str).value_counts().sort_index().items()]


def build_report(
    input_counts: dict[str, int],
    outputs: dict[str, pd.DataFrame],
    threshold: float,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    samples = outputs["samples"]
    rows: list[tuple[str, str]] = [
        ("input_thermoelectric_predictions_step14_rows", str(input_counts["thermoelectric_predictions"])),
        ("input_pf_zt_sample_results_step14_sample_count", str(input_counts["sample_results"])),
        ("input_validation_sample_results_step14_sample_count", str(input_counts.get("validation_sample_results", 0))),
        ("pf_zt_error_rows_step15_rows", str(len(outputs["rows"]))),
        ("pf_zt_error_samples_step15_sample_count", str(len(samples))),
        ("pf_zt_validation_error_samples_step15_sample_count", str(len(outputs["validation"]))),
        ("pf_zt_error_by_material_step15_rows", str(len(outputs["by_material"]))),
        ("manual_review_candidates_step15_sample_count", str(len(outputs["manual"]))),
        ("sintering_check_candidates_step15_sample_count", str(len(outputs["sintering"]))),
    ]
    rows.extend(value_counts_rows("pf_error_level_sample_step15", samples["pf_error_level_sample_step15"]))
    rows.extend(value_counts_rows("zt_obs_error_level_sample_step15", samples["zt_obs_error_level_sample_step15"]))
    rows.extend(value_counts_rows("zt_calc_error_level_sample_step15", samples["zt_calc_error_level_sample_step15"]))
    rows.extend(value_counts_rows("zt_error_analysis_category_step15", samples["zt_error_analysis_category_step15"]))
    classification = outputs["classification"]
    for _, row in classification[classification["evaluation_source_step14"].eq("step12_all_fit")].iterrows():
        t = row["threshold"]
        for metric in ["precision", "recall", "f1", "accuracy", "false_positive", "false_negative"]:
            rows.append((f"zt_ge_{t}_{metric}", str(row.get(metric, ""))))
    zt1 = classification[
        classification["evaluation_source_step14"].eq("step12_all_fit")
        & np.isclose(pd.to_numeric(classification["threshold"], errors="coerce"), threshold)
    ]
    if not zt1.empty:
        for metric in ["precision", "recall", "f1", "accuracy", "true_positive", "false_positive", "false_negative", "true_negative"]:
            rows.append((f"zt_ge_{threshold}_{metric}", str(zt1.iloc[0].get(metric, ""))))
    rows.extend(value_counts_rows("manual_review_priority_tier_step15", samples["manual_review_priority_tier_step15"]))
    rows.append(("needs_manual_review_step15_yes_sample_count", str(int(samples["needs_manual_review_step15"].eq("yes").sum()))))
    rows.append(("needs_sintering_check_later_step15_yes_sample_count", str(int(samples["needs_sintering_check_later_step15"].eq("yes").sum()))))
    rows.extend(value_counts_rows("sintering_check_reason_step15", samples["sintering_check_reason_step15"]))
    rows.append(("nanocarbon_keyword_detected_step9_true_sample_count", str(int(samples["nanocarbon_keyword_detected_step9"].map(normalize_bool).sum()))))
    rows.append(("rare_metal_flag_auto_step9_true_sample_count", str(int(samples["rare_metal_flag_auto_step9"].map(normalize_bool).sum()))))
    rows.append(("toxicity_flag_auto_step9_true_sample_count", str(int(samples["toxicity_flag_auto_step9"].map(normalize_bool).sum()))))
    rows.extend(group_median_rows(samples, "n_or_p", "zt_pred_vs_obs_mape_step14", "n_or_p_median_zt_vs_obs_mape"))
    rows.extend(group_median_rows(samples, "n_or_p", "zt_pred_vs_calc_mape_step14", "n_or_p_median_zt_vs_calc_mape"))
    material = outputs["by_material"].copy()
    for _, row in material.sort_values("median_zt_pred_vs_obs_mape_step15", ascending=True).head(20).iterrows():
        rows.append((f"material_system_{row['material_system']}_median_zt_vs_obs_mape", str(row["median_zt_pred_vs_obs_mape_step15"])))
    for _, row in material.sort_values("large_error_sample_count", ascending=False).head(20).iterrows():
        rows.append((f"material_system_{row['material_system']}_large_error_sample_count", str(row["large_error_sample_count"])))
    rows.extend(
        [
            ("sintering_method_unknown_rows", str(int(samples["sintering_method"].astype(str).str.casefold().eq("unknown").sum()))),
            ("sintering_checked_no_rows", str(int(samples["sintering_checked"].astype(str).str.casefold().eq("no").sum()))),
            ("record_checked_no_rows", str(int(samples["record_checked"].astype(str).str.casefold().eq("no").sum()))),
            ("n_p_changed_rows", "0"),
            ("sintering_changed_rows", str(sintering_changed_rows(samples))),
            ("note", "Step15 did not create new predictions or refit tau_eff."),
            ("note", "Step15 analyzed Step14 PF/ZT prediction results."),
            ("note", "Step14/15 ZT_pred uses predicted sigma_pred and observed S_obs/kappa_obs."),
            ("note", "Seebeck coefficient and thermal conductivity are not predicted yet."),
            ("note", "Sintering method has not been investigated; Step15 only extracts later-check candidates."),
        ]
    )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    return "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n", report_df


def group_median_rows(df: pd.DataFrame, group_col: str, value_col: str, prefix: str) -> list[tuple[str, str]]:
    rows = []
    for key, group in df.groupby(group_col, dropna=False):
        rows.append((f"{prefix}_{key}", str(median_numeric(group[value_col]))))
    return rows


def sintering_changed_rows(df: pd.DataFrame) -> int:
    return int(
        (
            ~df["sintering_method"].astype(str).str.casefold().eq("unknown")
            | ~df["sintering_checked"].astype(str).str.casefold().eq("no")
            | ~df["record_checked"].astype(str).str.casefold().eq("no")
        ).sum()
    )


def build_notes() -> str:
    return """# Step15 PF/ZT Error Analysis Notes

## Purpose

Step15 analyzes the PF and ZT prediction errors produced in Step14. It does not train a new model and does not refit tau_eff.

## Assumptions Up To Step14

The electrical conductivity prediction comes from fitted tau_eff. The Seebeck coefficient and thermal conductivity are not predicted in Step14 or Step15.

## Equations

PF_pred = S_obs^2 * sigma_pred

ZT_pred = S_obs^2 * sigma_pred * T / kappa_obs

Here, S_obs and kappa_obs are experimental values. Only sigma is predicted.

## Interpretation

ZT_pred vs ZT_calc_from_obs mainly shows the effect of sigma prediction error because both use standardized observed S and kappa.

ZT_pred vs ZT_obs compares against the Starrydata/literature ZT value and can include effects from reported ZT values, unit handling, temperature alignment, or data consistency.

Large errors can come from sigma prediction error, inconsistency in observed ZT, temperature matching, units, additives/structure information, or unconfirmed sintering method.

## Sintering

Sintering methods are still not investigated in Step15. Use `sintering_check_candidates_step15.csv` to check only important samples in later steps.
"""


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    csv_frame(df).to_csv(path, index=False)


def add_excel_preview_note(sheet_name: str, row_count: int, notes: list[str]) -> None:
    if row_count > EXCEL_PREVIEW_ROWS:
        notes.append(f"{sheet_name} has {row_count} rows; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook; full data is in CSV")


def fit_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    preview = df.head(200)
    for index, column in enumerate(df.columns, start=1):
        max_length = len(str(column))
        if not preview.empty:
            max_length = max(max_length, int(preview[column].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(max(max_length + 2, 12), 60)


def write_excel(output_dir: Path, report_df: pd.DataFrame) -> None:
    sheets = {
        "pf_zt_error_samples": "pf_zt_error_samples_step15.csv",
        "pf_zt_validation_error_samples": "pf_zt_validation_error_samples_step15.csv",
        "error_by_material": "pf_zt_error_by_material_step15.csv",
        "error_by_np_type": "pf_zt_error_by_np_type_step15.csv",
        "error_by_feature_flags": "pf_zt_error_by_feature_flags_step15.csv",
        "high_zt_classification": "high_zt_classification_analysis_step15.csv",
        "missed_false_positive_samples": "high_zt_missed_and_false_positive_samples_step15.csv",
        "best_candidate_samples": "best_candidate_samples_step15.csv",
        "manual_review_candidates": "manual_review_candidates_step15.csv",
        "sintering_check_candidates": "sintering_check_candidates_step15.csv",
    }
    with pd.ExcelWriter(output_dir / "starrydata2_step15_pf_zt_error_analysis.xlsx", engine="openpyxl") as writer:
        for sheet, filename in sheets.items():
            df = pd.read_csv(output_dir / filename, dtype=str, keep_default_na=False, nrows=EXCEL_PREVIEW_ROWS)
            df.to_excel(writer, sheet_name=sheet, index=False)
            fit_worksheet(writer, sheet, df)
        report_df.to_excel(writer, sheet_name="error_analysis_report", index=False)
        fit_worksheet(writer, "error_analysis_report", report_df)


def assert_acceptance(outputs: dict[str, pd.DataFrame]) -> None:
    rows = outputs["rows"]
    samples = outputs["samples"]
    if rows.duplicated(["sample_key", "temperature_K"]).any():
        raise ValueError("pf_zt_error_rows_step15 is not one row per sample-temperature")
    for column in ["pf_error_level_step15", "zt_obs_error_level_step15", "zt_error_direction_vs_obs_step15"]:
        if column not in rows.columns:
            raise KeyError(f"pf_zt_error_rows_step15 missing {column}")
    if samples["sample_key"].duplicated().any():
        raise ValueError("pf_zt_error_samples_step15 is not one row per sample")
    for column in [
        "zt_error_analysis_category_step15",
        "manual_review_priority_score_step15",
        "needs_manual_review_step15",
        "needs_sintering_check_later_step15",
    ]:
        if column not in samples.columns:
            raise KeyError(f"pf_zt_error_samples_step15 missing {column}")
    for threshold in ZT_THRESHOLDS:
        if not np.isclose(pd.to_numeric(outputs["classification"]["threshold"], errors="coerce"), threshold).any():
            raise ValueError(f"classification missing threshold {threshold}")
    for column in ["precision", "recall", "f1", "accuracy"]:
        if column not in outputs["classification"].columns:
            raise KeyError(f"classification missing {column}")
    for df_name in ["manual", "sintering"]:
        if outputs[df_name].empty:
            continue
        if "sintering_method" in outputs[df_name].columns and not outputs[df_name]["sintering_method"].astype(str).str.casefold().eq("unknown").all():
            raise ValueError(f"{df_name} changed sintering_method")
    if sintering_changed_rows(samples) != 0:
        raise ValueError("sintering values changed in sample outputs")


def main() -> None:
    args = parse_args()
    paths = input_paths(args.step14_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_counts = {label: count_csv_rows(path) for label, path in paths.items()}

    row_columns = sorted(
        set(
            ROW_REQUIRED
            + META_COLUMNS
            + [
                "sigma_obs_S_per_m_step11",
                "seebeck_obs_V_per_K_step11",
                "kappa_obs_W_per_mK_step11",
                "zt_obs_dimensionless_step11",
                "zt_calc_from_obs_step11",
                "power_factor_obs_W_per_mK2_step14",
                "pf_relative_error_step14",
                "pf_log_error_step14",
                "zt_pred_vs_obs_relative_error_step14",
                "zt_pred_vs_obs_log_error_step14",
                "zt_pred_vs_calc_relative_error_step14",
                "zt_pred_vs_calc_log_error_step14",
                "zt_pred_vs_obs_status_step14",
                "zt_pred_vs_calc_status_step14",
                "prediction_source_step14",
            ]
        )
    )
    rows_raw = read_csv_selected(paths["thermoelectric_predictions"], row_columns)
    validate_required(rows_raw, ROW_REQUIRED, "thermoelectric_predictions_step14.csv")
    error_rows = build_error_rows(rows_raw)

    sample_columns = sorted(
        set(
            SAMPLE_REQUIRED
            + META_COLUMNS
            + [
                "n_rows_step14",
                "n_pf_eval_rows_step14",
                "n_zt_calc_eval_rows_step14",
                "n_zt_obs_eval_rows_step14",
                "pf_mape_step14",
                "pf_log_rmse_step14",
                "zt_pred_vs_obs_mape_step14",
                "zt_pred_vs_obs_log_rmse_step14",
                "zt_pred_vs_calc_mape_step14",
                "zt_pred_vs_calc_log_rmse_step14",
                "zt_obs_max_step14",
                "zt_pred_max_step14",
                "zt_calc_from_obs_max_step14",
                "zt_pred_vs_obs_quality_step14",
                "zt_pred_vs_calc_quality_step14",
            ]
        )
    )
    sample_raw = read_csv_selected(paths["sample_results"], sample_columns)
    validate_required(sample_raw, SAMPLE_REQUIRED, "pf_zt_sample_results_step14.csv")
    problem_step14 = read_csv_selected(paths["problem_samples"], ["sample_key", "needs_sintering_check_later_step14"])
    error_samples = build_error_samples(sample_raw, rows_raw, problem_step14, args.zt_threshold)

    validation_samples = pd.DataFrame()
    if "validation_sample_results" in paths:
        validation_samples = read_csv_selected(paths["validation_sample_results"])
    validation_error_samples = build_validation_errors(validation_samples)
    by_material = build_summary_by_material(error_samples, args.zt_threshold)
    by_np = build_summary_by_np(error_samples, args.zt_threshold)
    by_feature = build_feature_summary(error_samples, args.zt_threshold)
    classification_step14 = read_csv_selected(paths["classification"])
    high_classification = build_high_zt_classification(error_samples, classification_step14)
    high_cases = build_missed_false_positive(error_samples, args.zt_threshold)
    best_candidates = build_best_candidates(error_samples, args.top_n_best_candidates, args.zt_threshold)
    manual_review = build_manual_review(error_samples, high_cases, args.top_n_manual_review)
    sintering_candidates = error_samples[error_samples["needs_sintering_check_later_step15"].eq("yes")].copy()
    sintering_candidates = sintering_candidates.sort_values("manual_review_priority_score_step15", ascending=False)

    outputs = {
        "rows": error_rows,
        "samples": error_samples,
        "validation": validation_error_samples,
        "by_material": by_material,
        "by_np": by_np,
        "by_feature": by_feature,
        "classification": high_classification,
        "high_cases": high_cases,
        "best": best_candidates,
        "manual": manual_review,
        "sintering": sintering_candidates,
    }
    assert_acceptance(outputs)

    write_csv(error_rows, args.output_dir / "pf_zt_error_rows_step15.csv")
    write_csv(error_samples, args.output_dir / "pf_zt_error_samples_step15.csv")
    write_csv(validation_error_samples, args.output_dir / "pf_zt_validation_error_samples_step15.csv")
    write_csv(by_material, args.output_dir / "pf_zt_error_by_material_step15.csv")
    write_csv(by_np, args.output_dir / "pf_zt_error_by_np_type_step15.csv")
    write_csv(by_feature, args.output_dir / "pf_zt_error_by_feature_flags_step15.csv")
    write_csv(high_classification, args.output_dir / "high_zt_classification_analysis_step15.csv")
    write_csv(high_cases, args.output_dir / "high_zt_missed_and_false_positive_samples_step15.csv")
    write_csv(best_candidates, args.output_dir / "best_candidate_samples_step15.csv")
    write_csv(manual_review, args.output_dir / "manual_review_candidates_step15.csv")
    write_csv(sintering_candidates, args.output_dir / "sintering_check_candidates_step15.csv")
    (args.output_dir / "step15_error_analysis_notes.md").write_text(build_notes(), encoding="utf-8")

    excel_notes: list[str] = []
    for name, df in [
        ("pf_zt_error_samples", error_samples),
        ("pf_zt_validation_error_samples", validation_error_samples),
        ("error_by_material", by_material),
        ("error_by_np_type", by_np),
        ("error_by_feature_flags", by_feature),
        ("high_zt_classification", high_classification),
        ("missed_false_positive_samples", high_cases),
        ("best_candidate_samples", best_candidates),
        ("manual_review_candidates", manual_review),
        ("sintering_check_candidates", sintering_candidates),
    ]:
        add_excel_preview_note(name, len(df), excel_notes)
    report_text, report_df = build_report(input_counts, outputs, args.zt_threshold, excel_notes)
    (args.output_dir / "step15_error_analysis_report.txt").write_text(report_text, encoding="utf-8")
    write_excel(args.output_dir, report_df)

    zt1 = high_classification[
        high_classification["evaluation_source_step14"].eq("step12_all_fit")
        & np.isclose(pd.to_numeric(high_classification["threshold"], errors="coerce"), args.zt_threshold)
    ]
    precision = float(zt1.iloc[0]["precision"]) if not zt1.empty else math.nan
    recall = float(zt1.iloc[0]["recall"]) if not zt1.empty else math.nan
    f1 = float(zt1.iloc[0]["f1"]) if not zt1.empty else math.nan

    def level_summary(column: str) -> str:
        counts = error_samples[column].value_counts()
        return "/".join(str(int(counts.get(level, 0))) for level in ["excellent", "good", "moderate", "poor"])

    print("Done.")
    print("Created:")
    print("- pf_zt_error_rows_step15.csv")
    print("- pf_zt_error_samples_step15.csv")
    print("- pf_zt_validation_error_samples_step15.csv")
    print("- pf_zt_error_by_material_step15.csv")
    print("- pf_zt_error_by_np_type_step15.csv")
    print("- pf_zt_error_by_feature_flags_step15.csv")
    print("- high_zt_classification_analysis_step15.csv")
    print("- high_zt_missed_and_false_positive_samples_step15.csv")
    print("- best_candidate_samples_step15.csv")
    print("- manual_review_candidates_step15.csv")
    print("- sintering_check_candidates_step15.csv")
    print("- step15_error_analysis_report.txt")
    print("- step15_error_analysis_notes.md")
    print("- starrydata2_step15_pf_zt_error_analysis.xlsx")
    print("")
    print("Summary:")
    print(f"sample results: {len(error_samples)}")
    print(f"PF excellent/good/moderate/poor samples: {level_summary('pf_error_level_sample_step15')}")
    print(f"ZT vs obs excellent/good/moderate/poor samples: {level_summary('zt_obs_error_level_sample_step15')}")
    print(f"ZT vs calc excellent/good/moderate/poor samples: {level_summary('zt_calc_error_level_sample_step15')}")
    print(f"ZT>=1 precision: {precision}")
    print(f"ZT>=1 recall: {recall}")
    print(f"ZT>=1 F1: {f1}")
    print(f"manual review candidates: {len(manual_review)}")
    print(f"sintering check candidates: {len(sintering_candidates)}")
    print(f"best candidate samples: {len(best_candidates)}")
    print(f"problem samples: {int(error_samples['needs_manual_review_step15'].eq('yes').sum())}")
    print("n/p changed rows: 0")
    print(f"sintering changed rows: {sintering_changed_rows(error_samples)}")


if __name__ == "__main__":
    main()
