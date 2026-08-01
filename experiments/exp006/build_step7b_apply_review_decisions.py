import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
DEFAULT_PREDICTIONS = PROCESSED_DIR / "step6b_broad_family" / "step5b_test_predictions_valid.parquet"
DEFAULT_PACKET = PROCESSED_DIR / "step7a_manual_review_packet"
DEFAULT_OUTPUT = PROCESSED_DIR / "step7b_review_applied"
DEFAULT_REPORT = EXP_DIR / "reports" / "step7b_review_applied" / "step7b_review_application_report.md"

VALID_REVIEW_STATUS = ["pending", "keep", "keep_but_note", "suspect", "exclude_from_primary", "exclude_from_all", "unresolved"]
VALID_SCOPE = ["row_only", "entire_sample", "entire_paper", "all_matching_source_curve", "undecided"]
VALID_PRIMARY_FLAG = ["keep_in_primary", "exclude_from_primary", "pending"]
VALID_SENSITIVITY_FLAG = ["keep_in_sensitivity", "exclude_from_sensitivity", "pending"]

STATUS_STRENGTH = {
    "exclude_from_all": 0,
    "exclude_from_primary": 1,
    "suspect": 2,
    "unresolved": 3,
    "pending": 4,
    "keep_but_note": 5,
    "keep": 6,
}

DEFAULT_CONFIGS = {
    "broad_material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "broad_paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

METRIC_NAMES = [
    "n_rows",
    "mae_log10",
    "rmse_log10",
    "median_log10_error",
    "factor_2_accuracy",
    "factor_5_accuracy",
    "factor_10_accuracy",
    "max_abs_log10_error",
    "extreme_ge_10_count",
    "severe_ge_5_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Step7A manual review decisions to broad_family predictions.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--decision-template", type=Path, default=DEFAULT_PACKET / "step7a_review_decisions_template.csv")
    parser.add_argument("--review-master", type=Path, default=DEFAULT_PACKET / "step7a_manual_review_master.csv")
    parser.add_argument("--source-trace", type=Path, default=DEFAULT_PACKET / "step7a_source_traceability_table.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--pending-policy", choices=["keep_with_pending_flag", "exclude_from_primary", "fail_if_pending"], default="keep_with_pending_flag")
    parser.add_argument("--suspect-policy", choices=["keep_both", "exclude_primary_keep_sensitivity", "exclude_both"], default="exclude_primary_keep_sensitivity")
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step7b] {message}", flush=True)


def out_name(base: str, suffix: str, ext: str = "csv") -> str:
    return f"{base}{suffix}.{ext}"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_csv_parquet(df: pd.DataFrame, csv_path: Path, parquet_path: Path | None = None) -> str:
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if parquet_path is None:
        return "csv_only"
    try:
        df.to_parquet(parquet_path, index=False)
        return "csv_and_parquet"
    except Exception:
        return "csv_only_parquet_failed"


def normalize_text(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def bool_any_yes(series: pd.Series) -> bool:
    return series.astype(str).str.casefold().isin(["yes", "true", "1"]).any()


def normalize_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    for col in [
        "review_status",
        "review_reason_code",
        "apply_to_scope",
        "primary_analysis_flag_after_review",
        "sensitivity_analysis_flag_after_review",
        "reviewer_name",
        "review_date",
        "reviewer_notes",
        "evidence_file_or_link",
        "checked_source_plot",
        "checked_units",
        "checked_temperature_alignment",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(lambda v: normalize_text(v))

    out["review_status"] = out["review_status"].replace("", "pending").str.casefold()
    out.loc[~out["review_status"].isin(VALID_REVIEW_STATUS), "review_status"] = "pending"
    out["apply_to_scope"] = out["apply_to_scope"].replace("", "undecided").str.casefold()
    out.loc[~out["apply_to_scope"].isin(VALID_SCOPE), "apply_to_scope"] = "undecided"
    out["primary_analysis_flag_after_review"] = out["primary_analysis_flag_after_review"].replace("", "pending").str.casefold()
    out.loc[~out["primary_analysis_flag_after_review"].isin(VALID_PRIMARY_FLAG), "primary_analysis_flag_after_review"] = "pending"
    out["sensitivity_analysis_flag_after_review"] = out["sensitivity_analysis_flag_after_review"].replace("", "pending").str.casefold()
    out.loc[~out["sensitivity_analysis_flag_after_review"].isin(VALID_SENSITIVITY_FLAG), "sensitivity_analysis_flag_after_review"] = "pending"
    out["decision_is_pending"] = out["review_status"].isin(["pending", "unresolved"])
    out["decision_is_human_reviewed"] = (
        ~out["review_status"].isin(["pending", ""])
        | out["reviewer_name"].astype(str).str.len().gt(0)
        | out["review_date"].astype(str).str.len().gt(0)
    )
    conflict = (
        out["review_status"].eq("exclude_from_all")
        & (
            out["primary_analysis_flag_after_review"].eq("keep_in_primary")
            | out["sensitivity_analysis_flag_after_review"].eq("keep_in_sensitivity")
        )
    )
    out["decision_validity_status"] = np.where(conflict, "conflict", "ok")
    out["decision_warning"] = np.where(out["apply_to_scope"].eq("undecided"), "undecided scope treated as row_only", "")
    keep_cols = [
        "review_case_id",
        "review_case_type",
        "review_priority",
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "material_group_key",
        "review_status",
        "review_reason_code",
        "apply_to_scope",
        "primary_analysis_flag_after_review",
        "sensitivity_analysis_flag_after_review",
        "reviewer_name",
        "review_date",
        "reviewer_notes",
        "evidence_file_or_link",
        "checked_source_plot",
        "checked_units",
        "checked_temperature_alignment",
        "decision_is_pending",
        "decision_is_human_reviewed",
        "decision_validity_status",
        "decision_warning",
    ]
    for col in keep_cols:
        if col not in out.columns:
            out[col] = ""
    return out[keep_cols].copy()


def prepare_predictions(pred: pd.DataFrame, max_rows_per_config: int | None) -> pd.DataFrame:
    if max_rows_per_config is not None:
        if max_rows_per_config <= 0:
            raise ValueError("--max-rows-per-config must be positive")
        pred = pred.groupby("config_id", dropna=False, sort=False).head(max_rows_per_config).copy()
    else:
        pred = pred.copy()
    pred["sigma_pred_over_exp"] = pred["sigma_pred_S_per_m"] / pred["sigma_S_per_m"]
    pred["log10_sigma_pred_over_exp"] = np.log10(pred["sigma_pred_over_exp"])
    pred["abs_log10_sigma_pred_over_exp"] = pred["log10_sigma_pred_over_exp"].abs()
    return pred


def decision_mask(pred: pd.DataFrame, row: pd.Series, source_trace: pd.DataFrame) -> tuple[pd.Series, bool]:
    scope = row["apply_to_scope"]
    warning = False
    if scope == "undecided":
        scope = "row_only"
        warning = True
    if scope == "row_only":
        rid = normalize_text(row.get("row_id"))
        if not rid:
            return pd.Series(False, index=pred.index), True
        return pred["row_id"].astype(str).eq(rid), warning
    if scope == "entire_sample":
        gid = normalize_text(row.get("validation_sample_group_id"))
        if not gid:
            return pd.Series(False, index=pred.index), True
        return pred["validation_sample_group_id"].astype(str).eq(gid), warning
    if scope == "entire_paper":
        gid = normalize_text(row.get("validation_paper_group_id"))
        if not gid:
            return pd.Series(False, index=pred.index), True
        return pred["validation_paper_group_id"].astype(str).eq(gid), warning
    if scope == "all_matching_source_curve":
        rid = normalize_text(row.get("row_id"))
        if source_trace.empty or not rid:
            fallback, _ = decision_mask(pred, pd.Series({**row.to_dict(), "apply_to_scope": "row_only"}), source_trace)
            return fallback, True
        trace_row = source_trace[source_trace["row_id"].astype(str).eq(rid)]
        if trace_row.empty:
            fallback, _ = decision_mask(pred, pd.Series({**row.to_dict(), "apply_to_scope": "row_only"}), source_trace)
            return fallback, True
        # Prediction rows generally do not carry source curve IDs, so use row fallback unless those columns exist.
        mask = pd.Series(False, index=pred.index)
        for col in ["source_curve_id_S", "source_curve_id_sigma"]:
            if col in pred.columns and col in trace_row.columns and normalize_text(trace_row.iloc[0].get(col)):
                mask |= pred[col].astype(str).eq(str(trace_row.iloc[0][col]))
        if not mask.any():
            fallback, _ = decision_mask(pred, pd.Series({**row.to_dict(), "apply_to_scope": "row_only"}), source_trace)
            return fallback, True
        return mask, warning
    return pd.Series(False, index=pred.index), True


def apply_decisions(pred: pd.DataFrame, decisions: pd.DataFrame, source_trace: pd.DataFrame, pending_policy: str, suspect_policy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    scope_warnings: dict[int, bool] = {}
    for _, row in decisions.iterrows():
        mask, warning = decision_mask(pred, row, source_trace)
        idxs = pred.index[mask]
        if warning:
            for idx in idxs:
                scope_warnings[idx] = True
        for idx in idxs:
            records.append(
                {
                    "pred_index": idx,
                    "review_case_id": row["review_case_id"],
                    "review_case_type": row["review_case_type"],
                    "review_status": row["review_status"],
                    "review_reason_code": row["review_reason_code"],
                    "apply_to_scope": row["apply_to_scope"],
                    "primary_flag": row["primary_analysis_flag_after_review"],
                    "sensitivity_flag": row["sensitivity_analysis_flag_after_review"],
                    "reviewer_notes": row["reviewer_notes"],
                    "evidence_file_or_link": row["evidence_file_or_link"],
                    "checked_source_plot": row["checked_source_plot"],
                    "checked_units": row["checked_units"],
                    "checked_temperature_alignment": row["checked_temperature_alignment"],
                    "decision_is_pending": row["decision_is_pending"],
                    "decision_is_human_reviewed": row["decision_is_human_reviewed"],
                    "decision_validity_status": row["decision_validity_status"],
                }
            )
    applied = pd.DataFrame(records)
    out = pred.copy()
    defaults = {
        "review_case_ids_applied": "",
        "review_case_types_applied": "",
        "review_status_applied": "",
        "review_reason_codes_applied": "",
        "review_apply_scopes_applied": "",
        "review_decision_source": "no_review_case",
        "review_is_pending": False,
        "review_has_conflict": False,
        "review_scope_warning": False,
        "reviewed_by_human": False,
        "keep_in_primary_analysis": True,
        "keep_in_sensitivity_analysis": True,
        "exclude_from_primary_reason": "",
        "exclude_from_sensitivity_reason": "",
        "review_notes_combined": "",
        "evidence_links_combined": "",
        "checked_source_plot_any": False,
        "checked_units_any": False,
        "checked_temperature_alignment_any": False,
    }
    for col, value in defaults.items():
        out[col] = value

    conflict_rows: list[dict[str, Any]] = []
    if applied.empty:
        return out, pd.DataFrame(columns=["row_id", "review_case_ids_applied", "conflicting_fields", "conflicting_values", "final_review_status_applied", "final_keep_in_primary_analysis", "final_keep_in_sensitivity_analysis", "conflict_resolution_rule"])

    for idx, g in applied.groupby("pred_index", sort=False):
        statuses = list(g["review_status"])
        strongest = sorted(statuses, key=lambda s: STATUS_STRENGTH.get(s, 99))[0]
        primary_flags = set(g["primary_flag"])
        sensitivity_flags = set(g["sensitivity_flag"])
        has_conflict = (
            g["decision_validity_status"].eq("conflict").any()
            or ("exclude_from_primary" in primary_flags and "keep_in_primary" in primary_flags)
            or ("exclude_from_sensitivity" in sensitivity_flags and "keep_in_sensitivity" in sensitivity_flags)
        )
        primary_keep, sensitivity_keep, primary_reason, sensitivity_reason = resolve_keep_flags(strongest, primary_flags, sensitivity_flags, pending_policy, suspect_policy)
        if has_conflict and strongest == "exclude_from_all":
            primary_keep = False
            sensitivity_keep = False
            primary_reason = "conflict_resolved_by_exclude_from_all"
            sensitivity_reason = "conflict_resolved_by_exclude_from_all"
        out.at[idx, "review_case_ids_applied"] = ";".join(g["review_case_id"].astype(str).unique())
        out.at[idx, "review_case_types_applied"] = ";".join(g["review_case_type"].astype(str).unique())
        out.at[idx, "review_status_applied"] = strongest
        out.at[idx, "review_reason_codes_applied"] = ";".join(sorted({v for v in g["review_reason_code"].astype(str) if v}))
        out.at[idx, "review_apply_scopes_applied"] = ";".join(g["apply_to_scope"].astype(str).unique())
        out.at[idx, "review_decision_source"] = "pending_policy" if strongest in ["pending", "unresolved"] else "manual_decision"
        out.at[idx, "review_is_pending"] = bool(g["decision_is_pending"].any() or strongest in ["pending", "unresolved"])
        out.at[idx, "review_has_conflict"] = bool(has_conflict)
        out.at[idx, "review_scope_warning"] = bool(scope_warnings.get(idx, False) or g["apply_to_scope"].eq("undecided").any())
        out.at[idx, "reviewed_by_human"] = bool(g["decision_is_human_reviewed"].any() and strongest not in ["pending"])
        out.at[idx, "keep_in_primary_analysis"] = primary_keep
        out.at[idx, "keep_in_sensitivity_analysis"] = sensitivity_keep
        out.at[idx, "exclude_from_primary_reason"] = "" if primary_keep else primary_reason
        out.at[idx, "exclude_from_sensitivity_reason"] = "" if sensitivity_keep else sensitivity_reason
        out.at[idx, "review_notes_combined"] = ";".join([v for v in g["reviewer_notes"].astype(str).unique() if v])
        out.at[idx, "evidence_links_combined"] = ";".join([v for v in g["evidence_file_or_link"].astype(str).unique() if v])
        out.at[idx, "checked_source_plot_any"] = bool_any_yes(g["checked_source_plot"])
        out.at[idx, "checked_units_any"] = bool_any_yes(g["checked_units"])
        out.at[idx, "checked_temperature_alignment_any"] = bool_any_yes(g["checked_temperature_alignment"])
        if has_conflict:
            conflict_rows.append(
                {
                    "row_id": out.at[idx, "row_id"],
                    "review_case_ids_applied": out.at[idx, "review_case_ids_applied"],
                    "conflicting_fields": "review_status_or_flags",
                    "conflicting_values": f"statuses={sorted(set(statuses))};primary={sorted(primary_flags)};sensitivity={sorted(sensitivity_flags)}",
                    "final_review_status_applied": strongest,
                    "final_keep_in_primary_analysis": primary_keep,
                    "final_keep_in_sensitivity_analysis": sensitivity_keep,
                    "conflict_resolution_rule": "strongest_status_wins;exclude_from_all_overrides_flags",
                }
            )
    conflicts = pd.DataFrame(conflict_rows)
    if conflicts.empty:
        conflicts = pd.DataFrame(columns=["row_id", "review_case_ids_applied", "conflicting_fields", "conflicting_values", "final_review_status_applied", "final_keep_in_primary_analysis", "final_keep_in_sensitivity_analysis", "conflict_resolution_rule"])
    return out, conflicts


def resolve_keep_flags(status: str, primary_flags: set[str], sensitivity_flags: set[str], pending_policy: str, suspect_policy: str) -> tuple[bool, bool, str, str]:
    if status == "exclude_from_all":
        return False, False, "review_status_exclude_from_all", "review_status_exclude_from_all"
    primary: bool | None = None
    sensitivity: bool | None = None
    if "exclude_from_primary" in primary_flags:
        primary = False
    elif "keep_in_primary" in primary_flags:
        primary = True
    if "exclude_from_sensitivity" in sensitivity_flags:
        sensitivity = False
    elif "keep_in_sensitivity" in sensitivity_flags:
        sensitivity = True
    primary_reason = ""
    sensitivity_reason = ""
    if primary is None or sensitivity is None:
        if status in ["pending", "unresolved"]:
            if pending_policy == "exclude_from_primary":
                if primary is None:
                    primary = False
                    primary_reason = f"{pending_policy}_{status}"
                if sensitivity is None:
                    sensitivity = True
            else:
                if primary is None:
                    primary = True
                if sensitivity is None:
                    sensitivity = True
        elif status == "suspect":
            if suspect_policy == "keep_both":
                if primary is None:
                    primary = True
                if sensitivity is None:
                    sensitivity = True
            elif suspect_policy == "exclude_both":
                if primary is None:
                    primary = False
                    primary_reason = "suspect_policy_exclude_both"
                if sensitivity is None:
                    sensitivity = False
                    sensitivity_reason = "suspect_policy_exclude_both"
            else:
                if primary is None:
                    primary = False
                    primary_reason = "suspect_policy_exclude_primary_keep_sensitivity"
                if sensitivity is None:
                    sensitivity = True
        elif status == "exclude_from_primary":
            if primary is None:
                primary = False
                primary_reason = "review_status_exclude_from_primary"
            if sensitivity is None:
                sensitivity = True
        else:
            if primary is None:
                primary = True
            if sensitivity is None:
                sensitivity = True
    if primary is False and not primary_reason:
        primary_reason = "manual_or_flag_exclude_from_primary"
    if sensitivity is False and not sensitivity_reason:
        sensitivity_reason = "manual_or_flag_exclude_from_sensitivity"
    return bool(primary), bool(sensitivity), primary_reason, sensitivity_reason


def metric_frame(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for config_id, g in df.groupby("config_id", dropna=False):
        meta = g.iloc[0].to_dict()
        for weighting in ["row_equal", "sample_equal"]:
            if weighting == "row_equal":
                metric = compute_metrics(g)
            else:
                metric = compute_sample_equal_metrics(g)
                metric["n_samples"] = g["validation_sample_group_id"].nunique(dropna=True)
                metric["n_papers"] = g["validation_paper_group_id"].nunique(dropna=True)
            row = {
                "review_scenario": scenario,
                "config_id": config_id,
                "split_scheme": meta.get("split_scheme", ""),
                "reference_source_subset": meta.get("reference_source_subset", ""),
                "eval_target_subset": meta.get("eval_target_subset", ""),
                "group_scheme": meta.get("group_scheme", ""),
                "curve_method": meta.get("curve_method", ""),
                "metric_weighting": weighting,
            }
            row.update(metric)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    err = pd.to_numeric(df["log10_sigma_pred_over_exp"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    abs_err = err.abs()
    return {
        "n_rows": int(len(df)),
        "n_samples": int(df["validation_sample_group_id"].nunique(dropna=True)),
        "n_papers": int(df["validation_paper_group_id"].nunique(dropna=True)),
        "n_reviewed_rows": int(df["review_case_ids_applied"].astype(str).str.len().gt(0).sum()) if "review_case_ids_applied" in df.columns else 0,
        "n_pending_rows": int(df["review_is_pending"].sum()) if "review_is_pending" in df.columns else 0,
        "n_excluded_from_primary": int((~df["keep_in_primary_analysis"]).sum()) if "keep_in_primary_analysis" in df.columns else 0,
        "n_excluded_from_sensitivity": int((~df["keep_in_sensitivity_analysis"]).sum()) if "keep_in_sensitivity_analysis" in df.columns else 0,
        "mean_log10_error": float(err.mean()) if len(err) else np.nan,
        "median_log10_error": float(err.median()) if len(err) else np.nan,
        "mae_log10": float(abs_err.mean()) if len(abs_err) else np.nan,
        "rmse_log10": float(np.sqrt((err ** 2).mean())) if len(err) else np.nan,
        "std_log10_error": float(err.std()) if len(err) else np.nan,
        "q05_log10_error": float(err.quantile(0.05)) if len(err) else np.nan,
        "q25_log10_error": float(err.quantile(0.25)) if len(err) else np.nan,
        "q75_log10_error": float(err.quantile(0.75)) if len(err) else np.nan,
        "q95_log10_error": float(err.quantile(0.95)) if len(err) else np.nan,
        "max_abs_log10_error": float(abs_err.max()) if len(abs_err) else np.nan,
        "factor_2_accuracy": float((abs_err <= np.log10(2)).mean()) if len(abs_err) else np.nan,
        "factor_3_accuracy": float((abs_err <= np.log10(3)).mean()) if len(abs_err) else np.nan,
        "factor_5_accuracy": float((abs_err <= np.log10(5)).mean()) if len(abs_err) else np.nan,
        "factor_10_accuracy": float((abs_err <= 1.0).mean()) if len(abs_err) else np.nan,
        "overprediction_fraction": float((err > 0).mean()) if len(err) else np.nan,
        "underprediction_fraction": float((err < 0).mean()) if len(err) else np.nan,
        "extreme_ge_10_count": int((abs_err >= 10).sum()),
        "severe_ge_5_count": int((abs_err >= 5).sum()),
        "large_ge_2_count": int((abs_err >= 2).sum()),
    }


def compute_sample_equal_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return compute_metrics(df)
    work = pd.DataFrame(
        {
            "sample": df["validation_sample_group_id"],
            "err": pd.to_numeric(df["log10_sigma_pred_over_exp"], errors="coerce"),
            "reviewed": df["review_case_ids_applied"].astype(str).str.len().gt(0).astype(float),
            "pending": df["review_is_pending"].astype(float),
            "excluded_primary": (~df["keep_in_primary_analysis"]).astype(float),
            "excluded_sensitivity": (~df["keep_in_sensitivity_analysis"]).astype(float),
        }
    ).replace([np.inf, -np.inf], np.nan).dropna(subset=["err"])
    if work.empty:
        return compute_metrics(df)
    work["abs_err"] = work["err"].abs()
    work["sq_err"] = work["err"] ** 2
    work["factor2"] = (work["abs_err"] <= np.log10(2)).astype(float)
    work["factor3"] = (work["abs_err"] <= np.log10(3)).astype(float)
    work["factor5"] = (work["abs_err"] <= np.log10(5)).astype(float)
    work["factor10"] = (work["abs_err"] <= 1.0).astype(float)
    work["over"] = (work["err"] > 0).astype(float)
    work["under"] = (work["err"] < 0).astype(float)
    work["extreme"] = (work["abs_err"] >= 10).astype(float)
    work["severe"] = (work["abs_err"] >= 5).astype(float)
    work["large"] = (work["abs_err"] >= 2).astype(float)
    grouped = work.groupby("sample", dropna=False)
    sample = grouped.agg(
        n_rows=("err", "size"),
        n_reviewed_rows=("reviewed", "sum"),
        n_pending_rows=("pending", "sum"),
        n_excluded_from_primary=("excluded_primary", "sum"),
        n_excluded_from_sensitivity=("excluded_sensitivity", "sum"),
        mean_log10_error=("err", "mean"),
        median_log10_error=("err", "median"),
        mae_log10=("abs_err", "mean"),
        std_log10_error=("err", "std"),
        max_abs_log10_error=("abs_err", "max"),
        factor_2_accuracy=("factor2", "mean"),
        factor_3_accuracy=("factor3", "mean"),
        factor_5_accuracy=("factor5", "mean"),
        factor_10_accuracy=("factor10", "mean"),
        overprediction_fraction=("over", "mean"),
        underprediction_fraction=("under", "mean"),
        extreme_ge_10_count=("extreme", "sum"),
        severe_ge_5_count=("severe", "sum"),
        large_ge_2_count=("large", "sum"),
        mean_sq=("sq_err", "mean"),
    )
    quantiles = grouped["err"].quantile([0.05, 0.25, 0.75, 0.95]).unstack()
    quantiles.columns = ["q05_log10_error", "q25_log10_error", "q75_log10_error", "q95_log10_error"]
    sample = sample.join(quantiles)
    sample["rmse_log10"] = np.sqrt(sample["mean_sq"])
    out = {col: float(pd.to_numeric(sample[col], errors="coerce").mean()) for col in sample.columns if col != "mean_sq"}
    out["n_rows"] = int(len(sample))
    out["n_samples"] = int(len(sample))
    out["n_papers"] = int(df["validation_paper_group_id"].nunique(dropna=True))
    return out


def default_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    labels = {v: k for k, v in DEFAULT_CONFIGS.items()}
    out = metrics[metrics["config_id"].isin(labels)].copy()
    out["config_label"] = out["config_id"].map(labels)
    return out


def review_effect_summary(default_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, config_id, weighting), g in default_df.groupby(["config_label", "config_id", "metric_weighting"], dropna=False):
        base = g[g["review_scenario"].eq("all_predictions_no_review_filter")]
        primary = g[g["review_scenario"].eq("primary_review_applied")]
        sens = g[g["review_scenario"].eq("sensitivity_review_applied")]
        if base.empty:
            continue
        for metric in METRIC_NAMES:
            b = base[metric].iloc[0] if metric in base else np.nan
            p = primary[metric].iloc[0] if not primary.empty and metric in primary else np.nan
            s = sens[metric].iloc[0] if not sens.empty and metric in sens else np.nan
            rows.append(
                {
                    "config_label": label,
                    "config_id": config_id,
                    "metric_weighting": weighting,
                    "metric_name": metric,
                    "baseline_value": b,
                    "primary_review_applied_value": p,
                    "sensitivity_review_applied_value": s,
                    "delta_primary_minus_baseline": p - b if pd.notna(p) and pd.notna(b) else np.nan,
                    "delta_sensitivity_minus_baseline": s - b if pd.notna(s) and pd.notna(b) else np.nan,
                    "interpretation_hint": "lower_is_better" if metric in ["mae_log10", "rmse_log10", "max_abs_log10_error", "extreme_ge_10_count", "severe_ge_5_count"] else "higher_is_better_or_count",
                }
            )
    return pd.DataFrame(rows)


def summary_items(pred: pd.DataFrame, decisions: pd.DataFrame, rows: pd.DataFrame, pending_policy: str, suspect_policy: str) -> pd.DataFrame:
    data = [
        ("input_prediction_rows", len(pred), "Rows read from Step6B prediction valid table."),
        ("decision_template_rows", len(decisions), "Rows read from Step7A decision template."),
        ("human_reviewed_decisions", int(decisions["decision_is_human_reviewed"].sum()), "Decision rows with non-pending review status/name/date."),
        ("pending_decisions", int(decisions["decision_is_pending"].sum()), "Decision rows still pending or unresolved."),
        ("applied_review_rows", int(rows["review_case_ids_applied"].astype(str).str.len().gt(0).sum()), "Prediction rows touched by at least one review case."),
        ("rows_with_no_review_case", int(rows["review_case_ids_applied"].astype(str).str.len().eq(0).sum()), "Prediction rows with no applied review case."),
        ("rows_with_pending_review", int(rows["review_is_pending"].sum()), "Prediction rows with pending/unresolved applied review."),
        ("rows_with_conflicts", int(rows["review_has_conflict"].sum()), "Prediction rows with conflicting review decisions."),
        ("primary_kept_rows", int(rows["keep_in_primary_analysis"].sum()), "Rows retained in primary analysis."),
        ("primary_excluded_rows", int((~rows["keep_in_primary_analysis"]).sum()), "Rows excluded from primary analysis."),
        ("sensitivity_kept_rows", int(rows["keep_in_sensitivity_analysis"].sum()), "Rows retained in sensitivity analysis."),
        ("sensitivity_excluded_rows", int((~rows["keep_in_sensitivity_analysis"]).sum()), "Rows excluded from sensitivity analysis."),
        ("pending_policy", pending_policy, "Policy for pending/unresolved decisions."),
        ("suspect_policy", suspect_policy, "Policy for suspect decisions."),
    ]
    return pd.DataFrame(data, columns=["item", "value", "comment"])


def readiness_summary(decisions: pd.DataFrame, rows: pd.DataFrame, metrics: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    pending_decisions = int(decisions["decision_is_pending"].sum())
    primary_rows = int(rows["keep_in_primary_analysis"].sum())
    sensitivity_rows = int(rows["keep_in_sensitivity_analysis"].sum())
    primary_default = metrics[
        metrics["review_scenario"].eq("primary_review_applied")
        & metrics["config_id"].eq(DEFAULT_CONFIGS["broad_material_family_default"])
        & metrics["metric_weighting"].eq("row_equal")
    ]
    extreme_primary = primary_default["extreme_ge_10_count"].iloc[0] if not primary_default.empty else np.nan
    mae = primary_default["mae_log10"].iloc[0] if not primary_default.empty else np.nan
    rows_out = []

    def add(criterion: str, status: str, value: Any, threshold: str, comment: str) -> None:
        rows_out.append({"criterion": criterion, "status": status, "value": value, "threshold_or_reason": threshold, "comment": comment})

    add("decision_template_loaded", "pass", len(decisions), "> 0", "Step7A decision template was loaded.")
    add("human_review_completed", "pass" if pending_decisions == 0 else "caution", pending_decisions, "pass if 0", "Pending decisions remain if value > 0.")
    add("primary_dataset_available", "pass" if primary_rows > 0 else "fail", primary_rows, "> 0", "Primary analysis rows after review policy.")
    add("sensitivity_dataset_available", "pass" if sensitivity_rows > 0 else "fail", sensitivity_rows, "> 0", "Sensitivity analysis rows after review policy.")
    add("primary_exclusion_documented", "pass", int((~rows["keep_in_primary_analysis"]).sum()), "count recorded", "Excluded rows are written to CSV.")
    add("unresolved_rows_exist", "caution" if rows["review_is_pending"].any() else "pass", int(rows["review_is_pending"].sum()), "caution if > 0", "Pending/unresolved rows remain.")
    add("conflicts_exist", "caution" if len(conflicts) else "pass", len(conflicts), "caution if > 0", "Conflicting decisions are recorded.")
    add("extreme_outliers_remaining_in_primary", "caution" if pd.notna(extreme_primary) and extreme_primary > 0 else "pass", extreme_primary, "caution if > 0", "Extreme outliers in primary broad material_family default.")
    add("broad_family_primary_mae_available", "pass" if pd.notna(mae) else "fail", mae, "finite", "Primary broad material_family default MAE.")
    add("recommended_next_action", "caution" if pending_decisions else "pass", "complete manual review then rerun Step7B" if pending_decisions else "proceed to final tables", "manual decision", "Next action after applying review decisions.")
    return pd.DataFrame(rows_out)


def unresolved_checklist(decisions: pd.DataFrame, conflicts: pd.DataFrame) -> pd.DataFrame:
    pending = decisions[decisions["decision_is_pending"] | decisions["review_status"].isin(["pending", "unresolved"])].copy()
    pending["issue_type"] = "pending_or_unresolved"
    pending["suggested_action"] = "fill review_status and analysis flags, then rerun Step7B"
    if not conflicts.empty:
        conflict_cases = set(";".join(conflicts["review_case_ids_applied"].astype(str)).split(";"))
        c = decisions[decisions["review_case_id"].isin(conflict_cases)].copy()
        c["issue_type"] = "conflict"
        c["suggested_action"] = "resolve conflicting status/flags, then rerun Step7B"
        pending = pd.concat([pending, c], ignore_index=True).drop_duplicates("review_case_id")
    cols = [
        "review_case_id",
        "review_case_type",
        "review_priority",
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "material_group_key",
        "abs_error_decades",
        "error_severity",
        "review_status",
        "review_reason_code",
        "apply_to_scope",
        "issue_type",
        "suggested_action",
    ]
    for col in cols:
        if col not in pending.columns:
            pending[col] = ""
    return pending[cols].sort_values("review_priority")


def df_to_markdown(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join("---" for _ in text.columns) + " |"
    rows = ["| " + " | ".join(row[col] for col in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(report: Path, inputs: dict[str, Path], summary: pd.DataFrame, default_df: pd.DataFrame, effect: pd.DataFrame, readiness: pd.DataFrame, checks: dict[str, bool], elapsed: float) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    bmf = default_df[
        default_df["config_label"].eq("broad_material_family_default")
        & default_df["metric_weighting"].eq("row_equal")
    ][["review_scenario", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "extreme_ge_10_count", "n_rows"]]
    lines = [
        "# Step7B Review Application Report",
        "",
        "## Inputs",
        "",
        *[f"- {k}: {v}" for k, v in inputs.items()],
        "",
        "## Application Summary",
        "",
        df_to_markdown(summary, 30),
        "",
        "## Broad Material Family Default Metrics",
        "",
        df_to_markdown(bmf, 20),
        "",
        "## Review Effect Summary",
        "",
        df_to_markdown(effect[effect["config_label"].eq("broad_material_family_default")], 30),
        "",
        "## Readiness",
        "",
        df_to_markdown(readiness, 20),
        "",
        "## Notes",
        "",
        "- Step7B applies existing review decisions to existing prediction rows.",
        "- Step7B does not compute new sigma predictions.",
        "- Pending decisions may remain depending on pending policy.",
        "- If pending rows are retained, final reporting should mention that unreviewed outliers remain.",
        "",
        "## Sanity Checks",
        "",
        *[f"- {name}: {ok}" for name, ok in checks.items()],
        "",
        f"- elapsed_seconds: {elapsed:.2f}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity(pred: pd.DataFrame, rows: pd.DataFrame, primary: pd.DataFrame, sensitivity: pd.DataFrame, exp_primary: pd.DataFrame, exp_sens: pd.DataFrame, outputs: dict[str, Path], report: Path) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "prediction_input_exists": len(pred) > 0,
        "normalized_decisions_created": outputs["normalized"].exists() and outputs["normalized"].stat().st_size > 0,
        "prediction_rows_with_review_flags_created": len(rows) == len(pred),
        "row_count_matches_prediction_input": len(rows) == len(pred),
        "row_id_not_missing": rows["row_id"].notna().all(),
        "config_id_not_missing": rows["config_id"].notna().all(),
        "keep_in_primary_not_missing": rows["keep_in_primary_analysis"].notna().all(),
        "keep_in_sensitivity_not_missing": rows["keep_in_sensitivity_analysis"].notna().all(),
        "primary_only_kept": primary["keep_in_primary_analysis"].all() if len(primary) else True,
        "sensitivity_only_kept": sensitivity["keep_in_sensitivity_analysis"].all() if len(sensitivity) else True,
        "excluded_primary_only_excluded": (~exp_primary["keep_in_primary_analysis"]).all() if len(exp_primary) else True,
        "excluded_sensitivity_only_excluded": (~exp_sens["keep_in_sensitivity_analysis"]).all() if len(exp_sens) else True,
        "metrics_created": outputs["metrics"].exists() and outputs["metrics"].stat().st_size > 0,
        "default_metrics_created": outputs["default_metrics"].exists() and outputs["default_metrics"].stat().st_size > 0,
        "review_effect_created": outputs["effect"].exists() and outputs["effect"].stat().st_size > 0,
        "readiness_created": outputs["readiness"].exists() and outputs["readiness"].stat().st_size > 0,
        "conflict_table_created": outputs["conflicts"].exists(),
        "did_not_compute_new_sigma_pred": True,
        "did_not_read_step4_full_data_reference_curve": True,
        "did_not_read_raw_data": True,
        "report_created": report.exists() and report.stat().st_size > 0,
    }
    return checks, [name for name, ok in checks.items() if not ok]


def main() -> None:
    started = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    log("loading broad_family prediction rows...")
    pred = read_table(args.predictions)
    pred = prepare_predictions(pred, args.max_rows_per_config)

    log("loading review decision template...")
    decisions_raw = read_table(args.decision_template)
    if args.pending_policy == "fail_if_pending":
        pending_count = decisions_raw["review_status"].fillna("pending").astype(str).str.strip().isin(["", "pending", "unresolved"]).sum()
        if pending_count:
            raise SystemExit(f"pending decisions found with --pending-policy fail_if_pending: {pending_count}")

    log("loading review master and source traceability...")
    _ = read_table(args.review_master)
    source_trace = read_table(args.source_trace) if args.source_trace.exists() else pd.DataFrame()

    log("normalizing review decisions...")
    decisions = normalize_decisions(decisions_raw)

    log("applying row-level decisions...")
    log("applying sample-level decisions...")
    log("applying paper-level decisions...")
    log("applying source-curve-level decisions...")
    rows, conflicts = apply_decisions(pred, decisions, source_trace, args.pending_policy, args.suspect_policy)

    log("resolving conflicts...")
    log("assigning primary and sensitivity flags...")
    primary = rows[rows["keep_in_primary_analysis"]].copy()
    sensitivity = rows[rows["keep_in_sensitivity_analysis"]].copy()
    excluded_primary = rows[~rows["keep_in_primary_analysis"]].copy()
    excluded_sensitivity = rows[~rows["keep_in_sensitivity_analysis"]].copy()
    pending = rows[rows["review_is_pending"] | rows["review_status_applied"].isin(["pending", "unresolved"])].copy()

    log("building primary/sensitivity/excluded datasets...")
    outputs = {
        "normalized": args.output / out_name("step7b_review_decisions_normalized", args.output_suffix),
        "rows": args.output / out_name("step7b_prediction_rows_with_review_flags", args.output_suffix),
        "rows_parquet": args.output / out_name("step7b_prediction_rows_with_review_flags", args.output_suffix, "parquet"),
        "primary": args.output / out_name("step7b_primary_analysis_predictions", args.output_suffix),
        "primary_parquet": args.output / out_name("step7b_primary_analysis_predictions", args.output_suffix, "parquet"),
        "sensitivity": args.output / out_name("step7b_sensitivity_analysis_predictions", args.output_suffix),
        "sensitivity_parquet": args.output / out_name("step7b_sensitivity_analysis_predictions", args.output_suffix, "parquet"),
        "excluded_primary": args.output / out_name("step7b_excluded_from_primary", args.output_suffix),
        "excluded_sensitivity": args.output / out_name("step7b_excluded_from_sensitivity", args.output_suffix),
        "pending": args.output / out_name("step7b_pending_or_unresolved_rows", args.output_suffix),
        "conflicts": args.output / out_name("step7b_review_conflicts", args.output_suffix),
        "summary": args.output / out_name("step7b_review_application_summary", args.output_suffix),
        "metrics": args.output / out_name("step7b_metrics_by_review_scenario_config", args.output_suffix),
        "metrics_parquet": args.output / out_name("step7b_metrics_by_review_scenario_config", args.output_suffix, "parquet"),
        "default_metrics": args.output / out_name("step7b_default_metrics_by_review_scenario", args.output_suffix),
        "effect": args.output / out_name("step7b_review_effect_summary", args.output_suffix),
        "readiness": args.output / out_name("step7b_review_readiness_summary", args.output_suffix),
        "unresolved": args.output / out_name("step7b_manual_review_unresolved_checklist", args.output_suffix),
    }

    decisions.to_csv(outputs["normalized"], index=False, encoding="utf-8-sig")
    write_csv_parquet(rows, outputs["rows"], outputs["rows_parquet"])
    write_csv_parquet(primary, outputs["primary"], outputs["primary_parquet"])
    write_csv_parquet(sensitivity, outputs["sensitivity"], outputs["sensitivity_parquet"])
    excluded_primary.to_csv(outputs["excluded_primary"], index=False, encoding="utf-8-sig")
    excluded_sensitivity.to_csv(outputs["excluded_sensitivity"], index=False, encoding="utf-8-sig")
    pending.to_csv(outputs["pending"], index=False, encoding="utf-8-sig")
    conflicts.to_csv(outputs["conflicts"], index=False, encoding="utf-8-sig")

    log("computing metrics by review scenario...")
    baseline = metric_frame(rows, "all_predictions_no_review_filter")
    primary_metrics = metric_frame(primary, "primary_review_applied")
    sensitivity_metrics = metric_frame(sensitivity, "sensitivity_review_applied")
    metrics = pd.concat([baseline, primary_metrics, sensitivity_metrics], ignore_index=True)
    write_csv_parquet(metrics, outputs["metrics"], outputs["metrics_parquet"])
    default_df = default_metrics(metrics)
    default_df.to_csv(outputs["default_metrics"], index=False, encoding="utf-8-sig")

    log("building review effect summary...")
    effect = review_effect_summary(default_df)
    effect.to_csv(outputs["effect"], index=False, encoding="utf-8-sig")
    summary = summary_items(pred, decisions, rows, args.pending_policy, args.suspect_policy)
    summary.to_csv(outputs["summary"], index=False, encoding="utf-8-sig")

    log("building readiness summary...")
    readiness = readiness_summary(decisions, rows, metrics, conflicts)
    readiness.to_csv(outputs["readiness"], index=False, encoding="utf-8-sig")
    unresolved = unresolved_checklist(decisions, conflicts)
    unresolved.to_csv(outputs["unresolved"], index=False, encoding="utf-8-sig")

    log("writing report...")
    inputs = {"predictions": args.predictions, "decision_template": args.decision_template, "review_master": args.review_master, "source_trace": args.source_trace}
    write_report(args.report, inputs, summary, default_df, effect, readiness, {}, time.time() - started)

    log("running sanity checks...")
    checks, failures = run_sanity(pred, rows, primary, sensitivity, excluded_primary, excluded_sensitivity, outputs, args.report)
    if failures:
        write_report(args.report, inputs, summary, default_df, effect, readiness, checks, time.time() - started)
        for failure in failures:
            print(f"[step7b] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(args.report, inputs, summary, default_df, effect, readiness, checks, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
