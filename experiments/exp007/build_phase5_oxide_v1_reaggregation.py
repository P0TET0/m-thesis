"""Reaggregate the frozen Phase 1 predictions using oxide taxonomy v1.

Phase 5 does not retrain or modify the predictor. It joins the existing held-out
row predictions to the frozen Phase 4 taxonomy and recomputes descriptive
metrics at row-equal and sample-equal levels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"

PHASE1_ROWS = DATA / "phase1_sigma_residual_rows.csv"
PHASE1_CHECK = REPORTS / "phase1_check_log.txt"
PHASE4_ROWS = DATA / "phase4_oxide_v1_row_assignments.csv"
PHASE4_CONFIG = DATA / "phase4_oxide_v1_classification_config.json"
PHASE4_CHECK = REPORTS / "phase4_check_log.txt"

DEFAULT_OUTPUT_DIR = DATA
DEFAULT_REPORT = REPORTS / "phase5_oxide_v1_reaggregation.md"
DEFAULT_VALIDATION_LOG = REPORTS / "phase5_validation_log.txt"
DEFAULT_WORKBOOK_PAYLOAD = REPORTS / ".phase5_workbook_payload.json"

PHASE5_DATE = "2026-10-14"
PRIMARY_BASELINE = "broad_family_100K"
ATOL = 1e-12
FACTOR_2_LOG10 = float(np.log10(2.0))
FACTOR_5_LOG10 = float(np.log10(5.0))
FACTOR_10_LOG10 = 1.0

SCOPE_ORDER = {
    "all_assigned": 0,
    "default_eligible": 1,
    "review_only": 2,
    "excluded_only": 3,
}

CLASSIFICATION_COLUMNS = [
    "taxonomy_version",
    "taxonomy_rules_sha256",
    "oxide_v1_rule_id",
    "oxide_v1_rule_priority",
    "oxide_v1_group_key",
    "oxide_v1_display_name_en",
    "oxide_v1_display_name_ja",
    "oxide_v1_scope",
    "oxide_v1_default_model_action",
    "oxide_v1_match_basis",
    "oxide_v1_inclusion_status",
    "oxide_v1_default_model_eligible",
    "oxide_v1_classification_confidence",
    "oxide_v1_review_priority",
    "oxide_v1_review_reasons",
    "oxide_v1_formula_like_label",
    "oxide_v1_multicomponent_notation",
    "oxide_v1_structural_class",
    "material_label_normalized",
    "candidate_oxide_family",
]

ROW_OUTPUT_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_label_normalized",
    "carrier_type",
    "T_K",
    "T_bin_label",
    "T_bin_center_K",
    "S_abs_uV_per_K",
    "sigma_observed_S_per_m",
    "sigma_predicted_S_per_m",
    "residual_log10_pred_over_observed",
    "abs_residual_log10",
    "absolute_error_factor",
    "prediction_direction",
    "residual_band",
    "within_factor_2",
    "within_factor_5",
    "within_factor_10",
    "beyond_factor_10",
    "taxonomy_version",
    "taxonomy_rules_sha256",
    "oxide_v1_rule_id",
    "oxide_v1_rule_priority",
    "oxide_v1_group_key",
    "oxide_v1_display_name_en",
    "oxide_v1_display_name_ja",
    "oxide_v1_scope",
    "oxide_v1_default_model_action",
    "oxide_v1_inclusion_status",
    "oxide_v1_default_model_eligible",
    "oxide_v1_classification_confidence",
    "oxide_v1_review_priority",
    "oxide_v1_review_reasons",
    "oxide_v1_formula_like_label",
    "oxide_v1_multicomponent_notation",
    "oxide_v1_structural_class",
    "candidate_oxide_family",
    "phase5_baseline_variant",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION_LOG)
    parser.add_argument("--workbook-payload", type=Path, default=DEFAULT_WORKBOOK_PAYLOAD)
    return parser.parse_args()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def first_nonempty(values: Iterable[Any]) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def join_unique(values: Iterable[Any], limit: int = 12) -> str:
    unique: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    if len(unique) <= limit:
        return " | ".join(unique)
    return " | ".join(unique[:limit]) + f" | ... (+{len(unique) - limit})"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_manifest() -> pd.DataFrame:
    sources = [
        ("Phase 1 held-out prediction rows", PHASE1_ROWS),
        ("Phase 1 independent check log", PHASE1_CHECK),
        ("Phase 4 row assignments", PHASE4_ROWS),
        ("Phase 4 taxonomy config", PHASE4_CONFIG),
        ("Phase 4 independent check log", PHASE4_CHECK),
    ]
    records = []
    for role, path in sources:
        records.append(
            {
                "role": role,
                "path": path.relative_to(ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(records)


def require_inputs() -> None:
    missing = [path for path in [PHASE1_ROWS, PHASE1_CHECK, PHASE4_ROWS, PHASE4_CONFIG, PHASE4_CHECK] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 5 inputs: {missing}")
    if "FAIL " in PHASE1_CHECK.read_text(encoding="utf-8"):
        raise RuntimeError("Phase 1 independent check is not PASS")
    if "status=PASS" not in PHASE4_CHECK.read_text(encoding="utf-8"):
        raise RuntimeError("Phase 4 independent check is not PASS")


def numeric_summary(residuals: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(residuals, errors="coerce").dropna().to_numpy(float)
    if not values.size:
        return {
            "mean_residual_log10": np.nan,
            "median_residual_log10": np.nan,
            "mae_log10": np.nan,
            "rmse_log10": np.nan,
            "max_abs_residual_log10": np.nan,
            "overprediction_fraction": np.nan,
            "underprediction_fraction": np.nan,
            "within_factor_2_fraction": np.nan,
            "within_factor_5_fraction": np.nan,
            "within_factor_10_fraction": np.nan,
            "beyond_factor_10_count": 0,
        }
    absolute = np.abs(values)
    return {
        "mean_residual_log10": float(np.mean(values)),
        "median_residual_log10": float(np.median(values)),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "max_abs_residual_log10": float(np.max(absolute)),
        "overprediction_fraction": float(np.mean(values > ATOL)),
        "underprediction_fraction": float(np.mean(values < -ATOL)),
        "within_factor_2_fraction": float(np.mean(absolute <= FACTOR_2_LOG10 + ATOL)),
        "within_factor_5_fraction": float(np.mean(absolute <= FACTOR_5_LOG10 + ATOL)),
        "within_factor_10_fraction": float(np.mean(absolute <= FACTOR_10_LOG10 + ATOL)),
        "beyond_factor_10_count": int(np.sum(absolute > FACTOR_10_LOG10 + ATOL)),
    }


def direction(value: float) -> str:
    if not np.isfinite(value):
        return "not_available"
    if value > ATOL:
        return "overprediction"
    if value < -ATOL:
        return "underprediction"
    return "matched"


def scope_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    eligible = as_bool(rows["oxide_v1_default_model_eligible"])
    status = rows["oxide_v1_inclusion_status"].astype(str)
    return {
        "all_assigned": pd.Series(True, index=rows.index),
        "default_eligible": eligible,
        "review_only": status.eq("review"),
        "excluded_only": status.eq("exclude"),
    }


def build_joined_rows(phase1: pd.DataFrame, phase4: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    oxide = phase1.loc[as_bool(phase1["is_oxide_broad"]) & as_bool(phase1["is_evaluable"])].copy()
    classification = phase4[["row_id"] + CLASSIFICATION_COLUMNS].copy()
    joined = oxide.merge(classification, on="row_id", how="left", validate="one_to_one")
    joined["phase5_baseline_variant"] = PRIMARY_BASELINE
    phase4_residual = phase4.set_index("row_id")["residual_log10_pred_over_observed"]
    aligned = joined["row_id"].map(phase4_residual)
    difference = np.abs(
        pd.to_numeric(joined["residual_log10_pred_over_observed"], errors="coerce").to_numpy(float)
        - pd.to_numeric(aligned, errors="coerce").to_numpy(float)
    )
    max_difference = float(np.nanmax(difference)) if difference.size else np.nan
    order = joined["oxide_v1_review_priority"].map({"high": 3, "medium": 2, "low": 1, "none": 0}).fillna(0)
    joined = joined.assign(_review_order=order).sort_values(
        ["_review_order", "abs_residual_log10", "oxide_v1_group_key", "validation_sample_group_id", "T_K"],
        ascending=[False, False, True, True, True],
        na_position="last",
    ).drop(columns="_review_order").reset_index(drop=True)
    return joined[ROW_OUTPUT_COLUMNS], max_difference


def build_sample_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_group, group in rows.groupby("validation_sample_group_id", sort=False):
        group_keys = group["oxide_v1_group_key"].dropna().unique()
        statuses = group["oxide_v1_inclusion_status"].dropna().unique()
        conflict = len(group_keys) != 1 or len(statuses) != 1
        record: dict[str, Any] = {
            "validation_sample_group_id": sample_group,
            "validation_paper_group_id": first_nonempty(group["validation_paper_group_id"]),
            "paper_id": first_nonempty(group["paper_id"]),
            "doi": first_nonempty(group["doi"]),
            "sample_id": first_nonempty(group["sample_id"]),
            "sample_key": first_nonempty(group["sample_key"]),
            "formula_raw": join_unique(group["formula_raw"]),
            "material_name_raw": join_unique(group["material_name_raw"]),
            "material_labels_normalized": join_unique(group["material_label_normalized"]),
            "carrier_types": join_unique(group["carrier_type"]),
            "taxonomy_version": first_nonempty(group["taxonomy_version"]),
            "taxonomy_rules_sha256": first_nonempty(group["taxonomy_rules_sha256"]),
            "oxide_v1_group_key": group_keys[0] if not conflict else "oxide_v1::sample_assignment_conflict",
            "oxide_v1_display_name_en": join_unique(group["oxide_v1_display_name_en"]),
            "oxide_v1_display_name_ja": join_unique(group["oxide_v1_display_name_ja"]),
            "oxide_v1_rule_id": join_unique(group["oxide_v1_rule_id"]),
            "oxide_v1_scope": join_unique(group["oxide_v1_scope"]),
            "oxide_v1_default_model_action": join_unique(group["oxide_v1_default_model_action"]),
            "oxide_v1_inclusion_status": statuses[0] if not conflict else "review",
            "oxide_v1_default_model_eligible": bool(as_bool(group["oxide_v1_default_model_eligible"]).all()) and not conflict,
            "oxide_v1_review_priority": join_unique(group["oxide_v1_review_priority"]),
            "oxide_v1_review_reasons": join_unique(group["oxide_v1_review_reasons"]),
            "sample_assignment_conflict": conflict,
            "n_rows": int(len(group)),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        }
        record.update(numeric_summary(group["residual_log10_pred_over_observed"]))
        median_residual = float(record["median_residual_log10"])
        record["sample_signed_bias_factor"] = float(10.0**median_residual)
        record["sample_absolute_error_factor"] = float(10.0 ** abs(median_residual))
        record["sample_prediction_direction"] = direction(median_residual)
        record["sample_level_within_factor_2"] = bool(abs(median_residual) <= FACTOR_2_LOG10 + ATOL)
        record["sample_level_within_factor_10"] = bool(abs(median_residual) <= FACTOR_10_LOG10 + ATOL)
        record["sample_has_beyond_factor_10_row"] = bool(record["beyond_factor_10_count"] > 0)
        worst_index = group["abs_residual_log10"].astype(float).idxmax()
        record["worst_row_id"] = group.loc[worst_index, "row_id"]
        record["worst_row_T_K"] = float(group.loc[worst_index, "T_K"])
        record["worst_row_residual_log10"] = float(group.loc[worst_index, "residual_log10_pred_over_observed"])
        records.append(record)
    output = pd.DataFrame(records)
    output["sample_abs_median_residual_rank"] = output["median_residual_log10"].abs().rank(
        method="first", ascending=False
    ).astype("Int64")
    return output.sort_values(
        ["sample_abs_median_residual_rank", "validation_sample_group_id"]
    ).reset_index(drop=True)


def summary_record(rows: pd.DataFrame) -> dict[str, Any]:
    record: dict[str, Any] = {
        "n_rows": int(len(rows)),
        "n_samples": int(rows["validation_sample_group_id"].nunique()) if len(rows) else 0,
        "n_papers": int(rows["validation_paper_group_id"].nunique()) if len(rows) else 0,
        "n_material_labels": int(rows["material_label_normalized"].nunique()) if len(rows) else 0,
        "T_min_K": float(pd.to_numeric(rows["T_K"], errors="coerce").min()) if len(rows) else np.nan,
        "T_max_K": float(pd.to_numeric(rows["T_K"], errors="coerce").max()) if len(rows) else np.nan,
    }
    row_metrics = numeric_summary(rows["residual_log10_pred_over_observed"])
    record.update({f"row_{key}": value for key, value in row_metrics.items()})
    sample_medians = rows.groupby("validation_sample_group_id")["residual_log10_pred_over_observed"].median() if len(rows) else pd.Series(dtype=float)
    sample_metrics = numeric_summary(sample_medians)
    record.update({f"sample_equal_{key}": value for key, value in sample_metrics.items()})
    record["row_signed_bias_factor"] = float(10.0 ** row_metrics["mean_residual_log10"]) if len(rows) else np.nan
    record["sample_equal_signed_bias_factor"] = float(10.0 ** sample_metrics["mean_residual_log10"]) if len(sample_medians) else np.nan
    record["row_bias_direction"] = direction(float(row_metrics["mean_residual_log10"]))
    record["sample_equal_bias_direction"] = direction(float(sample_metrics["mean_residual_log10"]))
    record["sample_equal_beyond_factor_10_fraction"] = (
        float(np.mean(np.abs(sample_medians.to_numpy(float)) > FACTOR_10_LOG10 + ATOL))
        if len(sample_medians)
        else np.nan
    )
    record["descriptive_support_minimum_30rows_3samples_3papers"] = bool(
        record["n_rows"] >= 30 and record["n_samples"] >= 3 and record["n_papers"] >= 3
    )
    return record


def build_overall_scope_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for scope, mask in scope_masks(rows).items():
        subset = rows.loc[mask]
        record = {
            "phase5_analysis_scope": scope,
            "scope_definition": {
                "all_assigned": "All Phase 4 oxide_v1 assignments",
                "default_eligible": "Only samples accepted by default in oxide_v1",
                "review_only": "Samples held for manual classification review",
                "excluded_only": "Organic/polymer/carbon-composite exclusion candidates",
            }[scope],
        }
        record.update(summary_record(subset))
        records.append(record)
    return pd.DataFrame(records)


def build_group_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "oxide_v1_group_key",
        "oxide_v1_display_name_en",
        "oxide_v1_display_name_ja",
        "oxide_v1_rule_id",
        "oxide_v1_scope",
        "oxide_v1_default_model_action",
    ]
    metadata = rows.groupby("oxide_v1_group_key", sort=False)[metadata_columns[1:]].first().reset_index()
    records: list[dict[str, Any]] = []
    masks = scope_masks(rows)
    for scope, scope_mask in masks.items():
        scope_rows = rows.loc[scope_mask]
        for meta in metadata.itertuples(index=False):
            subset = scope_rows.loc[scope_rows["oxide_v1_group_key"].eq(meta.oxide_v1_group_key)]
            record = {
                "phase5_analysis_scope": scope,
                "oxide_v1_group_key": meta.oxide_v1_group_key,
                "oxide_v1_display_name_en": meta.oxide_v1_display_name_en,
                "oxide_v1_display_name_ja": meta.oxide_v1_display_name_ja,
                "oxide_v1_rule_id": meta.oxide_v1_rule_id,
                "oxide_v1_scope": meta.oxide_v1_scope,
                "oxide_v1_default_model_action": meta.oxide_v1_default_model_action,
            }
            record.update(summary_record(subset))
            records.append(record)
    output = pd.DataFrame(records)
    output["_scope_order"] = output["phase5_analysis_scope"].map(SCOPE_ORDER)
    return output.sort_values(
        ["_scope_order", "n_samples", "n_rows", "oxide_v1_group_key"],
        ascending=[True, False, False, True],
    ).drop(columns="_scope_order").reset_index(drop=True)


def build_primary_group_metrics(group_metrics: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    primary = group_metrics.loc[group_metrics["phase5_analysis_scope"].eq("default_eligible")].copy()
    pool = overall.loc[overall["phase5_analysis_scope"].eq("default_eligible")].iloc[0]
    primary["delta_row_mae_vs_eligible_pool"] = primary["row_mae_log10"] - float(pool["row_mae_log10"])
    primary["delta_sample_equal_mae_vs_eligible_pool"] = primary["sample_equal_mae_log10"] - float(pool["sample_equal_mae_log10"])
    primary["abs_sample_equal_mean_bias_log10"] = primary["sample_equal_mean_residual_log10"].abs()
    supported = primary["descriptive_support_minimum_30rows_3samples_3papers"] & primary["n_rows"].gt(0)
    rank_specs = [
        ("row_mae_log10", False, "row_mae_high_error_rank"),
        ("sample_equal_mae_log10", False, "sample_equal_mae_high_error_rank"),
        ("row_within_factor_2_fraction", True, "row_factor2_low_accuracy_rank"),
        ("abs_sample_equal_mean_bias_log10", False, "sample_equal_abs_bias_rank"),
    ]
    for column, ascending, output_column in rank_specs:
        primary[output_column] = pd.Series(pd.NA, index=primary.index, dtype="Int64")
        primary.loc[supported, output_column] = (
            primary.loc[supported, column].rank(method="min", ascending=ascending).astype("Int64")
        )
    return primary.sort_values(
        ["sample_equal_mae_high_error_rank", "n_samples", "oxide_v1_group_key"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_stratified_metrics(rows: pd.DataFrame, stratifier: str) -> pd.DataFrame:
    eligible = rows.loc[as_bool(rows["oxide_v1_default_model_eligible"])].copy()
    group_columns = ["oxide_v1_group_key", "oxide_v1_display_name_ja", stratifier]
    if stratifier == "T_bin_label":
        group_columns.append("T_bin_center_K")
    records = []
    for keys, subset in eligible.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record.update(summary_record(subset))
        records.append(record)
    output = pd.DataFrame(records)
    sort_columns = ["oxide_v1_group_key"] + (["T_bin_center_K"] if stratifier == "T_bin_label" else [stratifier])
    return output.sort_values(sort_columns).reset_index(drop=True)


def build_summary_table(
    rows: pd.DataFrame,
    samples: pd.DataFrame,
    primary: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    all_metrics = overall.loc[overall["phase5_analysis_scope"].eq("all_assigned")].iloc[0]
    eligible_metrics = overall.loc[overall["phase5_analysis_scope"].eq("default_eligible")].iloc[0]
    supported = primary.loc[primary["descriptive_support_minimum_30rows_3samples_3papers"] & primary["n_rows"].gt(0)]
    worst = supported.sort_values("sample_equal_mae_log10", ascending=False).iloc[0]
    best = supported.sort_values("sample_equal_mae_log10", ascending=True).iloc[0]
    entries = [
        ("phase5_date", PHASE5_DATE, "", "Aggregation definition date"),
        ("baseline_variant", PRIMARY_BASELINE, "", "Existing held-out prediction baseline; no retraining"),
        ("taxonomy_version", first_nonempty(rows["taxonomy_version"]), "", "Frozen Phase 4 classification"),
        ("taxonomy_rules_sha256", first_nonempty(rows["taxonomy_rules_sha256"]), "", "Ordered-rule hash"),
        ("all_rows", len(rows), "rows", "All broad-oxide evaluable rows assigned to oxide_v1"),
        ("all_samples", len(samples), "samples", "All assigned validation samples"),
        ("default_eligible_rows", int(as_bool(rows["oxide_v1_default_model_eligible"]).sum()), "rows", "Rows retained in the primary reaggregation"),
        ("default_eligible_samples", int(as_bool(samples["oxide_v1_default_model_eligible"]).sum()), "samples", "Samples retained in the primary reaggregation"),
        ("review_samples", int(samples["oxide_v1_inclusion_status"].eq("review").sum()), "samples", "Classification review required"),
        ("excluded_samples", int(samples["oxide_v1_inclusion_status"].eq("exclude").sum()), "samples", "Excluded from inorganic-oxide v1 by default"),
        ("all_row_mae_log10", all_metrics["row_mae_log10"], "log10", "Row-equal MAE before status filtering"),
        ("eligible_row_mae_log10", eligible_metrics["row_mae_log10"], "log10", "Row-equal MAE for default-eligible rows"),
        ("eligible_sample_equal_mae_log10", eligible_metrics["sample_equal_mae_log10"], "log10", "MAE of per-sample median signed residuals"),
        ("eligible_row_within_factor_2", eligible_metrics["row_within_factor_2_fraction"], "fraction", "Default-eligible rows within factor 2"),
        ("eligible_row_within_factor_10", eligible_metrics["row_within_factor_10_fraction"], "fraction", "Default-eligible rows within factor 10"),
        ("highest_supported_sample_equal_mae_group", worst["oxide_v1_group_key"], "", "Highest sample-equal MAE among groups meeting 30-row/3-sample/3-paper support"),
        ("highest_supported_sample_equal_mae", worst["sample_equal_mae_log10"], "log10", "Existing prediction error; not post-retraining performance"),
        ("lowest_supported_sample_equal_mae_group", best["oxide_v1_group_key"], "", "Lowest sample-equal MAE among supported default-eligible groups"),
        ("lowest_supported_sample_equal_mae", best["sample_equal_mae_log10"], "log10", "Existing prediction error; group sizes differ"),
    ]
    return pd.DataFrame(entries, columns=["metric", "value", "unit", "definition"])


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def append_check(checks: list[dict[str, str]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def write_payload(path: Path, tables: dict[str, pd.DataFrame], insights: list[str]) -> None:
    payload = {
        "title": "Phase 5 oxide_v1 existing-prediction reaggregation",
        "scope_note": "Existing broad_family_100K held-out predictions regrouped by the frozen oxide_v1 taxonomy. No retraining.",
        "insights": insights,
        "tables": {name: json_records(frame) for name, frame in tables.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_report(
    path: Path,
    summary: pd.DataFrame,
    primary: pd.DataFrame,
    overall: pd.DataFrame,
) -> None:
    metric = dict(zip(summary["metric"], summary["value"]))
    supported = primary.loc[primary["descriptive_support_minimum_30rows_3samples_3papers"] & primary["n_rows"].gt(0)].copy()
    worst = supported.sort_values("sample_equal_mae_log10", ascending=False).head(5)
    eligible = overall.loc[overall["phase5_analysis_scope"].eq("default_eligible")].iloc[0]
    lines = [
        "# Phase 5: existing predictions reaggregated by oxide_v1",
        "",
        "## Scope",
        "",
        f"- Baseline: `{PRIMARY_BASELINE}` held-out predictions from Phase 1",
        "- Taxonomy: frozen `oxide_v1` assignments from Phase 4",
        "- Model fitting: none; this phase only changes the reporting groups",
        "- Primary view: default-eligible samples",
        "- Weighting: row-equal and sample-equal metrics are both reported",
        "",
        "## Cohort",
        "",
        f"- All assigned: {int(metric['all_rows']):,} rows / {int(metric['all_samples']):,} samples",
        f"- Default eligible: {int(metric['default_eligible_rows']):,} rows / {int(metric['default_eligible_samples']):,} samples",
        f"- Review: {int(metric['review_samples']):,} samples",
        f"- Excluded: {int(metric['excluded_samples']):,} samples",
        "",
        "## Primary pooled metrics",
        "",
        f"- Row-equal MAE: {float(eligible['row_mae_log10']):.3f} log10",
        f"- Sample-equal MAE: {float(eligible['sample_equal_mae_log10']):.3f} log10",
        f"- Within factor 2: {float(eligible['row_within_factor_2_fraction']):.1%}",
        f"- Within factor 10: {float(eligible['row_within_factor_10_fraction']):.1%}",
        "",
        "## Highest sample-equal errors among supported groups",
        "",
    ]
    for row in worst.itertuples(index=False):
        lines.append(
            f"- `{row.oxide_v1_group_key}`: {row.n_samples:,} samples, "
            f"sample-equal MAE {row.sample_equal_mae_log10:.3f}, "
            f"mean residual {row.sample_equal_mean_residual_log10:+.3f} log10"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Row-equal metrics weight every temperature row equally. Sample-equal metrics first take the median signed residual within each sample and then weight samples equally. Differences between the two indicate unequal temperature-row density or within-sample error structure.",
            "",
            "These results describe the existing broad-family predictor after regrouping. They do not estimate the performance of a future oxide_v1-specific retrained model. Small groups and review/exclusion subsets should not be compared as if they had equal statistical precision.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest_before = make_manifest()

    phase1 = pd.read_csv(PHASE1_ROWS, low_memory=False)
    phase4 = pd.read_csv(PHASE4_ROWS, low_memory=False)
    config = json.loads(PHASE4_CONFIG.read_text(encoding="utf-8-sig"))
    rows, max_residual_difference = build_joined_rows(phase1, phase4)
    samples = build_sample_metrics(rows)
    overall = build_overall_scope_metrics(rows)
    group_metrics = build_group_metrics(rows)
    primary = build_primary_group_metrics(group_metrics, overall)
    carrier = build_stratified_metrics(rows, "carrier_type")
    temperature = build_stratified_metrics(rows, "T_bin_label")
    outliers = samples.loc[
        samples["median_residual_log10"].abs().gt(FACTOR_10_LOG10 + ATOL)
        | samples["sample_has_beyond_factor_10_row"]
    ].copy().reset_index(drop=True)
    summary = build_summary_table(rows, samples, primary, overall)

    phase4_class = pd.read_csv(DATA / "phase4_oxide_v1_class_summary.csv")
    phase5_all = group_metrics.loc[group_metrics["phase5_analysis_scope"].eq("all_assigned")].set_index("oxide_v1_group_key")
    phase4_class = phase4_class.set_index("oxide_v1_group_key")
    common_groups = sorted(set(phase5_all.index) & set(phase4_class.index))
    class_mae_diff = float(
        np.nanmax(
            np.abs(
                phase5_all.loc[common_groups, "row_mae_log10"].to_numpy(float)
                - phase4_class.loc[common_groups, "mae_log10"].to_numpy(float)
            )
        )
    )

    checks: list[dict[str, str]] = []
    phase1_oxide = phase1.loc[as_bool(phase1["is_oxide_broad"]) & as_bool(phase1["is_evaluable"])]
    append_check(checks, "phase1_check_passed", "FAIL " not in PHASE1_CHECK.read_text(encoding="utf-8"), PHASE1_CHECK.name)
    append_check(checks, "phase4_check_passed", "status=PASS" in PHASE4_CHECK.read_text(encoding="utf-8"), PHASE4_CHECK.name)
    append_check(checks, "taxonomy_version_fixed", rows["taxonomy_version"].eq("oxide_v1").all(), join_unique(rows["taxonomy_version"]))
    append_check(checks, "taxonomy_hash_matches_config", rows["taxonomy_rules_sha256"].eq(config["taxonomy_rules_sha256"]).all(), config["taxonomy_rules_sha256"])
    append_check(checks, "all_phase1_oxide_rows_joined", len(rows) == len(phase1_oxide) == len(phase4) == 3437, f"phase5={len(rows):,}; phase1={len(phase1_oxide):,}; phase4={len(phase4):,}")
    append_check(checks, "row_ids_match_phase1", set(rows["row_id"].astype(str)) == set(phase1_oxide["row_id"].astype(str)), f"unique={rows['row_id'].nunique():,}")
    append_check(checks, "row_ids_unique", rows["row_id"].is_unique, f"duplicates={int(rows['row_id'].duplicated().sum())}")
    append_check(checks, "classification_join_complete", rows["oxide_v1_group_key"].notna().all(), f"missing={int(rows['oxide_v1_group_key'].isna().sum())}")
    append_check(checks, "residuals_match_phase4", max_residual_difference <= ATOL, f"max_abs_difference={max_residual_difference:.3e}")
    append_check(checks, "residuals_finite", np.isfinite(rows["residual_log10_pred_over_observed"].astype(float)).all(), f"rows={len(rows):,}")
    append_check(checks, "sample_count_reconciles", len(samples) == rows["validation_sample_group_id"].nunique() == 610, f"samples={len(samples):,}")
    append_check(checks, "no_sample_assignment_conflicts", not samples["sample_assignment_conflict"].any(), f"conflicts={int(samples['sample_assignment_conflict'].sum())}")
    append_check(checks, "status_sample_counts_frozen", samples["oxide_v1_inclusion_status"].value_counts().to_dict() == {"include": 553, "review": 36, "exclude": 21}, str(samples["oxide_v1_inclusion_status"].value_counts().to_dict()))
    append_check(checks, "default_eligible_rows_frozen", int(as_bool(rows["oxide_v1_default_model_eligible"]).sum()) == 3032, f"rows={int(as_bool(rows['oxide_v1_default_model_eligible']).sum()):,}")
    all_group_rows = int(group_metrics.loc[group_metrics["phase5_analysis_scope"].eq("all_assigned"), "n_rows"].sum())
    eligible_group_rows = int(group_metrics.loc[group_metrics["phase5_analysis_scope"].eq("default_eligible"), "n_rows"].sum())
    append_check(checks, "all_group_rows_reconcile", all_group_rows == len(rows), f"sum={all_group_rows:,}")
    append_check(checks, "eligible_group_rows_reconcile", eligible_group_rows == 3032, f"sum={eligible_group_rows:,}")
    scope_counts = overall.set_index("phase5_analysis_scope")["n_rows"].to_dict()
    append_check(checks, "scope_rows_partition", scope_counts["default_eligible"] + scope_counts["review_only"] + scope_counts["excluded_only"] == scope_counts["all_assigned"], str(scope_counts))
    append_check(checks, "phase4_class_mae_reproduced", class_mae_diff <= ATOL, f"max_abs_difference={class_mae_diff:.3e}")
    append_check(checks, "primary_contains_all_groups", len(primary) == rows["oxide_v1_group_key"].nunique() == 14, f"groups={len(primary)}")
    append_check(checks, "ranked_groups_meet_support", primary.loc[primary["sample_equal_mae_high_error_rank"].notna(), "descriptive_support_minimum_30rows_3samples_3papers"].all(), f"ranked={int(primary['sample_equal_mae_high_error_rank'].notna().sum())}")
    manifest_after = make_manifest()
    append_check(checks, "inputs_unchanged_during_build", manifest_before["sha256"].tolist() == manifest_after["sha256"].tolist(), "SHA-256 before and after build")
    validation = pd.DataFrame(checks)

    outputs = {
        "phase5_oxide_v1_inventory_summary.csv": summary,
        "phase5_oxide_v1_overall_scope_metrics.csv": overall,
        "phase5_oxide_v1_group_metrics.csv": group_metrics,
        "phase5_oxide_v1_primary_group_metrics.csv": primary,
        "phase5_oxide_v1_carrier_metrics.csv": carrier,
        "phase5_oxide_v1_temperature_metrics.csv": temperature,
        "phase5_oxide_v1_sample_metrics.csv": samples,
        "phase5_oxide_v1_outlier_samples.csv": outliers,
        "phase5_oxide_v1_existing_prediction_rows.csv": rows,
        "phase5_input_manifest.csv": manifest_after,
        "phase5_validation_checks.csv": validation,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    metric = dict(zip(summary["metric"], summary["value"]))
    supported = primary.loc[primary["descriptive_support_minimum_30rows_3samples_3papers"] & primary["n_rows"].gt(0)]
    worst = supported.sort_values("sample_equal_mae_log10", ascending=False).iloc[0]
    most_biased = supported.sort_values("abs_sample_equal_mean_bias_log10", ascending=False).iloc[0]
    insights = [
        f"Primary cohort: {int(metric['default_eligible_rows']):,} rows and {int(metric['default_eligible_samples']):,} samples. No retraining.",
        f"Pooled MAE: {float(metric['eligible_row_mae_log10']):.3f} row-equal and {float(metric['eligible_sample_equal_mae_log10']):.3f} sample-equal log10.",
        f"Highest supported sample-equal MAE: {worst.oxide_v1_display_name_ja}, {worst.sample_equal_mae_log10:.3f} ({int(worst.n_samples)} samples).",
        f"Largest supported sample-equal mean bias: {most_biased.oxide_v1_display_name_ja}, {most_biased.sample_equal_mean_residual_log10:+.3f} log10.",
        "Read both weightings because sample temperature grids are unequal.",
    ]
    payload_tables = {
        "summary": summary,
        "primary_groups": primary,
        "overall_scopes": overall,
        "all_group_scopes": group_metrics,
        "carrier": carrier,
        "temperature": temperature,
        "outliers": outliers,
        "samples": samples,
        "rows": rows,
        "validation": validation,
        "manifest": manifest_after,
    }
    write_payload(args.workbook_payload, payload_tables, insights)
    write_report(args.report, summary, primary, overall)

    args.validation_log.write_text(
        "\n".join(
            [
                "Phase 5 oxide_v1 reaggregation validation",
                f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}",
                f"date={PHASE5_DATE}",
                f"baseline={PRIMARY_BASELINE}",
                f"rows={len(rows)}",
                f"samples={len(samples)}",
                f"default_eligible_rows={int(as_bool(rows['oxide_v1_default_model_eligible']).sum())}",
                f"default_eligible_samples={int(as_bool(samples['oxide_v1_default_model_eligible']).sum())}",
                f"failed_checks={int(validation['status'].eq('FAIL').sum())}",
            ]
            + [f"{row.check}: {row.status} | {row.detail}" for row in validation.itertuples(index=False)]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Phase 5 reaggregation complete: rows={len(rows):,}, samples={len(samples):,}, "
        f"eligible_rows={int(as_bool(rows['oxide_v1_default_model_eligible']).sum()):,}, "
        f"eligible_samples={int(as_bool(samples['oxide_v1_default_model_eligible']).sum()):,}, "
        f"status={'PASS' if validation['status'].eq('PASS').all() else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
