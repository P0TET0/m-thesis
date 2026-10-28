"""Rebuild conductivity predictions after changing only the oxide taxonomy.

Phase 6 keeps the Phase 0 broad-family 100 K protocol fixed.  The only model
input changed is ``material_group_key`` for rows that were previously assigned
to ``broad::oxide``.  Those rows receive the frozen Phase 4 ``oxide_v1`` key;
all non-oxide keys, holdout assignments, temperature bins, reference subset,
aggregation method, and reliability thresholds remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp006 import build_step5b_assign_predictions as step5b
from oxide_taxonomy_v1 import TAXONOMY_VERSION, classify_oxide_v1, rules_sha256


EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"

SOURCE_ROWS = ROOT / "experiments" / "exp006" / "data" / "processed" / "step6a_validation_rows_with_splits_key_broad_family.csv"
BASELINE_PREDICTIONS = ROOT / "experiments" / "exp006" / "data" / "processed" / "step6b_broad_family" / "step5b_test_predictions_default.csv"
PHASE0_METRICS = DATA / "phase0_baseline_metrics.csv"
PHASE0_CHECK = REPORTS / "phase0_check_log.txt"
PHASE4_ASSIGNMENTS = DATA / "phase4_oxide_v1_row_assignments.csv"
PHASE4_CONFIG = DATA / "phase4_oxide_v1_classification_config.json"
PHASE4_CHECK = REPORTS / "phase4_check_log.txt"
TAXONOMY_MODULE = EXP007 / "oxide_taxonomy_v1.py"

DEFAULT_OUTPUT_DIR = DATA
DEFAULT_REPORT = REPORTS / "phase6_oxide_v1_reprediction.md"
DEFAULT_VALIDATION_LOG = REPORTS / "phase6_validation_log.txt"
DEFAULT_WORKBOOK_PAYLOAD = REPORTS / ".phase6_workbook_payload.json"

PHASE6_DATE = "2026-10-14"
BASELINE_VARIANT = "broad_family_100K"
NEW_VARIANT = "oxide_v1_100K"
BASELINE_CONFIG_ID = "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
NEW_CONFIG_ID = "phase6__sample_holdout__ref_conservative_valid__eval_all_valid__material_family_oxide_v1__sample_median"
ATOL = 1e-12
FACTOR_2_LOG10 = math.log10(2.0)
FACTOR_5_LOG10 = math.log10(5.0)

OUTPUT_FILES = {
    "assignments": "phase6_oxide_v1_all_row_assignments.csv",
    "reference": "phase6_oxide_v1_train_reference_bins.csv",
    "predictions": "phase6_oxide_v1_test_predictions.csv",
    "comparison_rows": "phase6_prediction_comparison_rows.csv",
    "overall_metrics": "phase6_overall_metrics.csv",
    "paired_metrics": "phase6_paired_metrics.csv",
    "metric_deltas": "phase6_metric_deltas.csv",
    "scope_metrics": "phase6_oxide_scope_metrics.csv",
    "group_metrics": "phase6_oxide_group_metrics.csv",
    "coverage": "phase6_oxide_group_coverage.csv",
    "sample_changes": "phase6_sample_prediction_changes.csv",
    "inventory": "phase6_inventory_summary.csv",
    "manifest": "phase6_input_manifest.csv",
    "validation": "phase6_validation_checks.csv",
}


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


def clean_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def first_nonempty(values: Iterable[Any]) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return value
    return None


def join_unique(values: Iterable[Any], limit: int = 10) -> str:
    unique: list[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in unique:
            unique.append(text)
    return " | ".join(unique[:limit]) + (f" | ... (+{len(unique) - limit})" if len(unique) > limit else "")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_inputs() -> None:
    required = [
        SOURCE_ROWS,
        BASELINE_PREDICTIONS,
        PHASE0_METRICS,
        PHASE0_CHECK,
        PHASE4_ASSIGNMENTS,
        PHASE4_CONFIG,
        PHASE4_CHECK,
        TAXONOMY_MODULE,
    ]
    missing = [relative(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 6 inputs: {missing}")
    phase0_check_text = PHASE0_CHECK.read_text(encoding="utf-8")
    if "FAIL " in phase0_check_text or "FAIL:" in phase0_check_text:
        raise RuntimeError("Phase 0 independent check is not PASS")
    if "status=PASS" not in PHASE4_CHECK.read_text(encoding="utf-8"):
        raise RuntimeError("Phase 4 independent check is not PASS")


def make_manifest() -> pd.DataFrame:
    sources = [
        ("Frozen broad-family validation rows and holdout assignments", SOURCE_ROWS),
        ("Frozen broad-family 100 K baseline predictions", BASELINE_PREDICTIONS),
        ("Phase 0 baseline metrics", PHASE0_METRICS),
        ("Phase 0 independent check", PHASE0_CHECK),
        ("Phase 4 held-out oxide assignments", PHASE4_ASSIGNMENTS),
        ("Phase 4 taxonomy configuration", PHASE4_CONFIG),
        ("Phase 4 independent check", PHASE4_CHECK),
        ("oxide_v1 classification module", TAXONOMY_MODULE),
    ]
    return pd.DataFrame(
        [
            {
                "role": role,
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in sources
        ]
    )


def apply_oxide_v1(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = source.copy()
    rows["material_group_key_phase0"] = rows["material_group_key"].astype(str)
    oxide_mask = rows["material_group_key_phase0"].eq("broad::oxide")
    assignments: list[dict[str, Any]] = []
    for record in rows.loc[
        oxide_mask,
        [
            "row_id",
            "validation_sample_group_id",
            "validation_paper_group_id",
            "sample_holdout_split",
            "formula_raw",
            "material_name_raw",
            "parsed_elements",
        ],
    ].itertuples(index=False):
        label = clean_label(record.formula_raw) or clean_label(record.material_name_raw)
        classification = classify_oxide_v1(label, record.parsed_elements)
        assignments.append(
            {
                "row_id": str(record.row_id),
                "validation_sample_group_id": record.validation_sample_group_id,
                "validation_paper_group_id": record.validation_paper_group_id,
                "sample_holdout_split": record.sample_holdout_split,
                "formula_raw": record.formula_raw,
                "material_name_raw": record.material_name_raw,
                "material_label_normalized": label,
                "parsed_elements": record.parsed_elements,
                "material_group_key_phase0": "broad::oxide",
                "material_group_key_phase6": classification["oxide_v1_group_key"],
                **classification,
            }
        )
    assignment_df = pd.DataFrame(assignments)
    mapping = assignment_df.set_index("row_id")["material_group_key_phase6"]
    rows.loc[oxide_mask, "material_group_key"] = rows.loc[oxide_mask, "row_id"].astype(str).map(mapping)
    if rows.loc[oxide_mask, "material_group_key"].isna().any():
        raise RuntimeError("oxide_v1 assignment is missing for one or more broad::oxide rows")
    return rows, assignment_df


def build_new_predictions(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    step5b.validate_columns(source)
    usable, dropped_raw = step5b.filter_usable_rows(source)
    config = pd.DataFrame(
        [
            {
                "config_id": NEW_CONFIG_ID,
                "split_scheme": "sample_holdout",
                "reference_source_subset": "conservative_valid",
                "eval_target_subset": "all_valid",
                "group_scheme": "material_family",
                "curve_method": "sample_median",
            }
        ]
    )
    reference = step5b.build_reference_curves(usable, config, min_rows=3, min_samples=3, min_papers=1)
    predictions = step5b.assign_test_predictions(usable, reference, config)
    coverage = step5b.build_coverage_summary(predictions, reference)
    dropped = step5b.build_dropped_output(dropped_raw)
    return reference, predictions, coverage, dropped


def metric_record(rows: pd.DataFrame, weighting: str) -> dict[str, Any]:
    if weighting == "sample_equal":
        source = (
            rows.groupby("validation_sample_group_id", dropna=False, sort=False)
            .agg(
                residual=("log10_sigma_pred_over_exp", "median"),
                validation_paper_group_id=("validation_paper_group_id", "first"),
            )
            .reset_index()
        )
    else:
        source = rows[["validation_sample_group_id", "validation_paper_group_id", "log10_sigma_pred_over_exp"]].rename(
            columns={"log10_sigma_pred_over_exp": "residual"}
        )
    values = pd.to_numeric(source["residual"], errors="coerce").dropna().to_numpy(float)
    absolute = np.abs(values)
    return {
        "metric_weighting": weighting,
        "n_rows": int(len(rows) if weighting == "row_equal" else len(source)),
        "n_source_rows": int(len(rows)),
        "n_samples": int(rows["validation_sample_group_id"].nunique()),
        "n_papers": int(rows["validation_paper_group_id"].nunique()),
        "mean_log10_error": float(np.mean(values)),
        "median_log10_error": float(np.median(values)),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "max_abs_log10_error": float(np.max(absolute)),
        "overprediction_fraction": float(np.mean(values > ATOL)),
        "underprediction_fraction": float(np.mean(values < -ATOL)),
        "factor_2_accuracy": float(np.mean(absolute <= FACTOR_2_LOG10 + ATOL)),
        "factor_5_accuracy": float(np.mean(absolute <= FACTOR_5_LOG10 + ATOL)),
        "factor_10_accuracy": float(np.mean(absolute <= 1.0 + ATOL)),
    }


def metric_rows(rows: pd.DataFrame, **dimensions: Any) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    return [{**dimensions, **metric_record(rows, weighting)} for weighting in ["row_equal", "sample_equal"]]


def prepare_baseline() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_PREDICTIONS, low_memory=False)
    baseline = baseline.loc[baseline["config_id"].eq(BASELINE_CONFIG_ID)].copy()
    baseline["prediction_variant"] = BASELINE_VARIANT
    baseline["prediction_available"] = baseline["prediction_status"].eq("ok")
    return baseline


def attach_assignment_metadata(rows: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "row_id",
        "oxide_v1_group_key",
        "oxide_v1_display_name_en",
        "oxide_v1_display_name_ja",
        "oxide_v1_inclusion_status",
        "oxide_v1_default_model_eligible",
        "oxide_v1_review_priority",
        "oxide_v1_review_reasons",
    ]
    return rows.merge(assignments[columns], on="row_id", how="left", validate="one_to_one")


def build_comparison_rows(baseline: pd.DataFrame, new: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "row_id",
        "paper_id",
        "doi",
        "sample_id",
        "sample_key",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "formula_raw",
        "material_name_raw",
        "carrier_type",
        "T_K",
        "T_bin_label",
        "T_bin_center_K",
        "S_abs_uV_per_K",
        "sigma_S_per_m",
    ]
    old_cols = id_cols + [
        "material_group_key_for_prediction",
        "prediction_status",
        "sigma0_ref_S_per_m",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_over_exp",
        "train_row_count",
        "train_sample_count",
        "train_paper_count",
    ]
    new_cols = [
        "row_id",
        "material_group_key_for_prediction",
        "prediction_status",
        "sigma0_ref_S_per_m",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_over_exp",
        "train_row_count",
        "train_sample_count",
        "train_paper_count",
    ]
    old = baseline[old_cols].rename(
        columns={
            "material_group_key_for_prediction": "baseline_group_key",
            "prediction_status": "baseline_prediction_status",
            "sigma0_ref_S_per_m": "baseline_sigma0_ref_S_per_m",
            "sigma_pred_S_per_m": "baseline_sigma_pred_S_per_m",
            "log10_sigma_pred_over_exp": "baseline_residual_log10",
            "train_row_count": "baseline_train_row_count",
            "train_sample_count": "baseline_train_sample_count",
            "train_paper_count": "baseline_train_paper_count",
        }
    )
    changed = new[new_cols].rename(
        columns={
            "material_group_key_for_prediction": "phase6_group_key",
            "prediction_status": "phase6_prediction_status",
            "sigma0_ref_S_per_m": "phase6_sigma0_ref_S_per_m",
            "sigma_pred_S_per_m": "phase6_sigma_pred_S_per_m",
            "log10_sigma_pred_over_exp": "phase6_residual_log10",
            "train_row_count": "phase6_train_row_count",
            "train_sample_count": "phase6_train_sample_count",
            "train_paper_count": "phase6_train_paper_count",
        }
    )
    comparison = old.merge(changed, on="row_id", how="outer", validate="one_to_one")
    comparison = attach_assignment_metadata(comparison, assignments)
    comparison["is_phase0_broad_oxide"] = comparison["baseline_group_key"].eq("broad::oxide")
    comparison["baseline_prediction_available"] = comparison["baseline_prediction_status"].eq("ok")
    comparison["phase6_prediction_available"] = comparison["phase6_prediction_status"].eq("ok")
    comparison["paired_prediction_available"] = comparison["baseline_prediction_available"] & comparison["phase6_prediction_available"]
    comparison["delta_log10_prediction_phase6_minus_baseline"] = np.where(
        comparison["paired_prediction_available"],
        np.log10(comparison["phase6_sigma_pred_S_per_m"] / comparison["baseline_sigma_pred_S_per_m"]),
        np.nan,
    )
    comparison["delta_abs_error_phase6_minus_baseline"] = np.where(
        comparison["paired_prediction_available"],
        comparison["phase6_residual_log10"].abs() - comparison["baseline_residual_log10"].abs(),
        np.nan,
    )
    comparison["phase6_error_change"] = np.select(
        [
            comparison["delta_abs_error_phase6_minus_baseline"] < -ATOL,
            comparison["delta_abs_error_phase6_minus_baseline"] > ATOL,
            comparison["paired_prediction_available"],
        ],
        ["improved", "worsened", "unchanged"],
        default="not_paired",
    )
    return comparison.sort_values(["is_phase0_broad_oxide", "row_id"], ascending=[False, True]).reset_index(drop=True)


def variant_rows(comparison: pd.DataFrame, variant: str, mask: pd.Series) -> pd.DataFrame:
    prefix = "baseline" if variant == BASELINE_VARIANT else "phase6"
    selected = comparison.loc[mask & comparison[f"{prefix}_prediction_available"]].copy()
    selected["log10_sigma_pred_over_exp"] = selected[f"{prefix}_residual_log10"]
    return selected


def build_overall_metrics(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall_records: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    all_mask = pd.Series(True, index=comparison.index)
    paired_mask = comparison["paired_prediction_available"]
    for variant in [BASELINE_VARIANT, NEW_VARIANT]:
        available = variant_rows(comparison, variant, all_mask)
        coverage = float(len(available) / len(comparison))
        for record in metric_rows(available, prediction_variant=variant, comparison_basis="available_as_is"):
            record.update({"test_rows": len(comparison), "prediction_ok_rows": len(available), "coverage_fraction": coverage})
            overall_records.append(record)
        paired = variant_rows(comparison, variant, paired_mask)
        for record in metric_rows(paired, prediction_variant=variant, comparison_basis="paired_common_rows"):
            record.update({"test_rows": len(comparison), "prediction_ok_rows": len(paired), "coverage_fraction": float(len(paired) / len(comparison))})
            paired_records.append(record)

    overall = pd.DataFrame(overall_records)
    paired = pd.DataFrame(paired_records)
    delta_records: list[dict[str, Any]] = []
    metric_names = [
        "mean_log10_error",
        "median_log10_error",
        "mae_log10",
        "rmse_log10",
        "factor_2_accuracy",
        "factor_5_accuracy",
        "factor_10_accuracy",
        "coverage_fraction",
    ]
    for basis, table in [("available_as_is", overall), ("paired_common_rows", paired)]:
        for weighting in ["row_equal", "sample_equal"]:
            old = table[(table["prediction_variant"].eq(BASELINE_VARIANT)) & table["metric_weighting"].eq(weighting)].iloc[0]
            new = table[(table["prediction_variant"].eq(NEW_VARIANT)) & table["metric_weighting"].eq(weighting)].iloc[0]
            for metric in metric_names:
                delta_records.append(
                    {
                        "comparison_basis": basis,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "baseline_value": float(old[metric]),
                        "phase6_value": float(new[metric]),
                        "delta_phase6_minus_baseline": float(new[metric] - old[metric]),
                        "lower_or_higher_is_better": "lower" if metric in {"mae_log10", "rmse_log10"} else "higher" if "accuracy" in metric or metric == "coverage_fraction" else "context",
                    }
                )
    return overall, paired, pd.DataFrame(delta_records)


def scope_masks(comparison: pd.DataFrame) -> dict[str, pd.Series]:
    oxide = as_bool(comparison["is_phase0_broad_oxide"])
    eligible = as_bool(comparison["oxide_v1_default_model_eligible"])
    status = comparison["oxide_v1_inclusion_status"].astype(str)
    return {
        "all_oxide_assigned": oxide,
        "default_eligible": oxide & eligible,
        "review_only": oxide & status.eq("review"),
        "excluded_only": oxide & status.eq("exclude"),
    }


def build_scope_metrics(comparison: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for scope, scope_mask in scope_masks(comparison).items():
        denominator = int(scope_mask.sum())
        for basis, basis_mask in [
            ("available_as_is", scope_mask),
            ("paired_common_rows", scope_mask & comparison["paired_prediction_available"]),
        ]:
            for variant in [BASELINE_VARIANT, NEW_VARIANT]:
                available = variant_rows(comparison, variant, basis_mask)
                for record in metric_rows(
                    available,
                    analysis_scope=scope,
                    comparison_basis=basis,
                    prediction_variant=variant,
                ):
                    record.update(
                        {
                            "test_rows_in_scope": denominator,
                            "prediction_ok_rows_in_scope": len(available),
                            "coverage_fraction_in_scope": float(len(available) / denominator) if denominator else np.nan,
                        }
                    )
                    records.append(record)
    return pd.DataFrame(records)


def build_group_metrics(comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    oxide = comparison.loc[as_bool(comparison["is_phase0_broad_oxide"])].copy()
    metric_records: list[dict[str, Any]] = []
    coverage_records: list[dict[str, Any]] = []
    group_cols = [
        "oxide_v1_group_key",
        "oxide_v1_display_name_en",
        "oxide_v1_display_name_ja",
    ]
    for keys, group in oxide.groupby(group_cols, dropna=False, sort=False):
        dims = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        denominator = len(group)
        baseline_ok = int(group["baseline_prediction_available"].sum())
        phase6_ok = int(group["phase6_prediction_available"].sum())
        paired_ok = int(group["paired_prediction_available"].sum())
        coverage_records.append(
            {
                **dims,
                "test_rows": denominator,
                "n_samples": group["validation_sample_group_id"].nunique(),
                "n_papers": group["validation_paper_group_id"].nunique(),
                "default_eligible_rows": int(as_bool(group["oxide_v1_default_model_eligible"]).sum()),
                "review_rows": int(group["oxide_v1_inclusion_status"].astype(str).eq("review").sum()),
                "excluded_rows": int(group["oxide_v1_inclusion_status"].astype(str).eq("exclude").sum()),
                "baseline_prediction_ok_rows": baseline_ok,
                "phase6_prediction_ok_rows": phase6_ok,
                "paired_prediction_rows": paired_ok,
                "baseline_coverage_fraction": baseline_ok / denominator,
                "phase6_coverage_fraction": phase6_ok / denominator,
                "coverage_delta_phase6_minus_baseline": (phase6_ok - baseline_ok) / denominator,
            }
        )
        eligible_mask = as_bool(group["oxide_v1_default_model_eligible"])
        for analysis_scope, scope_mask in [
            ("all_assigned", pd.Series(True, index=group.index)),
            ("default_eligible", eligible_mask),
        ]:
            scope_denominator = int(scope_mask.sum())
            if scope_denominator == 0:
                continue
            for basis, mask in [
                ("available_as_is", scope_mask),
                ("paired_common_rows", scope_mask & group["paired_prediction_available"]),
            ]:
                for variant in [BASELINE_VARIANT, NEW_VARIANT]:
                    subset = variant_rows(group, variant, mask)
                    for record in metric_rows(
                        subset,
                        **dims,
                        analysis_scope=analysis_scope,
                        comparison_basis=basis,
                        prediction_variant=variant,
                    ):
                        record["test_rows_in_group_scope"] = scope_denominator
                        metric_records.append(record)
    metrics = pd.DataFrame(metric_records)
    coverage = pd.DataFrame(coverage_records).sort_values("test_rows", ascending=False).reset_index(drop=True)
    return metrics, coverage


def build_sample_changes(comparison: pd.DataFrame) -> pd.DataFrame:
    paired = comparison.loc[comparison["paired_prediction_available"]].copy()
    records: list[dict[str, Any]] = []
    for sample_id, group in paired.groupby("validation_sample_group_id", sort=False):
        old = float(group["baseline_residual_log10"].median())
        new = float(group["phase6_residual_log10"].median())
        records.append(
            {
                "validation_sample_group_id": sample_id,
                "validation_paper_group_id": first_nonempty(group["validation_paper_group_id"]),
                "doi": first_nonempty(group["doi"]),
                "formula_raw": join_unique(group["formula_raw"]),
                "material_name_raw": join_unique(group["material_name_raw"]),
                "is_phase0_broad_oxide": bool(group["is_phase0_broad_oxide"].all()),
                "oxide_v1_group_key": first_nonempty(group["oxide_v1_group_key"]),
                "oxide_v1_display_name_ja": first_nonempty(group["oxide_v1_display_name_ja"]),
                "oxide_v1_inclusion_status": first_nonempty(group["oxide_v1_inclusion_status"]),
                "n_paired_rows": len(group),
                "baseline_median_residual_log10": old,
                "phase6_median_residual_log10": new,
                "baseline_abs_median_error_log10": abs(old),
                "phase6_abs_median_error_log10": abs(new),
                "delta_abs_median_error_phase6_minus_baseline": abs(new) - abs(old),
                "delta_median_prediction_log10_phase6_minus_baseline": new - old,
            }
        )
    output = pd.DataFrame(records)
    output["sample_error_change"] = np.select(
        [
            output["delta_abs_median_error_phase6_minus_baseline"] < -ATOL,
            output["delta_abs_median_error_phase6_minus_baseline"] > ATOL,
        ],
        ["improved", "worsened"],
        default="unchanged",
    )
    output["abs_error_change_rank"] = output["delta_abs_median_error_phase6_minus_baseline"].abs().rank(method="first", ascending=False).astype("Int64")
    return output.sort_values(["is_phase0_broad_oxide", "abs_error_change_rank"], ascending=[False, True]).reset_index(drop=True)


def build_inventory(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    oxide = as_bool(comparison["is_phase0_broad_oxide"])
    paired_oxide = oxide & comparison["paired_prediction_available"]
    changed = comparison.loc[paired_oxide, "delta_log10_prediction_phase6_minus_baseline"].abs().gt(ATOL)
    items = [
        ("source_rows", len(source), "rows", "Frozen Step6A broad-family validation rows"),
        ("source_samples", source["validation_sample_group_id"].nunique(), "samples", "All train and test samples"),
        ("phase0_material_groups", source["material_group_key_phase0"].nunique(), "groups", "Original broad-family keys"),
        ("phase6_material_groups", source["material_group_key"].nunique(), "groups", "Non-oxide keys unchanged; broad::oxide replaced by oxide_v1"),
        ("oxide_rows_reclassified", len(assignments), "rows", "Train and test broad::oxide rows"),
        ("oxide_samples_reclassified", assignments["validation_sample_group_id"].nunique(), "samples", "Train and test broad::oxide samples"),
        ("oxide_v1_nonempty_groups", assignments["oxide_v1_group_key"].nunique(), "groups", "Composition-only oxide_v1 groups"),
        ("new_reference_bins", len(reference), "bins", "All phase6 material/carrier/100 K bins"),
        ("test_rows", len(comparison), "rows", "Frozen sample-holdout test cohort"),
        ("baseline_prediction_ok_rows", int(comparison["baseline_prediction_available"].sum()), "rows", "broad_family_100K"),
        ("phase6_prediction_ok_rows", int(comparison["phase6_prediction_available"].sum()), "rows", "oxide_v1_100K"),
        ("paired_prediction_rows", int(comparison["paired_prediction_available"].sum()), "rows", "Predicted by both variants"),
        ("paired_oxide_rows", int(paired_oxide.sum()), "rows", "Oxide rows predicted by both variants"),
        ("paired_oxide_rows_with_changed_prediction", int(changed.sum()), "rows", "Absolute log10 prediction delta > 1e-12"),
        ("taxonomy_version", TAXONOMY_VERSION, "text", "Frozen Phase 4 taxonomy"),
        ("taxonomy_rules_sha256", rules_sha256(), "sha256", "Hash of ordered oxide_v1 rules"),
    ]
    return pd.DataFrame(items, columns=["metric", "value", "unit", "definition"])


def append_check(checks: list[dict[str, str]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def build_validation(
    source: pd.DataFrame,
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    phase4: pd.DataFrame,
    overall: pd.DataFrame,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    oxide_mask = source["material_group_key_phase0"].eq("broad::oxide")
    nonoxide = ~oxide_mask
    append_check(checks, "source_row_count_frozen", len(source) == 97086, f"rows={len(source)}")
    append_check(checks, "test_row_count_frozen", len(comparison) == 19008, f"rows={len(comparison)}")
    append_check(checks, "oxide_row_count_all_splits", len(assignments) == int(oxide_mask.sum()) == 17715, f"rows={len(assignments)}")
    append_check(checks, "all_oxide_rows_assigned", assignments["oxide_v1_group_key"].notna().all(), f"missing={int(assignments['oxide_v1_group_key'].isna().sum())}")
    append_check(checks, "nonoxide_keys_unchanged", source.loc[nonoxide, "material_group_key"].equals(source.loc[nonoxide, "material_group_key_phase0"]), f"rows={int(nonoxide.sum())}")
    append_check(checks, "broad_oxide_key_removed", not source["material_group_key"].eq("broad::oxide").any(), f"remaining={int(source['material_group_key'].eq('broad::oxide').sum())}")
    append_check(checks, "taxonomy_version_frozen", assignments["taxonomy_version"].eq(TAXONOMY_VERSION).all() and assignments["taxonomy_rules_sha256"].eq(rules_sha256()).all(), rules_sha256())
    test_assign = assignments.loc[assignments["sample_holdout_split"].eq("test"), ["row_id", "oxide_v1_group_key"]].set_index("row_id").sort_index()
    frozen_assign = phase4[["row_id", "oxide_v1_group_key"]].set_index("row_id").sort_index()
    append_check(checks, "heldout_assignments_match_phase4", test_assign.equals(frozen_assign), f"phase6={len(test_assign)}; phase4={len(frozen_assign)}")
    append_check(checks, "prediction_rows_complete", len(predictions) == 19008 and predictions["row_id"].is_unique, f"rows={len(predictions)}")
    append_check(checks, "reference_uses_train_only_keys", set(reference["material_group_key"]).issubset(set(source.loc[source["sample_holdout_split"].eq("train"), "material_group_key"])), f"bins={len(reference)}")
    append_check(checks, "prediction_status_allowed", set(predictions["prediction_status"]).issubset(step5b.PREDICTION_STATUSES), str(predictions["prediction_status"].value_counts().to_dict()))
    new_ok = predictions["prediction_status"].eq("ok")
    append_check(checks, "new_predictions_positive", (predictions.loc[new_ok, "sigma_pred_S_per_m"] > 0).all(), f"ok_rows={int(new_ok.sum())}")
    append_check(checks, "prediction_formula_recomputed", np.allclose(predictions.loc[new_ok, "sigma_pred_S_per_m"], predictions.loc[new_ok, "sigma0_ref_S_per_m"] * predictions.loc[new_ok, "F0_eta"], rtol=1e-10, atol=0), f"ok_rows={int(new_ok.sum())}")
    append_check(checks, "sample_holdout_no_leakage", not source.loc[source["sample_holdout_split"].eq("train"), "validation_sample_group_id"].isin(source.loc[source["sample_holdout_split"].eq("test"), "validation_sample_group_id"]).any(), "train versus test validation_sample_group_id")
    nonoxide_paired = comparison.loc[~comparison["is_phase0_broad_oxide"] & comparison["paired_prediction_available"]]
    nonoxide_delta = nonoxide_paired["delta_log10_prediction_phase6_minus_baseline"].abs()
    append_check(checks, "nonoxide_predictions_unchanged", bool((nonoxide_delta <= ATOL).all()), f"paired_rows={len(nonoxide_paired)}; max_abs_delta={float(nonoxide_delta.max() if len(nonoxide_delta) else 0):.3e}")
    baseline_metric = overall[(overall["prediction_variant"].eq(BASELINE_VARIANT)) & overall["metric_weighting"].eq("row_equal")].iloc[0]
    frozen_metric = pd.read_csv(PHASE0_METRICS)
    frozen_metric = frozen_metric[(frozen_metric["variant"].eq(BASELINE_VARIANT)) & frozen_metric["metric_weighting"].eq("row_equal")].iloc[0]
    append_check(checks, "baseline_metrics_reproduce_phase0", abs(float(baseline_metric["mae_log10"]) - float(frozen_metric["mae_log10"])) <= ATOL and int(baseline_metric["prediction_ok_rows"]) == int(frozen_metric["prediction_ok_rows"]), f"mae_diff={abs(float(baseline_metric['mae_log10']) - float(frozen_metric['mae_log10'])):.3e}")
    append_check(checks, "overall_metrics_complete", len(overall) == 4, f"rows={len(overall)}")
    return pd.DataFrame(checks)


def df_to_markdown(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "n/a"
    text = df.head(max_rows).copy()
    for column in text.columns:
        text[column] = text[column].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(str(row[column]) for column in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    inventory: pd.DataFrame,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    deltas: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paired_row = paired[paired["metric_weighting"].eq("row_equal")].set_index("prediction_variant")
    new_available = overall[(overall["prediction_variant"].eq(NEW_VARIANT)) & overall["metric_weighting"].eq("row_equal")].iloc[0]
    old_paired = paired_row.loc[BASELINE_VARIANT]
    new_paired = paired_row.loc[NEW_VARIANT]
    lines = [
        "# Phase 6 oxide_v1 re-prediction",
        "",
        "## Fixed comparison design",
        "",
        "- Changed factor: broad::oxide material keys were replaced by the frozen composition-only oxide_v1 keys.",
        "- Fixed factors: sample holdout assignments, 100 K bins, conservative-valid training references, all-valid test target, sample-median reference aggregation, and 3 rows / 3 samples / 1 paper bin thresholds.",
        "- Cohort policy: review and exclude labels were retained as separate predictive groups so the classification change is the only experimental intervention.",
        "- Prediction mechanism: train-only median sigma0 reference curves; no test-row sigma0 value was used to make a prediction.",
        "",
        "## Main result",
        "",
        f"- Phase 6 coverage: {int(new_available['prediction_ok_rows']):,}/{int(new_available['test_rows']):,} rows ({float(new_available['coverage_fraction']):.3%}).",
        f"- Paired common-row MAE: {float(old_paired['mae_log10']):.6f} -> {float(new_paired['mae_log10']):.6f} log10.",
        f"- Paired common-row RMSE: {float(old_paired['rmse_log10']):.6f} -> {float(new_paired['rmse_log10']):.6f} log10.",
        f"- Paired common-row factor-2 accuracy: {float(old_paired['factor_2_accuracy']):.3%} -> {float(new_paired['factor_2_accuracy']):.3%}.",
        f"- Paired common-row factor-10 accuracy: {float(old_paired['factor_10_accuracy']):.3%} -> {float(new_paired['factor_10_accuracy']):.3%}.",
        "",
        "These paired metrics isolate the prediction change from differences in coverage. Available-as-is metrics remain necessary for deployment-like interpretation.",
        "",
        "## Inventory",
        "",
        df_to_markdown(inventory),
        "",
        "## Available-as-is overall metrics",
        "",
        df_to_markdown(overall),
        "",
        "## Paired common-row metrics",
        "",
        df_to_markdown(paired),
        "",
        "## Metric deltas",
        "",
        df_to_markdown(deltas),
        "",
        "## Oxide-group coverage",
        "",
        df_to_markdown(coverage),
        "",
        "## Validation",
        "",
        df_to_markdown(validation),
        "",
        "## Interpretation limit",
        "",
        "The experiment measures the effect of the oxide_v1 grouping under the frozen Phase 0 reference-curve protocol. It does not test a new regression algorithm, new descriptors, 25 K bins, or manual crystal-structure labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.replace({np.nan: None, np.inf: None, -np.inf: None}).to_json(orient="records"))


def write_payload(
    path: Path,
    inventory: pd.DataFrame,
    overall: pd.DataFrame,
    paired: pd.DataFrame,
    deltas: pd.DataFrame,
    scopes: pd.DataFrame,
    groups: pd.DataFrame,
    coverage: pd.DataFrame,
    samples: pd.DataFrame,
    comparison: pd.DataFrame,
    assignments: pd.DataFrame,
    reference: pd.DataFrame,
    validation: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    paired_rows = paired[paired["metric_weighting"].eq("row_equal")].set_index("prediction_variant")
    old = paired_rows.loc[BASELINE_VARIANT]
    new = paired_rows.loc[NEW_VARIANT]
    phase6_overall = overall[(overall["prediction_variant"].eq(NEW_VARIANT)) & overall["metric_weighting"].eq("row_equal")].iloc[0]
    payload = {
        "title": "Phase 6: oxide_v1 classification-only re-prediction",
        "scope_note": "All Phase 0 settings remain fixed. Only broad::oxide material keys change to oxide_v1 before train-only reference curves are rebuilt.",
        "insights": [
            f"Phase 6 predicted {int(phase6_overall['prediction_ok_rows']):,} of {int(phase6_overall['test_rows']):,} test rows ({float(phase6_overall['coverage_fraction']):.1%}).",
            f"On rows predicted by both variants, row-equal MAE changed from {float(old['mae_log10']):.3f} to {float(new['mae_log10']):.3f} log10.",
            f"Paired factor-2 accuracy changed from {float(old['factor_2_accuracy']):.1%} to {float(new['factor_2_accuracy']):.1%}; factor-10 accuracy changed from {float(old['factor_10_accuracy']):.1%} to {float(new['factor_10_accuracy']):.1%}.",
            "Review and exclude labels were retained as separate groups so cohort filtering did not become a second experimental change.",
        ],
        "tables": {
            "inventory": safe_records(inventory),
            "overall": safe_records(overall),
            "paired": safe_records(paired),
            "deltas": safe_records(deltas),
            "scopes": safe_records(scopes),
            "groups": safe_records(groups),
            "coverage": safe_records(coverage),
            "samples": safe_records(samples),
            "comparison": safe_records(comparison),
            "assignments": safe_records(assignments),
            "reference": safe_records(reference),
            "validation": safe_records(validation),
            "manifest": safe_records(manifest),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    manifest = make_manifest()
    source = pd.read_csv(SOURCE_ROWS, low_memory=False)
    source, assignments = apply_oxide_v1(source)
    reference, predictions, coverage_config, dropped = build_new_predictions(source)
    predictions["prediction_variant"] = NEW_VARIANT
    predictions["taxonomy_version"] = TAXONOMY_VERSION
    predictions["taxonomy_rules_sha256"] = rules_sha256()

    baseline = prepare_baseline()
    phase4 = pd.read_csv(PHASE4_ASSIGNMENTS, low_memory=False)
    comparison = build_comparison_rows(baseline, predictions, assignments)
    overall, paired, deltas = build_overall_metrics(comparison)
    scopes = build_scope_metrics(comparison)
    groups, group_coverage = build_group_metrics(comparison)
    samples = build_sample_changes(comparison)
    inventory = build_inventory(source, assignments, reference, comparison)
    validation = build_validation(source, assignments, reference, predictions, comparison, phase4, overall)

    files = {
        OUTPUT_FILES["assignments"]: assignments,
        OUTPUT_FILES["reference"]: reference,
        OUTPUT_FILES["predictions"]: predictions,
        OUTPUT_FILES["comparison_rows"]: comparison,
        OUTPUT_FILES["overall_metrics"]: overall,
        OUTPUT_FILES["paired_metrics"]: paired,
        OUTPUT_FILES["metric_deltas"]: deltas,
        OUTPUT_FILES["scope_metrics"]: scopes,
        OUTPUT_FILES["group_metrics"]: groups,
        OUTPUT_FILES["coverage"]: group_coverage,
        OUTPUT_FILES["sample_changes"]: samples,
        OUTPUT_FILES["inventory"]: inventory,
        OUTPUT_FILES["manifest"]: manifest,
        OUTPUT_FILES["validation"]: validation,
    }
    for name, frame in files.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")

    write_report(args.report, inventory, overall, paired, deltas, group_coverage, validation)
    write_payload(
        args.workbook_payload,
        inventory,
        overall,
        paired,
        deltas,
        scopes,
        groups,
        group_coverage,
        samples,
        comparison,
        assignments,
        reference,
        validation,
        manifest,
    )

    failed = validation.loc[validation["status"].eq("FAIL")]
    log_lines = [
        "Phase 6 build validation",
        f"status={'PASS' if failed.empty else 'FAIL'}",
        f"checks={len(validation)}",
        f"failed={len(failed)}",
        f"date={PHASE6_DATE}",
        f"source_rows={len(source)}",
        f"oxide_rows_reclassified={len(assignments)}",
        f"test_rows={len(predictions)}",
        f"prediction_status_counts={predictions['prediction_status'].value_counts().to_dict()}",
        f"config_coverage={coverage_config.to_dict(orient='records')}",
        f"dropped_rows={len(dropped)}",
    ]
    log_lines.extend(f"{r.check}: {r.status} | {r.detail}" for r in validation.itertuples(index=False))
    args.validation_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines[:10]))
    if not failed.empty:
        raise SystemExit(f"Phase 6 build checks failed: {failed['check'].tolist()}")


if __name__ == "__main__":
    main()
