"""Build the exp007 Phase 1 electrical-conductivity residual datasets.

The primary baseline was frozen in Phase 0 as the exp006 broad-family 100 K
variant.  This script reads that prediction file without modifying it, keeps
all test rows, computes residual fields only for evaluable predictions, and
writes flat analysis tables plus validation evidence for later diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
EXP006_SOURCE = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step6b_broad_family"
    / "step5b_test_predictions_default.csv"
)
PHASE0_MANIFEST = EXP007 / "data" / "processed" / "phase0_input_manifest.csv"
PHASE0_METRICS = EXP007 / "data" / "processed" / "phase0_baseline_metrics.csv"
DEFAULT_OUTPUT = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase1_residual_dataset.md"
DEFAULT_VALIDATION = EXP007 / "reports" / "phase1_validation_log.txt"
DEFAULT_PAYLOAD = EXP007 / "reports" / ".phase1_workbook_payload.json"

PRIMARY_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid__"
    "material_family__sample_median"
)
PRIMARY_VARIANT = "broad_family_100K"
ATOL = 1e-12
FACTOR_2_LOG10 = math.log10(2.0)
FACTOR_5_LOG10 = math.log10(5.0)
FACTOR_10_LOG10 = 1.0

IDENTITY_COLUMNS = [
    "config_id",
    "split_scheme",
    "reference_source_subset",
    "eval_target_subset",
    "group_scheme",
    "curve_method",
    "prediction_status",
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_group_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "material_group_key_for_prediction",
]

CONTEXT_COLUMNS = [
    "T_K",
    "T_bin_index",
    "T_bin_left_K",
    "T_bin_right_K",
    "T_bin_center_K",
    "T_bin_label",
    "carrier_type",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "is_reference_bin_candidate",
    "reliability_level",
    "sigma_source",
    "match_method",
    "is_conservative_main_analysis",
    "sample_has_sign_change",
]

REQUIRED_SOURCE_COLUMNS = set(
    IDENTITY_COLUMNS
    + CONTEXT_COLUMNS
    + [
        "sigma_S_per_m",
        "log10_sigma_S_per_m",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "sigma_pred_over_exp",
        "log10_sigma_pred_over_exp",
        "abs_log10_sigma_pred_over_exp",
        "squared_log10_sigma_pred_over_exp",
    ]
)

RESIDUAL_COLUMNS = [
    "is_evaluable",
    "is_oxide_broad",
    "sigma_observed_S_per_m",
    "log10_sigma_observed_S_per_m",
    "sigma_predicted_S_per_m",
    "log10_sigma_predicted_S_per_m",
    "predicted_over_observed_ratio",
    "residual_S_per_m",
    "relative_residual_pred_minus_obs_over_obs",
    "residual_log10_pred_over_observed",
    "abs_residual_log10",
    "squared_residual_log10",
    "absolute_error_factor",
    "prediction_direction",
    "residual_band",
    "within_factor_2",
    "within_factor_5",
    "within_factor_10",
    "beyond_factor_10",
    "abs_residual_rank_overall",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--workbook-payload", type=Path, default=DEFAULT_PAYLOAD)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_inputs() -> None:
    missing = [
        path
        for path in (EXP006_SOURCE, PHASE0_MANIFEST, PHASE0_METRICS)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase 1 input files: {missing}")


def first_nonempty(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return None
    for value in values:
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def join_unique(series: pd.Series, limit: int = 8) -> str:
    values: list[str] = []
    for value in series.dropna():
        text = str(value).strip()
        if text and text not in values:
            values.append(text)
    if len(values) <= limit:
        return " | ".join(values)
    return " | ".join(values[:limit]) + f" | ... (+{len(values) - limit})"


def classify_direction(value: float) -> str:
    if value > ATOL:
        return "overprediction"
    if value < -ATOL:
        return "underprediction"
    return "matched"


def classify_band(abs_value: float) -> str:
    if abs_value <= FACTOR_2_LOG10 + ATOL:
        return "within_factor_2"
    if abs_value <= FACTOR_5_LOG10 + ATOL:
        return "factor_2_to_5"
    if abs_value <= FACTOR_10_LOG10 + ATOL:
        return "factor_5_to_10"
    return "beyond_factor_10"


def numeric_summary(errors: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(errors, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
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


def build_residual_rows(source: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    observed = pd.to_numeric(source["sigma_S_per_m"], errors="coerce")
    predicted = pd.to_numeric(source["sigma_pred_S_per_m"], errors="coerce")
    evaluable = (
        source["prediction_status"].eq("ok")
        & np.isfinite(observed)
        & np.isfinite(predicted)
        & observed.gt(0)
        & predicted.gt(0)
    )

    output = source[IDENTITY_COLUMNS + CONTEXT_COLUMNS].copy()
    output["is_evaluable"] = evaluable.to_numpy(dtype=bool)
    output["is_oxide_broad"] = source["material_group_key_for_prediction"].eq(
        "broad::oxide"
    )
    for column in [
        "sigma_observed_S_per_m",
        "log10_sigma_observed_S_per_m",
        "sigma_predicted_S_per_m",
        "log10_sigma_predicted_S_per_m",
        "predicted_over_observed_ratio",
        "residual_S_per_m",
        "relative_residual_pred_minus_obs_over_obs",
        "residual_log10_pred_over_observed",
        "abs_residual_log10",
        "squared_residual_log10",
        "absolute_error_factor",
        "abs_residual_rank_overall",
    ]:
        output[column] = np.nan

    index = source.index[evaluable]
    observed_ok = observed.loc[index]
    predicted_ok = predicted.loc[index]
    residual_log10 = np.log10(predicted_ok / observed_ok)
    abs_residual = residual_log10.abs()

    output.loc[index, "sigma_observed_S_per_m"] = observed_ok
    output.loc[index, "log10_sigma_observed_S_per_m"] = np.log10(observed_ok)
    output.loc[index, "sigma_predicted_S_per_m"] = predicted_ok
    output.loc[index, "log10_sigma_predicted_S_per_m"] = np.log10(predicted_ok)
    output.loc[index, "predicted_over_observed_ratio"] = predicted_ok / observed_ok
    output.loc[index, "residual_S_per_m"] = predicted_ok - observed_ok
    output.loc[index, "relative_residual_pred_minus_obs_over_obs"] = (
        predicted_ok / observed_ok - 1.0
    )
    output.loc[index, "residual_log10_pred_over_observed"] = residual_log10
    output.loc[index, "abs_residual_log10"] = abs_residual
    output.loc[index, "squared_residual_log10"] = residual_log10**2
    output.loc[index, "absolute_error_factor"] = 10.0**abs_residual

    output["prediction_direction"] = "not_evaluable"
    output.loc[index, "prediction_direction"] = [
        classify_direction(value) for value in residual_log10
    ]
    output["residual_band"] = "not_evaluable"
    output.loc[index, "residual_band"] = [classify_band(value) for value in abs_residual]

    for column, threshold in (
        ("within_factor_2", FACTOR_2_LOG10),
        ("within_factor_5", FACTOR_5_LOG10),
        ("within_factor_10", FACTOR_10_LOG10),
    ):
        output[column] = pd.Series(pd.NA, index=output.index, dtype="boolean")
        output.loc[index, column] = (abs_residual <= threshold + ATOL).to_numpy()
    output["beyond_factor_10"] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    output.loc[index, "beyond_factor_10"] = (
        abs_residual > FACTOR_10_LOG10 + ATOL
    ).to_numpy()
    output.loc[index, "abs_residual_rank_overall"] = abs_residual.rank(
        method="first", ascending=False
    ).to_numpy()
    output["abs_residual_rank_overall"] = output["abs_residual_rank_overall"].astype(
        "Int64"
    )

    stored = pd.to_numeric(
        source.loc[index, "log10_sigma_pred_over_exp"], errors="coerce"
    )
    max_stored_difference = float(
        np.max(np.abs(stored.to_numpy(dtype=float) - residual_log10.to_numpy(dtype=float)))
    )
    return output[IDENTITY_COLUMNS + CONTEXT_COLUMNS + RESIDUAL_COLUMNS], max_stored_difference


def sample_summary(residual_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in residual_rows.groupby(
        "validation_sample_group_id", dropna=False, sort=False
    ):
        evaluable = group.loc[group["is_evaluable"]].copy()
        record: dict[str, Any] = {
            "validation_sample_group_id": sample_key,
            "validation_paper_group_id": first_nonempty(group["validation_paper_group_id"]),
            "paper_id": first_nonempty(group["paper_id"]),
            "doi": first_nonempty(group["doi"]),
            "sample_id": first_nonempty(group["sample_id"]),
            "sample_key": first_nonempty(group["sample_key"]),
            "formula_raw": join_unique(group["formula_raw"]),
            "material_name_raw": join_unique(group["material_name_raw"]),
            "material_group_keys": join_unique(group["material_group_key_for_prediction"]),
            "n_distinct_material_groups": int(
                group["material_group_key_for_prediction"].nunique(dropna=False)
            ),
            "is_oxide_broad": bool(group["is_oxide_broad"].any()),
            "carrier_types": join_unique(group["carrier_type"]),
            "n_rows_total": int(len(group)),
            "n_rows_evaluable": int(len(evaluable)),
            "coverage_fraction": float(len(evaluable) / len(group)),
            "T_min_K": float(pd.to_numeric(group["T_K"], errors="coerce").min()),
            "T_max_K": float(pd.to_numeric(group["T_K"], errors="coerce").max()),
        }
        record.update(numeric_summary(evaluable["residual_log10_pred_over_observed"]))
        if evaluable.empty:
            record.update(
                {
                    "sample_prediction_direction": "not_evaluable",
                    "sample_residual_band": "not_evaluable",
                    "sample_signed_bias_factor": np.nan,
                    "sample_absolute_error_factor": np.nan,
                    "sample_has_factor_10_outlier_row": False,
                    "worst_row_id": None,
                    "worst_row_T_K": np.nan,
                    "worst_row_residual_log10": np.nan,
                    "worst_row_abs_residual_log10": np.nan,
                }
            )
        else:
            median_residual = float(record["median_residual_log10"])
            worst_index = evaluable["abs_residual_log10"].idxmax()
            record.update(
                {
                    "sample_prediction_direction": classify_direction(median_residual),
                    "sample_residual_band": classify_band(abs(median_residual)),
                    "sample_signed_bias_factor": float(10.0**median_residual),
                    "sample_absolute_error_factor": float(10.0 ** abs(median_residual)),
                    "sample_has_factor_10_outlier_row": bool(
                        (evaluable["abs_residual_log10"] > 1.0 + ATOL).any()
                    ),
                    "worst_row_id": evaluable.loc[worst_index, "row_id"],
                    "worst_row_T_K": float(evaluable.loc[worst_index, "T_K"]),
                    "worst_row_residual_log10": float(
                        evaluable.loc[worst_index, "residual_log10_pred_over_observed"]
                    ),
                    "worst_row_abs_residual_log10": float(
                        evaluable.loc[worst_index, "abs_residual_log10"]
                    ),
                }
            )
        records.append(record)
    output = pd.DataFrame(records)
    output["sample_abs_residual_rank"] = output["median_residual_log10"].abs().rank(
        method="first", ascending=False, na_option="bottom"
    ).astype("Int64")
    return output.sort_values(
        ["sample_abs_residual_rank", "validation_sample_group_id"], na_position="last"
    ).reset_index(drop=True)


def grouped_summary(residual_rows: pd.DataFrame, group_column: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_value, group in residual_rows.groupby(group_column, dropna=False, sort=False):
        evaluable = group.loc[group["is_evaluable"]].copy()
        record: dict[str, Any] = {
            group_column: group_value,
            "n_rows_total": int(len(group)),
            "n_rows_evaluable": int(len(evaluable)),
            "n_rows_not_evaluable": int(len(group) - len(evaluable)),
            "coverage_fraction": float(len(evaluable) / len(group)),
            "n_samples_total": int(group["validation_sample_group_id"].nunique()),
            "n_samples_evaluable": int(
                evaluable["validation_sample_group_id"].nunique()
            ),
            "n_papers_total": int(group["validation_paper_group_id"].nunique()),
            "n_papers_evaluable": int(
                evaluable["validation_paper_group_id"].nunique()
            ),
        }
        row_metrics = numeric_summary(evaluable["residual_log10_pred_over_observed"])
        record.update({f"row_{key}": value for key, value in row_metrics.items()})
        sample_medians = evaluable.groupby(
            "validation_sample_group_id", dropna=False
        )["residual_log10_pred_over_observed"].median()
        sample_metrics = numeric_summary(sample_medians)
        record.update({f"sample_equal_{key}": value for key, value in sample_metrics.items()})
        records.append(record)
    output = pd.DataFrame(records)
    return output.sort_values(
        ["n_rows_evaluable", group_column], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


def category_summary(residual_rows: pd.DataFrame) -> pd.DataFrame:
    evaluable = residual_rows.loc[residual_rows["is_evaluable"]]
    records: list[dict[str, Any]] = []
    for summary_type, column, order in (
        (
            "residual_band",
            "residual_band",
            [
                "within_factor_2",
                "factor_2_to_5",
                "factor_5_to_10",
                "beyond_factor_10",
            ],
        ),
        (
            "prediction_direction",
            "prediction_direction",
            ["underprediction", "matched", "overprediction"],
        ),
    ):
        counts = evaluable[column].value_counts(dropna=False)
        for category in order:
            count = int(counts.get(category, 0))
            records.append(
                {
                    "summary_type": summary_type,
                    "category": category,
                    "n_rows": count,
                    "fraction_of_evaluable_rows": float(count / len(evaluable)),
                }
            )
    status_counts = residual_rows["prediction_status"].value_counts(dropna=False)
    for status, count in status_counts.items():
        records.append(
            {
                "summary_type": "prediction_status",
                "category": str(status),
                "n_rows": int(count),
                "fraction_of_evaluable_rows": np.nan,
            }
        )
    return pd.DataFrame(records)


def summary_metrics(residual_rows: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    evaluable = residual_rows.loc[residual_rows["is_evaluable"]]
    row_metrics = numeric_summary(evaluable["residual_log10_pred_over_observed"])
    sample_metrics = numeric_summary(samples["median_residual_log10"])
    entries = [
        ("baseline_variant", PRIMARY_VARIANT, "", "Phase 0 primary baseline"),
        ("temperature_bin_width", 100, "K", "Broad-family reference-bin width"),
        ("test_rows", len(residual_rows), "rows", "All held-out test rows retained"),
        ("evaluable_rows", len(evaluable), "rows", "Rows with valid observed and predicted sigma"),
        (
            "prediction_coverage",
            len(evaluable) / len(residual_rows),
            "fraction",
            "Evaluable rows divided by all test rows",
        ),
        (
            "evaluable_samples",
            evaluable["validation_sample_group_id"].nunique(),
            "samples",
            "Unique evaluable held-out samples",
        ),
        (
            "evaluable_papers",
            evaluable["validation_paper_group_id"].nunique(),
            "papers",
            "Unique evaluable held-out papers",
        ),
        ("row_mae_log10", row_metrics["mae_log10"], "log10", "Row-equal MAE"),
        ("row_rmse_log10", row_metrics["rmse_log10"], "log10", "Row-equal RMSE"),
        (
            "row_mean_residual_log10",
            row_metrics["mean_residual_log10"],
            "log10",
            "Positive means overall overprediction",
        ),
        (
            "row_within_factor_2",
            row_metrics["within_factor_2_fraction"],
            "fraction",
            "Absolute log10 residual at most log10(2)",
        ),
        (
            "row_within_factor_10",
            row_metrics["within_factor_10_fraction"],
            "fraction",
            "Absolute log10 residual at most 1",
        ),
        (
            "row_beyond_factor_10",
            row_metrics["beyond_factor_10_count"],
            "rows",
            "Absolute log10 residual greater than 1",
        ),
        (
            "sample_equal_mae_log10",
            sample_metrics["mae_log10"],
            "log10",
            "MAE of per-sample median signed residuals",
        ),
        (
            "sample_equal_rmse_log10",
            sample_metrics["rmse_log10"],
            "log10",
            "RMSE of per-sample median signed residuals",
        ),
        (
            "oxide_evaluable_rows",
            int((evaluable["material_group_key_for_prediction"] == "broad::oxide").sum()),
            "rows",
            "Rows currently assigned to the broad oxide group",
        ),
    ]
    return pd.DataFrame(entries, columns=["metric", "value", "unit", "definition"])


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def append_validation(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: str
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
    )


def main() -> None:
    args = parse_args()
    require_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.validation_log.parent.mkdir(parents=True, exist_ok=True)
    args.workbook_payload.parent.mkdir(parents=True, exist_ok=True)

    source_hash_before = sha256(EXP006_SOURCE)
    source = pd.read_csv(EXP006_SOURCE, low_memory=False)
    missing_columns = sorted(REQUIRED_SOURCE_COLUMNS.difference(source.columns))
    if missing_columns:
        raise KeyError(f"Primary prediction source missing columns: {missing_columns}")
    configs = source["config_id"].dropna().unique().tolist()
    if configs != [PRIMARY_CONFIG_ID]:
        raise RuntimeError(f"Unexpected config IDs: {configs}")

    residual_rows, max_stored_difference = build_residual_rows(source)
    samples = sample_summary(residual_rows)
    materials = grouped_summary(residual_rows, "material_group_key_for_prediction")
    temperatures = grouped_summary(residual_rows, "T_bin_label")
    papers = grouped_summary(residual_rows, "validation_paper_group_id")
    categories = category_summary(residual_rows)
    metrics = summary_metrics(residual_rows, samples)
    outliers = residual_rows.loc[
        residual_rows["is_evaluable"] & residual_rows["beyond_factor_10"].fillna(False)
    ].sort_values("abs_residual_log10", ascending=False)

    outputs = {
        "phase1_sigma_residual_rows.csv": residual_rows,
        "phase1_sigma_sample_summary.csv": samples,
        "phase1_sigma_material_group_summary.csv": materials,
        "phase1_sigma_temperature_bin_summary.csv": temperatures,
        "phase1_sigma_paper_summary.csv": papers,
        "phase1_sigma_residual_category_summary.csv": categories,
        "phase1_sigma_summary_metrics.csv": metrics,
        "phase1_sigma_outlier_rows.csv": outliers,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    phase0_manifest = pd.read_csv(PHASE0_MANIFEST)
    source_relative = relative(EXP006_SOURCE)
    manifest_match = phase0_manifest.loc[
        phase0_manifest["source_file"].eq(source_relative)
    ]
    phase0_hash = (
        str(manifest_match.iloc[0]["sha256"]) if len(manifest_match) == 1 else None
    )
    phase0_metrics = pd.read_csv(PHASE0_METRICS)
    phase0_primary = phase0_metrics.loc[
        phase0_metrics["variant"].eq(PRIMARY_VARIANT)
        & phase0_metrics["metric_weighting"].eq("row_equal")
    ]
    if len(phase0_primary) != 1:
        raise RuntimeError("Phase 0 primary row-equal metric row is not unique")
    phase0_primary_row = phase0_primary.iloc[0]

    evaluable = residual_rows.loc[residual_rows["is_evaluable"]]
    current_numeric = numeric_summary(evaluable["residual_log10_pred_over_observed"])
    category_band_count = int(
        categories.loc[categories["summary_type"].eq("residual_band"), "n_rows"].sum()
    )
    checks: list[dict[str, Any]] = []
    append_validation(checks, "source_config_is_frozen", configs == [PRIMARY_CONFIG_ID], str(configs))
    append_validation(
        checks,
        "source_hash_matches_phase0_manifest",
        phase0_hash == source_hash_before,
        f"phase0={phase0_hash}; current={source_hash_before}",
    )
    append_validation(
        checks,
        "row_ids_are_unique",
        not residual_rows["row_id"].duplicated().any(),
        f"duplicate_count={int(residual_rows['row_id'].duplicated().sum())}",
    )
    append_validation(
        checks,
        "all_source_rows_preserved",
        len(residual_rows) == len(source)
        and residual_rows["row_id"].tolist() == source["row_id"].tolist(),
        f"source={len(source)}; output={len(residual_rows)}",
    )
    append_validation(
        checks,
        "stored_residual_reproduced",
        max_stored_difference <= ATOL,
        f"max_abs_difference={max_stored_difference:.3g}",
    )
    append_validation(
        checks,
        "residuals_only_on_evaluable_rows",
        residual_rows.loc[~residual_rows["is_evaluable"], "residual_log10_pred_over_observed"].isna().all(),
        f"not_evaluable_rows={int((~residual_rows['is_evaluable']).sum())}",
    )
    append_validation(
        checks,
        "residual_bands_partition_evaluable_rows",
        category_band_count == len(evaluable),
        f"band_total={category_band_count}; evaluable={len(evaluable)}",
    )
    append_validation(
        checks,
        "phase0_n_rows_reproduced",
        int(phase0_primary_row["n_rows"]) == len(evaluable),
        f"phase0={int(phase0_primary_row['n_rows'])}; phase1={len(evaluable)}",
    )
    for phase0_column, current_key in (
        ("mean_log10_error", "mean_residual_log10"),
        ("mae_log10", "mae_log10"),
        ("rmse_log10", "rmse_log10"),
        ("factor_2_accuracy", "within_factor_2_fraction"),
        ("factor_10_accuracy", "within_factor_10_fraction"),
    ):
        difference = abs(
            float(phase0_primary_row[phase0_column]) - float(current_numeric[current_key])
        )
        append_validation(
            checks,
            f"phase0_{phase0_column}_reproduced",
            difference <= ATOL,
            f"absolute_difference={difference:.3g}",
        )

    source_hash_after = sha256(EXP006_SOURCE)
    append_validation(
        checks,
        "exp006_source_unchanged_during_run",
        source_hash_before == source_hash_after,
        f"before={source_hash_before}; after={source_hash_after}",
    )
    validation = pd.DataFrame(checks)
    validation.to_csv(args.output_dir / "phase1_validation_checks.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "source_file": source_relative,
                "source_role": "phase0_primary_baseline_predictions",
                "baseline_variant": PRIMARY_VARIANT,
                "size_bytes": EXP006_SOURCE.stat().st_size,
                "sha256_before": source_hash_before,
                "sha256_after": source_hash_after,
                "source_unchanged_during_run": source_hash_before == source_hash_after,
                "matches_phase0_manifest": phase0_hash == source_hash_before,
            }
        ]
    )
    manifest.to_csv(args.output_dir / "phase1_input_manifest.csv", index=False)

    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS")]
        raise RuntimeError(f"Phase 1 validation failed:\n{failed.to_string(index=False)}")

    payload = {
        "summary_metrics": json_records(metrics),
        "category_summary": json_records(categories),
        "sample_summary": json_records(samples),
        "material_summary": json_records(materials),
        "temperature_summary": json_records(temperatures),
        "paper_summary": json_records(papers),
        "outlier_rows": json_records(outliers),
        "unavailable_rows": json_records(
            residual_rows.loc[~residual_rows["is_evaluable"]]
        ),
        "validation_checks": json_records(validation),
    }
    args.workbook_payload.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    status_counts = residual_rows["prediction_status"].value_counts().to_dict()
    category_rows = categories.loc[categories["summary_type"].eq("residual_band")]
    category_table = "\n".join(
        f"| {row.category} | {int(row.n_rows):,} | {row.fraction_of_evaluable_rows:.3%} |"
        for row in category_rows.itertuples(index=False)
    )
    report_lines = [
        "# exp007 Phase 1 residual dataset",
        "",
        "## Scope",
        "",
        "This phase converts the frozen Phase 0 primary baseline into analysis-ready electrical-conductivity residual tables. The exp006 source is read-only.",
        "",
        f"- Baseline variant: `{PRIMARY_VARIANT}`",
        f"- Config: `{PRIMARY_CONFIG_ID}`",
        f"- Source: `{source_relative}`",
        f"- Test rows retained: {len(residual_rows):,}",
        f"- Evaluable rows: {len(evaluable):,}",
        f"- Evaluable samples: {evaluable['validation_sample_group_id'].nunique():,}",
        f"- Evaluable papers: {evaluable['validation_paper_group_id'].nunique():,}",
        f"- Broad-oxide evaluable rows: {int((evaluable['is_oxide_broad']).sum()):,}",
        "",
        "## Residual definition",
        "",
        "`residual_log10_pred_over_observed = log10(sigma_predicted / sigma_observed)`",
        "",
        "Positive values are overpredictions. Negative values are underpredictions. Absolute residuals of log10(2), log10(5), and 1 delimit factor-2, factor-5, and factor-10 error bands.",
        "",
        "## Residual bands",
        "",
        "| band | rows | fraction of evaluable rows |",
        "| --- | ---: | ---: |",
        category_table,
        "",
        "## Prediction status",
        "",
        *[f"- `{key}`: {int(value):,}" for key, value in status_counts.items()],
        "",
        "## Output roles",
        "",
        "- `phase1_sigma_residual_rows.csv`: all test rows, including unavailable predictions.",
        "- `phase1_sigma_sample_summary.csv`: one row per held-out sample.",
        "- `phase1_sigma_material_group_summary.csv`: broad material-group aggregates.",
        "- `phase1_sigma_temperature_bin_summary.csv`: 100 K temperature-bin aggregates.",
        "- `phase1_sigma_paper_summary.csv`: held-out paper aggregates.",
        "- `phase1_sigma_outlier_rows.csv`: rows with error greater than factor 10.",
        "- `phase1_sigma_residual_analysis.xlsx`: reader-facing workbook built from these tables.",
        "",
        "Phase 1 prepares residual evidence only. Interpretation of error causes belongs to the next phase.",
        "",
        "## Validation",
        "",
        f"- Stored residual maximum absolute difference: {max_stored_difference:.3g}",
        f"- Phase 0 metrics reproduced: {validation['status'].eq('PASS').all()}",
        f"- exp006 source unchanged: {source_hash_before == source_hash_after}",
    ]
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    validation_lines = [
        "exp007 Phase 1 residual dataset validation",
        "============================================",
        *[
            f"{row.status} {row.check}: {row.detail}"
            for row in validation.itertuples(index=False)
        ],
        "",
        f"test_rows={len(residual_rows)}",
        f"evaluable_rows={len(evaluable)}",
        f"evaluable_samples={evaluable['validation_sample_group_id'].nunique()}",
        f"evaluable_papers={evaluable['validation_paper_group_id'].nunique()}",
        f"factor_10_outlier_rows={len(outliers)}",
        "Final status: PASS",
    ]
    args.validation_log.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")

    print(f"Wrote Phase 1 CSV outputs to {args.output_dir}")
    print(f"Workbook payload: {args.workbook_payload}")
    print(f"Report: {args.report}")
    print(f"Validation: {args.validation_log}")
    print(
        f"Rows: total={len(residual_rows):,}, evaluable={len(evaluable):,}, "
        f"factor-10 outliers={len(outliers):,}"
    )


if __name__ == "__main__":
    main()
