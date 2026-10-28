#!/usr/bin/env python3
"""Freeze and validate the exp007 baseline against three exp006 variants.

This script is intentionally read-only with respect to exp006.  It recomputes
the default row-equal and sample-equal metrics from CSV prediction rows,
validates them against the existing Step5C summaries, and writes an isolated
Phase 0 handoff under experiments/exp007.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP006_PROCESSED = ROOT / "experiments" / "exp006" / "data" / "processed"
EXP007 = ROOT / "experiments" / "exp007"
DEFAULT_OUTPUT = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase0_baseline_comparison.md"
DEFAULT_VALIDATION = EXP007 / "reports" / "phase0_validation_log.txt"

DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid__"
    "material_family__sample_median"
)
ATOL = 1e-12


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    directory: Path
    temperature_bin: str
    role: str
    decision_reason: str

    @property
    def predictions(self) -> Path:
        return self.directory / "step5b_test_predictions_default.csv"

    @property
    def reference_metrics(self) -> Path:
        return self.directory / "step5c_metrics_by_config.csv"


VARIANTS = (
    Variant(
        key="legacy_old_default",
        label="old/default (poster-era legacy)",
        directory=EXP006_PROCESSED,
        temperature_bin="100 K",
        role="legacy_reference",
        decision_reason=(
            "Retained only to reproduce the poster-era result; its prediction key "
            "collapses to one unknown material family."
        ),
    ),
    Variant(
        key="broad_family_100K",
        label="broad-family 100 K",
        directory=EXP006_PROCESSED / "step6b_broad_family",
        temperature_bin="100 K",
        role="primary_baseline",
        decision_reason=(
            "Selected for oxide-subclassification work because it isolates the material "
            "classification change while retaining the highest broad-family coverage."
        ),
    ),
    Variant(
        key="broad_family_25K",
        label="broad-family 25 K",
        directory=EXP006_PROCESSED / "step9a_25k_bin_broad_family",
        temperature_bin="25 K",
        role="sensitivity_baseline",
        decision_reason=(
            "Retained as a temperature-resolution sensitivity analysis; it slightly "
            "improves row-equal errors but reduces coverage and increases bin sparsity."
        ),
    ),
)

REQUIRED_PREDICTION_COLUMNS = {
    "config_id",
    "prediction_status",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key",
    "material_group_key_for_prediction",
    "T_bin_center_K",
    "sigma_S_per_m",
    "sigma_pred_S_per_m",
    "eta",
    "S_abs_uV_per_K",
    "T_K",
    "train_sample_count",
    "train_paper_count",
}

REFERENCE_CHECK_COLUMNS = (
    "n_rows",
    "n_samples",
    "n_papers",
    "n_material_families",
    "n_T_bins",
    "mean_log10_error",
    "median_log10_error",
    "mae_log10",
    "rmse_log10",
    "factor_2_accuracy",
    "factor_10_accuracy",
    "test_rows",
    "prediction_ok_rows",
    "prediction_unavailable_rows",
    "coverage_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def require_files() -> None:
    missing = [
        path
        for variant in VARIANTS
        for path in (variant.predictions, variant.reference_metrics)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Phase 0 source files: {missing}")


def prepare_prediction_rows(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    rows = pd.read_csv(path, low_memory=False)
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(rows.columns))
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    configs = rows["config_id"].dropna().unique().tolist()
    if configs != [DEFAULT_CONFIG_ID]:
        raise RuntimeError(f"Unexpected config IDs in {path}: {configs}")

    status_counts = {
        str(key): int(value)
        for key, value in rows["prediction_status"].value_counts(dropna=False).items()
    }
    exp = pd.to_numeric(rows["sigma_S_per_m"], errors="coerce")
    pred = pd.to_numeric(rows["sigma_pred_S_per_m"], errors="coerce")
    valid = (
        rows["prediction_status"].eq("ok")
        & np.isfinite(exp)
        & np.isfinite(pred)
        & exp.gt(0)
        & pred.gt(0)
    )
    evaluated = rows.loc[valid].copy()
    evaluated["sigma_S_per_m"] = exp.loc[valid]
    evaluated["sigma_pred_S_per_m"] = pred.loc[valid]
    evaluated["log10_sigma_pred_over_exp"] = np.log10(
        evaluated["sigma_pred_S_per_m"] / evaluated["sigma_S_per_m"]
    )
    evaluated["abs_log10_sigma_pred_over_exp"] = evaluated[
        "log10_sigma_pred_over_exp"
    ].abs()
    evaluated["squared_log10_sigma_pred_over_exp"] = evaluated[
        "log10_sigma_pred_over_exp"
    ] ** 2

    if "log10_sigma_pred_over_exp" in rows.columns:
        stored = pd.to_numeric(
            rows.loc[valid, "log10_sigma_pred_over_exp"], errors="coerce"
        )
        difference = (
            stored.to_numpy(dtype=float)
            - evaluated["log10_sigma_pred_over_exp"].to_numpy(dtype=float)
        )
        if not np.allclose(difference, 0.0, atol=ATOL, rtol=0.0, equal_nan=False):
            raise RuntimeError(f"Stored log10 errors do not reproduce in {path}")
    return rows, evaluated, status_counts


def sample_equal_rows(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in rows.groupby(
        "validation_sample_group_id", dropna=False, sort=False
    ):
        records.append(
            {
                "validation_sample_group_id": sample_key,
                "validation_paper_group_id": group[
                    "validation_paper_group_id"
                ].iloc[0],
                "material_group_key": group["material_group_key"].iloc[0],
                "T_bin_center_K": pd.to_numeric(
                    group["T_bin_center_K"], errors="coerce"
                ).median(),
                "log10_sigma_pred_over_exp": group[
                    "log10_sigma_pred_over_exp"
                ].median(),
                "sigma_S_per_m": pd.to_numeric(
                    group["sigma_S_per_m"], errors="coerce"
                ).median(),
                "sigma_pred_S_per_m": pd.to_numeric(
                    group["sigma_pred_S_per_m"], errors="coerce"
                ).median(),
                "eta": pd.to_numeric(group["eta"], errors="coerce").median(),
                "S_abs_uV_per_K": pd.to_numeric(
                    group["S_abs_uV_per_K"], errors="coerce"
                ).median(),
                "T_K": pd.to_numeric(group["T_K"], errors="coerce").median(),
                "train_sample_count": pd.to_numeric(
                    group["train_sample_count"], errors="coerce"
                ).median(),
                "train_paper_count": pd.to_numeric(
                    group["train_paper_count"], errors="coerce"
                ).median(),
            }
        )
    output = pd.DataFrame(records)
    output["abs_log10_sigma_pred_over_exp"] = output[
        "log10_sigma_pred_over_exp"
    ].abs()
    output["squared_log10_sigma_pred_over_exp"] = output[
        "log10_sigma_pred_over_exp"
    ] ** 2
    return output


def metric_row(
    source: pd.DataFrame,
    raw_test_rows: int,
    prediction_ok_rows: int,
    weighting: str,
) -> dict[str, Any]:
    errors = pd.to_numeric(source["log10_sigma_pred_over_exp"], errors="coerce")
    abs_errors = errors.abs()
    return {
        "metric_weighting": weighting,
        "n_rows": int(len(source)),
        "n_samples": int(
            source["validation_sample_group_id"].nunique(dropna=True)
        ),
        "n_papers": int(source["validation_paper_group_id"].nunique(dropna=True)),
        "n_material_families": int(source["material_group_key"].nunique(dropna=True)),
        "n_T_bins": int(source["T_bin_center_K"].nunique(dropna=True)),
        "mean_log10_error": float(errors.mean()),
        "median_log10_error": float(errors.median()),
        "mae_log10": float(abs_errors.mean()),
        "rmse_log10": float(math.sqrt((errors**2).mean())),
        "max_abs_log10_error": float(abs_errors.max()),
        "overprediction_fraction": float((errors > 0).mean()),
        "underprediction_fraction": float((errors < 0).mean()),
        "factor_2_accuracy": float((abs_errors <= math.log10(2.0)).mean()),
        "factor_5_accuracy": float((abs_errors <= math.log10(5.0)).mean()),
        "factor_10_accuracy": float((abs_errors <= 1.0).mean()),
        "test_rows": int(raw_test_rows),
        "prediction_ok_rows": int(prediction_ok_rows),
        "prediction_unavailable_rows": int(raw_test_rows - prediction_ok_rows),
        "coverage_fraction": float(prediction_ok_rows / raw_test_rows),
    }


def compute_variant_metrics(
    evaluated: pd.DataFrame, raw_test_rows: int
) -> pd.DataFrame:
    sample_rows = sample_equal_rows(evaluated)
    return pd.DataFrame(
        [
            metric_row(evaluated, raw_test_rows, len(evaluated), "row_equal"),
            metric_row(sample_rows, raw_test_rows, len(evaluated), "sample_equal"),
        ]
    )


def validate_reference(
    variant: Variant, recomputed: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[str]]:
    reference = pd.read_csv(variant.reference_metrics, low_memory=False)
    selected = reference.loc[reference["config_id"].eq(DEFAULT_CONFIG_ID)].copy()
    if set(selected["metric_weighting"]) != {"row_equal", "sample_equal"}:
        raise RuntimeError(f"Could not identify both reference rows for {variant.key}")
    selected = selected.set_index("metric_weighting")
    calculated = recomputed.set_index("metric_weighting")
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for weighting in ("row_equal", "sample_equal"):
        for column in REFERENCE_CHECK_COLUMNS:
            observed = float(calculated.loc[weighting, column])
            expected = float(selected.loc[weighting, column])
            difference = abs(observed - expected)
            passed = math.isclose(observed, expected, rel_tol=0.0, abs_tol=ATOL)
            records.append(
                {
                    "variant": variant.key,
                    "metric_weighting": weighting,
                    "metric": column,
                    "recomputed": observed,
                    "reference": expected,
                    "abs_difference": difference,
                    "passed": passed,
                }
            )
            lines.append(
                f"{'PASS' if passed else 'FAIL'} {variant.key} {weighting} "
                f"{column}: calculated={observed:.15g}, reference={expected:.15g}, "
                f"abs_diff={difference:.3g}"
            )
            if not passed:
                raise RuntimeError(
                    f"Reference validation failed: {variant.key} {weighting} {column}"
                )
    return records, lines


def markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 6) -> str:
    view = frame.loc[:, columns].copy()
    rendered: list[list[str]] = []
    for _, row in view.iterrows():
        values: list[str] = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        rendered.append(values)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rendered]
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    require_files()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.validation_log.parent.mkdir(parents=True, exist_ok=True)

    source_paths = [
        path
        for variant in VARIANTS
        for path in (variant.predictions, variant.reference_metrics)
    ]
    hashes_before = {path: sha256(path) for path in source_paths}

    metric_frames: list[pd.DataFrame] = []
    validation_records: list[dict[str, Any]] = []
    validation_lines: list[str] = []
    status_records: list[dict[str, Any]] = []
    evaluated_by_variant: dict[str, pd.DataFrame] = {}

    for variant in VARIANTS:
        raw, evaluated, status_counts = prepare_prediction_rows(variant.predictions)
        evaluated_by_variant[variant.key] = evaluated
        metrics = compute_variant_metrics(evaluated, len(raw))
        metrics.insert(0, "config_id", DEFAULT_CONFIG_ID)
        metrics.insert(0, "role", variant.role)
        metrics.insert(0, "temperature_bin", variant.temperature_bin)
        metrics.insert(0, "variant_label", variant.label)
        metrics.insert(0, "variant", variant.key)
        metric_frames.append(metrics)
        records, lines = validate_reference(variant, metrics)
        validation_records.extend(records)
        validation_lines.extend(lines)
        status_records.append(
            {
                "variant": variant.key,
                "test_rows": int(len(raw)),
                "prediction_ok_rows": int(len(evaluated)),
                "missing_reference_bin": status_counts.get("missing_reference_bin", 0),
                "unreliable_reference_bin": status_counts.get(
                    "unreliable_reference_bin", 0
                ),
                "material_prediction_groups": int(
                    raw["material_group_key_for_prediction"].nunique(dropna=False)
                ),
                "temperature_bins_all_statuses": int(
                    raw["T_bin_center_K"].nunique(dropna=False)
                ),
            }
        )

    metrics_all = pd.concat(metric_frames, ignore_index=True)
    validation = pd.DataFrame(validation_records)
    status = pd.DataFrame(status_records)

    common_ids = set.intersection(
        *[set(frame["row_id"].tolist()) for frame in evaluated_by_variant.values()]
    )
    common_frames: list[pd.DataFrame] = []
    common_metrics_frames: list[pd.DataFrame] = []
    common_observed: pd.Series | None = None
    for variant in VARIANTS:
        frame = evaluated_by_variant[variant.key]
        common = frame.loc[frame["row_id"].isin(common_ids)].copy()
        common = common.sort_values("row_id").reset_index(drop=True)
        if common_observed is None:
            common_observed = common.set_index("row_id")["sigma_S_per_m"]
        else:
            check = common.set_index("row_id")["sigma_S_per_m"]
            aligned = common_observed.loc[check.index]
            if not np.allclose(
                aligned.to_numpy(dtype=float),
                check.to_numpy(dtype=float),
                rtol=0.0,
                atol=0.0,
            ):
                raise RuntimeError("Observed sigma differs between baseline variants")
        common_frames.append(
            common[
                [
                    "row_id",
                    "validation_sample_group_id",
                    "validation_paper_group_id",
                    "sigma_S_per_m",
                    "sigma_pred_S_per_m",
                ]
            ].rename(
                columns={
                    "sigma_pred_S_per_m": f"sigma_pred_{variant.key}_S_per_m"
                }
            )
        )
        computed = compute_variant_metrics(common, len(common))
        computed.insert(0, "cohort", "three_variant_common_prediction_rows")
        computed.insert(0, "variant", variant.key)
        common_metrics_frames.append(computed)

    common_predictions = common_frames[0]
    for frame in common_frames[1:]:
        common_predictions = common_predictions.merge(
            frame.drop(
                columns=[
                    "validation_sample_group_id",
                    "validation_paper_group_id",
                    "sigma_S_per_m",
                ]
            ),
            on="row_id",
            how="inner",
            validate="one_to_one",
        )
    common_metrics = pd.concat(common_metrics_frames, ignore_index=True)

    primary = "broad_family_100K"
    comparison_metrics = [
        "coverage_fraction",
        "mae_log10",
        "rmse_log10",
        "mean_log10_error",
        "factor_2_accuracy",
        "factor_10_accuracy",
    ]
    deltas: list[dict[str, Any]] = []
    for weighting in ("row_equal", "sample_equal"):
        subset = metrics_all.loc[metrics_all["metric_weighting"].eq(weighting)].set_index(
            "variant"
        )
        for variant in VARIANTS:
            for metric in comparison_metrics:
                deltas.append(
                    {
                        "comparison": f"{variant.key}_minus_{primary}",
                        "variant": variant.key,
                        "reference_variant": primary,
                        "metric_weighting": weighting,
                        "metric": metric,
                        "variant_value": float(subset.loc[variant.key, metric]),
                        "reference_value": float(subset.loc[primary, metric]),
                        "delta": float(
                            subset.loc[variant.key, metric] - subset.loc[primary, metric]
                        ),
                    }
                )
    delta_frame = pd.DataFrame(deltas)

    decision = pd.DataFrame(
        [
            {
                "variant": variant.key,
                "variant_label": variant.label,
                "temperature_bin": variant.temperature_bin,
                "role": variant.role,
                "selected_primary": variant.key == primary,
                "decision_reason": variant.decision_reason,
            }
            for variant in VARIANTS
        ]
    )

    hashes_after = {path: sha256(path) for path in source_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("One or more exp006 source files changed during Phase 0")
    manifest = pd.DataFrame(
        [
            {
                "source_file": relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashes_before[path],
                "source_unchanged_during_run": hashes_before[path] == hashes_after[path],
            }
            for path in source_paths
        ]
    )

    outputs = {
        "phase0_baseline_metrics.csv": metrics_all,
        "phase0_prediction_status_summary.csv": status,
        "phase0_reference_validation.csv": validation,
        "phase0_common_row_metrics.csv": common_metrics,
        "phase0_common_row_predictions.csv": common_predictions,
        "phase0_metric_deltas_vs_100K.csv": delta_frame,
        "phase0_baseline_decision.csv": decision,
        "phase0_input_manifest.csv": manifest,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.output_dir / name, index=False, encoding="utf-8-sig")

    row_metrics = metrics_all.loc[metrics_all["metric_weighting"].eq("row_equal")]
    sample_metrics = metrics_all.loc[
        metrics_all["metric_weighting"].eq("sample_equal")
    ]
    report = "\n".join(
        [
            "# exp007 Phase 0 baseline comparison",
            "",
            "## Scope",
            "",
            "This phase freezes the comparison basis for the AROB study. It reads existing exp006 outputs, recomputes the default metrics, and does not modify exp006.",
            "",
            f"Default config: `{DEFAULT_CONFIG_ID}`",
            "",
            "## Row-equal metrics on each available cohort",
            "",
            markdown_table(
                row_metrics,
                [
                    "variant",
                    "n_rows",
                    "n_samples",
                    "coverage_fraction",
                    "mae_log10",
                    "rmse_log10",
                    "mean_log10_error",
                    "factor_2_accuracy",
                    "factor_10_accuracy",
                ],
            ),
            "",
            "## Sample-equal metrics on each available cohort",
            "",
            markdown_table(
                sample_metrics,
                [
                    "variant",
                    "n_rows",
                    "n_papers",
                    "coverage_fraction",
                    "mae_log10",
                    "rmse_log10",
                    "mean_log10_error",
                    "factor_2_accuracy",
                    "factor_10_accuracy",
                ],
            ),
            "",
            "## Prediction coverage and grouping context",
            "",
            markdown_table(
                status,
                [
                    "variant",
                    "test_rows",
                    "prediction_ok_rows",
                    "missing_reference_bin",
                    "unreliable_reference_bin",
                    "material_prediction_groups",
                    "temperature_bins_all_statuses",
                ],
                digits=0,
            ),
            "",
            "## Common-row comparison",
            "",
            f"The intersection contains {len(common_predictions):,} rows with predictions from all three variants.",
            "",
            markdown_table(
                common_metrics.loc[
                    common_metrics["metric_weighting"].eq("row_equal")
                ],
                [
                    "variant",
                    "n_rows",
                    "n_samples",
                    "mae_log10",
                    "rmse_log10",
                    "mean_log10_error",
                    "factor_2_accuracy",
                    "factor_10_accuracy",
                ],
            ),
            "",
            "## Baseline decision",
            "",
            "- Primary baseline: `broad_family_100K`.",
            "- Sensitivity baseline: `broad_family_25K`.",
            "- Legacy reproduction reference: `legacy_old_default`.",
            "",
            "The 100 K broad-family variant is selected for the oxide-subclassification experiment because it changes the material taxonomy without simultaneously imposing the sparsity of 25 K temperature bins. The 25 K variant remains a required sensitivity analysis because its row-equal errors are slightly lower, while its coverage is lower. The old/default variant is not suitable as the primary baseline because its material-family prediction key is collapsed to a single unknown group.",
            "",
            "## Validation",
            "",
            f"- Existing Step5C values reproduced: {bool(validation['passed'].all())}",
            f"- Maximum absolute difference from Step5C: {validation['abs_difference'].max():.3g}",
            f"- exp006 source files unchanged during the run: {bool(manifest['source_unchanged_during_run'].all())}",
            "",
        ]
    )
    args.report.write_text(report, encoding="utf-8")

    log = "\n".join(
        [
            "exp007 Phase 0 baseline validation",
            "=" * 45,
            f"config_id={DEFAULT_CONFIG_ID}",
            f"variants={','.join(variant.key for variant in VARIANTS)}",
            f"common_prediction_rows={len(common_predictions)}",
            "",
            *validation_lines,
            "",
            f"PASS all reference checks={bool(validation['passed'].all())}",
            f"PASS source files unchanged={bool(manifest['source_unchanged_during_run'].all())}",
            "Final status: PASS",
            "",
        ]
    )
    args.validation_log.write_text(log, encoding="utf-8")
    print(f"Wrote Phase 0 outputs to: {args.output_dir}")
    print(f"Report: {args.report}")
    print(f"Validation: {args.validation_log}")
    print(row_metrics[["variant", "n_rows", "coverage_fraction", "mae_log10", "rmse_log10", "factor_2_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
