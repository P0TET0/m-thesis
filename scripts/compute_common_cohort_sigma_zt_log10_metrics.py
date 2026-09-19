#!/usr/bin/env python3
"""Create unified log10-ratio metrics for the two common-cohort parity figures."""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from compute_poster_unified_log10_metrics import compute_log10_ratio_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison"
    / "three_method_row_comparisons.csv"
)
DEFAULT_SIGMA_REFERENCE = (
    ROOT
    / "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison"
    / "three_method_sigma_model_metrics.csv"
)
DEFAULT_ZT_REFERENCE = (
    ROOT
    / "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison"
    / "three_method_model_metrics.csv"
)
DEFAULT_OUTPUT = ROOT / "data/output/common_cohort_sigma_zt_unified_log10_metrics"

METHODS = (
    (
        "exp005_DF",
        "exp005 DF",
        "sigma_pred_exp005_DF_S_per_m",
        "ZT_pred_exp005_DF",
        "Reference only: direct in-sample curve fitting.",
    ),
    (
        "exp005_ML",
        "exp005 ML",
        "sigma_pred_exp005_S_per_m",
        "ZT_pred_exp005_ML",
        "Held-out prediction method.",
    ),
    (
        "exp006",
        "exp006",
        "sigma_pred_exp006_S_per_m",
        "ZT_pred_exp006",
        "Held-out old/default exp006 predictions used in the displayed parity figures; not the later broad_family rerun.",
    ),
)
DATASET = "common_cohort_383_rows_109_samples_83_DOIs_reported_ZT_pm2K"
ATOL = 1e-12


def public_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in metrics.items() if key != "exclusion_reasons"}


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def metric_row(
    target: str,
    method_key: str,
    method_label: str,
    metrics: dict[str, object],
    source: Path,
    exp_column: str,
    pred_column: str,
    note: str,
) -> dict[str, object]:
    return {
        "target": target,
        "dataset": DATASET,
        "evaluation_level": "row",
        "method_key": method_key,
        "method": method_label,
        **public_metrics(metrics),
        "source_file": relative(source),
        "y_exp_column": exp_column,
        "y_pred_column": pred_column,
        "notes": note,
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def table_for(frame: pd.DataFrame, symbol: str) -> list[str]:
    indexed = frame.set_index("method")
    methods = ["exp005 DF", "exp005 ML", "exp006"]
    return [
        "| Metric | exp005 DF | exp005 ML | exp006 |",
        "|---|---:|---:|---:|",
        "| RMSE(log10 pred/exp) | "
        + " | ".join(f"{indexed.loc[m, 'rmse_log10_pred_over_exp']:.3f}" for m in methods)
        + " |",
        "| MAE(log10 pred/exp) | "
        + " | ".join(f"{indexed.loc[m, 'mae_log10_pred_over_exp']:.3f}" for m in methods)
        + " |",
        "| Factor-2 accuracy | "
        + " | ".join(percent(indexed.loc[m, "factor2_accuracy"]) for m in methods)
        + " |",
        "| Factor-10 accuracy | "
        + " | ".join(percent(indexed.loc[m, "factor10_accuracy"]) for m in methods)
        + " |",
        "| Valid rows | "
        + " | ".join(
            f"{int(indexed.loc[m, 'n_valid'])}/{int(indexed.loc[m, 'n_total'])}" for m in methods
        )
        + " |",
        "",
        f"All {symbol} metrics use finite, positive experimental/predicted pairs only.",
    ]


def validate_against_existing(
    sigma: pd.DataFrame,
    zt: pd.DataFrame,
    sigma_reference_path: Path,
    zt_reference_path: Path,
) -> list[str]:
    sigma_ref = pd.read_csv(sigma_reference_path).set_index("method_key")
    zt_ref = pd.read_csv(zt_reference_path).set_index("method_key")
    sigma_by_key = sigma.set_index("method_key")
    zt_by_key = zt.set_index("method_key")
    lines: list[str] = []
    for method_key, *_ in METHODS:
        checks = (
            (
                "sigma RMSE(log10)",
                float(sigma_by_key.loc[method_key, "rmse_log10_pred_over_exp"]),
                float(sigma_ref.loc[method_key, "sigma_log10_RMSE"]),
            ),
            (
                "sigma MAE(log10)",
                float(sigma_by_key.loc[method_key, "mae_log10_pred_over_exp"]),
                float(sigma_ref.loc[method_key, "sigma_log10_MAE"]),
            ),
            (
                "sigma factor-2",
                float(sigma_by_key.loc[method_key, "factor2_accuracy"]),
                float(sigma_ref.loc[method_key, "sigma_within_factor_2_rate"]),
            ),
            (
                "ZT RMSE(log10) vs existing ln/ln(10)",
                float(zt_by_key.loc[method_key, "rmse_log10_pred_over_exp"]),
                float(zt_ref.loc[method_key, "row_log_RMSE"]) / math.log(10.0),
            ),
            (
                "ZT factor-2",
                float(zt_by_key.loc[method_key, "factor2_accuracy"]),
                float(zt_ref.loc[method_key, "row_within_factor_2_rate"]),
            ),
        )
        for label, calculated, expected in checks:
            difference = abs(calculated - expected)
            if not math.isclose(calculated, expected, rel_tol=0.0, abs_tol=ATOL):
                raise RuntimeError(
                    f"Validation failed: {method_key} {label}: "
                    f"calculated={calculated}, expected={expected}"
                )
            lines.append(
                f"PASS {method_key} {label}: calculated={calculated:.15g}, "
                f"reference={expected:.15g}, abs_diff={difference:.3g}"
            )
    return lines


def exclusion_lines(
    label: str, method_metrics: dict[str, dict[str, object]]
) -> list[str]:
    lines: list[str] = []
    for method_key, metrics in method_metrics.items():
        lines.append(
            f"{label} {method_key}: total={metrics['n_total']}, "
            f"valid={metrics['n_valid']}, excluded={metrics['n_excluded']}"
        )
        reasons = metrics["exclusion_reasons"]
        assert isinstance(reasons, dict)
        lines.extend(f"  {reason}: {count}" for reason, count in reasons.items())
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sigma-reference", type=Path, default=DEFAULT_SIGMA_REFERENCE)
    parser.add_argument("--zt-reference", type=Path, default=DEFAULT_ZT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.input, args.sigma_reference, args.zt_reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}; use --overwrite to replace it")

    rows = pd.read_csv(args.input, low_memory=False)
    required = {
        "sample_key",
        "exp006_DOI_normalized",
        "sigma_obs_S_per_m",
        "ZT_obs",
        *[method[2] for method in METHODS],
        *[method[3] for method in METHODS],
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    counts = (
        len(rows),
        rows["sample_key"].nunique(dropna=True),
        rows["exp006_DOI_normalized"].nunique(dropna=True),
    )
    if counts != (383, 109, 83):
        raise RuntimeError(f"Unexpected cohort counts: rows/samples/DOIs={counts}")

    sigma_records: list[dict[str, object]] = []
    zt_records: list[dict[str, object]] = []
    sigma_diagnostics: dict[str, dict[str, object]] = {}
    zt_diagnostics: dict[str, dict[str, object]] = {}
    for method_key, method_label, sigma_pred, zt_pred, note in METHODS:
        sigma_metrics = compute_log10_ratio_metrics(
            rows["sigma_obs_S_per_m"], rows[sigma_pred]
        )
        zt_metrics = compute_log10_ratio_metrics(rows["ZT_obs"], rows[zt_pred])
        sigma_diagnostics[method_key] = sigma_metrics
        zt_diagnostics[method_key] = zt_metrics
        sigma_records.append(
            metric_row(
                "sigma",
                method_key,
                method_label,
                sigma_metrics,
                args.input,
                "sigma_obs_S_per_m",
                sigma_pred,
                note,
            )
        )
        zt_records.append(
            metric_row(
                "ZT",
                method_key,
                method_label,
                zt_metrics,
                args.input,
                "ZT_obs",
                zt_pred,
                note,
            )
        )

    sigma = pd.DataFrame(sigma_records)
    zt = pd.DataFrame(zt_records)
    validation = validate_against_existing(
        sigma, zt, args.sigma_reference, args.zt_reference
    )
    summary = "\n".join(
        [
            "# Unified common-cohort metrics for the displayed parity figures",
            "",
            "Cohort: same 383 rows / 109 samples / 83 DOIs; reported ZT within ±2 K.",
            "",
            "## Electrical conductivity (sigma)",
            "",
            *table_for(sigma, "sigma"),
            "",
            "## ZT",
            "",
            *table_for(zt, "ZT"),
            "",
            "## Important scope note",
            "",
            "The exp006 values here reproduce the old/default exp006 predictions used in the two displayed parity figures. They are not the later broad_family predictions. DF is reference-only direct in-sample fitting.",
            "",
        ]
    )
    log = "\n".join(
        [
            "Common-cohort sigma/ZT unified log10 metric validation",
            "=" * 58,
            f"input={args.input.resolve()}",
            "cohort rows=383; samples=109; DOIs=83; evaluation_level=row",
            "formula=log10(pred/exp); valid=finite(exp,pred) and exp>0 and pred>0",
            "exp006 variant=old/default used in displayed parity figures (not broad_family)",
            "",
            *validation,
            "",
            "Exclusions",
            *exclusion_lines("sigma", sigma_diagnostics),
            *exclusion_lines("ZT", zt_diagnostics),
            "",
            "Final status: PASS",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="common_cohort_log10_", dir=output.parent))
    try:
        sigma.to_csv(
            stage / "sigma_common_cohort_unified_log10_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        zt.to_csv(
            stage / "zt_common_cohort_unified_log10_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        (stage / "common_cohort_unified_log10_metrics_summary.md").write_text(
            summary, encoding="utf-8"
        )
        (stage / "validation_log.txt").write_text(log, encoding="utf-8")
        if output.exists():
            shutil.rmtree(output)
        stage.replace(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    print(f"Wrote: {output}")
    print("\nSIGMA")
    print(sigma.to_string(index=False))
    print("\nZT")
    print(zt.to_string(index=False))


if __name__ == "__main__":
    main()
