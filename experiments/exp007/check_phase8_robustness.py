"""Independent output checks for Phase 8 robustness artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
FIGURES = EXP007 / "figures"

PREDICTIONS = DATA / "phase8_model_sensitivity_predictions.csv"
METRICS = DATA / "phase8_model_sensitivity_metrics.csv"
EVALUATION = DATA / "phase8_evaluation_sensitivity.csv"
LEAVE_ONE_OUT = DATA / "phase8_leave_one_group_out.csv"
GROUPS = DATA / "phase8_group_direction_consistency.csv"
DECISION = DATA / "phase8_decision_summary.csv"
BUILD_CHECKS = DATA / "phase8_validation_checks.csv"
FROZEN = DATA / "phase6_prediction_comparison_rows.csv"
REPORT = REPORTS / "phase8_robustness_check.md"
WORKBOOK = REPORTS / "phase8_robustness_check.xlsx"
WORKBOOK_LOG = REPORTS / "phase8_workbook_validation_log.txt"
OUTPUT_LOG = REPORTS / "phase8_check_log.txt"

FIGURES_EXPECTED = [
    FIGURES / "figure_phase8_model_sensitivity.png",
    FIGURES / "figure_phase8_evaluation_sensitivity.png",
    FIGURES / "figure_phase8_leave_one_group_out.png",
]


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def residual_metrics(values: pd.Series) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    return float(np.mean(np.abs(array))), float(np.sqrt(np.mean(array**2)))


def main() -> None:
    expected = [
        PREDICTIONS, METRICS, EVALUATION, LEAVE_ONE_OUT, GROUPS, DECISION,
        BUILD_CHECKS, REPORT, WORKBOOK, WORKBOOK_LOG, *FIGURES_EXPECTED,
    ]
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file()]
    add("all_expected_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 8 outputs: " + ", ".join(missing))

    predictions = pd.read_csv(PREDICTIONS, low_memory=False)
    metrics = pd.read_csv(METRICS, low_memory=False)
    evaluation = pd.read_csv(EVALUATION, low_memory=False)
    loo = pd.read_csv(LEAVE_ONE_OUT, low_memory=False)
    groups = pd.read_csv(GROUPS, low_memory=False)
    decision = pd.read_csv(DECISION, low_memory=False)
    build = pd.read_csv(BUILD_CHECKS, low_memory=False)
    frozen = pd.read_csv(FROZEN, low_memory=False)

    add("build_checks_pass", build["status"].eq("PASS").all(), f"pass={int(build['status'].eq('PASS').sum())}/{len(build)}")
    add("scenario_shape", predictions["scenario_id"].nunique() == 13 and len(predictions) == 13 * 3032, f"scenarios={predictions['scenario_id'].nunique()}; rows={len(predictions):,}")
    add("scenario_row_keys_unique", not predictions.duplicated(["scenario_id", "row_id"]).any(), "scenario_id + row_id is unique")
    add("metric_grid_complete", len(metrics) == 13 * 3 and set(metrics["metric_weighting"]) == {"row_equal", "sample_equal", "paper_equal"}, f"records={len(metrics)}")

    scenario = predictions[predictions["scenario_id"].eq("bw100_frozen_sample_median")].copy()
    expected_frozen = frozen[
        as_bool(frozen["is_phase0_broad_oxide"]) & as_bool(frozen["oxide_v1_default_model_eligible"])
    ][["row_id", "baseline_prediction_status", "phase6_prediction_status", "baseline_residual_log10", "phase6_residual_log10"]]
    joined = scenario.merge(expected_frozen, on="row_id", suffixes=("", "_expected"), validate="one_to_one")
    statuses = joined["baseline_prediction_status"].eq(joined["baseline_prediction_status_expected"]).all() and joined["oxide_v1_prediction_status"].eq(joined["phase6_prediction_status"]).all()
    add("frozen_status_reproduced", statuses, f"rows={len(joined):,}")
    old_ok = joined["baseline_prediction_status"].eq("ok")
    new_ok = joined["oxide_v1_prediction_status"].eq("ok")
    old_diff = float(np.max(np.abs(joined.loc[old_ok, "baseline_residual_log10"] - joined.loc[old_ok, "baseline_residual_log10_expected"])))
    new_diff = float(np.max(np.abs(joined.loc[new_ok, "oxide_v1_residual_log10"] - joined.loc[new_ok, "phase6_residual_log10"])))
    add("frozen_residuals_reproduced", old_diff < 1e-10 and new_diff < 1e-10, f"max_abs_diff={max(old_diff, new_diff):.3e}")

    paired = scenario[as_bool(scenario["paired_prediction_available"])]
    base_mae, base_rmse = residual_metrics(paired["baseline_residual_log10"])
    new_mae, new_rmse = residual_metrics(paired["oxide_v1_residual_log10"])
    metric_row = metrics[(metrics["scenario_id"].eq("bw100_frozen_sample_median")) & metrics["metric_weighting"].eq("row_equal")].iloc[0]
    metric_match = all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
        for a, b in [
            (base_mae, metric_row["baseline_mae_log10"]),
            (new_mae, metric_row["oxide_v1_mae_log10"]),
            (base_rmse, metric_row["baseline_rmse_log10"]),
            (new_rmse, metric_row["oxide_v1_rmse_log10"]),
        ]
    )
    add("frozen_metrics_recomputed", metric_match, f"MAE {base_mae:.6f}->{new_mae:.6f}; RMSE {base_rmse:.6f}->{new_rmse:.6f}")
    add("all_model_comparisons_improve", ((metrics["delta_mae_log10"] < 0) & (metrics["delta_rmse_log10"] < 0)).all(), f"improved={int(((metrics['delta_mae_log10'] < 0) & (metrics['delta_rmse_log10'] < 0)).sum())}/{len(metrics)}")
    add("evaluation_sensitivities_improve", len(evaluation) == 6 and ((evaluation["delta_mae_log10"] < 0) & (evaluation["delta_rmse_log10"] < 0)).all(), f"improved={int(((evaluation['delta_mae_log10'] < 0) & (evaluation['delta_rmse_log10'] < 0)).sum())}/{len(evaluation)}")
    add("leave_one_out_improves", len(loo) == 24 and ((loo["delta_mae_log10"] < 0) & (loo["delta_rmse_log10"] < 0)).all(), f"improved={int(((loo['delta_mae_log10'] < 0) & (loo['delta_rmse_log10'] < 0)).sum())}/{len(loo)}")
    add("group_summary_complete", len(groups) == 12 and groups["evaluated_scenarios"].eq(13).all(), f"groups={len(groups)}")
    add("decision_verdict_present", decision.loc[decision["item"].eq("overall_verdict"), "result"].eq("robust_with_coverage_and_group_specific_caveats").any(), "Expected qualified robustness verdict")

    workbook_log = WORKBOOK_LOG.read_text(encoding="utf-8")
    add("workbook_validation_pass", "status=PASS" in workbook_log and "formula_error_matches=0" in workbook_log and "previews=6" in workbook_log, "Workbook export, render, and formula scan passed")
    add("figures_nonempty", all(path.stat().st_size > 20_000 for path in FIGURES_EXPECTED), "; ".join(f"{path.name}={path.stat().st_size}" for path in FIGURES_EXPECTED))
    report_text = REPORT.read_text(encoding="utf-8")
    add("report_contains_limits", "not constitute external validation" in report_text and "coverage" in report_text.casefold(), "Report states external-validation and coverage limitations")

    passed = all(condition for _, condition, _ in checks)
    lines = [
        "Phase 8 independent validation",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"pass={sum(condition for _, condition, _ in checks)}",
        *[f"{'PASS' if condition else 'FAIL'} {name}: {detail}" for name, condition, detail in checks],
    ]
    OUTPUT_LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
