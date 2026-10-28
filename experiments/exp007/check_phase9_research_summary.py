"""Independent validation of Phase 9 research-synthesis outputs."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
FIGURES = EXP007 / "figures"

KEY = DATA / "phase9_key_results.csv"
CLAIMS = DATA / "phase9_claim_evidence_matrix.csv"
GROUPS = DATA / "phase9_group_result_summary.csv"
FLOW = DATA / "phase9_study_flow.csv"
MANIFEST = DATA / "phase9_input_manifest.csv"
BUILD_CHECKS = DATA / "phase9_validation_checks.csv"
REPORT = REPORTS / "phase9_research_results_summary.md"
ABSTRACT = REPORTS / "phase9_abstract_ja_en.md"
FIGURE = FIGURES / "figure_phase9_research_summary.png"
OUTPUT_LOG = REPORTS / "phase9_check_log.txt"

SOURCE_METRICS = DATA / "phase7_paired_metric_evaluation.csv"
SOURCE_BOOTSTRAP = DATA / "phase7_bootstrap_intervals.csv"
SOURCE_GROUPS = DATA / "phase7_oxide_group_evaluation.csv"
SOURCE_COVERAGE = DATA / "phase7_coverage_loss_summary.csv"
SOURCE_MODEL = DATA / "phase8_model_sensitivity_metrics.csv"


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)


def main() -> None:
    expected = [KEY, CLAIMS, GROUPS, FLOW, MANIFEST, BUILD_CHECKS, REPORT, ABSTRACT, FIGURE]
    checks: list[tuple[str, bool, str]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file()]
    add("all_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 9 outputs")

    key = pd.read_csv(KEY, low_memory=False)
    claims = pd.read_csv(CLAIMS, low_memory=False)
    groups = pd.read_csv(GROUPS, low_memory=False)
    flow = pd.read_csv(FLOW, low_memory=False)
    manifest = pd.read_csv(MANIFEST, low_memory=False)
    build = pd.read_csv(BUILD_CHECKS, low_memory=False)
    metrics = pd.read_csv(SOURCE_METRICS, low_memory=False)
    bootstrap = pd.read_csv(SOURCE_BOOTSTRAP, low_memory=False)
    source_groups = pd.read_csv(SOURCE_GROUPS, low_memory=False)
    coverage = pd.read_csv(SOURCE_COVERAGE, low_memory=False)
    model = pd.read_csv(SOURCE_MODEL, low_memory=False)

    add("build_checks_pass", build["status"].eq("PASS").all(), f"pass={int(build['status'].eq('PASS').sum())}/{len(build)}")
    add("key_results_complete", len(key) == 12 and key["result_id"].is_unique, f"records={len(key)}")
    add("claims_complete", len(claims) == 9 and claims["claim_id"].is_unique, f"claims={len(claims)}")
    add("group_summary_complete", len(groups) == 12 and groups["oxide_v1_group_key"].is_unique, f"groups={len(groups)}")
    add("study_flow_complete", list(flow["study_role"]) == ["Observation", "Intervention", "Descriptive regrouping", "Controlled re-prediction", "Primary evaluation", "Robustness", "Research synthesis"], f"steps={len(flow)}")
    add("manifest_complete", len(manifest) == 14 and manifest["sha256"].str.fullmatch(r"[0-9a-f]{64}").all(), f"inputs={len(manifest)}")

    primary = metrics[
        metrics["analysis_scope"].eq("oxide_default_eligible")
        & metrics["metric_weighting"].eq("row_equal")
        & metrics["metric_name"].isin(["mae_log10", "rmse_log10", "factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"])
    ].set_index("metric_name")
    for metric_name in primary.index:
        result = key[key["result_id"].eq(f"row_equal_{metric_name}")].iloc[0]
        source = primary.loc[metric_name]
        matched = (
            close(result["baseline_value"], source["baseline_value"])
            and close(result["oxide_v1_value"], source["oxide_v1_value"])
            and close(result["delta_oxide_v1_minus_baseline"], source["delta_oxide_v1_minus_baseline"])
        )
        add(f"key_metric_matches_{metric_name}", matched, f"delta={source['delta_oxide_v1_minus_baseline']:.6f}")

    boot = bootstrap[
        bootstrap["analysis_scope"].eq("oxide_default_eligible")
        & bootstrap["metric_weighting"].eq("row_equal")
        & bootstrap["metric_name"].isin(["mae_log10", "rmse_log10", "factor_2_accuracy"])
    ].set_index("metric_name")
    for metric_name in boot.index:
        result = key[key["result_id"].eq(f"row_equal_{metric_name}")].iloc[0]
        matched = close(result["ci95_lower"], boot.loc[metric_name, "ci95_lower"]) and close(result["ci95_upper"], boot.loc[metric_name, "ci95_upper"])
        add(f"bootstrap_interval_matches_{metric_name}", matched, f"CI=[{boot.loc[metric_name, 'ci95_lower']:.6f},{boot.loc[metric_name, 'ci95_upper']:.6f}]")

    coverage_source = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    coverage_result = key[key["result_id"].eq("prediction_coverage")].iloc[0]
    add("coverage_matches", close(coverage_result["baseline_value"], coverage_source["baseline_coverage_fraction"]) and close(coverage_result["oxide_v1_value"], coverage_source["oxide_v1_coverage_fraction"]), f"coverage={coverage_source['baseline_coverage_fraction']:.6f}->{coverage_source['oxide_v1_coverage_fraction']:.6f}")

    group_merge = groups.merge(source_groups[["oxide_v1_group_key", "row_delta_mae_log10"]], on="oxide_v1_group_key", suffixes=("", "_source"), validate="one_to_one")
    max_group_diff = float((group_merge["row_delta_mae_log10"] - group_merge["row_delta_mae_log10_source"]).abs().max())
    add("group_deltas_match", max_group_diff < 1e-12, f"max_abs_diff={max_group_diff:.3e}")
    as_p = groups[groups["oxide_v1_group_key"].eq("oxide_v1::mixed_anion_O_As_P")].iloc[0]
    add("group_exception_stated", as_p["robust_direction"] == "consistently_worsened" and as_p["paired_samples"] == 6, f"direction={as_p['robust_direction']}; n={int(as_p['paired_samples'])}")
    add("robustness_counts_match", len(model) == 39 and ((model["delta_mae_log10"] < 0) & (model["delta_rmse_log10"] < 0)).all(), "39/39 model-setting x weighting comparisons improve")

    report_text = REPORT.read_text(encoding="utf-8")
    abstract_text = ABSTRACT.read_text(encoding="utf-8")
    add("report_main_numbers_present", all(token in report_text for token in ["2,984行", "552試料", "395論文", "1.023", "0.875", "98.417%"]), "Primary cohort, performance, and coverage numbers are present")
    add("report_balances_claims", all(token in report_text for token in ["Factor-2精度の改善は不確実", "予測可能率は低下", "DOI単位ホールドアウト", "結晶構造"]), "Uncertainty, coverage, external validation, and structure limitations are explicit")
    add("abstract_bilingual_and_consistent", "## 日本語要旨" in abstract_text and "## English abstract" in abstract_text and "2,984" in abstract_text and "14.5%" in abstract_text, "Both abstracts contain the primary cohort and effect size")
    add("figure_nonempty", FIGURE.stat().st_size > 100_000, f"bytes={FIGURE.stat().st_size}")

    passed = all(condition for _, condition, _ in checks)
    lines = [
        "Phase 9 independent validation",
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
