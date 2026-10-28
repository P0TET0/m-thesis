"""Build the Phase 9 research synthesis from frozen Phase 3-8 evidence.

Phase 9 does not fit a new model. It consolidates the controlled oxide
classification experiment into publication-ready claims, result tables,
abstract text, a summary figure, and explicit limitations.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
FIGURES = EXP007 / "figures"

PHASE3_INVENTORY = DATA / "phase3_oxide_inventory_summary.csv"
PHASE4_INVENTORY = DATA / "phase4_oxide_v1_inventory_summary.csv"
PHASE7_METRICS = DATA / "phase7_paired_metric_evaluation.csv"
PHASE7_BOOTSTRAP = DATA / "phase7_bootstrap_intervals.csv"
PHASE7_GROUPS = DATA / "phase7_oxide_group_evaluation.csv"
PHASE7_SAMPLE_CHANGE = DATA / "phase7_sample_change_summary.csv"
PHASE7_COVERAGE = DATA / "phase7_coverage_loss_summary.csv"
PHASE7_CHECK = REPORTS / "phase7_check_log.txt"
PHASE8_MODEL = DATA / "phase8_model_sensitivity_metrics.csv"
PHASE8_EVALUATION = DATA / "phase8_evaluation_sensitivity.csv"
PHASE8_LOO = DATA / "phase8_leave_one_group_out.csv"
PHASE8_GROUPS = DATA / "phase8_group_direction_consistency.csv"
PHASE8_CHECK = REPORTS / "phase8_check_log.txt"
POSTER = Path(r"C:\Users\miots\OneDrive - 山陽小野田市立山口東京理科大学 (1)\USBにいたヤツ\大学\1_研究\TSJ2026\ポスター\熱電学会2026_塚本ver6.pptx")

OUT_KEY_RESULTS = DATA / "phase9_key_results.csv"
OUT_CLAIMS = DATA / "phase9_claim_evidence_matrix.csv"
OUT_GROUPS = DATA / "phase9_group_result_summary.csv"
OUT_STUDY_FLOW = DATA / "phase9_study_flow.csv"
OUT_MANIFEST = DATA / "phase9_input_manifest.csv"
OUT_VALIDATION = DATA / "phase9_validation_checks.csv"
OUT_REPORT = REPORTS / "phase9_research_results_summary.md"
OUT_ABSTRACT = REPORTS / "phase9_abstract_ja_en.md"
OUT_LOG = REPORTS / "phase9_validation_log.txt"
OUT_FIGURE = FIGURES / "figure_phase9_research_summary.png"

ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def require_inputs() -> None:
    paths = [
        PHASE3_INVENTORY, PHASE4_INVENTORY, PHASE7_METRICS, PHASE7_BOOTSTRAP,
        PHASE7_GROUPS, PHASE7_SAMPLE_CHANGE, PHASE7_COVERAGE, PHASE7_CHECK,
        PHASE8_MODEL, PHASE8_EVALUATION, PHASE8_LOO, PHASE8_GROUPS,
        PHASE8_CHECK, POSTER,
    ]
    missing = [display_path(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 9 inputs: {missing}")
    for label, path in [("Phase 7", PHASE7_CHECK), ("Phase 8", PHASE8_CHECK)]:
        if "status=PASS" not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"{label} independent validation is not PASS")


def build_manifest() -> pd.DataFrame:
    sources = [
        ("Prior conference poster and study framing", POSTER),
        ("Broad-oxide cohort inventory", PHASE3_INVENTORY),
        ("oxide_v1 cohort inventory", PHASE4_INVENTORY),
        ("Primary paired metrics", PHASE7_METRICS),
        ("Sample-cluster bootstrap intervals", PHASE7_BOOTSTRAP),
        ("Oxide-group evaluation", PHASE7_GROUPS),
        ("Sample-level change summary", PHASE7_SAMPLE_CHANGE),
        ("Coverage summary", PHASE7_COVERAGE),
        ("Phase 7 independent validation", PHASE7_CHECK),
        ("Model-setting robustness metrics", PHASE8_MODEL),
        ("Weighting and outlier sensitivity", PHASE8_EVALUATION),
        ("Leave-one-group-out results", PHASE8_LOO),
        ("Group direction consistency", PHASE8_GROUPS),
        ("Phase 8 independent validation", PHASE8_CHECK),
    ]
    return pd.DataFrame(
        [{"role": role, "path": display_path(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for role, path in sources]
    )


def metric_lookup(metrics: pd.DataFrame, weighting: str, metric: str) -> pd.Series:
    return metrics[
        metrics["analysis_scope"].eq("oxide_default_eligible")
        & metrics["metric_weighting"].eq(weighting)
        & metrics["metric_name"].eq(metric)
    ].iloc[0]


def bootstrap_lookup(bootstrap: pd.DataFrame, weighting: str, metric: str) -> pd.Series:
    return bootstrap[
        bootstrap["analysis_scope"].eq("oxide_default_eligible")
        & bootstrap["metric_weighting"].eq(weighting)
        & bootstrap["metric_name"].eq(metric)
    ].iloc[0]


def make_metric_result(metrics: pd.DataFrame, bootstrap: pd.DataFrame, metric: str, label_ja: str, unit: str, weighting: str = "row_equal") -> dict[str, Any]:
    row = metric_lookup(metrics, weighting, metric)
    boot = bootstrap_lookup(bootstrap, weighting, metric)
    return {
        "result_id": f"{weighting}_{metric}",
        "section": "primary_paired_performance",
        "result_label_ja": label_ja,
        "analysis_scope": "oxide_default_eligible_paired",
        "metric_weighting": weighting,
        "baseline_value": row["baseline_value"],
        "oxide_v1_value": row["oxide_v1_value"],
        "delta_oxide_v1_minus_baseline": row["delta_oxide_v1_minus_baseline"],
        "relative_change_percent": 100.0 * row["relative_delta_fraction"] if pd.notna(row["relative_delta_fraction"]) else np.nan,
        "ci95_lower": boot["ci95_lower"],
        "ci95_upper": boot["ci95_upper"],
        "unit": unit,
        "evidence_interpretation": boot["interval_interpretation"],
        "summary_text_ja": "",
        "evidence_source": f"{display_path(PHASE7_METRICS)}; {display_path(PHASE7_BOOTSTRAP)}",
    }


def build_key_results(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    sample_change: pd.DataFrame,
    coverage: pd.DataFrame,
    model: pd.DataFrame,
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        make_metric_result(metrics, bootstrap, "mean_log10_error", "平均符号付きlog10誤差", "log10", "row_equal"),
        make_metric_result(metrics, bootstrap, "mae_log10", "MAE", "log10", "row_equal"),
        make_metric_result(metrics, bootstrap, "rmse_log10", "RMSE", "log10", "row_equal"),
        make_metric_result(metrics, bootstrap, "factor_2_accuracy", "Factor-2精度", "fraction", "row_equal"),
        make_metric_result(metrics, bootstrap, "factor_5_accuracy", "Factor-5精度", "fraction", "row_equal"),
        make_metric_result(metrics, bootstrap, "factor_10_accuracy", "Factor-10精度", "fraction", "row_equal"),
        make_metric_result(metrics, bootstrap, "mae_log10", "試料均等MAE", "log10", "sample_equal"),
    ]
    primary_samples = sample_change[
        sample_change["summary_level"].eq("primary_scope")
        & sample_change["scope_or_group"].eq("oxide_default_eligible")
    ].iloc[0]
    primary_coverage = coverage[
        coverage["summary_level"].eq("analysis_scope")
        & coverage["scope_or_group"].eq("oxide_default_eligible")
    ].iloc[0]
    row_model = model[model["metric_weighting"].eq("row_equal")]
    extra = [
        {
            "result_id": "prediction_coverage",
            "section": "coverage",
            "result_label_ja": "予測可能率",
            "analysis_scope": "oxide_default_eligible_all_test_rows",
            "metric_weighting": "row_equal",
            "baseline_value": primary_coverage["baseline_coverage_fraction"],
            "oxide_v1_value": primary_coverage["oxide_v1_coverage_fraction"],
            "delta_oxide_v1_minus_baseline": primary_coverage["coverage_delta_oxide_v1_minus_baseline"],
            "relative_change_percent": 100.0 * primary_coverage["coverage_delta_oxide_v1_minus_baseline"] / primary_coverage["baseline_coverage_fraction"],
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "unit": "fraction",
            "evidence_interpretation": "coverage_cost",
            "summary_text_ja": f"{int(primary_coverage['lost_prediction_rows'])}行・{int(primary_coverage['lost_prediction_samples'])}試料で予測を失った",
            "evidence_source": display_path(PHASE7_COVERAGE),
        },
        {
            "result_id": "sample_level_direction",
            "section": "sample_level",
            "result_label_ja": "試料単位の改善方向",
            "analysis_scope": "oxide_default_eligible_paired_samples",
            "metric_weighting": "sample_equal",
            "baseline_value": np.nan,
            "oxide_v1_value": primary_samples["improved_fraction_excluding_ties"],
            "delta_oxide_v1_minus_baseline": np.nan,
            "relative_change_percent": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "unit": "fraction",
            "evidence_interpretation": "supported_improvement",
            "summary_text_ja": f"{int(primary_samples['improved_samples'])}改善 / {int(primary_samples['worsened_samples'])}悪化 / {int(primary_samples['unchanged_samples'])}不変、exact sign p={primary_samples['two_sided_exact_sign_test_p']:.3g}",
            "evidence_source": display_path(PHASE7_SAMPLE_CHANGE),
        },
        {
            "result_id": "model_setting_robustness",
            "section": "robustness",
            "result_label_ja": "モデル設定感度",
            "analysis_scope": "13_model_settings_x_3_weightings",
            "metric_weighting": "multiple",
            "baseline_value": np.nan,
            "oxide_v1_value": float(((model["delta_mae_log10"] < 0) & (model["delta_rmse_log10"] < 0)).mean()),
            "delta_oxide_v1_minus_baseline": np.nan,
            "relative_change_percent": np.nan,
            "ci95_lower": row_model["delta_mae_log10"].min(),
            "ci95_upper": row_model["delta_mae_log10"].max(),
            "unit": "fraction_and_delta_range",
            "evidence_interpretation": "robust_direction",
            "summary_text_ja": f"39/39比較でMAEとRMSEが同時に改善。行均等MAE差の範囲 {row_model['delta_mae_log10'].min():+.6f}〜{row_model['delta_mae_log10'].max():+.6f}",
            "evidence_source": display_path(PHASE8_MODEL),
        },
        {
            "result_id": "evaluation_sensitivity",
            "section": "robustness",
            "result_label_ja": "重み付け・外れ値感度",
            "analysis_scope": "6_predefined_sensitivities",
            "metric_weighting": "multiple",
            "baseline_value": np.nan,
            "oxide_v1_value": float(((evaluation["delta_mae_log10"] < 0) & (evaluation["delta_rmse_log10"] < 0)).mean()),
            "delta_oxide_v1_minus_baseline": np.nan,
            "relative_change_percent": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "unit": "fraction",
            "evidence_interpretation": "robust_direction",
            "summary_text_ja": "6/6条件でMAEとRMSEが同時に改善",
            "evidence_source": display_path(PHASE8_EVALUATION),
        },
        {
            "result_id": "leave_one_group_out",
            "section": "robustness",
            "result_label_ja": "一群除外感度",
            "analysis_scope": "12_groups_x_2_weightings",
            "metric_weighting": "row_and_sample_equal",
            "baseline_value": np.nan,
            "oxide_v1_value": float(((loo["delta_mae_log10"] < 0) & (loo["delta_rmse_log10"] < 0)).mean()),
            "delta_oxide_v1_minus_baseline": np.nan,
            "relative_change_percent": np.nan,
            "ci95_lower": np.nan,
            "ci95_upper": np.nan,
            "unit": "fraction",
            "evidence_interpretation": "not_driven_by_single_group",
            "summary_text_ja": "24/24比較でMAEとRMSEが同時に改善",
            "evidence_source": display_path(PHASE8_LOO),
        },
    ]
    return pd.DataFrame([*rows, *extra])


def build_group_summary(groups: pd.DataFrame, robustness: pd.DataFrame) -> pd.DataFrame:
    merged = groups.merge(robustness, on=["oxide_v1_group_key", "oxide_v1_display_name_en"], how="left", validate="one_to_one")
    merged["relative_mae_change_percent"] = 100.0 * merged["row_delta_mae_log10"] / merged["row_baseline_mae_log10"]

    def interpret(row: pd.Series) -> str:
        if row["robust_direction"] == "consistently_improved":
            return "consistent_gain"
        if row["robust_direction"] == "consistently_worsened":
            return "consistent_loss"
        if row["row_delta_mae_log10"] < 0:
            return "gain_in_frozen_setting_but_setting_sensitive"
        return "loss_in_frozen_setting_and_setting_sensitive"

    merged["research_interpretation"] = merged.apply(interpret, axis=1)
    columns = [
        "oxide_v1_group_key", "oxide_v1_display_name_en", "oxide_v1_display_name_ja",
        "eligible_test_rows", "paired_rows", "eligible_samples", "paired_samples", "n_papers",
        "baseline_coverage_fraction", "oxide_v1_coverage_fraction", "coverage_delta_oxide_v1_minus_baseline",
        "row_baseline_mae_log10", "row_oxide_v1_mae_log10", "row_delta_mae_log10", "relative_mae_change_percent",
        "row_baseline_rmse_log10", "row_oxide_v1_rmse_log10", "row_delta_rmse_log10",
        "sample_baseline_mae_log10", "sample_oxide_v1_mae_log10", "sample_delta_mae_log10",
        "improved_samples", "worsened_samples", "unchanged_samples", "exact_sign_test_p", "support_note",
        "evaluated_scenarios", "scenarios_mae_improved", "scenarios_mae_worsened", "improvement_fraction",
        "median_delta_mae_log10", "minimum_delta_mae_log10", "maximum_delta_mae_log10", "minimum_paired_samples",
        "robust_direction", "research_interpretation",
    ]
    return merged[columns].sort_values("row_delta_mae_log10").reset_index(drop=True)


def build_claims(key_results: pd.DataFrame, group_summary: pd.DataFrame) -> pd.DataFrame:
    lookup = key_results.set_index("result_id")
    mae = lookup.loc["row_equal_mae_log10"]
    rmse = lookup.loc["row_equal_rmse_log10"]
    f2 = lookup.loc["row_equal_factor_2_accuracy"]
    f5 = lookup.loc["row_equal_factor_5_accuracy"]
    f10 = lookup.loc["row_equal_factor_10_accuracy"]
    coverage = lookup.loc["prediction_coverage"]
    as_p = group_summary[group_summary["oxide_v1_group_key"].eq("oxide_v1::mixed_anion_O_As_P")].iloc[0]
    records = [
        ("C01", "酸化物を一括した分類よりもoxide_v1の細分類の方が、共通予測可能行における電気伝導率予測のMAEを低下させた。", "supported", f"MAE {mae['baseline_value']:.6f}→{mae['oxide_v1_value']:.6f}; 差 {mae['delta_oxide_v1_minus_baseline']:+.6f}; 95% CI [{mae['ci95_lower']:+.6f}, {mae['ci95_upper']:+.6f}]", "default-eligible oxide paired rows", "同一データと固定sample holdoutによる内部比較", mae["evidence_source"]),
        ("C02", "oxide_v1はRMSEも低下させ、特に大きな予測誤差を抑えた。", "supported", f"RMSE {rmse['baseline_value']:.6f}→{rmse['oxide_v1_value']:.6f}; 差 {rmse['delta_oxide_v1_minus_baseline']:+.6f}; 95% CI [{rmse['ci95_lower']:+.6f}, {rmse['ci95_upper']:+.6f}]", "default-eligible oxide paired rows", "RMSEは少数の大誤差の影響を受ける", rmse["evidence_source"]),
        ("C03", "Factor-5およびFactor-10精度は改善した。", "supported", f"Factor-5 {100*f5['baseline_value']:.1f}%→{100*f5['oxide_v1_value']:.1f}%; Factor-10 {100*f10['baseline_value']:.1f}%→{100*f10['oxide_v1_value']:.1f}%", "default-eligible oxide paired rows", "Factor-2より緩い許容誤差での評価", f5["evidence_source"]),
        ("C04", "Factor-2精度が改善したとは断定できない。", "inconclusive", f"差 {100*f2['delta_oxide_v1_minus_baseline']:+.2f} percentage points; 95% CI [{100*f2['ci95_lower']:+.2f}, {100*f2['ci95_upper']:+.2f}]", "default-eligible oxide paired rows", "信頼区間が0をまたぐ", f2["evidence_source"]),
        ("C05", "全体の改善は特定の重み付け、外れ値、または単一の酸化物群だけでは説明できない。", "supported_internal_robustness", "39/39モデル設定×重み付け、6/6評価感度、24/24一群除外比較でMAEとRMSEが改善", "predefined internal sensitivity analyses", "外部データやDOI holdoutではない", f"{display_path(PHASE8_MODEL)}; {display_path(PHASE8_EVALUATION)}; {display_path(PHASE8_LOO)}"),
        ("C06", "細分類には予測可能率の低下という代償がある。", "supported", f"coverage {100*coverage['baseline_value']:.3f}%→{100*coverage['oxide_v1_value']:.3f}% ({100*coverage['delta_oxide_v1_minus_baseline']:+.3f} pp)", "all default-eligible oxide test rows", "細かい参照群で学習支持が不足する", coverage["evidence_source"]),
        ("C07", "すべての酸化物群が改善したわけではない。", "supported_descriptive", f"Mixed-anion O + As/P: MAE差 {as_p['row_delta_mae_log10']:+.6f}; 13/13設定で悪化", "group-level descriptive analysis", "当該群は6 paired samplesで非常に小さい", f"{display_path(PHASE7_GROUPS)}; {display_path(PHASE8_GROUPS)}"),
        ("C08", "本結果は、組成に基づく材料分類の粒度が電気伝導率予測の誤差要因の一つであることを示す。", "supported_inference", "分類以外の主要条件を固定した比較でMAEとRMSEが低下", "current dataset and sample holdout", "分類の因果効果や外部一般化を直接証明するものではない", f"{display_path(PHASE7_METRICS)}; {display_path(PHASE8_MODEL)}"),
        ("C09", "結晶構造、ドーピング、焼結条件などの追加情報が不要になったとは言えない。", "limitation", "formula-only oxide_v1でも群別例外と大きな残差が残った", "study interpretation", "oxide_v1は結晶構造を割り当てていない", f"{display_path(PHASE7_GROUPS)}; {display_path(PHASE8_GROUPS)}"),
    ]
    return pd.DataFrame(records, columns=["claim_id", "claim_text_ja", "evidence_status", "quantitative_basis", "scope", "caveat", "evidence_source"])


def build_study_flow() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("Observation", "Prior poster / Phases 1-3", "Broad oxide showed large errors and overprediction; 3,437 held-out rows, 610 samples, 439 papers were inventoried."),
            ("Intervention", "Phase 4", "Define formula-only oxide_v1 with explicit include, review, and exclude rules; no structure labels are inferred."),
            ("Descriptive regrouping", "Phase 5", "Reaggregate frozen predictions under oxide_v1 to identify group-specific residual patterns without retraining."),
            ("Controlled re-prediction", "Phase 6", "Change only the oxide material-group key before rebuilding train-only reference curves; keep the sample holdout and other model conditions fixed."),
            ("Primary evaluation", "Phase 7", "Compare paired default-eligible rows using row- and sample-equal metrics, sample-cluster bootstrap intervals, and separate coverage accounting."),
            ("Robustness", "Phase 8", "Vary bin width, support thresholds, aggregation, weighting, extreme-row inclusion, and omitted oxide group."),
            ("Research synthesis", "Phase 9", "State supported claims, exceptions, limitations, and conference-ready abstract text."),
        ],
        columns=["study_role", "phase", "method_and_output"],
    )


def make_summary_figure(metrics: pd.DataFrame, groups: pd.DataFrame, model: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"baseline": "#6C7A89", "oxide": "#2A9D8F"}
    fig = plt.figure(figsize=(15, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.35], hspace=0.38, wspace=0.30)

    ax1 = fig.add_subplot(grid[0, 0])
    error_names = ["mae_log10", "rmse_log10"]
    labels = ["MAE", "RMSE"]
    old = [metric_lookup(metrics, "row_equal", name)["baseline_value"] for name in error_names]
    new = [metric_lookup(metrics, "row_equal", name)["oxide_v1_value"] for name in error_names]
    x = np.arange(len(labels))
    width = 0.34
    ax1.bar(x - width / 2, old, width, color=colors["baseline"], label="Broad oxide")
    ax1.bar(x + width / 2, new, width, color=colors["oxide"], label="oxide_v1")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("log10 ratio error")
    ax1.set_title("A  Paired prediction error")
    ax1.legend(frameon=False)
    for i, values in enumerate(zip(old, new)):
        for offset, value in zip([-width / 2, width / 2], values):
            ax1.text(i + offset, value + 0.035, f"{value:.3f}", ha="center", va="bottom", fontsize=10)

    ax2 = fig.add_subplot(grid[0, 1])
    acc_names = ["factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy"]
    acc_labels = ["Factor 2", "Factor 5", "Factor 10"]
    old_acc = 100 * np.array([metric_lookup(metrics, "row_equal", name)["baseline_value"] for name in acc_names])
    new_acc = 100 * np.array([metric_lookup(metrics, "row_equal", name)["oxide_v1_value"] for name in acc_names])
    x = np.arange(len(acc_labels))
    ax2.bar(x - width / 2, old_acc, width, color=colors["baseline"])
    ax2.bar(x + width / 2, new_acc, width, color=colors["oxide"])
    ax2.set_xticks(x, acc_labels)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 100)
    ax2.set_title("B  Tolerance accuracy")
    for i, values in enumerate(zip(old_acc, new_acc)):
        for offset, value in zip([-width / 2, width / 2], values):
            ax2.text(i + offset, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    ax3 = fig.add_subplot(grid[1, 0])
    ordered = groups.sort_values("row_delta_mae_log10", ascending=True).reset_index(drop=True)
    y = np.arange(len(ordered))
    color_map = {"consistently_improved": "#2A9D8F", "mixed": "#E9C46A", "consistently_worsened": "#E76F51"}
    ax3.barh(y, ordered["row_delta_mae_log10"], color=[color_map[value] for value in ordered["robust_direction"]])
    ax3.axvline(0, color="#333333", linewidth=1)
    ax3.set_yticks(y, ordered["oxide_v1_display_name_en"])
    ax3.invert_yaxis()
    ax3.set_xlabel("Delta MAE (oxide_v1 - broad), log10")
    ax3.set_title("C  Oxide-group effects at the frozen setting")
    for idx, row in ordered.iterrows():
        delta = float(row["row_delta_mae_log10"])
        if delta <= -0.12:
            x_text, ha, text_color = delta / 2.0, "center", "white"
        elif delta < 0:
            x_text, ha, text_color = delta - 0.012, "right", "#222222"
        else:
            x_text, ha, text_color = delta + 0.012, "left", "#222222"
        ax3.text(x_text, idx, f"n={int(row['paired_samples'])}", va="center", ha=ha, fontsize=8, color=text_color)

    ax4 = fig.add_subplot(grid[1, 1])
    row_model = model[model["metric_weighting"].eq("row_equal")].copy()
    scatter_colors = row_model["temperature_bin_width_K"].map({50: "#457B9D", 100: "#2A9D8F", 150: "#E9C46A", 200: "#E76F51"})
    ax4.scatter(
        100 * row_model["coverage_delta_oxide_v1_minus_baseline"],
        row_model["delta_mae_log10"],
        c=scatter_colors,
        s=80,
        edgecolor="white",
        linewidth=0.8,
    )
    ax4.axhline(0, color="#333333", linewidth=1)
    ax4.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax4.set_xlabel("Coverage change (percentage points)")
    ax4.set_ylabel("Delta MAE (oxide_v1 - broad), log10")
    ax4.set_title("D  Model-setting robustness")
    ax4.text(0.03, 0.04, "All 13 settings improve MAE\nCoverage loss grows in sparse settings", transform=ax4.transAxes, fontsize=10, va="bottom")
    for width_value, color in [(50, "#457B9D"), (100, "#2A9D8F"), (150, "#E9C46A"), (200, "#E76F51")]:
        ax4.scatter([], [], color=color, s=55, label=f"{width_value} K")
    ax4.legend(frameon=False, ncol=2, loc="upper left")

    fig.suptitle("Finer oxide classification reduces held-out conductivity error, with coverage and group-specific caveats", fontsize=16, fontweight="bold", y=0.98)
    fig.text(0.5, 0.015, "Primary cohort: 2,984 paired rows, 552 samples, 395 papers; fixed sample holdout and formula-only oxide_v1 taxonomy", ha="center", fontsize=10)
    fig.savefig(OUT_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_abstract(metrics: pd.DataFrame, bootstrap: pd.DataFrame, coverage: pd.DataFrame) -> str:
    mae = metric_lookup(metrics, "row_equal", "mae_log10")
    rmse = metric_lookup(metrics, "row_equal", "rmse_log10")
    f5 = metric_lookup(metrics, "row_equal", "factor_5_accuracy")
    f10 = metric_lookup(metrics, "row_equal", "factor_10_accuracy")
    mae_ci = bootstrap_lookup(bootstrap, "row_equal", "mae_log10")
    coverage_row = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    return f"""# 酸化物の細分類が電気伝導率予測に与える影響

English title: Effect of Finer Oxide Classification on Electrical Conductivity Prediction Using Starrydata2

## 日本語要旨

Starrydata2のゼーベック係数と電気伝導率から推定したフェルミ準位情報を用いる予測法では、酸化物群の低電気伝導率領域に大きな誤差が残っていた。本研究では、従来一括していた酸化物を組成表記に基づくoxide_v1へ細分類し、材料分類以外の条件を固定して電気伝導率予測を再計算した。試料単位で分離したホールドアウトのうち、両分類で予測可能な2,984行、552試料、395論文を主解析とした。行均等MAEは{mae['baseline_value']:.3f}から{mae['oxide_v1_value']:.3f}へ{abs(100*mae['relative_delta_fraction']):.1f}%低下し、差の試料クラスターブートストラップ95%信頼区間は{mae_ci['ci95_lower']:.3f}〜{mae_ci['ci95_upper']:.3f}であった。RMSEは{rmse['baseline_value']:.3f}から{rmse['oxide_v1_value']:.3f}へ低下し、Factor-5精度は{100*f5['baseline_value']:.1f}%から{100*f5['oxide_v1_value']:.1f}%、Factor-10精度は{100*f10['baseline_value']:.1f}%から{100*f10['oxide_v1_value']:.1f}%へ向上した。温度ビン幅、参照支持数、集約法、重み付け、外れ値除外、一群除外を変えた内部感度分析でも全体の改善方向は維持された。一方、予測可能率は{100*coverage_row['baseline_coverage_fraction']:.1f}%から{100*coverage_row['oxide_v1_coverage_fraction']:.1f}%へ低下し、Mixed-anion O + As/Pでは改善しなかった。以上より、組成に基づく材料分類の粒度は電気伝導率予測誤差の一要因であり、酸化物の細分類は大きな誤差の抑制に有効である。ただし、未知論文への一般化、結晶構造、ドーピング、焼結条件の影響は今後の検証課題である。

キーワード: Starrydata2、電気伝導率、フェルミ準位、材料分類、酸化物、機械学習

## English abstract

Electrical conductivity predictions using Fermi-level information inferred from Seebeck coefficient and conductivity data in Starrydata2 retained large errors for oxides, particularly in the low-conductivity region. We tested whether a finer composition-based oxide taxonomy could reduce these errors while keeping the sample holdout, reference-data subset, temperature protocol, aggregation method, and support thresholds fixed. The primary paired analysis included 2,984 rows from 552 samples and 395 papers that were predictable under both the broad-oxide baseline and oxide_v1. Row-weighted MAE decreased from {mae['baseline_value']:.3f} to {mae['oxide_v1_value']:.3f}, a {abs(100*mae['relative_delta_fraction']):.1f}% reduction, with a sample-cluster bootstrap 95% confidence interval of {mae_ci['ci95_lower']:.3f} to {mae_ci['ci95_upper']:.3f} for the paired difference. RMSE decreased from {rmse['baseline_value']:.3f} to {rmse['oxide_v1_value']:.3f}. Factor-5 accuracy increased from {100*f5['baseline_value']:.1f}% to {100*f5['oxide_v1_value']:.1f}%, and factor-10 accuracy increased from {100*f10['baseline_value']:.1f}% to {100*f10['oxide_v1_value']:.1f}%. The overall improvement persisted across predefined changes in temperature-bin width, reference support, aggregation, weighting, extreme-row inclusion, and leave-one-group-out analyses. Prediction coverage decreased from {100*coverage_row['baseline_coverage_fraction']:.1f}% to {100*coverage_row['oxide_v1_coverage_fraction']:.1f}%, and Mixed-anion O + As/P did not improve. These results indicate that material-classification granularity contributes to conductivity-prediction error and that finer oxide grouping reduces large errors, although external DOI-held-out validation and explicit structural and processing descriptors remain necessary.

Keywords: Starrydata2, electrical conductivity, Fermi level, material classification, oxide, machine learning
"""


def build_report(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    groups: pd.DataFrame,
    sample_change: pd.DataFrame,
    coverage: pd.DataFrame,
    model: pd.DataFrame,
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
    claims: pd.DataFrame,
) -> str:
    mae = metric_lookup(metrics, "row_equal", "mae_log10")
    rmse = metric_lookup(metrics, "row_equal", "rmse_log10")
    bias = metric_lookup(metrics, "row_equal", "mean_log10_error")
    sample_mae = metric_lookup(metrics, "sample_equal", "mae_log10")
    f2 = metric_lookup(metrics, "row_equal", "factor_2_accuracy")
    f5 = metric_lookup(metrics, "row_equal", "factor_5_accuracy")
    f10 = metric_lookup(metrics, "row_equal", "factor_10_accuracy")
    mae_ci = bootstrap_lookup(bootstrap, "row_equal", "mae_log10")
    rmse_ci = bootstrap_lookup(bootstrap, "row_equal", "rmse_log10")
    f2_ci = bootstrap_lookup(bootstrap, "row_equal", "factor_2_accuracy")
    primary_samples = sample_change[(sample_change["summary_level"].eq("primary_scope")) & sample_change["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    primary_coverage = coverage[(coverage["summary_level"].eq("analysis_scope")) & coverage["scope_or_group"].eq("oxide_default_eligible")].iloc[0]
    row_model = model[model["metric_weighting"].eq("row_equal")]
    top = groups.sort_values("row_delta_mae_log10").head(4)
    exceptions = groups[groups["robust_direction"].ne("consistently_improved")]

    top_text = "、".join(
        f"{row.oxide_v1_display_name_en}（ΔMAE {row.row_delta_mae_log10:+.3f}）" for row in top.itertuples(index=False)
    )
    exception_text = "、".join(
        f"{row.oxide_v1_display_name_en}（{row.robust_direction}）" for row in exceptions.itertuples(index=False)
    )
    claim_rows = "\n".join(
        f"- **{row.claim_id} {row.evidence_status}**: {row.claim_text_ja} 根拠: {row.quantitative_basis} 注意: {row.caveat}"
        for row in claims.itertuples(index=False)
    )
    return f"""# 酸化物の細分類が電気伝導率予測に与える影響

Starrydata2の輸送物性データとフェルミ準位情報を用いた電気伝導率予測について、従来の一括した酸化物分類を組成ベースのoxide_v1へ細分化した効果を検証した。材料分類以外の主要条件を固定した比較では、oxide_v1がMAEとRMSEを約14%低下させた。この改善は事前に定めた内部感度分析でも維持された。一方、細分類によって予測可能率が低下し、改善しない酸化物群も残った。したがって、材料分類の粒度は予測誤差の一要因だが、細分類だけで誤差問題を解消したとは結論できない。

## 研究背景と研究課題

先日の学会ポスターでは、フェルミ準位情報を考慮した手法が電気伝導率とZTの全体誤差を低下させる一方、酸化物・硫化物などの低電気伝導率領域で過大評価が残ることを示した。その結果から、結晶構造、ドーピング、焼結条件などの材料情報を増やす必要性を指摘した。

本研究では、その前段階として「酸化物を一つの材料群として扱う粗い分類自体が誤差を増やしていないか」を検証した。構造情報を新たに加えず、組成表記から再現可能なoxide_v1だけを導入することで、分類粒度の効果を分離した。

## 研究設計

- 初期のbroad::oxide群は3,437評価行、610試料、439論文で構成された。
- oxide_v1は、組成表記と元素集合から酸化物を細分類し、有機・高分子候補、混合アニオン系、通常の無機酸化物を優先規則で分離した。結晶構造名は付与していない。
- 主比較では、試料ホールドアウト、学習参照データ、100 K温度ビン、sample-median集約、最低支持数を固定し、材料群キーだけを変更した。
- 主解析対象は、default-eligibleで両方式が予測可能な2,984行、552試料、395論文、12酸化物群である。
- 不確実性は、試料をクラスターとする2,000回のpaired bootstrapで評価した。予測可能率はpaired filteringの前に別途評価した。

## 主結果

| 指標 | broad oxide | oxide_v1 | 変化 | 解釈 |
|---|---:|---:|---:|---|
| 平均符号付きlog10誤差 | {bias['baseline_value']:.3f} | {bias['oxide_v1_value']:.3f} | {bias['delta_oxide_v1_minus_baseline']:+.3f} | 過大評価バイアスが縮小 |
| 行均等MAE | {mae['baseline_value']:.3f} | {mae['oxide_v1_value']:.3f} | {100*mae['relative_delta_fraction']:+.1f}% | 95% CI of delta [{mae_ci['ci95_lower']:+.3f}, {mae_ci['ci95_upper']:+.3f}] |
| 行均等RMSE | {rmse['baseline_value']:.3f} | {rmse['oxide_v1_value']:.3f} | {100*rmse['relative_delta_fraction']:+.1f}% | 95% CI of delta [{rmse_ci['ci95_lower']:+.3f}, {rmse_ci['ci95_upper']:+.3f}] |
| 試料均等MAE | {sample_mae['baseline_value']:.3f} | {sample_mae['oxide_v1_value']:.3f} | {100*sample_mae['relative_delta_fraction']:+.1f}% | 温度点の多い試料への偏りを抑えても改善 |
| Factor-2精度 | {100*f2['baseline_value']:.1f}% | {100*f2['oxide_v1_value']:.1f}% | {100*f2['delta_oxide_v1_minus_baseline']:+.2f} pp | 95% CI [{100*f2_ci['ci95_lower']:+.2f}, {100*f2_ci['ci95_upper']:+.2f}]で不確実 |
| Factor-5精度 | {100*f5['baseline_value']:.1f}% | {100*f5['oxide_v1_value']:.1f}% | {100*f5['delta_oxide_v1_minus_baseline']:+.2f} pp | 改善を支持 |
| Factor-10精度 | {100*f10['baseline_value']:.1f}% | {100*f10['oxide_v1_value']:.1f}% | {100*f10['delta_oxide_v1_minus_baseline']:+.2f} pp | 改善を支持 |

試料単位では、{int(primary_samples['improved_samples'])}試料が改善、{int(primary_samples['worsened_samples'])}試料が悪化、{int(primary_samples['unchanged_samples'])}試料が不変だった。タイを除く改善割合は{100*primary_samples['improved_fraction_excluding_ties']:.1f}%で、exact sign testはp={primary_samples['two_sided_exact_sign_test_p']:.3g}だった。

## 酸化物群ごとの結果

改善量が大きかった群は、{top_text}だった。これらの結果は、酸化物内部の組成差を一つの参照曲線に混在させることが、一部の大誤差につながっていた可能性と整合する。

ただし、群別結果は一様ではない。Phase 8の13モデル設定に対し、7群は一貫して改善、4群は設定依存、1群は一貫して悪化した。例外は{exception_text}である。特にMixed-anion O + As/Pは主設定でΔMAEが{groups.loc[groups['oxide_v1_group_key'].eq('oxide_v1::mixed_anion_O_As_P'), 'row_delta_mae_log10'].iloc[0]:+.3f}であり、13設定すべてで悪化した。ただしpaired samplesは6試料であり、推定は不安定である。

## 頑健性

- 温度ビン幅、最低支持数、集約法を変えた13モデル設定において、行・試料・論文の3重み付けを組み合わせた39/39比較でMAEとRMSEが改善した。
- 行均等MAE差は{row_model['delta_mae_log10'].min():+.3f}〜{row_model['delta_mae_log10'].max():+.3f}の範囲で、すべてoxide_v1側が小さかった。
- 重み付け変更と1%、2.5%、5%の対称的な極端値除外を含む{len(evaluation)}/{len(evaluation)}条件で改善方向を維持した。
- 12酸化物群を一群ずつ除外した行均等・試料均等の{len(loo)}/{len(loo)}比較で改善方向を維持した。

これらの結果から、全体改善は特定の酸化物群、一部の外れ値、または行数の多い試料だけに依存していないと判断できる。

## 予測可能率とのトレードオフ

default-eligible全3,032行に対する予測可能率は{100*primary_coverage['baseline_coverage_fraction']:.3f}%から{100*primary_coverage['oxide_v1_coverage_fraction']:.3f}%へ{100*primary_coverage['coverage_delta_oxide_v1_minus_baseline']:+.3f}ポイント変化した。細分類後は{int(primary_coverage['lost_prediction_rows'])}行、{int(primary_coverage['lost_prediction_samples'])}試料で、参照ビンの欠損または支持不足により予測できなかった。Phase 8で最も疎な50 K・strict設定では、予測可能率の差が最大約−5.01ポイントまで拡大した。

精度向上と予測可能率は別の評価軸である。細分類を実用モデルへ採用する場合は、希少群を統合する階層型分類、近接群へのフォールバック、または最低支持数を考慮した正則化が必要になる。

## 研究結果として主張できる範囲

{claim_rows}

## 結論

組成表記だけを用いた酸化物の細分類は、固定sample holdout上の電気伝導率予測で、MAEとRMSEを約14%低下させた。改善は試料均等評価、論文均等評価、外れ値感度、モデル設定感度、一群除外評価でも維持された。この結果は、材料分類の粒度が予測誤差の一要因であり、酸化物を単一群として扱うと異なる輸送特性を持つ組成が混在することを示唆する。

一方、Factor-2精度の改善は不確実で、予測可能率は低下し、Mixed-anion O + As/Pでは悪化した。oxide_v1は結晶構造を表していないため、学会ポスターで示した今後の課題は残る。次の優先課題は、DOI単位ホールドアウトによる未知論文への一般化確認と、結晶構造、ドーピング、焼結条件を含む説明変数の導入である。

## 発表用の短い結論

酸化物を組成に基づいて細分類すると、分類以外の条件を固定した内部ホールドアウトで電気伝導率予測のMAEとRMSEが約14%低下した。改善は複数の感度分析で維持されたが、予測可能率の低下と群別例外が残った。材料分類の細粒度化は有効だが、未知論文への一般化と構造・プロセス情報の導入が次の課題である。

## 成果物

- `{display_path(OUT_KEY_RESULTS)}`: 主要数値と信頼区間
- `{display_path(OUT_CLAIMS)}`: 主張、根拠、適用範囲、注意点
- `{display_path(OUT_GROUPS)}`: 酸化物群別の主評価と頑健性
- `{display_path(OUT_STUDY_FLOW)}`: Phase 3〜9の研究設計
- `{display_path(OUT_ABSTRACT)}`: 日英要旨
- `{display_path(OUT_FIGURE)}`: 統合結果図
"""


def append_check(checks: list[dict[str, str]], name: str, condition: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})


def build_validation(
    key_results: pd.DataFrame,
    claims: pd.DataFrame,
    groups: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    model: pd.DataFrame,
    evaluation: pd.DataFrame,
    loo: pd.DataFrame,
    report: str,
    abstract: str,
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    append_check(checks, "phase7_independent_validation", "status=PASS" in PHASE7_CHECK.read_text(encoding="utf-8"), "Phase 7 check log is PASS")
    append_check(checks, "phase8_independent_validation", "status=PASS" in PHASE8_CHECK.read_text(encoding="utf-8"), "Phase 8 check log is PASS")
    append_check(checks, "key_result_ids_unique", key_results["result_id"].is_unique, f"records={len(key_results)}")
    append_check(checks, "claim_ids_unique", claims["claim_id"].is_unique, f"claims={len(claims)}")
    append_check(checks, "group_summary_complete", len(groups) == 12 and groups["oxide_v1_group_key"].is_unique, f"groups={len(groups)}")
    mae = metric_lookup(metrics, "row_equal", "mae_log10")
    rmse = metric_lookup(metrics, "row_equal", "rmse_log10")
    mae_ci = bootstrap_lookup(bootstrap, "row_equal", "mae_log10")
    rmse_ci = bootstrap_lookup(bootstrap, "row_equal", "rmse_log10")
    append_check(checks, "primary_mae_supported", mae["delta_oxide_v1_minus_baseline"] < 0 and mae_ci["ci95_upper"] < 0, f"delta={mae['delta_oxide_v1_minus_baseline']:.6f}; CI=[{mae_ci['ci95_lower']:.6f},{mae_ci['ci95_upper']:.6f}]")
    append_check(checks, "primary_rmse_supported", rmse["delta_oxide_v1_minus_baseline"] < 0 and rmse_ci["ci95_upper"] < 0, f"delta={rmse['delta_oxide_v1_minus_baseline']:.6f}; CI=[{rmse_ci['ci95_lower']:.6f},{rmse_ci['ci95_upper']:.6f}]")
    f2_ci = bootstrap_lookup(bootstrap, "row_equal", "factor_2_accuracy")
    append_check(checks, "factor2_uncertainty_preserved", f2_ci["ci95_lower"] < 0 < f2_ci["ci95_upper"], f"CI=[{f2_ci['ci95_lower']:.6f},{f2_ci['ci95_upper']:.6f}]")
    append_check(checks, "model_robustness_preserved", len(model) == 39 and ((model["delta_mae_log10"] < 0) & (model["delta_rmse_log10"] < 0)).all(), f"improved={int(((model['delta_mae_log10'] < 0) & (model['delta_rmse_log10'] < 0)).sum())}/{len(model)}")
    append_check(checks, "evaluation_robustness_preserved", len(evaluation) == 6 and ((evaluation["delta_mae_log10"] < 0) & (evaluation["delta_rmse_log10"] < 0)).all(), f"improved={int(((evaluation['delta_mae_log10'] < 0) & (evaluation['delta_rmse_log10'] < 0)).sum())}/{len(evaluation)}")
    append_check(checks, "leave_one_out_preserved", len(loo) == 24 and ((loo["delta_mae_log10"] < 0) & (loo["delta_rmse_log10"] < 0)).all(), f"improved={int(((loo['delta_mae_log10'] < 0) & (loo['delta_rmse_log10'] < 0)).sum())}/{len(loo)}")
    as_p = groups[groups["oxide_v1_group_key"].eq("oxide_v1::mixed_anion_O_As_P")].iloc[0]
    append_check(checks, "group_exception_preserved", as_p["robust_direction"] == "consistently_worsened" and as_p["row_delta_mae_log10"] > 0, f"delta={as_p['row_delta_mae_log10']:.6f}; direction={as_p['robust_direction']}")
    append_check(checks, "report_required_sections", all(title in report for title in ["## 研究背景と研究課題", "## 研究設計", "## 主結果", "## 頑健性", "## 予測可能率とのトレードオフ", "## 結論"]), "All required research sections are present")
    append_check(checks, "report_limits_external_generalization", "DOI単位ホールドアウト" in report and "外部" in report, "External-generalization limitation is explicit")
    append_check(checks, "abstract_bilingual", "## 日本語要旨" in abstract and "## English abstract" in abstract, "Japanese and English abstracts are present")
    append_check(checks, "claim_sources_resolve", claims["evidence_source"].astype(str).str.len().gt(0).all(), "Every claim has an evidence source")
    return pd.DataFrame(checks)


def main() -> None:
    require_inputs()
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(PHASE7_METRICS, low_memory=False)
    bootstrap = pd.read_csv(PHASE7_BOOTSTRAP, low_memory=False)
    phase7_groups = pd.read_csv(PHASE7_GROUPS, low_memory=False)
    sample_change = pd.read_csv(PHASE7_SAMPLE_CHANGE, low_memory=False)
    coverage = pd.read_csv(PHASE7_COVERAGE, low_memory=False)
    model = pd.read_csv(PHASE8_MODEL, low_memory=False)
    evaluation = pd.read_csv(PHASE8_EVALUATION, low_memory=False)
    loo = pd.read_csv(PHASE8_LOO, low_memory=False)
    phase8_groups = pd.read_csv(PHASE8_GROUPS, low_memory=False)

    key_results = build_key_results(metrics, bootstrap, sample_change, coverage, model, evaluation, loo)
    group_summary = build_group_summary(phase7_groups, phase8_groups)
    claims = build_claims(key_results, group_summary)
    study_flow = build_study_flow()
    manifest = build_manifest()
    abstract = build_abstract(metrics, bootstrap, coverage)
    report = build_report(metrics, bootstrap, group_summary, sample_change, coverage, model, evaluation, loo, claims)
    validation = build_validation(key_results, claims, group_summary, metrics, bootstrap, model, evaluation, loo, report, abstract)

    key_results.to_csv(OUT_KEY_RESULTS, index=False)
    claims.to_csv(OUT_CLAIMS, index=False)
    group_summary.to_csv(OUT_GROUPS, index=False)
    study_flow.to_csv(OUT_STUDY_FLOW, index=False)
    manifest.to_csv(OUT_MANIFEST, index=False)
    validation.to_csv(OUT_VALIDATION, index=False)
    OUT_REPORT.write_text(report, encoding="utf-8")
    OUT_ABSTRACT.write_text(abstract, encoding="utf-8")
    make_summary_figure(metrics, group_summary, model)

    status = "PASS" if validation["status"].eq("PASS").all() else "FAIL"
    OUT_LOG.write_text(
        "Phase 9 research synthesis validation\n"
        f"status={status}\n"
        f"checks={len(validation)}\n"
        f"pass={int(validation['status'].eq('PASS').sum())}\n"
        + "\n".join(f"{row.status} {row.check}: {row.detail}" for row in validation.itertuples(index=False))
        + "\n",
        encoding="utf-8",
    )
    print(f"Phase 9 build: {status} ({int(validation['status'].eq('PASS').sum())}/{len(validation)} checks)")
    print(claims[["claim_id", "evidence_status", "claim_text_ja"]].to_string(index=False))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
