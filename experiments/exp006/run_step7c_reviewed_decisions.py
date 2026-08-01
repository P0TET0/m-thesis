import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
DEFAULT_PACKET = PROCESSED_DIR / "step7a_manual_review_packet"
DEFAULT_OUTPUT = PROCESSED_DIR / "step7c_reviewed_decisions_applied"
DEFAULT_REPORT_DIR = EXP_DIR / "reports" / "step7c_reviewed_decisions_applied"

VALID_REVIEW_STATUS = {"pending", "keep", "keep_but_note", "suspect", "exclude_from_primary", "exclude_from_all", "unresolved"}
VALID_REASON = {
    "",
    "physically_plausible_keep",
    "source_trace_ok",
    "source_trace_missing",
    "suspicious_sigma_unit",
    "suspicious_resistivity_conversion",
    "suspicious_temperature_match",
    "suspicious_curve_pairing",
    "possible_digitization_error",
    "possible_sample_mismatch",
    "duplicate_or_near_duplicate",
    "extreme_but_unresolved",
    "other",
}
VALID_SCOPE = {"row_only", "entire_sample", "entire_paper", "all_matching_source_curve", "undecided"}
VALID_PRIMARY_FLAG = {"keep_in_primary", "exclude_from_primary", "pending"}
VALID_SENSITIVITY_FLAG = {"keep_in_sensitivity", "exclude_from_sensitivity", "pending"}

DEFAULT_CONFIG_LABELS = {
    "broad_material_family_default": "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_global_default": "sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
    "broad_paper_material_family_default": "paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median",
    "broad_paper_global_default": "paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median",
}

COMPARE_METRICS = [
    "n_rows",
    "n_pending_rows",
    "n_excluded_from_primary",
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
    parser = argparse.ArgumentParser(description="Run Step7C with reviewed manual decisions.")
    parser.add_argument("--predictions", type=Path, default=PROCESSED_DIR / "step6b_broad_family" / "step5b_test_predictions_valid.parquet")
    parser.add_argument("--reviewed-decision-template", type=Path, default=DEFAULT_PACKET / "step7a_review_decisions_template_reviewed.csv")
    parser.add_argument("--review-master", type=Path, default=DEFAULT_PACKET / "step7a_manual_review_master.csv")
    parser.add_argument("--source-trace", type=Path, default=DEFAULT_PACKET / "step7a_source_traceability_table.csv")
    parser.add_argument("--previous-step7b-dir", type=Path, default=PROCESSED_DIR / "step7b_review_applied")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--max-rows-per-config", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step7c] {message}", flush=True)


def out_name(base: str, suffix: str, ext: str = "csv") -> str:
    return f"{base}{suffix}.{ext}"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def normalize_text(value: Any, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def validate_reviewed_decisions(path: Path, output: Path, suffix: str, is_test: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Reviewed decision template is required and was not found: {path}. "
            "Create step7a_review_decisions_template_reviewed.csv from the Step7A template before running Step7C."
        )
    decisions = read_csv(path)
    required = [
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
        "T_K",
        "abs_error_decades",
        "error_severity",
        "likely_error_origin_hint",
        "reviewer_name",
        "review_date",
        "review_status",
        "review_reason_code",
        "apply_to_scope",
        "primary_analysis_flag_after_review",
        "sensitivity_analysis_flag_after_review",
        "reviewer_notes",
        "evidence_file_or_link",
        "checked_source_plot",
        "checked_units",
        "checked_temperature_alignment",
    ]
    missing = sorted(set(required) - set(decisions.columns))
    if missing:
        raise ValueError(f"reviewed decision template missing required columns: {missing}")
    df = decisions.copy()
    for col in required:
        df[col] = df[col].map(lambda value: normalize_text(value))
    df["review_status"] = df["review_status"].replace("", "pending").str.casefold()
    df["review_reason_code"] = df["review_reason_code"].str.casefold()
    df["apply_to_scope"] = df["apply_to_scope"].replace("", "undecided").str.casefold()
    df["primary_analysis_flag_after_review"] = df["primary_analysis_flag_after_review"].replace("", "pending").str.casefold()
    df["sensitivity_analysis_flag_after_review"] = df["sensitivity_analysis_flag_after_review"].replace("", "pending").str.casefold()
    df["decision_is_pending"] = df["review_status"].isin(["pending", "unresolved"])
    df["decision_is_human_reviewed"] = (
        ~df["review_status"].isin(["pending", ""])
        | df["reviewer_name"].astype(str).str.len().gt(0)
        | df["review_date"].astype(str).str.len().gt(0)
        | df["reviewer_notes"].astype(str).str.len().gt(0)
        | df["review_reason_code"].astype(str).str.len().gt(0)
    )
    df["decision_is_exclusion"] = df["review_status"].isin(["exclude_from_primary", "exclude_from_all"])
    df["decision_is_suspect"] = df["review_status"].eq("suspect")
    reviewed_status = df["review_status"].isin(["keep", "keep_but_note", "suspect", "exclude_from_primary", "exclude_from_all"])
    df["decision_has_required_review_fields"] = (~reviewed_status) | (
        df["review_reason_code"].astype(str).str.len().gt(0) & df["apply_to_scope"].astype(str).str.len().gt(0)
    )
    invalid_value = (
        ~df["review_status"].isin(VALID_REVIEW_STATUS)
        | ~df["review_reason_code"].isin(VALID_REASON)
        | ~df["apply_to_scope"].isin(VALID_SCOPE)
        | ~df["primary_analysis_flag_after_review"].isin(VALID_PRIMARY_FLAG)
        | ~df["sensitivity_analysis_flag_after_review"].isin(VALID_SENSITIVITY_FLAG)
    )
    conflict = (
        df["review_status"].eq("exclude_from_all")
        & (
            df["primary_analysis_flag_after_review"].eq("keep_in_primary")
            | df["sensitivity_analysis_flag_after_review"].eq("keep_in_sensitivity")
        )
    )
    df["decision_validation_status"] = "ok"
    df.loc[df["decision_is_pending"], "decision_validation_status"] = "pending"
    df.loc[~df["decision_has_required_review_fields"], "decision_validation_status"] = "missing_required_fields"
    df.loc[conflict, "decision_validation_status"] = "conflicting_flags"
    df.loc[invalid_value, "decision_validation_status"] = "invalid_value"
    df["decision_validation_warning"] = ""
    df.loc[df["apply_to_scope"].eq("undecided"), "decision_validation_warning"] = "undecided scope will be treated as row_only by Step7B"

    extreme = df[(df["error_severity"].eq("extreme_ge_10_decades")) | (pd.to_numeric(df["abs_error_decades"], errors="coerce") >= 10)]
    extreme_summary = pd.DataFrame(
        [
            {
                "extreme_case_count": len(extreme),
                "extreme_human_reviewed_count": int(extreme["decision_is_human_reviewed"].sum()) if len(extreme) else 0,
                "extreme_pending_count": int(extreme["decision_is_pending"].sum()) if len(extreme) else 0,
                "extreme_exclude_from_primary_count": int(extreme["review_status"].eq("exclude_from_primary").sum()) if len(extreme) else 0,
                "extreme_exclude_from_all_count": int(extreme["review_status"].eq("exclude_from_all").sum()) if len(extreme) else 0,
                "extreme_keep_count": int(extreme["review_status"].isin(["keep", "keep_but_note"]).sum()) if len(extreme) else 0,
                "extreme_unresolved_count": int(extreme["review_status"].eq("unresolved").sum()) if len(extreme) else 0,
            }
        ]
    )
    summary_rows = [
        ("decision_template_rows", len(df), "Rows in reviewed decision template."),
        ("human_reviewed_decisions", int(df["decision_is_human_reviewed"].sum()), "Rows with human review evidence."),
        ("pending_decisions", int(df["decision_is_pending"].sum()), "Rows pending or unresolved."),
        ("unresolved_decisions", int(df["review_status"].eq("unresolved").sum()), "Rows marked unresolved."),
        ("keep_decisions", int(df["review_status"].eq("keep").sum()), "Rows marked keep."),
        ("keep_but_note_decisions", int(df["review_status"].eq("keep_but_note").sum()), "Rows marked keep_but_note."),
        ("suspect_decisions", int(df["review_status"].eq("suspect").sum()), "Rows marked suspect."),
        ("exclude_from_primary_decisions", int(df["review_status"].eq("exclude_from_primary").sum()), "Rows marked exclude_from_primary."),
        ("exclude_from_all_decisions", int(df["review_status"].eq("exclude_from_all").sum()), "Rows marked exclude_from_all."),
        ("invalid_decisions", int(df["decision_validation_status"].isin(["invalid_value", "missing_required_fields", "conflicting_flags"]).sum()), "Rows with invalid or incomplete reviewed decision values."),
        ("extreme_case_count", int(extreme_summary["extreme_case_count"].iloc[0]), "Extreme cases in reviewed template."),
        ("extreme_human_reviewed_count", int(extreme_summary["extreme_human_reviewed_count"].iloc[0]), "Extreme cases with human review evidence."),
        ("extreme_pending_count", int(extreme_summary["extreme_pending_count"].iloc[0]), "Extreme cases still pending."),
        ("extreme_exclude_count", int(extreme["review_status"].isin(["exclude_from_primary", "exclude_from_all"]).sum()) if len(extreme) else 0, "Extreme cases excluded by review status."),
        ("full_run_ready", bool((df["decision_is_human_reviewed"].sum() > 0) and (df["decision_validation_status"].isin(["invalid_value", "missing_required_fields", "conflicting_flags"]).sum() == 0) and (not len(extreme) or extreme["decision_is_human_reviewed"].any())), "Full run readiness based on reviewed decisions."),
    ]
    summary = pd.DataFrame(summary_rows, columns=["item", "value", "comment"])
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / out_name("step7c_reviewed_decision_validation", suffix), index=False, encoding="utf-8-sig")
    summary.to_csv(output / out_name("step7c_reviewed_decision_validation_summary", suffix), index=False, encoding="utf-8-sig")
    extreme_summary.to_csv(output / out_name("step7c_extreme_review_completion_summary", suffix), index=False, encoding="utf-8-sig")

    errors: list[str] = []
    if int(summary.loc[summary["item"].eq("invalid_decisions"), "value"].iloc[0]) > 0:
        errors.append("invalid reviewed decisions are present")
    if int(summary.loc[summary["item"].eq("human_reviewed_decisions"), "value"].iloc[0]) == 0 and not is_test:
        errors.append("no human reviewed decisions found in full run")
    if int(summary.loc[summary["item"].eq("extreme_case_count"), "value"].iloc[0]) > 0 and int(summary.loc[summary["item"].eq("extreme_human_reviewed_count"), "value"].iloc[0]) == 0 and not is_test:
        errors.append("all extreme cases remain unreviewed in full run")
    return df, summary, extreme_summary, errors


def is_test_run(args: argparse.Namespace) -> bool:
    return bool(args.output_suffix) or args.max_rows_per_config is not None


def run_command(cmd: list[str]) -> None:
    log("running: " + " ".join(str(part) for part in cmd))
    subprocess.run([str(part) for part in cmd], check=True)


def step7b_build_cmd(args: argparse.Namespace, out_dir: Path, report: Path, pending_policy: str) -> list[str]:
    cmd = [
        sys.executable,
        str(EXP_DIR / "build_step7b_apply_review_decisions.py"),
        "--predictions",
        args.predictions,
        "--decision-template",
        args.reviewed_decision_template,
        "--review-master",
        args.review_master,
        "--source-trace",
        args.source_trace,
        "--output",
        out_dir,
        "--report",
        report,
        "--pending-policy",
        pending_policy,
        "--suspect-policy",
        "exclude_primary_keep_sensitivity",
    ]
    if args.max_rows_per_config is not None:
        cmd.extend(["--max-rows-per-config", str(args.max_rows_per_config)])
    if args.output_suffix:
        cmd.extend(["--output-suffix", args.output_suffix])
    return cmd


def step7b_check_cmd(out_dir: Path, report: Path, suffix: str) -> list[str]:
    return [
        sys.executable,
        str(EXP_DIR / "check_step7b_review_applied.py"),
        "--output",
        out_dir,
        "--rows",
        out_dir / out_name("step7b_prediction_rows_with_review_flags", suffix),
        "--primary",
        out_dir / out_name("step7b_primary_analysis_predictions", suffix),
        "--sensitivity",
        out_dir / out_name("step7b_sensitivity_analysis_predictions", suffix),
        "--metrics",
        out_dir / out_name("step7b_metrics_by_review_scenario_config", suffix),
        "--default-metrics",
        out_dir / out_name("step7b_default_metrics_by_review_scenario", suffix),
        "--readiness",
        out_dir / out_name("step7b_review_readiness_summary", suffix),
        "--report",
        report,
    ]


def policy_dir(base: Path, policy: str, suffix: str) -> Path:
    if suffix:
        return base / f"{policy}{suffix}"
    return base / policy


def compare_policy_metrics(keep_dir: Path, exclude_dir: Path, output: Path, suffix: str) -> pd.DataFrame:
    keep = read_csv(keep_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    ex = read_csv(exclude_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    rows = []
    for label, config_id in DEFAULT_CONFIG_LABELS.items():
        for weighting in ["row_equal", "sample_equal"]:
            kp = keep[
                keep["config_id"].eq(config_id)
                & keep["metric_weighting"].eq(weighting)
                & keep["review_scenario"].eq("primary_review_applied")
            ]
            ep = ex[
                ex["config_id"].eq(config_id)
                & ex["metric_weighting"].eq(weighting)
                & ex["review_scenario"].eq("primary_review_applied")
            ]
            if kp.empty or ep.empty:
                continue
            for metric in COMPARE_METRICS:
                rows.append(
                    {
                        "config_label": label,
                        "config_id": config_id,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "keep_pending_value": kp[metric].iloc[0],
                        "exclude_pending_primary_value": ep[metric].iloc[0],
                        "delta_exclude_pending_minus_keep_pending": ep[metric].iloc[0] - kp[metric].iloc[0],
                        "interpretation_hint": "lower_is_better" if metric in ["mae_log10", "rmse_log10", "max_abs_log10_error", "extreme_ge_10_count", "severe_ge_5_count"] else "higher_is_better_or_count",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(output / out_name("step7c_policy_comparison_metrics", suffix), index=False, encoding="utf-8-sig")
    return out


def compare_previous_baseline(previous_dir: Path, keep_dir: Path, exclude_dir: Path, output: Path, suffix: str) -> pd.DataFrame:
    rows = []
    previous_path = previous_dir / "step7b_default_metrics_by_review_scenario.csv"
    if not previous_path.exists():
        out = pd.DataFrame()
        out.to_csv(output / out_name("step7c_reviewed_vs_pending_baseline_comparison", suffix), index=False, encoding="utf-8-sig")
        return out
    prev = read_csv(previous_path)
    keep = read_csv(keep_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    ex = read_csv(exclude_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    for label, config_id in DEFAULT_CONFIG_LABELS.items():
        for weighting in ["row_equal", "sample_equal"]:
            old = prev[
                prev["config_id"].eq(config_id)
                & prev["metric_weighting"].eq(weighting)
                & prev["review_scenario"].eq("primary_review_applied")
            ]
            kp = keep[
                keep["config_id"].eq(config_id)
                & keep["metric_weighting"].eq(weighting)
                & keep["review_scenario"].eq("primary_review_applied")
            ]
            ep = ex[
                ex["config_id"].eq(config_id)
                & ex["metric_weighting"].eq(weighting)
                & ex["review_scenario"].eq("primary_review_applied")
            ]
            if old.empty or kp.empty or ep.empty:
                continue
            for metric in ["n_rows", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_5_accuracy", "factor_10_accuracy", "max_abs_log10_error", "extreme_ge_10_count"]:
                rows.append(
                    {
                        "config_label": label,
                        "config_id": config_id,
                        "metric_weighting": weighting,
                        "metric_name": metric,
                        "previous_pending_baseline_value": old[metric].iloc[0],
                        "keep_pending_reviewed_value": kp[metric].iloc[0],
                        "exclude_pending_primary_reviewed_value": ep[metric].iloc[0],
                        "delta_keep_pending_minus_previous": kp[metric].iloc[0] - old[metric].iloc[0],
                        "delta_exclude_pending_minus_previous": ep[metric].iloc[0] - old[metric].iloc[0],
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(output / out_name("step7c_reviewed_vs_pending_baseline_comparison", suffix), index=False, encoding="utf-8-sig")
    return out


def build_manifest(output: Path, keep_dir: Path, exclude_dir: Path, suffix: str) -> pd.DataFrame:
    rows = []
    for policy, directory in [("keep_pending", keep_dir), ("exclude_pending_primary", exclude_dir)]:
        rows.extend(
            [
                ("primary predictions", directory / out_name("step7b_primary_analysis_predictions", suffix), "Primary prediction rows after review policy.", "Use for primary final tables if this policy is selected."),
                ("sensitivity predictions", directory / out_name("step7b_sensitivity_analysis_predictions", suffix), "Sensitivity prediction rows after review policy.", "Use for sensitivity comparisons."),
                ("pending/unresolved rows", directory / out_name("step7b_pending_or_unresolved_rows", suffix), "Rows still pending or unresolved.", "Review before final reporting."),
                ("excluded from primary", directory / out_name("step7b_excluded_from_primary", suffix), "Rows excluded from primary by policy.", "Document exclusion rule if used."),
                ("excluded from sensitivity", directory / out_name("step7b_excluded_from_sensitivity", suffix), "Rows excluded from sensitivity by policy.", "Document exclusion rule if used."),
                ("metrics by scenario", directory / out_name("step7b_metrics_by_review_scenario_config", suffix), "Review scenario metrics for all configs.", "Use for method appendix."),
                ("default metrics", directory / out_name("step7b_default_metrics_by_review_scenario", suffix), "Review scenario metrics for default configs.", "Use for final comparison table."),
            ]
        )
    df = pd.DataFrame(
        [
            {
                "dataset_role": role,
                "dataset_path": str(path),
                "policy": policy,
                "description": description,
                "recommended_use": use,
            }
            for policy, items in [("keep_pending", rows[:7]), ("exclude_pending_primary", rows[7:])]
            for role, path, description, use in items
        ]
    )
    df.to_csv(output / out_name("step7c_final_candidate_dataset_manifest", suffix), index=False, encoding="utf-8-sig")
    return df


def build_unresolved_checklist(exclude_dir: Path, output: Path, suffix: str) -> pd.DataFrame:
    src = exclude_dir / out_name("step7b_manual_review_unresolved_checklist", suffix)
    if src.exists():
        df = read_csv(src)
    else:
        df = pd.DataFrame()
    df.to_csv(output / out_name("step7c_unresolved_after_review_checklist", suffix), index=False, encoding="utf-8-sig")
    return df


def build_readiness(summary: pd.DataFrame, extreme: pd.DataFrame, keep_dir: Path, exclude_dir: Path, output: Path, suffix: str, policy_compare: pd.DataFrame) -> pd.DataFrame:
    value = {row["item"]: row["value"] for _, row in summary.iterrows()}
    human = int(value.get("human_reviewed_decisions", 0))
    invalid = int(value.get("invalid_decisions", 0))
    extreme_count = int(extreme["extreme_case_count"].iloc[0]) if not extreme.empty else 0
    extreme_reviewed = int(extreme["extreme_human_reviewed_count"].iloc[0]) if not extreme.empty else 0
    pending = int(value.get("pending_decisions", 0))
    keep_primary = keep_dir / out_name("step7b_primary_analysis_predictions", suffix)
    exclude_primary = exclude_dir / out_name("step7b_primary_analysis_predictions", suffix)
    keep_default = read_csv(keep_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    exclude_default = read_csv(exclude_dir / out_name("step7b_default_metrics_by_review_scenario", suffix))
    bmf_keep = keep_default[
        keep_default["config_label"].eq("broad_material_family_default")
        & keep_default["review_scenario"].eq("primary_review_applied")
        & keep_default["metric_weighting"].eq("row_equal")
    ]
    bmf_ex = exclude_default[
        exclude_default["config_label"].eq("broad_material_family_default")
        & exclude_default["review_scenario"].eq("primary_review_applied")
        & exclude_default["metric_weighting"].eq("row_equal")
    ]
    extreme_primary = bmf_ex["extreme_ge_10_count"].iloc[0] if not bmf_ex.empty else None
    recommended = "exclude_pending_primary" if pending else "keep_pending"
    ready = invalid == 0 and human > 0 and (extreme_count == 0 or extreme_reviewed > 0) and exclude_primary.exists()
    rows = [
        ("reviewed_template_exists", "pass", True, "file exists", "Reviewed decision template was loaded."),
        ("human_reviewed_decisions_exist", "pass" if human > 0 else "fail", human, "> 0", "At least one human reviewed decision is required for full Step7C."),
        ("extreme_cases_reviewed", "pass" if extreme_count == 0 or extreme_reviewed > 0 else "fail", f"{extreme_reviewed}/{extreme_count}", "at least one extreme reviewed", "Extreme outlier review completion."),
        ("invalid_decisions_absent", "pass" if invalid == 0 else "fail", invalid, "0", "Invalid decisions must be corrected."),
        ("keep_pending_policy_outputs_exist", "pass" if keep_primary.exists() else "fail", keep_primary.exists(), "exists", "keep_pending Step7B output."),
        ("exclude_pending_policy_outputs_exist", "pass" if exclude_primary.exists() else "fail", exclude_primary.exists(), "exists", "exclude_pending_primary Step7B output."),
        ("primary_dataset_available", "pass" if exclude_primary.exists() and exclude_primary.stat().st_size > 0 else "fail", str(exclude_primary), "non-empty", "Primary candidate dataset exists."),
        ("sensitivity_dataset_available", "pass", str(exclude_dir / out_name("step7b_sensitivity_analysis_predictions", suffix)), "non-empty", "Sensitivity candidate dataset exists."),
        ("unresolved_or_pending_cases_remaining", "caution" if pending else "pass", pending, "caution if > 0", "Pending is allowed but must be disclosed."),
        ("extreme_outliers_remaining_in_primary", "caution" if extreme_primary and extreme_primary > 0 else "pass", extreme_primary, "caution if > 0", "Extreme outliers in exclude_pending_primary primary."),
        ("recommended_policy_for_step8", "caution" if pending else "pass", recommended, "manual decision", "Recommended policy for Step8 candidate dataset."),
        ("ready_for_step8", "pass" if ready and not pending else "caution" if ready else "fail", ready, "no invalid decisions and reviewed evidence", "Step8 readiness."),
    ]
    df = pd.DataFrame(rows, columns=["criterion", "status", "value", "threshold_or_reason", "comment"])
    df.to_csv(output / out_name("step7c_final_readiness_summary", suffix), index=False, encoding="utf-8-sig")
    return df


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


def write_report(report: Path, summary: pd.DataFrame, extreme: pd.DataFrame, policy_compare: pd.DataFrame, baseline_compare: pd.DataFrame, manifest: pd.DataFrame, readiness: pd.DataFrame, elapsed: float) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Step7C Reviewed Decisions Report",
        "",
        "## Decision Validation Summary",
        "",
        df_to_markdown(summary, 40),
        "",
        "## Extreme Review Completion",
        "",
        df_to_markdown(extreme, 10),
        "",
        "## Policy Comparison",
        "",
        df_to_markdown(policy_compare[policy_compare["config_label"].eq("broad_material_family_default")], 30),
        "",
        "## Previous Baseline Comparison",
        "",
        df_to_markdown(baseline_compare[baseline_compare["config_label"].eq("broad_material_family_default")], 30),
        "",
        "## Final Candidate Dataset Manifest",
        "",
        df_to_markdown(manifest, 30),
        "",
        "## Final Readiness",
        "",
        df_to_markdown(readiness, 20),
        "",
        "## Notes",
        "",
        "- Step7C does not compute new sigma predictions.",
        "- Step7C applies reviewed decisions by rerunning the existing Step7B decision application script.",
        "- No figures are created.",
        "- If pending decisions remain, final reporting should disclose the policy used.",
        "",
        f"- elapsed_seconds: {elapsed:.2f}",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.time()
    args = parse_args()
    test = is_test_run(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    log("loading reviewed decision template...")
    log("validating decision values...")
    validation, summary, extreme, errors = validate_reviewed_decisions(args.reviewed_decision_template, args.output, args.output_suffix, test)

    log("checking human reviewed decision counts...")
    log("checking extreme outlier review completion...")
    if errors and not test:
        for error in errors:
            print(f"[step7c] FAIL: {error}", flush=True)
        raise SystemExit(1)
    if errors:
        for error in errors:
            print(f"[step7c] WARNING: {error}", flush=True)

    keep_dir = policy_dir(args.output, "keep_pending", args.output_suffix)
    ex_dir = policy_dir(args.output, "exclude_pending_primary", args.output_suffix)
    keep_report = args.report_dir / keep_dir.name / "step7b_review_application_report.md"
    ex_report = args.report_dir / ex_dir.name / "step7b_review_application_report.md"

    log("running Step7B keep_pending policy...")
    run_command(step7b_build_cmd(args, keep_dir, keep_report, "keep_with_pending_flag"))
    log("checking Step7B keep_pending outputs...")
    run_command(step7b_check_cmd(keep_dir, keep_report, args.output_suffix))

    log("running Step7B exclude_pending_primary policy...")
    run_command(step7b_build_cmd(args, ex_dir, ex_report, "exclude_from_primary"))
    log("checking Step7B exclude_pending_primary outputs...")
    run_command(step7b_check_cmd(ex_dir, ex_report, args.output_suffix))

    log("comparing policies...")
    policy_compare = compare_policy_metrics(keep_dir, ex_dir, args.output, args.output_suffix)
    log("comparing with previous all-pending baseline...")
    baseline_compare = compare_previous_baseline(args.previous_step7b_dir, keep_dir, ex_dir, args.output, args.output_suffix)
    log("writing final dataset manifest...")
    manifest = build_manifest(args.output, keep_dir, ex_dir, args.output_suffix)
    unresolved = build_unresolved_checklist(ex_dir, args.output, args.output_suffix)
    log("writing final readiness summary...")
    readiness = build_readiness(summary, extreme, keep_dir, ex_dir, args.output, args.output_suffix, policy_compare)

    log("writing report...")
    report = args.report_dir / out_name("step7c_reviewed_decisions_report", args.output_suffix, "md")
    write_report(report, summary, extreme, policy_compare, baseline_compare, manifest, readiness, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
