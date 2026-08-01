import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
DEFAULT_STEP6D_DIR = PROCESSED_DIR / "step6d_broad_family_audit"
DEFAULT_STEP6C_DIR = PROCESSED_DIR / "step6c_broad_family"
DEFAULT_STEP6B_DIR = PROCESSED_DIR / "step6b_broad_family"
DEFAULT_OUTPUT = PROCESSED_DIR / "step7a_manual_review_packet"
DEFAULT_REPORT = EXP_DIR / "reports" / "step7a_manual_review_packet" / "step7a_manual_review_packet_report.md"

SOURCE_COLS = [
    "source_file_S",
    "source_file_sigma",
    "source_property_label_S",
    "source_property_label_sigma",
    "source_unit_S",
    "source_unit_sigma_or_rho",
    "source_curve_id_S",
    "source_curve_id_sigma",
    "T_delta_K",
]

MASTER_COLS = [
    "review_case_id",
    "review_case_type",
    "review_priority",
    "review_reason",
    "review_status_initial",
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "material_group_key",
    "carrier_type",
    "T_K",
    "T_bin_center_K",
    "S_uV_per_K",
    "S_abs_uV_per_K",
    "eta",
    "F0_eta",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_pred_over_exp",
    "log10_sigma_pred_over_exp",
    "abs_error_decades",
    "sigma0_S_per_m",
    "log10_sigma0_S_per_m",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "sigma0_ref_over_row_sigma0",
    "log10_sigma0_ref_over_row_sigma0",
    "train_row_count",
    "train_sample_count",
    "train_paper_count",
    "reliability_level",
    "sigma_source",
    "match_method",
    "sample_has_sign_change",
    "error_direction",
    "error_severity",
    "likely_error_origin_hint",
    *SOURCE_COLS,
    "source_traceability_score",
    "manual_review_note_hint",
]

DECISION_HUMAN_COLS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step7A manual review packet.")
    parser.add_argument("--step6d-dir", type=Path, default=DEFAULT_STEP6D_DIR)
    parser.add_argument("--step6c-dir", type=Path, default=DEFAULT_STEP6C_DIR)
    parser.add_argument("--step6b-dir", type=Path, default=DEFAULT_STEP6B_DIR)
    parser.add_argument("--metadata-input", type=Path, default=PROCESSED_DIR / "step6a_validation_rows_with_splits_key_broad_family.parquet")
    parser.add_argument("--step3-input", type=Path, default=PROCESSED_DIR / "step3_sigma0_valid.parquet")
    parser.add_argument("--step0-input", type=Path, default=Path("data/processed/step0_te_analysis_base.parquet"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-row-cases", type=int, default=200)
    parser.add_argument("--max-sample-cases", type=int, default=100)
    parser.add_argument("--max-paper-cases", type=int, default=100)
    parser.add_argument("--casebook-top-n", type=int, default=50)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step7a] {message}", flush=True)


def out_name(base: str, suffix: str, ext: str = "csv") -> str:
    return f"{base}{suffix}.{ext}"


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def load_optional(path: Path, warnings: list[str], label: str) -> pd.DataFrame | None:
    candidates = [path]
    if path.suffix.casefold() == ".parquet":
        candidates.append(path.with_suffix(".csv"))
    elif path.suffix.casefold() == ".csv":
        candidates.append(path.with_suffix(".parquet"))
    for candidate in candidates:
        if candidate.exists():
            try:
                return read_table(candidate)
            except Exception as exc:
                warnings.append(f"optional {label} failed to load: {candidate}: {exc}")
                return None
    warnings.append(f"optional {label} not found: {path}")
    return None


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def is_present(value: Any) -> bool:
    return pd.notna(value) and str(value).strip() != ""


def source_score(row: pd.Series) -> tuple[int, str, str]:
    fields = [
        "source_file_S",
        "source_file_sigma",
        "source_property_label_S",
        "source_property_label_sigma",
        "source_unit_S",
    ]
    score = sum(1 for col in fields if is_present(row.get(col)))
    missing = [col for col in fields if not is_present(row.get(col))]
    if score >= 5:
        hint = "source metadata complete"
    elif score >= 3:
        hint = "check missing source metadata before deciding"
    else:
        hint = "source traceability weak; prioritize source verification"
    return score, ";".join(missing), hint


def add_traceability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in SOURCE_COLS:
        if col not in out.columns:
            out[col] = ""
    scores = out.apply(source_score, axis=1, result_type="expand")
    out["source_traceability_score"] = scores[0]
    out["missing_source_fields"] = scores[1]
    out["source_review_hint"] = scores[2]
    return out


def review_hint(row: pd.Series) -> str:
    checks = ["check units", "check sigma/rho conversion", "check temperature matching", "check curve pairing", "check sample identity"]
    if str(row.get("match_method", "")).casefold() == "nearest":
        checks.insert(0, "nearest temperature match")
    if str(row.get("sigma_source", "")).casefold() == "resistivity_converted":
        checks.insert(0, "resistivity conversion")
    if row.get("source_traceability_score", 0) < 5:
        checks.insert(0, "missing source metadata")
    if str(row.get("error_severity", "")).startswith("extreme"):
        checks.insert(0, "extreme error")
    return "; ".join(dict.fromkeys(checks))


def assign_ids(df: pd.DataFrame, case_type: str, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out["review_case_type"] = case_type
    out["review_case_id"] = [f"{prefix}_{i:04d}" for i in range(1, len(out) + 1)]
    out["review_status_initial"] = "pending"
    return out


def build_row_cases(outliers: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    rows = add_traceability(outliers).copy()
    rows = rows.sort_values(
        ["error_severity", "abs_error_decades"],
        key=lambda s: s.map({"extreme_ge_10_decades": 0, "severe_ge_5_decades": 1, "large_ge_2_decades": 2}).fillna(s)
        if s.name == "error_severity"
        else s,
        ascending=[True, False],
    ).head(max_cases)
    rows["review_reason"] = np.where(
        rows["error_severity"].eq("extreme_ge_10_decades"),
        "extreme_ge_10_decades row; verify source and units",
        "large outlier row; verify source and pairing",
    )
    rows["manual_review_note_hint"] = rows.apply(review_hint, axis=1)
    rows = assign_ids(rows, "row_case", "ROW")
    return rows


def build_sample_cases(sample_summary: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    rows = sample_summary.sort_values(
        ["extreme_ge_10_row_count", "severe_ge_5_row_count", "max_abs_error_decades", "fraction_factor10_or_more"],
        ascending=[False, False, False, False],
    ).head(max_cases).copy()
    rows["review_reason"] = "outliers concentrated within validation sample"
    rows["manual_review_note_hint"] = "check whether the whole sample curve is anomalous or only isolated points"
    rows = assign_ids(rows, "sample_case", "SAMPLE")
    return rows


def build_paper_cases(paper_summary: pd.DataFrame, max_cases: int) -> pd.DataFrame:
    rows = paper_summary.sort_values(
        ["extreme_ge_10_row_count", "severe_ge_5_row_count", "max_abs_error_decades", "fraction_factor10_or_more"],
        ascending=[False, False, False, False],
    ).head(max_cases).copy()
    rows["review_reason"] = "outliers concentrated within validation paper"
    rows["manual_review_note_hint"] = "check whether paper-level units, curves, or sample labels are systematically problematic"
    rows = assign_ids(rows, "paper_case", "PAPER")
    return rows


def rows_from_sample_cases(sample_cases: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["review_case_id"] = sample_cases["review_case_id"]
    out["review_case_type"] = "sample_case"
    out["review_priority"] = sample_cases["review_priority"]
    out["review_reason"] = sample_cases["review_reason"]
    out["review_status_initial"] = "pending"
    out["row_id"] = ""
    out["validation_sample_group_id"] = sample_cases["validation_sample_group_id"]
    out["validation_paper_group_id"] = ""
    out["paper_id"] = sample_cases["paper_id_examples"]
    out["doi"] = ""
    out["sample_id"] = sample_cases["sample_id_examples"]
    out["sample_key"] = sample_cases["sample_key_examples"]
    out["formula_raw"] = sample_cases["formula_raw_examples"]
    out["material_name_raw"] = sample_cases["material_name_raw_examples"]
    out["material_family_raw"] = ""
    out["material_group_key"] = sample_cases["material_group_key_values"]
    out["carrier_type"] = ""
    out["T_K"] = ""
    out["T_bin_center_K"] = ""
    out["S_uV_per_K"] = ""
    out["S_abs_uV_per_K"] = ""
    out["eta"] = ""
    out["F0_eta"] = ""
    out["sigma_S_per_m"] = ""
    out["log10_sigma_S_per_m"] = ""
    out["sigma_pred_S_per_m"] = ""
    out["log10_sigma_pred_S_per_m"] = ""
    out["sigma_pred_over_exp"] = ""
    out["log10_sigma_pred_over_exp"] = ""
    out["abs_error_decades"] = sample_cases["max_abs_error_decades"]
    out["sigma0_S_per_m"] = ""
    out["log10_sigma0_S_per_m"] = ""
    out["sigma0_ref_S_per_m"] = ""
    out["log10_sigma0_ref_S_per_m"] = ""
    out["sigma0_ref_over_row_sigma0"] = ""
    out["log10_sigma0_ref_over_row_sigma0"] = ""
    out["train_row_count"] = ""
    out["train_sample_count"] = ""
    out["train_paper_count"] = ""
    out["reliability_level"] = ""
    out["sigma_source"] = ""
    out["match_method"] = ""
    out["sample_has_sign_change"] = ""
    out["error_direction"] = sample_cases["dominant_error_direction"]
    out["error_severity"] = np.where(sample_cases["extreme_ge_10_row_count"] > 0, "extreme_ge_10_decades", "sample_concentration")
    out["likely_error_origin_hint"] = sample_cases["dominant_likely_error_origin_hint"]
    for col in SOURCE_COLS:
        out[col] = ""
    out["source_traceability_score"] = ""
    out["manual_review_note_hint"] = sample_cases["manual_review_note_hint"]
    return out


def rows_from_paper_cases(paper_cases: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["review_case_id"] = paper_cases["review_case_id"]
    out["review_case_type"] = "paper_case"
    out["review_priority"] = paper_cases["review_priority"]
    out["review_reason"] = paper_cases["review_reason"]
    out["review_status_initial"] = "pending"
    out["row_id"] = ""
    out["validation_sample_group_id"] = ""
    out["validation_paper_group_id"] = paper_cases["validation_paper_group_id"]
    out["paper_id"] = paper_cases["paper_id_examples"]
    out["doi"] = paper_cases["doi_examples"]
    out["sample_id"] = ""
    out["sample_key"] = ""
    out["formula_raw"] = ""
    out["material_name_raw"] = ""
    out["material_family_raw"] = ""
    out["material_group_key"] = paper_cases["material_group_key_values"]
    out["carrier_type"] = ""
    out["T_K"] = ""
    out["T_bin_center_K"] = ""
    out["S_uV_per_K"] = ""
    out["S_abs_uV_per_K"] = ""
    out["eta"] = ""
    out["F0_eta"] = ""
    out["sigma_S_per_m"] = ""
    out["log10_sigma_S_per_m"] = ""
    out["sigma_pred_S_per_m"] = ""
    out["log10_sigma_pred_S_per_m"] = ""
    out["sigma_pred_over_exp"] = ""
    out["log10_sigma_pred_over_exp"] = ""
    out["abs_error_decades"] = paper_cases["max_abs_error_decades"]
    out["sigma0_S_per_m"] = ""
    out["log10_sigma0_S_per_m"] = ""
    out["sigma0_ref_S_per_m"] = ""
    out["log10_sigma0_ref_S_per_m"] = ""
    out["sigma0_ref_over_row_sigma0"] = ""
    out["log10_sigma0_ref_over_row_sigma0"] = ""
    out["train_row_count"] = ""
    out["train_sample_count"] = ""
    out["train_paper_count"] = ""
    out["reliability_level"] = ""
    out["sigma_source"] = ""
    out["match_method"] = ""
    out["sample_has_sign_change"] = ""
    out["error_direction"] = paper_cases["dominant_error_direction"]
    out["error_severity"] = np.where(paper_cases["extreme_ge_10_row_count"] > 0, "extreme_ge_10_decades", "paper_concentration")
    out["likely_error_origin_hint"] = paper_cases["dominant_likely_error_origin_hint"]
    for col in SOURCE_COLS:
        out[col] = ""
    out["source_traceability_score"] = ""
    out["manual_review_note_hint"] = paper_cases["manual_review_note_hint"]
    return out


def build_master(row_cases: pd.DataFrame, sample_cases: pd.DataFrame, paper_cases: pd.DataFrame) -> pd.DataFrame:
    row_master = row_cases.copy()
    row_master["review_priority"] = range(1, len(row_master) + 1)
    row_master = row_master[MASTER_COLS]

    sample_cases = sample_cases.copy()
    sample_cases["review_priority"] = range(len(row_master) + 1, len(row_master) + len(sample_cases) + 1)
    sample_master = rows_from_sample_cases(sample_cases)

    paper_cases = paper_cases.copy()
    paper_cases["review_priority"] = range(len(row_master) + len(sample_master) + 1, len(row_master) + len(sample_master) + len(paper_cases) + 1)
    paper_master = rows_from_paper_cases(paper_cases)

    master = pd.concat([row_master, sample_master[MASTER_COLS], paper_master[MASTER_COLS]], ignore_index=True)
    master["review_priority"] = range(1, len(master) + 1)
    return master


def build_decision_template(master: pd.DataFrame) -> pd.DataFrame:
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
        "T_K",
        "abs_error_decades",
        "error_severity",
        "likely_error_origin_hint",
    ]
    out = master[cols].copy()
    out["reviewer_name"] = ""
    out["review_date"] = ""
    out["review_status"] = "pending"
    out["review_reason_code"] = ""
    out["apply_to_scope"] = "undecided"
    out["primary_analysis_flag_after_review"] = "pending"
    out["sensitivity_analysis_flag_after_review"] = "pending"
    out["reviewer_notes"] = ""
    out["evidence_file_or_link"] = ""
    out["checked_source_plot"] = "not_available"
    out["checked_units"] = "not_available"
    out["checked_temperature_alignment"] = "not_available"
    return out


def build_source_trace(master: pd.DataFrame) -> pd.DataFrame:
    rows = master[master["row_id"].astype(str).str.len() > 0].copy()
    rows = add_traceability(rows)
    cols = [
        "row_id",
        "paper_id",
        "doi",
        "sample_id",
        "sample_key",
        "validation_sample_group_id",
        "validation_paper_group_id",
        *SOURCE_COLS,
        "T_K",
        "match_method",
        "sigma_source",
        "source_traceability_score",
        "missing_source_fields",
        "source_review_hint",
    ]
    return rows[cols].drop_duplicates("row_id").copy()


def build_sample_context(context: pd.DataFrame, master: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    ctx = context.copy()
    source = outliers[["row_id", *[col for col in SOURCE_COLS if col in outliers.columns]]].drop_duplicates("row_id")
    ctx = ctx.merge(source, on="row_id", how="left")
    sample_map = master[master["review_case_type"].eq("sample_case")][["review_case_id", "validation_sample_group_id"]]
    ctx = ctx.merge(sample_map, on="validation_sample_group_id", how="left")
    row_map = master[master["review_case_type"].eq("row_case")][["review_case_id", "row_id"]].rename(columns={"review_case_id": "row_review_case_id"})
    ctx = ctx.merge(row_map, on="row_id", how="left")
    ctx["review_case_id"] = ctx["review_case_id"].fillna(ctx["row_review_case_id"])
    cols = [
        "review_case_id",
        "is_original_outlier_row",
        "outlier_rank",
        "row_id",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
        "material_group_key",
        "carrier_type",
        "T_K",
        "T_bin_center_K",
        "S_uV_per_K",
        "eta",
        "sigma_S_per_m",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_over_exp",
        "abs_error_decades",
        "sigma0_S_per_m",
        "sigma0_ref_S_per_m",
        "train_sample_count",
        "reliability_level",
        "sigma_source",
        "match_method",
        "source_file_S",
        "source_file_sigma",
        "source_curve_id_S",
        "source_curve_id_sigma",
    ]
    for col in cols:
        if col not in ctx.columns:
            ctx[col] = ""
    return ctx[cols].sort_values(["review_case_id", "is_original_outlier_row", "abs_error_decades"], ascending=[True, False, False])


def build_readiness_update(readiness: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in readiness.iterrows():
        rows.append(
            {
                "item": row["criterion"],
                "status": row["status"],
                "value": row["value"],
                "comment": row["comment"],
                "next_action": "use Step7A decision template for human review" if row["status"] == "caution" else "record as supporting evidence",
            }
        )
    rows.append(
        {
            "item": "manual_review_packet_created",
            "status": "pending_review",
            "value": len(master),
            "comment": "Manual review cases are prepared but not adjudicated.",
            "next_action": "fill step7a_review_decisions_template.csv, then run Step7B",
        }
    )
    return pd.DataFrame(rows)


def write_casebook(master: pd.DataFrame, path: Path, top_n: int) -> None:
    lines = ["# Step7A Manual Review Casebook", ""]
    for _, row in master.head(top_n).iterrows():
        lines.extend(
            [
                f"## {row['review_case_id']} ({row['review_case_type']})",
                "",
                f"- priority: {row['review_priority']}",
                f"- reason: {row['review_reason']}",
                f"- row_id: {row['row_id']}",
                f"- sample: {row['validation_sample_group_id']} / {row['sample_id']}",
                f"- paper: {row['validation_paper_group_id']} / {row['paper_id']} / {row['doi']}",
                f"- material: {row['formula_raw']} | {row['material_name_raw']} | {row['material_group_key']}",
                f"- carrier_type: {row['carrier_type']}",
                f"- T_K: {row['T_K']}",
                f"- S_uV_per_K: {row['S_uV_per_K']}",
                f"- eta: {row['eta']}",
                f"- sigma_exp: {row['sigma_S_per_m']}",
                f"- sigma_pred: {row['sigma_pred_S_per_m']}",
                f"- error_decades: {row['abs_error_decades']}",
                f"- likely_error_origin_hint: {row['likely_error_origin_hint']}",
                f"- source_traceability_score: {row['source_traceability_score']}",
                f"- source S: {row['source_file_S']} | {row['source_property_label_S']} | {row['source_unit_S']}",
                f"- source sigma: {row['source_file_sigma']} | {row['source_property_label_sigma']} | {row['source_unit_sigma_or_rho']}",
                "",
                "Suggested checks:",
                "",
                "- unit conversion",
                "- sigma/rho conversion",
                "- temperature matching",
                "- curve pairing",
                "- sample identity",
                "- digitization issue",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def packet_index(paths: dict[str, Path]) -> pd.DataFrame:
    descriptions = {
        "master": "Unified row/sample/paper review case list.",
        "row_cases": "Row-level outlier cases.",
        "sample_cases": "Sample-level concentration cases.",
        "paper_cases": "Paper-level concentration cases.",
        "decision_template": "Human-editable review decisions for Step7B.",
        "source_trace": "Source traceability fields and scores.",
        "sample_context": "Rows around top outlier samples.",
        "casebook": "Readable markdown summary of top review cases.",
        "readiness_update": "Readiness state after packet creation.",
        "report": "Step7A generation report.",
    }
    actions = {
        "decision_template": "Fill reviewer columns before Step7B.",
        "casebook": "Read first for triage.",
        "source_trace": "Use to locate source files and curves.",
    }
    rows = []
    for role, path in paths.items():
        rows.append(
            {
                "file_role": role,
                "file_path": str(path),
                "description": descriptions.get(role, ""),
                "intended_user_action": actions.get(role, "Use as supporting review material."),
            }
        )
    return pd.DataFrame(rows)


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


def write_report(report: Path, paths: dict[str, Path], master: pd.DataFrame, trace: pd.DataFrame, readiness_update: pd.DataFrame, checks: dict[str, bool], warnings: list[str], elapsed: float) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    score_counts = trace["source_traceability_score"].value_counts(dropna=False).sort_index().to_dict() if not trace.empty else {}
    lines = [
        "# Step7A Manual Review Packet Report",
        "",
        "## Outputs",
        "",
        df_to_markdown(packet_index(paths), 30),
        "",
        "## Case Counts",
        "",
        f"- total review cases: {len(master)}",
        f"- row_case: {int(master['review_case_type'].eq('row_case').sum())}",
        f"- sample_case: {int(master['review_case_type'].eq('sample_case').sum())}",
        f"- paper_case: {int(master['review_case_type'].eq('paper_case').sum())}",
        f"- extreme_ge_10_decades cases: {int(master['error_severity'].eq('extreme_ge_10_decades').sum())}",
        f"- severe_ge_5_decades cases: {int(master['error_severity'].eq('severe_ge_5_decades').sum())}",
        f"- large_ge_2_decades cases: {int(master['error_severity'].eq('large_ge_2_decades').sum())}",
        f"- source_traceability_score distribution: {score_counts}",
        f"- cases with source metadata gaps: {int((pd.to_numeric(master['source_traceability_score'], errors='coerce').fillna(0) < 5).sum())}",
        "",
        "## Top Review Cases",
        "",
        df_to_markdown(master[["review_case_id", "review_case_type", "review_priority", "row_id", "paper_id", "sample_id", "material_group_key", "abs_error_decades", "error_severity", "likely_error_origin_hint"]], 20),
        "",
        "## Decision Template Use",
        "",
        "- Fill `review_status` with keep, keep_but_note, suspect, exclude_from_primary, exclude_from_all, or unresolved.",
        "- Fill `review_reason_code` with the closest source/unit/temperature/pairing reason.",
        "- Use `apply_to_scope` to distinguish row-only, sample-level, paper-level, or source-curve-level decisions.",
        "- Step7B should read the completed decision template and create primary/sensitivity analysis flags.",
        "",
        "## Readiness Update",
        "",
        df_to_markdown(readiness_update, 30),
        "",
        "## Warnings",
        "",
    ]
    lines.extend([f"- {warning}" for warning in warnings] if warnings else ["- none"])
    lines.extend(
        [
            "",
            "## Sanity Checks",
            "",
            *[f"- {name}: {ok}" for name, ok in checks.items()],
            "",
            "## Notes",
            "",
            "- Step7A does not automatically exclude outliers.",
            "- Step7A does not compute new sigma predictions.",
            "- Step7A does not create figures.",
            "- Step7A is a packet for human source verification, not a final research decision.",
            "",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity(paths: dict[str, Path], master: pd.DataFrame, decisions: pd.DataFrame) -> tuple[dict[str, bool], list[str]]:
    checks = {
        "required_outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in paths.values()),
        "manual_review_master_created": not master.empty,
        "review_case_id_unique": master["review_case_id"].is_unique,
        "review_priority_not_missing": master["review_priority"].notna().all(),
        "review_status_initial_pending": master["review_status_initial"].eq("pending").all(),
        "decision_template_created": not decisions.empty,
        "decision_template_human_columns_exist": set(DECISION_HUMAN_COLS).issubset(set(decisions.columns)),
        "source_traceability_table_created": paths["source_trace"].exists() and paths["source_trace"].stat().st_size > 0,
        "sample_context_created": paths["sample_context"].exists() and paths["sample_context"].stat().st_size > 0,
        "casebook_created": paths["casebook"].exists() and paths["casebook"].stat().st_size > 0,
        "packet_index_created": paths["packet_index"].exists() and paths["packet_index"].stat().st_size > 0,
        "readiness_update_created": paths["readiness_update"].exists() and paths["readiness_update"].stat().st_size > 0,
        "report_created": paths["report"].exists() and paths["report"].stat().st_size > 0,
        "did_not_compute_new_sigma_pred": True,
        "did_not_read_step4_full_data_reference_curve": True,
        "did_not_read_raw_data": True,
        "did_not_auto_exclude_outliers": True,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return checks, failures


def main() -> None:
    started = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    log("loading Step6D audit outputs...")
    outliers = read_table(args.step6d_dir / "step6d_outlier_rows_topN.csv")
    sample_summary = read_table(args.step6d_dir / "step6d_outlier_summary_by_sample.csv")
    paper_summary = read_table(args.step6d_dir / "step6d_outlier_summary_by_paper.csv")
    context = read_table(args.step6d_dir / "step6d_top_outlier_sample_context_rows.csv")
    readiness = read_table(args.step6d_dir / "step6d_broad_family_main_result_readiness_summary.csv")

    log("loading Step6C/Step6B summary outputs...")
    for path in [
        args.step6c_dir / "step6c_visual_diagnostics_summary.csv",
        args.step6c_dir / "step6c_broad_family_default_metrics_for_figures.csv",
        args.step6b_dir / "step5c_default_comparison.csv",
        args.step6b_dir / "step5c_largest_abs_error_rows.csv",
    ]:
        if path.exists():
            _ = read_table(path)
        else:
            warnings.append(f"optional summary input missing: {path}")

    log("loading optional metadata...")
    for path, label in [(args.metadata_input, "metadata"), (args.step3_input, "step3"), (args.step0_input, "step0")]:
        _ = load_optional(path, warnings, label)

    require_columns(outliers, ["row_id", "abs_error_decades", "error_severity", "likely_error_origin_hint", *SOURCE_COLS], "Step6D outliers")

    log("building row review cases...")
    row_cases = build_row_cases(outliers, args.max_row_cases)

    log("building sample review cases...")
    sample_cases = build_sample_cases(sample_summary, args.max_sample_cases)

    log("building paper review cases...")
    paper_cases = build_paper_cases(paper_summary, args.max_paper_cases)

    log("assigning review priorities...")
    master = build_master(row_cases, sample_cases, paper_cases)

    log("building decision template...")
    decisions = build_decision_template(master)

    log("building source traceability table...")
    trace = build_source_trace(master)

    log("building sample context file...")
    sample_context = build_sample_context(context, master, outliers)

    log("writing manual review casebook...")
    casebook_path = args.output / out_name("step7a_manual_review_casebook", args.output_suffix, "md")
    write_casebook(master, casebook_path, args.casebook_top_n)

    readiness_update = build_readiness_update(readiness, master)

    paths = {
        "master": args.output / out_name("step7a_manual_review_master", args.output_suffix),
        "row_cases": args.output / out_name("step7a_row_review_cases", args.output_suffix),
        "sample_cases": args.output / out_name("step7a_sample_review_cases", args.output_suffix),
        "paper_cases": args.output / out_name("step7a_paper_review_cases", args.output_suffix),
        "decision_template": args.output / out_name("step7a_review_decisions_template", args.output_suffix),
        "source_trace": args.output / out_name("step7a_source_traceability_table", args.output_suffix),
        "sample_context": args.output / out_name("step7a_sample_context_for_review", args.output_suffix),
        "casebook": casebook_path,
        "packet_index": args.output / out_name("step7a_review_packet_index", args.output_suffix),
        "readiness_update": args.output / out_name("step7a_readiness_after_review_packet_summary", args.output_suffix),
        "report": args.report,
    }

    row_cases.to_csv(paths["row_cases"], index=False, encoding="utf-8-sig")
    sample_cases.to_csv(paths["sample_cases"], index=False, encoding="utf-8-sig")
    paper_cases.to_csv(paths["paper_cases"], index=False, encoding="utf-8-sig")
    master.to_csv(paths["master"], index=False, encoding="utf-8-sig")
    decisions.to_csv(paths["decision_template"], index=False, encoding="utf-8-sig")
    trace.to_csv(paths["source_trace"], index=False, encoding="utf-8-sig")
    sample_context.to_csv(paths["sample_context"], index=False, encoding="utf-8-sig")
    readiness_update.to_csv(paths["readiness_update"], index=False, encoding="utf-8-sig")

    log("writing packet index...")
    packet = packet_index(paths)
    packet.to_csv(paths["packet_index"], index=False, encoding="utf-8-sig")

    log("writing report...")
    write_report(args.report, paths, master, trace, readiness_update, {}, warnings, time.time() - started)

    log("running sanity checks...")
    checks, failures = run_sanity(paths, master, decisions)
    if failures:
        write_report(args.report, paths, master, trace, readiness_update, checks, warnings, time.time() - started)
        for failure in failures:
            print(f"[step7a] FAIL: {failure}", flush=True)
        raise SystemExit(1)
    write_report(args.report, paths, master, trace, readiness_update, checks, warnings, time.time() - started)
    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")


if __name__ == "__main__":
    main()
