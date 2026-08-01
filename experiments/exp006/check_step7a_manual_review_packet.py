import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step7a_manual_review_packet"
DEFAULT_REPORT = EXP_DIR / "reports" / "step7a_manual_review_packet" / "step7a_manual_review_packet_report.md"

HUMAN_COLS = [
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
    parser = argparse.ArgumentParser(description="Check Step7A manual review packet outputs.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--master", type=Path, default=DEFAULT_OUTPUT / "step7a_manual_review_master.csv")
    parser.add_argument("--decision-template", type=Path, default=DEFAULT_OUTPUT / "step7a_review_decisions_template.csv")
    parser.add_argument("--source-trace", type=Path, default=DEFAULT_OUTPUT / "step7a_source_traceability_table.csv")
    parser.add_argument("--casebook", type=Path, default=DEFAULT_OUTPUT / "step7a_manual_review_casebook.md")
    parser.add_argument("--packet-index", type=Path, default=DEFAULT_OUTPUT / "step7a_review_packet_index.csv")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def sibling(path: Path, old: str, new: str) -> Path:
    return path.with_name(path.name.replace(old, new))


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    paths = {
        "master": args.master,
        "row_cases": sibling(args.master, "manual_review_master", "row_review_cases"),
        "sample_cases": sibling(args.master, "manual_review_master", "sample_review_cases"),
        "paper_cases": sibling(args.master, "manual_review_master", "paper_review_cases"),
        "decision_template": args.decision_template,
        "source_trace": args.source_trace,
        "sample_context": sibling(args.master, "manual_review_master", "sample_context_for_review"),
        "casebook": args.casebook,
        "packet_index": args.packet_index,
        "readiness_update": sibling(args.master, "manual_review_master", "readiness_after_review_packet_summary"),
        "report": args.report,
    }
    for label, path in paths.items():
        if not path.exists() or path.stat().st_size == 0:
            failures.append(f"missing or empty {label}: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    master = pd.read_csv(args.master, low_memory=False)
    row_cases = pd.read_csv(paths["row_cases"], low_memory=False)
    sample_cases = pd.read_csv(paths["sample_cases"], low_memory=False)
    paper_cases = pd.read_csv(paths["paper_cases"], low_memory=False)
    decisions = pd.read_csv(args.decision_template, low_memory=False)
    trace = pd.read_csv(args.source_trace, low_memory=False)
    context = pd.read_csv(paths["sample_context"], low_memory=False)
    packet_index = pd.read_csv(args.packet_index, low_memory=False)
    readiness = pd.read_csv(paths["readiness_update"], low_memory=False)

    require_columns(
        master,
        [
            "review_case_id",
            "review_case_type",
            "review_priority",
            "review_reason",
            "review_status_initial",
            "row_id",
            "validation_sample_group_id",
            "validation_paper_group_id",
            "abs_error_decades",
            "source_traceability_score",
            "manual_review_note_hint",
        ],
        "master",
        failures,
    )
    require_columns(decisions, ["review_case_id", "review_case_type", "review_priority", *HUMAN_COLS], "decision_template", failures)
    require_columns(trace, ["row_id", "source_traceability_score", "missing_source_fields", "source_review_hint"], "source_trace", failures)
    require_columns(context, ["review_case_id", "row_id", "validation_sample_group_id", "abs_error_decades"], "sample_context", failures)
    require_columns(packet_index, ["file_role", "file_path", "description", "intended_user_action"], "packet_index", failures)
    require_columns(readiness, ["item", "status", "value", "comment", "next_action"], "readiness_update", failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if master.empty:
        failures.append("master is empty")
    if not master["review_case_id"].is_unique:
        failures.append("review_case_id is not unique")
    if master["review_priority"].isna().any():
        failures.append("review_priority has missing values")
    if not master["review_status_initial"].eq("pending").all():
        failures.append("review_status_initial must be pending for all cases")
    if decisions.empty:
        failures.append("decision template is empty")
    if set(HUMAN_COLS) - set(decisions.columns):
        failures.append("decision template missing human columns")
    if not decisions["review_status"].eq("pending").all():
        failures.append("decision template review_status must start as pending")
    if trace.empty:
        failures.append("source traceability table is empty")
    if context.empty:
        failures.append("sample context is empty")
    if packet_index.empty:
        failures.append("packet index is empty")
    if readiness.empty:
        failures.append("readiness update is empty")
    if args.casebook.stat().st_size == 0:
        failures.append("casebook is empty")
    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    print(f"manual review master rows: {len(master)}")
    print(f"row_case rows: {len(row_cases)}")
    print(f"sample_case rows: {len(sample_cases)}")
    print(f"paper_case rows: {len(paper_cases)}")
    print(f"decision template rows: {len(decisions)}")
    print(f"source traceability rows: {len(trace)}")
    print(f"sample context rows: {len(context)}")
    print(f"packet index rows: {len(packet_index)}")
    print(f"readiness update rows: {len(readiness)}")
    print(f"source_traceability_score median: {pd.to_numeric(trace['source_traceability_score'], errors='coerce').median()}")
    print(f"cases with source metadata gaps: {(pd.to_numeric(master['source_traceability_score'], errors='coerce').fillna(0) < 5).sum()}")
    print(f"extreme cases: {master['error_severity'].eq('extreme_ge_10_decades').sum()}")
    print("step7a manual review packet checks passed")


if __name__ == "__main__":
    main()
