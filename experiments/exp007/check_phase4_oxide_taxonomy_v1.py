"""Independent checks for exp007 Phase 4 oxide taxonomy v1 outputs."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from oxide_taxonomy_v1 import TAXONOMY_VERSION, rules_sha256


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"

REQUIRED = [
    DATA / "phase4_oxide_v1_inventory_summary.csv",
    DATA / "phase4_oxide_v1_rules.csv",
    DATA / "phase4_oxide_v1_class_summary.csv",
    DATA / "phase4_oxide_v1_row_assignments.csv",
    DATA / "phase4_oxide_v1_sample_assignments.csv",
    DATA / "phase4_oxide_v1_label_assignments.csv",
    DATA / "phase4_oxide_v1_review_queue.csv",
    DATA / "phase4_oxide_v1_transition_from_phase3.csv",
    DATA / "phase4_oxide_v1_rule_examples.csv",
    DATA / "phase4_oxide_v1_classification_config.json",
    DATA / "phase4_input_manifest.csv",
    DATA / "phase4_validation_checks.csv",
    REPORTS / "phase4_oxide_v1_definition.md",
    REPORTS / "phase4_validation_log.txt",
    REPORTS / "phase4_workbook_validation_log.txt",
    REPORTS / "phase4_oxide_v1_definition.xlsx",
]

EXPECTED_SHEETS = {
    "Summary",
    "Rules",
    "Class summary",
    "Review queue",
    "Sample assignments",
    "Label assignments",
    "Phase3 transition",
    "Rule examples",
    "Row assignments",
    "Validation",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.casefold().isin({"true", "1", "yes", "y"})


def workbook_sheet_names(path: Path) -> set[str]:
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("xl/workbook.xml"))
    return {node.attrib["name"] for node in root.findall("x:sheets/x:sheet", namespace)}


def workbook_has_error_text(path: Path) -> bool:
    pattern = re.compile(rb"#REF!|#DIV/0!|#VALUE!|#NAME\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!")
    with zipfile.ZipFile(path) as archive:
        return any(
            pattern.search(archive.read(name))
            for name in archive.namelist()
            if name.startswith("xl/") and name.endswith(".xml")
        )


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    check("required_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 4 outputs; run the build steps first.")

    phase3_rows = pd.read_csv(DATA / "phase3_oxide_row_inventory.csv", usecols=["row_id"], low_memory=False)
    rows = pd.read_csv(DATA / "phase4_oxide_v1_row_assignments.csv", low_memory=False)
    samples = pd.read_csv(DATA / "phase4_oxide_v1_sample_assignments.csv", low_memory=False)
    labels = pd.read_csv(DATA / "phase4_oxide_v1_label_assignments.csv", low_memory=False)
    review = pd.read_csv(DATA / "phase4_oxide_v1_review_queue.csv", low_memory=False)
    classes = pd.read_csv(DATA / "phase4_oxide_v1_class_summary.csv", low_memory=False)
    transitions = pd.read_csv(DATA / "phase4_oxide_v1_transition_from_phase3.csv", low_memory=False)
    rules = pd.read_csv(DATA / "phase4_oxide_v1_rules.csv", low_memory=False)
    validation = pd.read_csv(DATA / "phase4_validation_checks.csv")
    manifest = pd.read_csv(DATA / "phase4_input_manifest.csv")
    config = json.loads((DATA / "phase4_oxide_v1_classification_config.json").read_text(encoding="utf-8-sig"))

    expected_hash = rules_sha256()
    check("all_build_validation_passes", validation["status"].eq("PASS").all(), f"failed={int(validation['status'].eq('FAIL').sum())}")
    check("taxonomy_version_fixed", TAXONOMY_VERSION == "oxide_v1" and rows["taxonomy_version"].eq("oxide_v1").all(), TAXONOMY_VERSION)
    check("row_count_reconciles", len(rows) == len(phase3_rows) == 3437, f"phase4={len(rows)}; phase3={len(phase3_rows)}")
    check("row_ids_match_phase3", set(rows["row_id"].astype(str)) == set(phase3_rows["row_id"].astype(str)), f"unique={rows['row_id'].nunique()}")
    check("row_ids_unique", rows["row_id"].is_unique, f"duplicates={int(rows['row_id'].duplicated().sum())}")
    check("all_rows_have_one_rule", rows["oxide_v1_rule_id"].notna().all(), f"missing={int(rows['oxide_v1_rule_id'].isna().sum())}")
    check("sample_count_reconciles", len(samples) == rows["validation_sample_group_id"].nunique() == 610, f"samples={len(samples)}")
    check("label_count_reconciles", len(labels) == rows["material_label_normalized"].nunique() == 506, f"labels={len(labels)}")
    check("sample_assignment_unique", samples["validation_sample_group_id"].is_unique, f"duplicates={int(samples['validation_sample_group_id'].duplicated().sum())}")
    check("no_sample_assignment_conflicts", not as_bool(samples["sample_assignment_conflict"]).any(), f"conflicts={int(as_bool(samples['sample_assignment_conflict']).sum())}")
    check("class_rows_reconcile", int(classes["n_rows"].sum()) == len(rows), f"sum={int(classes['n_rows'].sum())}")
    check("transition_rows_reconcile", int(transitions["n_rows"].sum()) == len(rows), f"sum={int(transitions['n_rows'].sum())}")
    check("rule_ids_unique", rules["rule_id"].is_unique and rules["priority"].is_unique, f"rules={len(rules)}")
    check("rules_hash_matches_implementation", rules["taxonomy_rules_sha256"].eq(expected_hash).all(), expected_hash)
    check("rules_hash_matches_config", config["taxonomy_rules_sha256"] == expected_hash, config["taxonomy_rules_sha256"])
    check("fail_safe_rule_unused", int(rules.loc[rules["rule_id"].eq("OXV1_R999"), "matched_rows"].iloc[0]) == 0, "OXV1_R999")
    check("organic_exclusion_present", int(rules.loc[rules["rule_id"].eq("OXV1_R010"), "matched_samples"].iloc[0]) == 21, "excluded samples=21")
    mixed = set(rows.loc[rows["oxide_v1_scope"].eq("mixed_anion_O_containing"), "oxide_v1_rule_id"])
    check("mixed_anion_rules_present", mixed == {"OXV1_R020", "OXV1_R030", "OXV1_R040"}, f"rules={sorted(mixed)}")
    check("inclusion_status_complete", samples["oxide_v1_inclusion_status"].isin({"include", "review", "exclude"}).all(), ", ".join(sorted(samples["oxide_v1_inclusion_status"].unique())))
    eligible = as_bool(samples["oxide_v1_default_model_eligible"])
    check("eligible_only_when_included", eligible.eq(samples["oxide_v1_inclusion_status"].eq("include")).all(), f"eligible={int(eligible.sum())}")
    check("structural_class_not_inferred", samples["oxide_v1_structural_class"].eq("not_assigned_formula_only_v1").all(), "composition-only v1")
    needs_attention = samples["oxide_v1_inclusion_status"].ne("include")
    check("review_queue_reconciles", len(review) == int(needs_attention.sum()) == 57, f"queue={len(review)}")
    status_counts = samples["oxide_v1_inclusion_status"].value_counts().to_dict()
    check("status_counts_frozen", status_counts == {"include": 553, "review": 36, "exclude": 21}, f"counts={status_counts}")

    current_hashes: list[str | None] = []
    missing_manifest_sources: list[str] = []
    for item in manifest.itertuples(index=False):
        source = ROOT / str(item.path)
        if not source.is_file():
            missing_manifest_sources.append(str(item.path))
            current_hashes.append(None)
        else:
            current_hashes.append(sha256(source))
    check("manifest_sources_exist", not missing_manifest_sources, f"missing={missing_manifest_sources}")
    check("manifest_hashes_match", current_hashes == manifest["sha256"].tolist(), "current SHA-256 versus recorded")

    workbook = REPORTS / "phase4_oxide_v1_definition.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        bad_zip_member = archive.testzip()
    sheets = workbook_sheet_names(workbook)
    check("workbook_zip_integrity", bad_zip_member is None, f"bad_member={bad_zip_member}")
    check("workbook_sheets_complete", sheets == EXPECTED_SHEETS, f"sheets={sorted(sheets)}")
    check("workbook_has_no_error_text", not workbook_has_error_text(workbook), "scan of xl/*.xml")
    check("workbook_nonempty", workbook.stat().st_size > 100_000, f"size={workbook.stat().st_size}")
    check("workbook_validation_passes", "status=PASS" in (REPORTS / "phase4_workbook_validation_log.txt").read_text(encoding="utf-8"), "phase4_workbook_validation_log.txt")

    report = (REPORTS / "phase4_oxide_v1_definition.md").read_text(encoding="utf-8")
    check("report_records_scope_and_limit", "composition-only" in report and "does not distinguish perovskite" in report, "scope and interpretation limit")

    passed = all(condition for _, condition, _ in checks)
    log_lines = [
        "Phase 4 independent checks",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"failed={sum(not condition for _, condition, _ in checks)}",
    ]
    log_lines.extend(
        f"{name}: {'PASS' if condition else 'FAIL'} | {detail}"
        for name, condition, detail in checks
    )
    (REPORTS / "phase4_check_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines[:4]))
    if not passed:
        failed = [name for name, condition, _ in checks if not condition]
        raise SystemExit(f"Phase 4 checks failed: {failed}")


if __name__ == "__main__":
    main()
