"""Independent checks for exp007 Phase 3 oxide-inventory outputs."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
PHASE1_ROWS = DATA / "phase1_sigma_residual_rows.csv"

REQUIRED = [
    DATA / "phase3_oxide_inventory_summary.csv",
    DATA / "phase3_oxide_row_inventory.csv",
    DATA / "phase3_oxide_sample_inventory.csv",
    DATA / "phase3_oxide_label_inventory.csv",
    DATA / "phase3_oxide_review_queue.csv",
    DATA / "phase3_oxide_candidate_family_summary.csv",
    DATA / "phase3_oxide_element_summary.csv",
    DATA / "phase3_oxide_temperature_support.csv",
    DATA / "phase3_input_manifest.csv",
    DATA / "phase3_validation_checks.csv",
    REPORTS / "phase3_oxide_inventory.md",
    REPORTS / "phase3_validation_log.txt",
    REPORTS / "phase3_workbook_validation_log.txt",
    REPORTS / "phase3_oxide_inventory.xlsx",
]

EXPECTED_SHEETS = {
    "Summary",
    "Candidate families",
    "Review queue",
    "Samples",
    "Material labels",
    "Elements",
    "Temperature support",
    "Oxide rows",
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
        for name in archive.namelist():
            if name.startswith("xl/") and name.endswith(".xml"):
                if pattern.search(archive.read(name)):
                    return True
    return False


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, bool(condition), detail))

    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.is_file()]
    check("required_outputs_exist", not missing, f"missing={missing}")
    if missing:
        raise SystemExit("Missing Phase 3 outputs; run the build steps first.")

    rows = pd.read_csv(DATA / "phase3_oxide_row_inventory.csv", low_memory=False)
    samples = pd.read_csv(DATA / "phase3_oxide_sample_inventory.csv", low_memory=False)
    labels = pd.read_csv(DATA / "phase3_oxide_label_inventory.csv", low_memory=False)
    review = pd.read_csv(DATA / "phase3_oxide_review_queue.csv", low_memory=False)
    families = pd.read_csv(DATA / "phase3_oxide_candidate_family_summary.csv", low_memory=False)
    elements = pd.read_csv(DATA / "phase3_oxide_element_summary.csv", low_memory=False)
    temperature = pd.read_csv(DATA / "phase3_oxide_temperature_support.csv", low_memory=False)
    manifest = pd.read_csv(DATA / "phase3_input_manifest.csv")
    validation = pd.read_csv(DATA / "phase3_validation_checks.csv")
    phase1 = pd.read_csv(PHASE1_ROWS, usecols=["row_id", "is_oxide_broad"], low_memory=False)
    phase1_oxide = phase1.loc[as_bool(phase1["is_oxide_broad"])]

    check("all_build_validation_passes", validation["status"].eq("PASS").all(), f"failed={int(validation['status'].eq('FAIL').sum())}")
    check("phase1_oxide_rows_reconcile", len(rows) == len(phase1_oxide), f"phase3={len(rows)}; phase1={len(phase1_oxide)}")
    check("row_ids_match_phase1", set(rows["row_id"].astype(str)) == set(phase1_oxide["row_id"].astype(str)), f"unique={rows['row_id'].nunique()}")
    check("row_ids_unique", rows["row_id"].is_unique, f"duplicates={int(rows['row_id'].duplicated().sum())}")
    check("sample_rows_reconcile", len(samples) == rows["validation_sample_group_id"].nunique(), f"samples={len(samples)}")
    check("label_rows_reconcile", len(labels) == rows["material_label_normalized"].nunique(), f"labels={len(labels)}")
    check("family_rows_reconcile", int(families["n_rows"].sum()) == len(rows), f"sum={int(families['n_rows'].sum())}")
    check("review_queue_reconciles", len(review) == int(as_bool(samples["manual_review_required"]).sum()), f"review={len(review)}")
    check("all_rows_have_O", rows["parsed_elements"].astype(str).str.split(";").map(lambda values: "O" in values).all(), "parsed_elements")
    check("all_rows_parse_from_formula", rows["formula_parse_source"].eq("formula_raw").all(), ", ".join(sorted(rows["formula_parse_source"].unique())))
    check("all_current_reasons_contains_O", rows["current_broad_family_reason"].eq("contains_O").all(), ", ".join(sorted(rows["current_broad_family_reason"].unique())))
    check("candidate_families_nonempty", rows["candidate_oxide_family"].notna().all() and len(families) >= 5, f"families={len(families)}")
    check("organic_false_positive_candidates_present", as_bool(samples["likely_not_inorganic_oxide"]).any(), f"samples={int(as_bool(samples['likely_not_inorganic_oxide']).sum())}")
    check("mixed_anion_candidates_present", as_bool(samples["has_chalcogen_anion_with_O"]).any(), f"samples={int(as_bool(samples['has_chalcogen_anion_with_O']).sum())}")
    check("temperature_support_has_both_pass_and_fail", as_bool(temperature["meets_reference_like_minimum_3rows_3samples_1paper"]).any() and (~as_bool(temperature["meets_reference_like_minimum_3rows_3samples_1paper"])).any(), f"cells={len(temperature)}")
    check("element_inventory_contains_O", "O" in set(elements["element"]), f"elements={len(elements)}")
    check("residuals_finite", np.isfinite(pd.to_numeric(rows["residual_log10_pred_over_observed"], errors="coerce")).all(), f"rows={len(rows)}")

    current_hashes = []
    missing_manifest_sources = []
    for item in manifest.itertuples(index=False):
        source = ROOT / str(item.path)
        if not source.is_file():
            missing_manifest_sources.append(str(item.path))
            current_hashes.append(None)
        else:
            current_hashes.append(sha256(source))
    check("manifest_sources_exist", not missing_manifest_sources, f"missing={missing_manifest_sources}")
    check("manifest_hashes_match", current_hashes == manifest["sha256"].tolist(), "current SHA-256 versus recorded")

    workbook = REPORTS / "phase3_oxide_inventory.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        bad_zip_member = archive.testzip()
    sheets = workbook_sheet_names(workbook)
    check("workbook_zip_integrity", bad_zip_member is None, f"bad_member={bad_zip_member}")
    check("workbook_sheets_complete", sheets == EXPECTED_SHEETS, f"sheets={sorted(sheets)}")
    check("workbook_has_no_error_text", not workbook_has_error_text(workbook), "scan of xl/*.xml")
    check("workbook_nonempty", workbook.stat().st_size > 100_000, f"size={workbook.stat().st_size}")

    report = (REPORTS / "phase3_oxide_inventory.md").read_text(encoding="utf-8")
    check("report_records_scope_and_limit", "broad_family_100K" in report and "not a training-feasibility test" in report, "scope and interpretation limit")
    check("workbook_validation_passes", "status=PASS" in (REPORTS / "phase3_workbook_validation_log.txt").read_text(encoding="utf-8"), "phase3_workbook_validation_log.txt")

    passed = all(condition for _, condition, _ in checks)
    log_lines = [
        "Phase 3 independent checks",
        f"status={'PASS' if passed else 'FAIL'}",
        f"checks={len(checks)}",
        f"failed={sum(not condition for _, condition, _ in checks)}",
    ]
    log_lines.extend(
        f"{name}: {'PASS' if condition else 'FAIL'} | {detail}"
        for name, condition, detail in checks
    )
    (REPORTS / "phase3_check_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print("\n".join(log_lines[:4]))
    if not passed:
        failed = [name for name, condition, _ in checks if not condition]
        raise SystemExit(f"Phase 3 checks failed: {failed}")


if __name__ == "__main__":
    main()
