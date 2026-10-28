"""Independently verify exp007 Phase 1 residual outputs."""

from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
DATA = EXP007 / "data" / "processed"
REPORTS = EXP007 / "reports"
SOURCE = (
    ROOT
    / "experiments"
    / "exp006"
    / "data"
    / "processed"
    / "step6b_broad_family"
    / "step5b_test_predictions_default.csv"
)

REQUIRED_FILES = [
    DATA / "phase1_sigma_residual_rows.csv",
    DATA / "phase1_sigma_sample_summary.csv",
    DATA / "phase1_sigma_material_group_summary.csv",
    DATA / "phase1_sigma_temperature_bin_summary.csv",
    DATA / "phase1_sigma_paper_summary.csv",
    DATA / "phase1_sigma_residual_category_summary.csv",
    DATA / "phase1_sigma_summary_metrics.csv",
    DATA / "phase1_sigma_outlier_rows.csv",
    DATA / "phase1_validation_checks.csv",
    DATA / "phase1_input_manifest.csv",
    REPORTS / "phase1_residual_dataset.md",
    REPORTS / "phase1_validation_log.txt",
    REPORTS / "phase1_sigma_residual_analysis.xlsx",
    REPORTS / "phase1_workbook_validation_log.txt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workbook_sheet_names(path: Path) -> list[str]:
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    return [sheet.attrib["name"] for sheet in root.findall("m:sheets/m:sheet", namespace)]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 1 outputs: {missing}")

    source = pd.read_csv(SOURCE, low_memory=False)
    residuals = pd.read_csv(DATA / "phase1_sigma_residual_rows.csv", low_memory=False)
    samples = pd.read_csv(DATA / "phase1_sigma_sample_summary.csv", low_memory=False)
    materials = pd.read_csv(DATA / "phase1_sigma_material_group_summary.csv")
    temperatures = pd.read_csv(DATA / "phase1_sigma_temperature_bin_summary.csv")
    papers = pd.read_csv(DATA / "phase1_sigma_paper_summary.csv")
    categories = pd.read_csv(DATA / "phase1_sigma_residual_category_summary.csv")
    outliers = pd.read_csv(DATA / "phase1_sigma_outlier_rows.csv", low_memory=False)
    validations = pd.read_csv(DATA / "phase1_validation_checks.csv")
    manifest = pd.read_csv(DATA / "phase1_input_manifest.csv")

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append((name, bool(passed), detail))

    check(
        "all_source_rows_preserved_in_order",
        len(source) == len(residuals)
        and source["row_id"].tolist() == residuals["row_id"].tolist(),
        f"source={len(source)}; residuals={len(residuals)}",
    )
    check(
        "row_ids_unique",
        not residuals["row_id"].duplicated().any(),
        f"duplicates={int(residuals['row_id'].duplicated().sum())}",
    )

    evaluable = residuals["is_evaluable"].eq(True)
    observed = pd.to_numeric(residuals.loc[evaluable, "sigma_observed_S_per_m"])
    predicted = pd.to_numeric(residuals.loc[evaluable, "sigma_predicted_S_per_m"])
    recomputed = np.log10(predicted / observed)
    stored = pd.to_numeric(
        residuals.loc[evaluable, "residual_log10_pred_over_observed"]
    )
    max_difference = float(np.max(np.abs(recomputed.to_numpy() - stored.to_numpy())))
    check(
        "row_residuals_recompute",
        max_difference <= 1e-12,
        f"max_abs_difference={max_difference:.3g}",
    )
    check(
        "non_evaluable_rows_have_blank_residual",
        residuals.loc[~evaluable, "residual_log10_pred_over_observed"].isna().all(),
        f"non_evaluable={int((~evaluable).sum())}",
    )

    ranks = pd.to_numeric(
        residuals.loc[evaluable, "abs_residual_rank_overall"], errors="coerce"
    ).dropna()
    expected_ranks = set(range(1, int(evaluable.sum()) + 1))
    check(
        "residual_ranks_are_complete",
        set(ranks.astype(int)) == expected_ranks,
        f"rank_count={len(ranks)}; evaluable={int(evaluable.sum())}",
    )

    expected_outlier_ids = set(
        residuals.loc[
            evaluable
            & (
                pd.to_numeric(residuals["abs_residual_log10"], errors="coerce")
                > 1.0 + 1e-12
            ),
            "row_id",
        ]
    )
    check(
        "outlier_export_matches_factor_10_rule",
        set(outliers["row_id"]) == expected_outlier_ids,
        f"expected={len(expected_outlier_ids)}; output={len(outliers)}",
    )

    check(
        "sample_summary_has_unique_samples",
        samples["validation_sample_group_id"].is_unique
        and len(samples) == residuals["validation_sample_group_id"].nunique(),
        f"sample_rows={len(samples)}",
    )
    check(
        "material_summary_reconciles_rows",
        int(materials["n_rows_total"].sum()) == len(residuals)
        and int(materials["n_rows_evaluable"].sum()) == int(evaluable.sum()),
        f"total={int(materials['n_rows_total'].sum())}; evaluable={int(materials['n_rows_evaluable'].sum())}",
    )
    check(
        "temperature_summary_reconciles_rows",
        int(temperatures["n_rows_total"].sum()) == len(residuals)
        and int(temperatures["n_rows_evaluable"].sum()) == int(evaluable.sum()),
        f"total={int(temperatures['n_rows_total'].sum())}; evaluable={int(temperatures['n_rows_evaluable'].sum())}",
    )
    check(
        "paper_summary_reconciles_rows",
        int(papers["n_rows_total"].sum()) == len(residuals)
        and int(papers["n_rows_evaluable"].sum()) == int(evaluable.sum()),
        f"total={int(papers['n_rows_total'].sum())}; evaluable={int(papers['n_rows_evaluable'].sum())}",
    )

    band_rows = categories.loc[categories["summary_type"].eq("residual_band")]
    check(
        "residual_bands_reconcile",
        int(band_rows["n_rows"].sum()) == int(evaluable.sum()),
        f"bands={int(band_rows['n_rows'].sum())}; evaluable={int(evaluable.sum())}",
    )
    check(
        "builder_validation_all_pass",
        validations["status"].eq("PASS").all(),
        f"failed={int(validations['status'].ne('PASS').sum())}",
    )

    current_hash = sha256(SOURCE)
    check(
        "source_hash_matches_manifest",
        len(manifest) == 1
        and manifest.iloc[0]["sha256_before"] == current_hash
        and manifest.iloc[0]["sha256_after"] == current_hash,
        f"current={current_hash}",
    )

    workbook = REPORTS / "phase1_sigma_residual_analysis.xlsx"
    expected_sheets = [
        "Summary",
        "Outliers",
        "Unavailable rows",
        "Sample summary",
        "Material summary",
        "Temperature summary",
        "Paper summary",
    ]
    sheet_names = workbook_sheet_names(workbook)
    check(
        "workbook_has_expected_sheets",
        sheet_names == expected_sheets,
        f"sheets={sheet_names}",
    )
    check(
        "workbook_is_nonempty",
        workbook.stat().st_size > 100_000,
        f"size_bytes={workbook.stat().st_size}",
    )

    report = (REPORTS / "phase1_residual_dataset.md").read_text(encoding="utf-8")
    check(
        "report_records_primary_baseline",
        "broad_family_100K" in report and "residual_log10_pred_over_observed" in report,
        "baseline and residual definition present",
    )
    workbook_log = (REPORTS / "phase1_workbook_validation_log.txt").read_text(
        encoding="utf-8"
    )
    check(
        "workbook_validation_passed",
        "Final status: PASS" in workbook_log,
        "workbook validation log final status",
    )

    failed = [name for name, passed, _ in checks if not passed]
    lines = [
        "exp007 Phase 1 independent check",
        "=====================================",
        *[
            f"{'PASS' if passed else 'FAIL'} {name}: {detail}"
            for name, passed, detail in checks
        ],
        "",
        f"test_rows={len(residuals)}",
        f"evaluable_rows={int(evaluable.sum())}",
        f"samples={len(samples)}",
        f"factor_10_outlier_rows={len(outliers)}",
        f"maximum_residual_recompute_difference={max_difference:.3g}",
        f"Final status: {'PASS' if not failed else 'FAIL'}",
    ]
    output = "\n".join(lines) + "\n"
    (REPORTS / "phase1_check_log.txt").write_text(output, encoding="utf-8")
    print(output)
    if failed:
        raise RuntimeError(f"Phase 1 independent checks failed: {failed}")


if __name__ == "__main__":
    main()
