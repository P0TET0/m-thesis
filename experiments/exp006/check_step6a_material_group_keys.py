import argparse
from pathlib import Path

import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EXP_DIR / "data" / "processed"
REPORT_DIR = EXP_DIR / "reports"

VARIANTS = {
    "formula_system": "material_group_key_formula_system",
    "broad_family": "material_group_key_broad_family",
    "hybrid_v1": "material_group_key_hybrid_v1",
    "hybrid_v2_broad_first": "material_group_key_hybrid_v2_broad_first",
    "formula_system_collapsed": "material_group_key_formula_system_collapsed",
    "hybrid_v1_collapsed": "material_group_key_hybrid_v1_collapsed",
}

REQUIRED_CANDIDATE_COLUMNS = [
    "row_id",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "material_group_key_original",
    "material_family_raw_clean",
    "material_text_combined",
    "parsed_elements",
    "parsed_element_count",
    "formula_parse_status",
    "formula_system_key",
    *VARIANTS.values(),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Step6A material group key outputs.")
    parser.add_argument("--candidate-rows", type=Path, default=PROCESSED_DIR / "step6a_material_group_candidate_rows.csv")
    parser.add_argument("--summary", type=Path, default=PROCESSED_DIR / "step6a_material_group_key_summary.csv")
    parser.add_argument("--preflight", type=Path, default=PROCESSED_DIR / "step6a_material_group_key_preflight_coverage.csv")
    parser.add_argument("--recommended", type=Path, default=PROCESSED_DIR / "step6a_recommended_material_key_variants.csv")
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "step6a_material_group_key_rebuild_report.md")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], label: str, failures: list[str]) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        failures.append(f"{label} missing columns: {missing}")


def variant_path(candidate_path: Path, variant: str) -> Path:
    suffix = ""
    stem = candidate_path.stem
    base = "step6a_material_group_candidate_rows"
    if stem.startswith(base):
        suffix = stem[len(base):]
    return candidate_path.parent / f"step6a_validation_rows_with_splits_key_{variant}{suffix}.csv"


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for path in [args.candidate_rows, args.summary, args.preflight, args.recommended, args.report]:
        if not path.exists():
            failures.append(f"missing output: {path}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    candidates = pd.read_csv(args.candidate_rows, low_memory=False)
    summary = pd.read_csv(args.summary, low_memory=False)
    preflight = pd.read_csv(args.preflight, low_memory=False)
    recommended = pd.read_csv(args.recommended, low_memory=False)
    require_columns(candidates, REQUIRED_CANDIDATE_COLUMNS, "candidate_rows", failures)
    require_columns(
        summary,
        ["material_key_variant", "unique_group_count", "unknown_row_fraction", "row_count", "sample_count"],
        "summary",
        failures,
    )
    require_columns(
        preflight,
        ["material_key_variant", "split_scheme", "reference_source_subset", "eval_target_subset", "coverage_fraction"],
        "preflight",
        failures,
    )
    require_columns(
        recommended,
        ["rank", "material_key_variant", "unique_group_count", "unknown_row_fraction"],
        "recommended",
        failures,
    )
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)

    if len(candidates) == 0:
        failures.append("candidate_rows is empty")
    if not candidates["row_id"].is_unique:
        failures.append("row_id is not unique")
    if not set(candidates["formula_parse_status"].dropna()).issubset({"ok", "low_confidence", "failed"}):
        failures.append("formula_parse_status contains unexpected values")
    if (pd.to_numeric(candidates["parsed_element_count"], errors="coerce") < 0).any():
        failures.append("parsed_element_count contains negative values")
    for col in VARIANTS.values():
        if candidates[col].isna().any():
            failures.append(f"{col} contains missing values")
    if preflight.empty:
        failures.append("preflight is empty")
    if not pd.to_numeric(preflight["coverage_fraction"], errors="coerce").dropna().between(0, 1).all():
        failures.append("coverage_fraction is outside 0..1")
    if (summary[summary["material_key_variant"].isin(VARIANTS.keys())]["unique_group_count"].astype(float) > 1).sum() == 0:
        failures.append("no candidate variant has unique_group_count > 1")
    if (
        summary[summary["material_key_variant"].isin(["formula_system", "hybrid_v1", "hybrid_v2_broad_first"])]
        ["unknown_row_fraction"]
        .astype(float)
        .ge(1.0)
        .all()
    ):
        failures.append("formula/hybrid variants are all unknown")

    for variant, col in VARIANTS.items():
        path = variant_path(args.candidate_rows, variant)
        if not path.exists():
            failures.append(f"missing variant file: {path}")
            continue
        frame = pd.read_csv(path, usecols=["row_id", "material_group_key", "material_group_key_original", "sample_holdout_split", "paper_holdout_split"], low_memory=False)
        if len(frame) != len(candidates):
            failures.append(f"variant {variant} row count mismatch")
        if not frame["material_group_key"].equals(candidates[col].reset_index(drop=True)):
            failures.append(f"variant {variant} material_group_key replacement mismatch")
        if "material_group_key_original" not in frame.columns:
            failures.append(f"variant {variant} missing material_group_key_original")
        if frame["sample_holdout_split"].isna().any() or frame["paper_holdout_split"].isna().any():
            failures.append(f"variant {variant} split columns contain missing values")

    if args.report.stat().st_size == 0:
        failures.append("report is empty")

    print(f"candidate rows: {len(candidates)}")
    print(f"summary rows: {len(summary)}")
    print(f"preflight rows: {len(preflight)}")
    print(f"recommended rows: {len(recommended)}")
    print(f"formula_parse_status counts: {candidates['formula_parse_status'].value_counts().to_dict()}")
    print(summary[["material_key_variant", "unique_group_count", "unknown_row_fraction"]].to_string(index=False))
    print(recommended.head(6).to_string(index=False))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print("step6a material group key checks passed")


if __name__ == "__main__":
    main()
