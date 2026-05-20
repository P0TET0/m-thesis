import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT / "data" / "output" / "starrydata2_prepared_for_relaxation_time_csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step3_fixed"
DEFAULT_SAMPLE_MASTER = DEFAULT_INPUT_DIR / "sample_master.csv"
DEFAULT_PROPERTY_DATA = DEFAULT_INPUT_DIR / "property_data.csv"

SAMPLE_REQUIRED_COLUMNS = {
    "sample_key",
    "DOI",
    "sample_id",
    "composition",
}
PROPERTY_REQUIRED_COLUMNS = {
    "curve_key",
    "sample_key",
    "DOI",
    "sample_id",
    "composition",
    "prop_y_canonical",
    "property_family",
    "unit_y",
    "y_values_json",
}
TARGET_FAMILIES = (
    "seebeck",
    "electrical_conductivity",
    "electrical_resistivity",
    "thermal_conductivity",
    "zt",
)
DIMENSIONLESS_UNIT_NORMALIZED = {"", "-", "1", "dimensionless"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fix and extend Starrydata2 step-3 CSV outputs for relaxation-time fitting "
            "and accuracy-check preparation."
        )
    )
    parser.add_argument(
        "--sample-master",
        type=Path,
        default=DEFAULT_SAMPLE_MASTER,
        help="input sample_master.csv path",
    )
    parser.add_argument(
        "--property-data",
        type=Path,
        default=DEFAULT_PROPERTY_DATA,
        help="input property_data.csv path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for fixed outputs",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def load_numeric_list(raw_value: Any) -> list[float]:
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []

    parsed: Any = raw_value
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []

    if not isinstance(parsed, (list, tuple)):
        return []

    values: list[float] = []
    for value in parsed:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{label} is missing required columns: {sorted(missing)}")


def seebeck_sign_class(values: list[float], tolerance: float = 1e-15) -> str:
    if not values:
        return "unknown"
    positive = all(value > tolerance for value in values)
    negative = all(value < -tolerance for value in values)
    if positive:
        return "positive"
    if negative:
        return "negative"
    if all(abs(value) <= tolerance for value in values):
        return "zero"
    return "mixed"


def carrier_type_from_sign(sign_class: str) -> str:
    if sign_class == "positive":
        return "p-type"
    if sign_class == "negative":
        return "n-type"
    if sign_class == "zero":
        return "unknown"
    if sign_class == "mixed":
        return "mixed"
    return "unknown"


def normalize_unit(unit_y: Any) -> str:
    return normalize_text(unit_y).casefold()


def zt_unit_ok(unit_y: Any) -> bool:
    return normalize_unit(unit_y) in DIMENSIONLESS_UNIT_NORMALIZED


def yes_no(value: bool) -> str:
    return "yes" if bool(value) else "no"


def join_unique(values: pd.Series) -> str:
    items = sorted({normalize_text(value) for value in values if normalize_text(value)})
    return " | ".join(items)


def build_property_fixed(property_df: pd.DataFrame) -> pd.DataFrame:
    property_fixed = property_df.copy()
    property_fixed["sample_key"] = property_fixed["sample_key"].map(normalize_text)
    property_fixed["curve_key"] = property_fixed["curve_key"].map(normalize_text)
    property_fixed["prop_y_canonical"] = property_fixed["prop_y_canonical"].map(normalize_text)
    property_fixed["property_family"] = property_fixed["property_family"].map(normalize_text)
    property_fixed["unit_y"] = property_fixed["unit_y"].map(normalize_text)

    property_fixed["is_seebeck_curve"] = property_fixed["property_family"].eq("seebeck")
    property_fixed["is_electrical_conductivity_curve"] = property_fixed["property_family"].eq(
        "electrical_conductivity"
    )
    property_fixed["is_electrical_resistivity_curve"] = property_fixed["property_family"].eq(
        "electrical_resistivity"
    )
    property_fixed["is_thermal_conductivity_curve"] = property_fixed["property_family"].eq(
        "thermal_conductivity"
    )
    property_fixed["is_zt_curve"] = property_fixed["property_family"].eq("zt")

    seebeck_values = property_fixed["y_values_json"].map(load_numeric_list)
    property_fixed["seebeck_curve_sign_class"] = "not_seebeck"
    property_fixed.loc[property_fixed["is_seebeck_curve"], "seebeck_curve_sign_class"] = (
        seebeck_values[property_fixed["is_seebeck_curve"]].map(seebeck_sign_class)
    )
    property_fixed["curve_carrier_type_guess"] = "not_applicable"
    property_fixed.loc[property_fixed["is_seebeck_curve"], "curve_carrier_type_guess"] = (
        property_fixed.loc[property_fixed["is_seebeck_curve"], "seebeck_curve_sign_class"]
        .map(carrier_type_from_sign)
    )

    property_fixed["zt_unit_is_dimensionless"] = property_fixed["is_zt_curve"] & property_fixed[
        "unit_y"
    ].map(zt_unit_ok)
    property_fixed["zt_unit_needs_check"] = property_fixed["is_zt_curve"] & ~property_fixed[
        "unit_y"
    ].map(zt_unit_ok)
    property_fixed["zt_unit_check_status"] = "not_zt"
    property_fixed.loc[property_fixed["is_zt_curve"], "zt_unit_check_status"] = (
        property_fixed.loc[property_fixed["is_zt_curve"], "zt_unit_needs_check"]
        .map(lambda flag: "needs_check" if flag else "ok")
    )

    property_fixed = property_fixed.sort_values(
        ["sample_key", "property_family", "prop_y_canonical", "curve_key"], kind="stable"
    ).reset_index(drop=True)
    return property_fixed


def summarize_sample_seebeck(property_fixed: pd.DataFrame) -> pd.DataFrame:
    seebeck_df = property_fixed[property_fixed["is_seebeck_curve"]].copy()
    if seebeck_df.empty:
        return pd.DataFrame(
            columns=[
                "sample_key",
                "seebeck_sign_class",
                "carrier_type_guess",
            ]
        )

    grouped = seebeck_df.groupby("sample_key", sort=False)
    records: list[dict[str, Any]] = []
    for sample_key, group in grouped:
        signs = group["seebeck_curve_sign_class"].tolist()
        sign_set = set(signs)
        if sign_set == {"positive"}:
            sign_class = "positive"
        elif sign_set == {"negative"}:
            sign_class = "negative"
        elif sign_set == {"zero"}:
            sign_class = "zero"
        else:
            sign_class = "mixed"

        records.append(
            {
                "sample_key": sample_key,
                "seebeck_sign_class": sign_class,
                "carrier_type_guess": carrier_type_from_sign(sign_class),
            }
        )
    return pd.DataFrame.from_records(records)


def build_sample_fixed(sample_df: pd.DataFrame, property_fixed: pd.DataFrame) -> pd.DataFrame:
    sample_fixed = sample_df.copy()
    sample_fixed["sample_key"] = sample_fixed["sample_key"].map(normalize_text)
    sample_fixed["DOI"] = sample_fixed["DOI"].map(normalize_text)
    sample_fixed["sample_id"] = sample_fixed["sample_id"].map(normalize_text)
    sample_fixed["composition"] = sample_fixed["composition"].map(normalize_text)

    counts = (
        property_fixed.groupby(["sample_key", "property_family"], sort=False)
        .size()
        .unstack(fill_value=0)
    )
    for family in TARGET_FAMILIES:
        if family not in counts.columns:
            counts[family] = 0
    counts = counts.reset_index()
    counts = counts.rename(
        columns={
            column: f"{column}_curve_count"
            for column in counts.columns
            if column != "sample_key"
        }
    )

    zt_df = property_fixed[property_fixed["is_zt_curve"]].copy()
    if zt_df.empty:
        zt_summary = pd.DataFrame(
            columns=[
                "sample_key",
                "zt_unit_values",
                "zt_unit_issue_count",
                "zt_unit_all_dimensionless",
            ]
        )
    else:
        zt_summary = (
            zt_df.groupby("sample_key", sort=False)
            .agg(
                zt_unit_values=("unit_y", join_unique),
                zt_unit_issue_count=("zt_unit_needs_check", "sum"),
                zt_unit_all_dimensionless=("zt_unit_needs_check", lambda s: not bool(s.any())),
            )
            .reset_index()
        )

    seebeck_summary = summarize_sample_seebeck(property_fixed)

    sample_fixed = sample_fixed.merge(counts, on="sample_key", how="left")
    sample_fixed = sample_fixed.merge(seebeck_summary, on="sample_key", how="left")
    sample_fixed = sample_fixed.merge(zt_summary, on="sample_key", how="left")

    fill_zero_columns = [
        "seebeck_curve_count",
        "electrical_conductivity_curve_count",
        "electrical_resistivity_curve_count",
        "thermal_conductivity_curve_count",
        "zt_curve_count",
        "zt_unit_issue_count",
    ]
    for column in fill_zero_columns:
        if column not in sample_fixed.columns:
            sample_fixed[column] = 0
        sample_fixed[column] = sample_fixed[column].fillna(0).astype(int)

    for column, default_value in (
        ("seebeck_sign_class", "unknown"),
        ("carrier_type_guess", "unknown"),
        ("zt_unit_values", ""),
        ("zt_unit_all_dimensionless", False),
    ):
        if column not in sample_fixed.columns:
            sample_fixed[column] = default_value

    sample_fixed["seebeck_sign_class"] = sample_fixed["seebeck_sign_class"].fillna("unknown")
    sample_fixed["carrier_type_guess"] = sample_fixed["carrier_type_guess"].fillna("unknown")
    sample_fixed["zt_unit_values"] = sample_fixed["zt_unit_values"].fillna("")
    sample_fixed["zt_unit_all_dimensionless"] = sample_fixed["zt_unit_all_dimensionless"].map(
        lambda value: bool(value) if pd.notna(value) else False
    )

    sample_fixed["sample_key_is_primary_key"] = ~sample_fixed["sample_key"].duplicated()
    sample_fixed["has_seebeck_curve"] = sample_fixed["seebeck_curve_count"] > 0
    sample_fixed["has_electrical_conductivity_curve"] = (
        sample_fixed["electrical_conductivity_curve_count"] > 0
    )
    sample_fixed["has_electrical_resistivity_curve"] = (
        sample_fixed["electrical_resistivity_curve_count"] > 0
    )
    sample_fixed["has_any_electrical_transport_curve"] = (
        sample_fixed["has_electrical_conductivity_curve"]
        | sample_fixed["has_electrical_resistivity_curve"]
    )
    sample_fixed["has_thermal_conductivity_curve"] = (
        sample_fixed["thermal_conductivity_curve_count"] > 0
    )
    sample_fixed["has_zt_curve"] = sample_fixed["zt_curve_count"] > 0

    sample_fixed["sintering_method"] = "unknown"
    sample_fixed["sintering_method_checked"] = "no"
    sample_fixed["record_checked"] = "no"

    sample_fixed["is_relaxation_fit_candidate"] = (
        sample_fixed["has_seebeck_curve"] & sample_fixed["has_electrical_conductivity_curve"]
    )
    sample_fixed["is_accuracy_check_candidate"] = (
        sample_fixed["is_relaxation_fit_candidate"]
        & sample_fixed["has_thermal_conductivity_curve"]
        & sample_fixed["has_zt_curve"]
    )
    sample_fixed["is_extended_transport_candidate"] = (
        sample_fixed["has_seebeck_curve"]
        & sample_fixed["has_any_electrical_transport_curve"]
        & sample_fixed["has_thermal_conductivity_curve"]
        & sample_fixed["has_zt_curve"]
    )
    sample_fixed["candidate_priority"] = 0
    sample_fixed.loc[sample_fixed["is_relaxation_fit_candidate"], "candidate_priority"] = 1
    sample_fixed.loc[sample_fixed["is_accuracy_check_candidate"], "candidate_priority"] = 2

    sample_fixed["missing_for_relaxation_fit"] = sample_fixed.apply(
        lambda row: build_missing_components(
            row,
            ("has_seebeck_curve", "has_electrical_conductivity_curve"),
            ("seebeck", "electrical_conductivity"),
        ),
        axis=1,
    )
    sample_fixed["missing_for_accuracy_check"] = sample_fixed.apply(
        lambda row: build_missing_components(
            row,
            (
                "has_seebeck_curve",
                "has_electrical_conductivity_curve",
                "has_thermal_conductivity_curve",
                "has_zt_curve",
            ),
            ("seebeck", "electrical_conductivity", "thermal_conductivity", "zt"),
        ),
        axis=1,
    )

    sample_fixed["zt_unit_check_status"] = sample_fixed.apply(
        lambda row: "not_applicable"
        if row["zt_curve_count"] == 0
        else ("needs_check" if row["zt_unit_issue_count"] > 0 else "ok"),
        axis=1,
    )

    sample_fixed = sample_fixed.sort_values(
        ["candidate_priority", "DOI", "sample_id", "sample_key"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return sample_fixed


def build_missing_components(row: pd.Series, flag_columns: tuple[str, ...], labels: tuple[str, ...]) -> str:
    missing = [
        label for flag_column, label in zip(flag_columns, labels) if not bool(row[flag_column])
    ]
    if not missing:
        return "complete"
    return " | ".join(missing)


def build_candidate_samples(sample_fixed: pd.DataFrame) -> pd.DataFrame:
    candidate_df = sample_fixed[sample_fixed["is_relaxation_fit_candidate"]].copy()
    columns = [
        "sample_key",
        "DOI",
        "sample_id",
        "composition",
        "carrier_type_guess",
        "seebeck_sign_class",
        "seebeck_curve_count",
        "electrical_conductivity_curve_count",
        "electrical_resistivity_curve_count",
        "thermal_conductivity_curve_count",
        "zt_curve_count",
        "is_relaxation_fit_candidate",
        "is_accuracy_check_candidate",
        "is_extended_transport_candidate",
        "candidate_priority",
        "zt_unit_check_status",
        "missing_for_accuracy_check",
    ]
    candidate_df = candidate_df.loc[:, columns].reset_index(drop=True)
    return candidate_df


def enrich_property_with_sample_flags(
    property_fixed: pd.DataFrame, sample_fixed: pd.DataFrame
) -> pd.DataFrame:
    merge_columns = [
        "sample_key",
        "carrier_type_guess",
        "seebeck_sign_class",
        "is_relaxation_fit_candidate",
        "is_accuracy_check_candidate",
        "is_extended_transport_candidate",
        "candidate_priority",
        "sintering_method",
        "sintering_method_checked",
        "record_checked",
    ]
    property_fixed = property_fixed.merge(
        sample_fixed.loc[:, merge_columns], on="sample_key", how="left"
    )
    return property_fixed


def build_quality_report(
    sample_fixed: pd.DataFrame, property_fixed: pd.DataFrame, candidate_df: pd.DataFrame
) -> str:
    lines: list[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append("Starrydata2 Step3 Fixed Quality Report")
    lines.append(f"generated_at: {timestamp}")
    lines.append("")
    lines.append("[Row counts]")
    lines.append(f"sample_master_fixed_rows: {len(sample_fixed)}")
    lines.append(f"property_data_fixed_rows: {len(property_fixed)}")
    lines.append(f"candidate_samples_rows: {len(candidate_df)}")
    lines.append("")
    lines.append("[Key integrity]")
    lines.append(f"sample_key_unique_rows: {sample_fixed['sample_key'].nunique()}")
    lines.append(
        f"sample_key_duplicate_rows: {int(sample_fixed['sample_key'].duplicated().sum())}"
    )
    lines.append(f"curve_key_unique_rows: {property_fixed['curve_key'].nunique()}")
    lines.append(
        f"curve_key_duplicate_rows: {int(property_fixed['curve_key'].duplicated().sum())}"
    )
    lines.append("")
    lines.append("[Candidate counts]")
    lines.append(
        f"relaxation_fit_candidates: {int(sample_fixed['is_relaxation_fit_candidate'].sum())}"
    )
    lines.append(
        f"accuracy_check_candidates: {int(sample_fixed['is_accuracy_check_candidate'].sum())}"
    )
    lines.append(
        f"extended_transport_candidates: {int(sample_fixed['is_extended_transport_candidate'].sum())}"
    )
    lines.append("")
    lines.append("[Carrier type guess]")
    lines.append(sample_fixed["carrier_type_guess"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("[ZT unit check]")
    zt_rows = property_fixed[property_fixed["is_zt_curve"]]
    lines.append(f"zt_curve_rows: {len(zt_rows)}")
    lines.append(
        f"zt_curve_rows_needing_check: {int(zt_rows['zt_unit_needs_check'].sum())}"
    )
    lines.append(
        f"samples_with_zt_unit_issue: {int((sample_fixed['zt_unit_issue_count'] > 0).sum())}"
    )
    if len(zt_rows) > 0:
        lines.append(zt_rows["unit_y"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("[Property-family coverage]")
    lines.append(property_fixed["property_family"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("[Notes]")
    lines.append("n-type / p-type is guessed only from Seebeck sign.")
    lines.append("sintering_method is intentionally set to unknown for all samples.")
    lines.append("record_checked is intentionally set to no for all samples.")
    return "\n".join(lines) + "\n"


def fit_column_widths(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    preview = df.head(200)
    for column_index, column_name in enumerate(df.columns, start=1):
        max_length = len(str(column_name))
        if not preview.empty:
            max_length = max(max_length, int(preview[column_name].astype(str).map(len).max()))
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=column_index).column_letter
        ].width = min(max(max_length + 2, 12), 60)


def write_outputs(
    output_dir: Path,
    sample_fixed: pd.DataFrame,
    property_fixed: pd.DataFrame,
    candidate_df: pd.DataFrame,
    quality_report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = output_dir / "sample_master_fixed.csv"
    property_csv = output_dir / "property_data_fixed.csv"
    candidate_csv = output_dir / "candidate_samples.csv"
    workbook_path = output_dir / "starrydata2_step3_fixed.xlsx"
    report_path = output_dir / "quality_report.txt"

    sample_fixed.to_csv(sample_csv, index=False)
    property_fixed.to_csv(property_csv, index=False)
    candidate_df.to_csv(candidate_csv, index=False)
    report_path.write_text(quality_report, encoding="utf-8")

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        sample_fixed.to_excel(writer, sheet_name="sample_master_fixed", index=False)
        property_fixed.to_excel(writer, sheet_name="property_data_fixed", index=False)
        candidate_df.to_excel(writer, sheet_name="candidate_samples", index=False)
        fit_column_widths(writer, "sample_master_fixed", sample_fixed)
        fit_column_widths(writer, "property_data_fixed", property_fixed)
        fit_column_widths(writer, "candidate_samples", candidate_df)


def main() -> None:
    args = parse_args()
    sample_df = pd.read_csv(args.sample_master)
    property_df = pd.read_csv(args.property_data)

    validate_columns(sample_df, SAMPLE_REQUIRED_COLUMNS, "sample_master.csv")
    validate_columns(property_df, PROPERTY_REQUIRED_COLUMNS, "property_data.csv")

    property_fixed = build_property_fixed(property_df)
    sample_fixed = build_sample_fixed(sample_df, property_fixed)
    property_fixed = enrich_property_with_sample_flags(property_fixed, sample_fixed)
    candidate_df = build_candidate_samples(sample_fixed)
    quality_report = build_quality_report(sample_fixed, property_fixed, candidate_df)

    write_outputs(args.output_dir, sample_fixed, property_fixed, candidate_df, quality_report)

    print(f"saved_dir: {args.output_dir}")
    print(f"sample_master_fixed_rows: {len(sample_fixed)}")
    print(f"property_data_fixed_rows: {len(property_fixed)}")
    print(f"candidate_samples_rows: {len(candidate_df)}")


if __name__ == "__main__":
    main()
