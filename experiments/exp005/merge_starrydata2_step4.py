import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step3_fixed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step4_merged"

DEFAULT_SAMPLE_MASTER = DEFAULT_INPUT_DIR / "sample_master_fixed.csv"
DEFAULT_PROPERTY_DATA = DEFAULT_INPUT_DIR / "property_data_fixed.csv"
DEFAULT_CANDIDATE_SAMPLES = DEFAULT_INPUT_DIR / "candidate_samples.csv"

EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000

TEXT_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "sample_id",
    "curve_key",
    "curve_id",
    "composition",
]
TARGET_PROPERTIES = {
    "Electrical conductivity",
    "Electrical resistivity",
    "Seebeck coefficient",
    "Thermal conductivity",
    "ZT",
}
DIMENSIONLESS_ZT_UNITS = {"", "-", "1", "dimensionless"}

SAMPLE_REQUIRED_COLUMNS = {
    "sample_key",
    "SID",
    "sample_id",
    "DOI",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "sintering_method",
    "is_learning_candidate",
}
PROPERTY_REQUIRED_COLUMNS = {
    "sample_key",
    "curve_id",
    "SID",
    "sample_id",
    "DOI",
    "property",
    "property_family",
    "unit",
    "unit_x",
    "unit_y",
    "n_points",
    "x_values_json",
    "y_values_json",
}
CANDIDATE_REQUIRED_COLUMNS = {"sample_key"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Starrydata2 step-3 sample metadata and property curves."
    )
    parser.add_argument("--sample-master", type=Path, default=DEFAULT_SAMPLE_MASTER)
    parser.add_argument("--property-data", type=Path, default=DEFAULT_PROPERTY_DATA)
    parser.add_argument("--candidate-samples", type=Path, default=DEFAULT_CANDIDATE_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def normalize_key(value: Any) -> str:
    return normalize_text(value)


def normalize_doi(value: Any) -> str:
    return normalize_text(value).casefold()


def normalize_composition(value: Any) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def parse_bool(value: Any) -> bool:
    text = normalize_text(value).casefold()
    return text in {"true", "1", "yes", "y"}


def load_numeric_list(raw_value: Any) -> list[float]:
    text = normalize_text(raw_value)
    if not text:
        return []
    parsed: Any
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


def point_count(row: pd.Series) -> int:
    numeric = pd.to_numeric(row.get("n_points", ""), errors="coerce")
    if pd.notna(numeric):
        return int(numeric)
    return len(load_numeric_list(row.get("x_values_json", "")))


def unique_join(values: pd.Series) -> str:
    items = sorted({normalize_text(value) for value in values if normalize_text(value)})
    return " | ".join(items)


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def validate_required_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


def add_property_compatibility_columns(property_df: pd.DataFrame) -> pd.DataFrame:
    df = property_df.copy()
    if "curve_id" not in df.columns:
        df["curve_id"] = df.get("curve_key", "")
    if "property" not in df.columns:
        df["property"] = df.get("prop_y_canonical", df.get("prop_y_raw", ""))
    if "unit" not in df.columns:
        df["unit"] = df.get("unit_y", "")
    if "prop_y" not in df.columns:
        df["prop_y"] = df.get("prop_y_canonical", df.get("prop_y_raw", ""))
    if "unit_check_note" not in df.columns:
        df["unit_check_note"] = df.get("zt_unit_check_status", "")
    if "note" not in df.columns:
        df["note"] = df.get("comments", "")

    for column in PROPERTY_REQUIRED_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_text)
    return df


def add_sample_compatibility_columns(
    sample_df: pd.DataFrame, property_df: pd.DataFrame
) -> pd.DataFrame:
    df = sample_df.copy()

    if "SID" not in df.columns:
        sid_by_sample = (
            property_df.groupby("sample_key", sort=False)["SID"]
            .agg(unique_join)
            .reset_index()
        )
        df = df.merge(sid_by_sample, on="sample_key", how="left")

    defaults = {
        "paper_title": "",
        "year": "",
        "sample_name": "",
        "composition_detail": "",
        "material_system": "unknown",
        "additive": "",
        "note": "",
        "learning_candidate_reason": "",
    }
    for column, value in defaults.items():
        if column not in df.columns:
            df[column] = value

    if "sample_name" in df.columns:
        df["sample_name"] = df["sample_name"].where(
            df["sample_name"].map(normalize_text).ne(""), df.get("composition", "")
        )

    if "n_or_p" not in df.columns:
        df["n_or_p"] = df.get("carrier_type_guess", "").map(n_or_p_from_carrier_guess)
    if "n_or_p_basis" not in df.columns:
        df["n_or_p_basis"] = df.get("seebeck_sign_class", "").map(n_or_p_basis)

    if "sintering_checked" not in df.columns:
        if "record_checked" in df.columns:
            df["sintering_checked"] = df["record_checked"]
        else:
            df["sintering_checked"] = "no"
    if "record_checked" not in df.columns:
        df["record_checked"] = "no"

    if "is_learning_candidate" not in df.columns:
        df["is_learning_candidate"] = df.get("is_relaxation_fit_candidate", "False")
    if "learning_candidate_reason" not in df.columns:
        df["learning_candidate_reason"] = df.get("missing_for_relaxation_fit", "")

    df = add_sample_point_counts(df, property_df)

    for column in SAMPLE_REQUIRED_COLUMNS | {"sintering_checked", "record_checked"}:
        if column not in df.columns:
            df[column] = ""

    for column in TEXT_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_text)
    return df


def n_or_p_from_carrier_guess(value: Any) -> str:
    text = normalize_text(value).casefold()
    if text in {"n-type", "n"}:
        return "n"
    if text in {"p-type", "p"}:
        return "p"
    if text == "mixed":
        return "mixed"
    return "unknown"


def n_or_p_basis(value: Any) -> str:
    text = normalize_text(value)
    if text:
        return f"Seebeck sign: {text}"
    return "unknown"


def add_sample_point_counts(sample_df: pd.DataFrame, property_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    prop = property_df.copy()
    prop["n_points_counted"] = prop.apply(point_count, axis=1)

    families = {
        "has_sigma_or_rho": ["electrical_conductivity", "electrical_resistivity"],
        "has_seebeck": ["seebeck"],
        "has_kappa_or_zt": ["thermal_conductivity", "zt"],
    }
    point_columns = {
        "sigma_or_rho_point_count": ["electrical_conductivity", "electrical_resistivity"],
        "seebeck_point_count": ["seebeck"],
        "kappa_or_zt_point_count": ["thermal_conductivity", "zt"],
    }

    for output_column, family_list in families.items():
        if output_column in df.columns:
            continue
        keys = set(prop.loc[prop["property_family"].isin(family_list), "sample_key"])
        df[output_column] = df["sample_key"].isin(keys)

    for output_column, family_list in point_columns.items():
        if output_column in df.columns:
            continue
        counts = (
            prop[prop["property_family"].isin(family_list)]
            .groupby("sample_key", sort=False)["n_points_counted"]
            .sum()
            .reset_index(name=output_column)
        )
        df = df.merge(counts, on="sample_key", how="left")
        df[output_column] = pd.to_numeric(df[output_column], errors="coerce").fillna(0).astype(int)

    return df


def add_candidate_compatibility_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    df = candidate_df.copy()
    if "is_learning_candidate" not in df.columns:
        df["is_learning_candidate"] = "True"
    if "learning_candidate_reason" not in df.columns:
        df["learning_candidate_reason"] = "listed_in_candidate_samples"
    df["sample_key"] = df["sample_key"].map(normalize_text)
    return df


def prepare_sample_master(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    duplicate_count = int(sample_df["sample_key"].duplicated().sum())
    if duplicate_count:
        sample_df = sample_df.drop_duplicates(subset=["sample_key"], keep="first").copy()
    sample_df["__sample_joined"] = True
    return sample_df, duplicate_count


def rename_for_merge(property_df: pd.DataFrame, sample_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    property_renames = {
        "SID": "SID_curve",
        "DOI": "DOI_curve",
        "sample_id": "sample_id_curve",
        "composition": "composition_curve",
        "note": "curve_note",
        "sintering_method": "sintering_method_curve",
        "sintering_method_checked": "sintering_checked_curve",
        "record_checked": "record_checked_curve",
        "carrier_type_guess": "carrier_type_guess_curve",
        "seebeck_sign_class": "seebeck_sign_class_curve",
    }
    sample_renames = {
        "SID": "SID_sample",
        "DOI": "DOI_sample",
        "sample_id": "sample_id_sample",
        "composition": "composition_sample",
        "note": "sample_note",
    }
    return property_df.rename(columns=property_renames), sample_df.rename(columns=sample_renames)


def values_match(left: Any, right: Any, normalizer=normalize_key) -> bool:
    left_text = normalizer(left)
    right_text = normalizer(right)
    if not left_text or not right_text:
        return False
    return left_text == right_text


def coalesce_columns(df: pd.DataFrame, output: str, preferred: str, fallback: str) -> None:
    df[output] = df[preferred].where(df[preferred].map(normalize_text).ne(""), df[fallback])


def build_merged_curves(
    property_df: pd.DataFrame,
    sample_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    property_for_merge, sample_for_merge = rename_for_merge(property_df, sample_df)
    merged = property_for_merge.merge(sample_for_merge, on="sample_key", how="left")

    coalesce_columns(merged, "SID", "SID_sample", "SID_curve")
    coalesce_columns(merged, "DOI", "DOI_sample", "DOI_curve")
    coalesce_columns(merged, "sample_id", "sample_id_sample", "sample_id_curve")
    coalesce_columns(merged, "composition", "composition_sample", "composition_curve")

    merged["SID_match"] = [
        values_match(left, right)
        for left, right in zip(merged["SID_curve"], merged["SID_sample"])
    ]
    merged["DOI_match"] = [
        values_match(left, right, normalize_doi)
        for left, right in zip(merged["DOI_curve"], merged["DOI_sample"])
    ]
    merged["sample_id_match"] = [
        values_match(left, right)
        for left, right in zip(merged["sample_id_curve"], merged["sample_id_sample"])
    ]
    merged["composition_match"] = [
        values_match(left, right, normalize_composition)
        for left, right in zip(merged["composition_curve"], merged["composition_sample"])
    ]

    joined = merged["__sample_joined"].map(parse_bool)
    merged["merge_status"] = "matched"
    merged.loc[~joined, "merge_status"] = "unmatched_sample_master"
    mismatch = joined & (~merged["SID_match"] | ~merged["sample_id_match"])
    merged.loc[mismatch, "merge_status"] = "key_mismatch"

    candidate_keys = set(candidate_df["sample_key"].map(normalize_text))
    merged["is_candidate_sample"] = merged["sample_key"].isin(candidate_keys)
    merged["is_target_property_for_relaxation"] = merged["property"].isin(TARGET_PROPERTIES)

    ordered_columns = build_output_columns(merged)
    return merged.loc[:, ordered_columns]


def build_output_columns(df: pd.DataFrame) -> list[str]:
    primary_columns = [
        "curve_id",
        "curve_key",
        "sample_key",
        "SID",
        "DOI",
        "sample_id",
        "composition",
        "SID_curve",
        "SID_sample",
        "SID_match",
        "DOI_curve",
        "DOI_sample",
        "DOI_match",
        "sample_id_curve",
        "sample_id_sample",
        "sample_id_match",
        "paper_title",
        "year",
        "sample_name",
        "composition_curve",
        "composition_sample",
        "composition_match",
        "composition_detail",
        "material_system",
        "n_or_p",
        "n_or_p_basis",
        "sintering_method",
        "sintering_checked",
        "record_checked",
        "additive",
        "sample_note",
        "figure_id",
        "prop_x",
        "prop_y_raw",
        "prop_y_canonical",
        "property_family",
        "property",
        "unit",
        "prop_y",
        "unit_x",
        "unit_y",
        "n_points",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "x_values_json",
        "y_values_json",
        "unit_check_note",
        "curve_note",
        "has_sigma_or_rho",
        "has_seebeck",
        "has_kappa_or_zt",
        "sigma_or_rho_point_count",
        "seebeck_point_count",
        "kappa_or_zt_point_count",
        "is_learning_candidate",
        "learning_candidate_reason",
        "is_candidate_sample",
        "is_target_property_for_relaxation",
        "merge_status",
    ]
    remaining = [column for column in df.columns if column not in primary_columns and not column.startswith("__")]
    return [column for column in primary_columns if column in df.columns] + remaining


def zt_unit_needs_check(row: pd.Series) -> bool:
    if normalize_text(row.get("property")) != "ZT":
        return False
    if "zt_unit_needs_check" in row.index and normalize_text(row.get("zt_unit_needs_check")):
        return parse_bool(row.get("zt_unit_needs_check"))
    return normalize_text(row.get("unit_y")).casefold() not in DIMENSIONLESS_ZT_UNITS


def summarize_property_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["property", "curve_count", "point_count"])
    work = df.copy()
    work["n_points_counted"] = work.apply(point_count, axis=1)
    return (
        work.groupby("property", sort=True)
        .agg(curve_count=("curve_id", "count"), point_count=("n_points_counted", "sum"))
        .reset_index()
    )


def build_report(
    sample_input_rows: int,
    property_input_rows: int,
    candidate_input_rows: int,
    sample_duplicate_count: int,
    property_df: pd.DataFrame,
    merged_df: pd.DataFrame,
    core_df: pd.DataFrame,
    candidate_curves_df: pd.DataFrame,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    unmatched = merged_df["merge_status"].eq("unmatched_sample_master")
    zt_issue_mask = merged_df.apply(zt_unit_needs_check, axis=1)
    candidate_samples = merged_df[merged_df["is_candidate_sample"]].drop_duplicates("sample_key")

    lines: list[tuple[str, str]] = [
        ("input_sample_master_fixed_rows", str(sample_input_rows)),
        ("input_property_data_fixed_rows", str(property_input_rows)),
        ("input_candidate_samples_rows", str(candidate_input_rows)),
        ("property_curves_merged_rows", str(len(merged_df))),
        ("property_core_curves_merged_rows", str(len(core_df))),
        ("candidate_property_curves_rows", str(len(candidate_curves_df))),
        ("sample_master_sample_key_duplicate_rows", str(sample_duplicate_count)),
        ("property_data_fixed_sample_key_unique_count", str(property_df["sample_key"].nunique())),
        ("unmatched_sample_master_rows", str(int(unmatched.sum()))),
        ("unmatched_sample_master_sample_key_count", str(merged_df.loc[unmatched, "sample_key"].nunique())),
        ("SID_match_false_rows", str(int((~merged_df["SID_match"]).sum()))),
        ("DOI_match_false_rows", str(int((~merged_df["DOI_match"]).sum()))),
        ("sample_id_match_false_rows", str(int((~merged_df["sample_id_match"]).sum()))),
        ("composition_match_false_rows", str(int((~merged_df["composition_match"]).sum()))),
    ]

    for status, count in merged_df["merge_status"].value_counts(dropna=False).sort_index().items():
        lines.append((f"merge_status_{status}", str(int(count))))

    lines.extend(
        [
            ("candidate_sample_curve_count", str(int(merged_df["is_candidate_sample"].sum()))),
            (
                "target_property_curve_count",
                str(int(merged_df["is_target_property_for_relaxation"].sum())),
            ),
        ]
    )

    for label, frame in (
        ("property_core_curves_merged", core_df),
        ("candidate_property_curves", candidate_curves_df),
    ):
        summary = summarize_property_counts(frame)
        for row in summary.itertuples(index=False):
            lines.append((f"{label}_{row.property}_curve_count", str(int(row.curve_count))))
            lines.append((f"{label}_{row.property}_point_count", str(int(row.point_count))))

    lines.extend(
        [
            ("zt_unit_abnormal_curve_count", str(int(zt_issue_mask.sum()))),
            ("zt_unit_abnormal_sample_count", str(merged_df.loc[zt_issue_mask, "sample_key"].nunique())),
            (
                "n_type_candidate_sample_count",
                str(int(candidate_samples["n_or_p"].eq("n").sum())),
            ),
            (
                "p_type_candidate_sample_count",
                str(int(candidate_samples["n_or_p"].eq("p").sum())),
            ),
            (
                "mixed_candidate_sample_count",
                str(int(candidate_samples["n_or_p"].eq("mixed").sum())),
            ),
            (
                "unknown_candidate_sample_count",
                str(int(candidate_samples["n_or_p"].eq("unknown").sum())),
            ),
        ]
    )

    for note in excel_notes:
        lines.append(("excel_note", note))

    report_df = pd.DataFrame(lines, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in lines) + "\n"
    return report_text, report_df


def write_csv_outputs(
    output_dir: Path,
    merged_df: pd.DataFrame,
    core_df: pd.DataFrame,
    candidate_curves_df: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_dir / "property_curves_merged.csv", index=False)
    core_df.to_csv(output_dir / "property_core_curves_merged.csv", index=False)
    candidate_curves_df.to_csv(output_dir / "candidate_property_curves.csv", index=False)
    (output_dir / "step4_merge_report.txt").write_text(report_text, encoding="utf-8")


def dataframe_for_excel(df: pd.DataFrame, sheet_name: str, notes: list[str]) -> pd.DataFrame:
    if len(df) <= EXCEL_MAX_ROWS - 1:
        return df
    notes.append(
        f"{sheet_name} exceeded Excel row limit; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook"
    )
    return df.head(EXCEL_PREVIEW_ROWS)


def fit_worksheet(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
    preview = df.head(200)
    for index, column_name in enumerate(df.columns, start=1):
        max_length = len(str(column_name))
        if not preview.empty:
            max_length = max(max_length, int(preview[column_name].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(
            max(max_length + 2, 12), 60
        )


def write_excel_output(
    output_dir: Path,
    merged_df: pd.DataFrame,
    core_df: pd.DataFrame,
    candidate_curves_df: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    workbook_path = output_dir / "starrydata2_step4_merged.xlsx"
    sheets = {
        "property_curves_merged": dataframe_for_excel(
            merged_df, "property_curves_merged", excel_notes
        ),
        "property_core_curves_merged": dataframe_for_excel(
            core_df, "property_core_curves_merged", excel_notes
        ),
        "candidate_property_curves": dataframe_for_excel(
            candidate_curves_df, "candidate_property_curves", excel_notes
        ),
        "merge_report": report_df,
    }
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def assert_acceptance(
    property_input_rows: int,
    merged_df: pd.DataFrame,
    core_df: pd.DataFrame,
    candidate_curves_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> None:
    if len(merged_df) < property_input_rows:
        raise ValueError("property_curves_merged row count decreased")
    if len(merged_df) > property_input_rows:
        raise ValueError("property_curves_merged row count increased after de-duplication")

    required_merged = {
        "sample_key",
        "curve_id",
        "paper_title",
        "composition",
        "material_system",
        "n_or_p",
        "sintering_method",
        "is_candidate_sample",
        "is_target_property_for_relaxation",
        "merge_status",
    }
    missing = required_merged - set(merged_df.columns)
    if missing:
        raise KeyError(f"property_curves_merged missing columns: {sorted(missing)}")
    if "sintering_checked" not in merged_df.columns and "record_checked" not in merged_df.columns:
        raise KeyError("property_curves_merged missing sintering_checked or record_checked")

    core_properties = set(core_df["property"].drop_duplicates())
    if core_properties - TARGET_PROPERTIES:
        raise ValueError(f"property_core_curves_merged has non-target properties: {sorted(core_properties - TARGET_PROPERTIES)}")

    candidate_keys = set(candidate_df["sample_key"])
    if set(candidate_curves_df["sample_key"]) - candidate_keys:
        raise ValueError("candidate_property_curves contains sample_key not in candidate_samples")
    candidate_properties = set(candidate_curves_df["property"].drop_duplicates())
    if candidate_properties - TARGET_PROPERTIES:
        raise ValueError(f"candidate_property_curves has non-target properties: {sorted(candidate_properties - TARGET_PROPERTIES)}")
    for column in ("x_values_json", "y_values_json"):
        if column not in candidate_curves_df.columns:
            raise KeyError(f"candidate_property_curves missing {column}")


def main() -> None:
    args = parse_args()

    sample_raw = read_csv_text(args.sample_master)
    property_raw = read_csv_text(args.property_data)
    candidate_raw = read_csv_text(args.candidate_samples)

    property_df = add_property_compatibility_columns(property_raw)
    sample_df = add_sample_compatibility_columns(sample_raw, property_df)
    candidate_df = add_candidate_compatibility_columns(candidate_raw)

    validate_required_columns(sample_df, SAMPLE_REQUIRED_COLUMNS, "sample_master_fixed.csv")
    if "sintering_checked" not in sample_df.columns and "record_checked" not in sample_df.columns:
        raise KeyError("sample_master_fixed.csv missing sintering_checked or record_checked")
    validate_required_columns(property_df, PROPERTY_REQUIRED_COLUMNS, "property_data_fixed.csv")
    validate_required_columns(candidate_df, CANDIDATE_REQUIRED_COLUMNS, "candidate_samples.csv")

    sample_df, sample_duplicate_count = prepare_sample_master(sample_df)
    merged_df = build_merged_curves(property_df, sample_df, candidate_df)
    core_df = merged_df[merged_df["is_target_property_for_relaxation"]].copy()
    candidate_curves_df = merged_df[
        merged_df["is_candidate_sample"] & merged_df["is_target_property_for_relaxation"]
    ].copy()

    assert_acceptance(len(property_df), merged_df, core_df, candidate_curves_df, candidate_df)

    excel_notes: list[str] = []
    report_text, report_df = build_report(
        sample_input_rows=len(sample_raw),
        property_input_rows=len(property_raw),
        candidate_input_rows=len(candidate_raw),
        sample_duplicate_count=sample_duplicate_count,
        property_df=property_df,
        merged_df=merged_df,
        core_df=core_df,
        candidate_curves_df=candidate_curves_df,
        excel_notes=excel_notes,
    )

    write_csv_outputs(args.output_dir, merged_df, core_df, candidate_curves_df, report_text)
    write_excel_output(
        args.output_dir, merged_df, core_df, candidate_curves_df, report_df, excel_notes
    )
    if excel_notes:
        report_text, report_df = build_report(
            sample_input_rows=len(sample_raw),
            property_input_rows=len(property_raw),
            candidate_input_rows=len(candidate_raw),
            sample_duplicate_count=sample_duplicate_count,
            property_df=property_df,
            merged_df=merged_df,
            core_df=core_df,
            candidate_curves_df=candidate_curves_df,
            excel_notes=excel_notes,
        )
        (args.output_dir / "step4_merge_report.txt").write_text(report_text, encoding="utf-8")

    unmatched_rows = int(merged_df["merge_status"].eq("unmatched_sample_master").sum())
    sid_mismatches = int((~merged_df["SID_match"]).sum())
    sample_id_mismatches = int((~merged_df["sample_id_match"]).sum())

    print("Done.")
    print("Created:")
    print("- property_curves_merged.csv")
    print("- property_core_curves_merged.csv")
    print("- candidate_property_curves.csv")
    print("- step4_merge_report.txt")
    print("- starrydata2_step4_merged.xlsx")
    print("")
    print("Summary:")
    print(f"property_data_fixed rows: {len(property_df)}")
    print(f"property_curves_merged rows: {len(merged_df)}")
    print(f"property_core_curves_merged rows: {len(core_df)}")
    print(f"candidate_property_curves rows: {len(candidate_curves_df)}")
    print(f"unmatched sample_master rows: {unmatched_rows}")
    print(f"SID mismatches: {sid_mismatches}")
    print(f"sample_id mismatches: {sample_id_mismatches}")
    print(f"candidate samples: {candidate_df['sample_key'].nunique()}")
    print(f"candidate curves: {int(merged_df['is_candidate_sample'].sum())}")
    print(f"target property curves: {int(merged_df['is_target_property_for_relaxation'].sum())}")


if __name__ == "__main__":
    main()
