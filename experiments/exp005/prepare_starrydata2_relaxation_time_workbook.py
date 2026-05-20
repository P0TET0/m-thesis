import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGE_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "sige"
DEFAULT_INPUT_PATH = SIGE_OUTPUT_DIR / "starrydata_curves_fixed.csv"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT / "data" / "output" / "starrydata2_prepared_for_relaxation_time.xlsx"
)

BASE_REQUIRED_COLUMNS = {
    "DOI",
    "composition",
    "sample_id",
    "figure_id",
    "prop_x",
    "prop_y",
}
OPTIONAL_COLUMNS = [
    "SID",
    "unit_x",
    "unit_y",
    "created_at",
    "updated_at",
    "project_names",
    "comments",
]
PROPERTY_FAMILY_RULES = [
    ("seebeck", ("seebeck", "thermopower")),
    ("electrical_conductivity", ("electrical conductivity",)),
    ("electrical_resistivity", ("electrical resistivity", "resistance")),
    (
        "thermal_conductivity",
        (
            "thermal conductivity",
            "lattice thermal conductivity",
            "electronic thermal conductivity",
            "electron thermal conductivity",
            "carrier thermal conductivity",
            "total thermal conductivity",
            "lattice contribution",
        ),
    ),
    ("carrier_concentration", ("carrier concentration",)),
    ("carrier_mobility", ("carrier mobility",)),
    ("hall_coefficient", ("hall coefficient",)),
    ("hall_mobility", ("hall mobility",)),
    ("power_factor", ("power factor", "calculated pf")),
    ("zt", ("figure of merit", "zt")),
]
CORE_FAMILIES = {
    "seebeck",
    "electrical_conductivity",
    "electrical_resistivity",
    "thermal_conductivity",
    "carrier_concentration",
    "carrier_mobility",
    "hall_coefficient",
    "hall_mobility",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an Excel workbook for relaxation-time fitting from downloaded "
            "Starrydata2 curve data."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="source CSV or Excel file exported from Starrydata2",
    )
    parser.add_argument(
        "--input-sheet",
        default="0",
        help="sheet index or name when the input file is Excel; ignored for CSV",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="output Excel workbook path",
    )
    parser.add_argument(
        "--project-name",
        default="ThermoelectricMaterials",
        help="filter project_names by this substring; pass an empty string to disable",
    )
    parser.add_argument(
        "--prop-x",
        default="Temperature",
        help="keep only rows whose prop_x matches this value; pass an empty string to disable",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def sheet_selector(raw_value: str) -> int | str:
    text = raw_value.strip()
    if re.fullmatch(r"\d+", text):
        return int(text)
    return text


def detect_xy_columns(columns: list[str]) -> tuple[str, str]:
    x_candidates = ["x", "x_list"]
    y_candidates = ["y", "y_list"]
    x_col = next((column for column in x_candidates if column in columns), None)
    y_col = next((column for column in y_candidates if column in columns), None)
    if x_col is None or y_col is None:
        raise KeyError("input data must contain either x/y or x_list/y_list columns")
    return x_col, y_col


def load_input_dataframe(input_path: Path, input_sheet: str) -> tuple[pd.DataFrame, str]:
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path), "csv"
    if suffix in {".xlsx", ".xls"}:
        selector = sheet_selector(input_sheet)
        excel_file = pd.ExcelFile(input_path)
        if isinstance(selector, int):
            if selector < 0 or selector >= len(excel_file.sheet_names):
                raise IndexError(
                    f"sheet index {selector} is out of range for {input_path.name}"
                )
            sheet_name = excel_file.sheet_names[selector]
        else:
            if selector not in excel_file.sheet_names:
                raise KeyError(f"sheet {selector!r} not found in {input_path.name}")
            sheet_name = selector
        return excel_file.parse(sheet_name=sheet_name), sheet_name
    raise ValueError("input-path must be a .csv, .xlsx, or .xls file")


def parse_numeric_list(raw_value: Any) -> list[float]:
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
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
    if not isinstance(parsed, (list, tuple)):
        return []

    values: list[float] = []
    for value in parsed:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            values.append(numeric)
    return values


def clean_xy_values(x_raw: Any, y_raw: Any) -> tuple[list[float], list[float]]:
    x_values = parse_numeric_list(x_raw)
    y_values = parse_numeric_list(y_raw)
    n = min(len(x_values), len(y_values))
    if n == 0:
        return [], []

    pairs = [(x_values[index], y_values[index]) for index in range(n)]
    pairs = [
        (x_value, y_value)
        for x_value, y_value in pairs
        if math.isfinite(x_value) and math.isfinite(y_value)
    ]
    if not pairs:
        return [], []

    pairs.sort(key=lambda pair: pair[0])
    x_clean = [float(x_value) for x_value, _ in pairs]
    y_clean = [float(y_value) for _, y_value in pairs]
    return x_clean, y_clean


def json_dumps(values: list[float]) -> str:
    return json.dumps(values, ensure_ascii=True, separators=(",", ":"))


def slug_key_part(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return "unknown"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def sample_key(doi: Any, sample_id: Any) -> str:
    return f"{slug_key_part(doi)}__sample_{slug_key_part(sample_id)}"


def curve_key(
    doi: Any,
    sample_id: Any,
    sid: Any,
    figure_id: Any,
    prop_y: Any,
    source_row_index: Any,
) -> str:
    return (
        f"{sample_key(doi, sample_id)}"
        f"__figure_{slug_key_part(figure_id)}"
        f"__prop_{slug_key_part(prop_y)}"
        f"__sid_{slug_key_part(sid)}"
        f"__row_{slug_key_part(source_row_index)}"
    )


def infer_property_family(prop_y: Any) -> str:
    lowered = normalize_text(prop_y).casefold()
    if lowered == "s":
        return "seebeck"
    for family, keywords in PROPERTY_FAMILY_RULES:
        if any(keyword in lowered for keyword in keywords):
            return family
    return "other"


def canonical_property_name(prop_y: Any, family: str) -> str:
    raw_name = normalize_text(prop_y)
    if family == "seebeck":
        return "Seebeck coefficient"
    if family == "electrical_conductivity":
        return "Electrical conductivity"
    if family == "electrical_resistivity":
        return "Electrical resistivity"
    if family == "thermal_conductivity":
        return "Thermal conductivity"
    if family == "carrier_concentration":
        return "Carrier concentration"
    if family == "carrier_mobility":
        return "Carrier mobility"
    if family == "hall_coefficient":
        return "Hall coefficient"
    if family == "hall_mobility":
        return "Hall mobility"
    if family == "power_factor":
        return "Power factor"
    if family == "zt":
        return "ZT"
    return raw_name


def unique_join(values: pd.Series) -> str:
    items = sorted({normalize_text(value) for value in values if normalize_text(value)})
    return " | ".join(items)


def validate_columns(df: pd.DataFrame) -> tuple[str, str]:
    missing = BASE_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"input data is missing required columns: {sorted(missing)}")
    return detect_xy_columns(df.columns.tolist())


def ensure_optional_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output


def filter_dataframe(df: pd.DataFrame, project_name: str, prop_x_filter: str) -> pd.DataFrame:
    filtered = df.copy()
    if project_name.strip():
        filtered = filtered[
            filtered["project_names"]
            .astype(str)
            .str.contains(re.escape(project_name.strip()), case=False, na=False)
        ].copy()
    if prop_x_filter.strip():
        filtered = filtered[
            filtered["prop_x"].astype(str).str.strip() == prop_x_filter.strip()
        ].copy()
    return filtered


def build_property_dataframe(
    df: pd.DataFrame, x_col: str, y_col: str, source_file: Path, source_sheet: str
) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, Any]] = []
    skipped_rows = 0

    for row in df.reset_index(names="source_row_index").itertuples(index=False):
        x_values, y_values = clean_xy_values(getattr(row, x_col), getattr(row, y_col))
        if not x_values:
            skipped_rows += 1
            continue

        doi = getattr(row, "DOI")
        source_sample_id = getattr(row, "sample_id")
        source_sid = getattr(row, "SID", "")
        source_figure_id = getattr(row, "figure_id")
        raw_property_name = getattr(row, "prop_y")
        family = infer_property_family(raw_property_name)
        canonical_name = canonical_property_name(raw_property_name, family)
        source_row_index = getattr(row, "source_row_index")

        records.append(
            {
                "curve_key": curve_key(
                    doi,
                    source_sample_id,
                    source_sid,
                    source_figure_id,
                    raw_property_name,
                    source_row_index,
                ),
                "sample_key": sample_key(doi, source_sample_id),
                "source_row_index": source_row_index,
                "SID": normalize_text(source_sid),
                "DOI": normalize_text(doi),
                "sample_id": normalize_text(source_sample_id),
                "composition": normalize_text(getattr(row, "composition")),
                "figure_id": normalize_text(source_figure_id),
                "prop_x": normalize_text(getattr(row, "prop_x")),
                "prop_y_raw": normalize_text(raw_property_name),
                "prop_y_canonical": canonical_name,
                "property_family": family,
                "is_core_property": family in CORE_FAMILIES,
                "unit_x": normalize_text(getattr(row, "unit_x", "")),
                "unit_y": normalize_text(getattr(row, "unit_y", "")),
                "n_points": len(x_values),
                "x_min": min(x_values),
                "x_max": max(x_values),
                "y_min": min(y_values),
                "y_max": max(y_values),
                "x_values_json": json_dumps(x_values),
                "y_values_json": json_dumps(y_values),
                "created_at": normalize_text(getattr(row, "created_at", "")),
                "updated_at": normalize_text(getattr(row, "updated_at", "")),
                "project_names": normalize_text(getattr(row, "project_names", "")),
                "comments": normalize_text(getattr(row, "comments", "")),
                "source_file": str(source_file),
                "source_sheet": source_sheet,
            }
        )

    property_df = pd.DataFrame.from_records(records)
    if property_df.empty:
        return property_df, skipped_rows

    property_df = property_df.sort_values(
        ["DOI", "sample_id", "prop_y_canonical", "figure_id", "SID", "curve_key"],
        kind="stable",
    ).reset_index(drop=True)
    return property_df, skipped_rows


def build_sample_master(property_df: pd.DataFrame) -> pd.DataFrame:
    grouped = property_df.groupby("sample_key", sort=True, dropna=False)
    sample_df = grouped.agg(
        DOI=("DOI", "first"),
        sample_id=("sample_id", "first"),
        composition=("composition", "first"),
        curve_count=("curve_key", "count"),
        total_point_count=("n_points", "sum"),
        property_count=("prop_y_canonical", lambda series: series.nunique()),
        figure_ids=("figure_id", unique_join),
        property_names=("prop_y_raw", unique_join),
        canonical_property_names=("prop_y_canonical", unique_join),
        property_families=("property_family", unique_join),
        core_property_curve_count=("is_core_property", "sum"),
        temperature_min=("x_min", "min"),
        temperature_max=("x_max", "max"),
        created_at_values=("created_at", unique_join),
        updated_at_values=("updated_at", unique_join),
        project_names=("project_names", unique_join),
        comments=("comments", unique_join),
        source_file=("source_file", "first"),
        source_sheet=("source_sheet", "first"),
    ).reset_index()

    sample_df["has_core_property"] = sample_df["core_property_curve_count"] > 0
    sample_df = sample_df.sort_values(
        ["DOI", "sample_id", "sample_key"], kind="stable"
    ).reset_index(drop=True)
    return sample_df


def fit_column_widths(writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame) -> None:
    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    preview = df.head(200)
    for column_index, column_name in enumerate(df.columns, start=1):
        max_length = len(str(column_name))
        if not preview.empty:
            preview_lengths = preview[column_name].astype(str).map(len)
            max_length = max(max_length, int(preview_lengths.max()))
        worksheet.column_dimensions[
            worksheet.cell(row=1, column=column_index).column_letter
        ].width = min(max(max_length + 2, 12), 60)


def write_workbook(
    sample_df: pd.DataFrame, property_df: pd.DataFrame, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sample_df.to_excel(writer, sheet_name="sample_master", index=False)
        property_df.to_excel(writer, sheet_name="property_data", index=False)
        fit_column_widths(writer, "sample_master", sample_df)
        fit_column_widths(writer, "property_data", property_df)


def main() -> None:
    args = parse_args()

    try:
        df, source_sheet = load_input_dataframe(args.input_path, args.input_sheet)
        x_col, y_col = validate_columns(df)
        df = ensure_optional_columns(df)
        filtered_df = filter_dataframe(df, args.project_name, args.prop_x)
        property_df, skipped_rows = build_property_dataframe(
            filtered_df, x_col, y_col, args.input_path, source_sheet
        )
        if property_df.empty:
            raise ValueError("no valid curves remained after filtering and cleaning")
        sample_df = build_sample_master(property_df)
        write_workbook(sample_df, property_df, args.output_path)
    except Exception as exc:
        raise SystemExit(f"failed to prepare workbook: {exc}") from exc

    print(f"saved: {args.output_path}")
    print(f"input_rows: {len(df)}")
    print(f"filtered_rows: {len(filtered_df)}")
    print(f"property_rows_written: {len(property_df)}")
    print(f"sample_rows_written: {len(sample_df)}")
    print(f"skipped_rows_without_valid_xy: {skipped_rows}")
    print(f"source_sheet: {source_sheet}")


if __name__ == "__main__":
    main()
