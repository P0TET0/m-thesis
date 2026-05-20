import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEP4_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step4_merged"
STEP3_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step3_fixed"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step5_core_properties"

TARGET_PROPERTIES = {
    "Electrical conductivity",
    "Electrical resistivity",
    "Seebeck coefficient",
    "Thermal conductivity",
    "ZT",
}
PROPERTY_ORDER = [
    "Electrical conductivity",
    "Electrical resistivity",
    "Seebeck coefficient",
    "Thermal conductivity",
    "ZT",
]
PROPERTY_SOURCE_COLUMNS = [
    "property_family",
    "property",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
]
MIN_REQUIRED_COLUMNS = {"sample_key", "x_values_json", "y_values_json"}
OPTIONAL_KEEP_COLUMNS = [
    "curve_id",
    "curve_key",
    "sample_key",
    "SID",
    "SID_curve",
    "SID_sample",
    "DOI",
    "DOI_curve",
    "DOI_sample",
    "sample_id",
    "sample_id_curve",
    "sample_id_sample",
    "paper_title",
    "year",
    "composition",
    "composition_curve",
    "composition_sample",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "figure_id",
    "prop_x",
    "property_family",
    "property",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
    "unit",
    "unit_x",
    "unit_y",
    "n_points",
    "n_points_step5",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "x_values_json",
    "y_values_json",
    "unit_check_note",
    "unit_check_note_step5",
    "xy_length_check",
    "property_step5",
    "property_step5_source",
    "is_target_property_step5",
    "property_filter_reason",
    "merge_status",
    "is_candidate_sample",
    "is_learning_candidate",
    "learning_candidate_reason",
]
SAMPLE_AVAILABILITY_COLUMNS = [
    "sample_key",
    "SID",
    "DOI",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "has_electrical_conductivity",
    "has_electrical_resistivity",
    "has_sigma_or_rho",
    "has_seebeck",
    "has_thermal_conductivity",
    "has_zt",
    "has_kappa_or_zt",
    "electrical_conductivity_curve_count",
    "electrical_resistivity_curve_count",
    "seebeck_curve_count",
    "thermal_conductivity_curve_count",
    "zt_curve_count",
    "electrical_conductivity_point_count",
    "electrical_resistivity_point_count",
    "sigma_or_rho_point_count",
    "seebeck_point_count",
    "thermal_conductivity_point_count",
    "zt_point_count",
    "kappa_or_zt_point_count",
    "is_learning_candidate_step5",
    "learning_candidate_reason_step5",
]
DIMENSIONLESS_ZT_UNITS = {"", "-", "1", "dimensionless", "nan", "none"}
EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter Starrydata2 merged curves to the five target properties."
    )
    parser.add_argument("--input", type=Path, default=None, help="input curve CSV")
    parser.add_argument(
        "--candidate_input",
        "--candidate-input",
        dest="candidate_input",
        type=Path,
        default=None,
        help="optional candidate curve CSV",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output directory",
    )
    return parser.parse_args()


def resolve_input_path(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    candidates = [
        STEP4_DIR / "property_curves_merged.csv",
        STEP3_DIR / "property_data_fixed.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("property_curves_merged.csv or property_data_fixed.csv not found")


def resolve_candidate_path(explicit_path: Path | None, input_path: Path) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    candidates = [
        input_path.parent / "candidate_property_curves.csv",
        STEP4_DIR / "candidate_property_curves.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def read_csv_text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none"}:
        return ""
    return text


def normalize_bool(value: Any) -> bool:
    return normalize_text(value).casefold() in {"true", "1", "yes", "y"}


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).casefold())


def parse_numeric_list(raw_value: Any) -> tuple[list[float], bool]:
    text = normalize_text(raw_value)
    if not text:
        return [], False
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return [], True
    if not isinstance(parsed, (list, tuple)):
        return [], True
    values: list[float] = []
    for value in parsed:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            return [], True
    return values, False


def validate_input(df: pd.DataFrame, label: str) -> None:
    missing = sorted(MIN_REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")
    if not any(column in df.columns for column in PROPERTY_SOURCE_COLUMNS):
        raise KeyError(
            f"{label} needs at least one property source column: {PROPERTY_SOURCE_COLUMNS}"
        )


def source_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        return ""
    return normalize_text(row[column])


def property_texts(row: pd.Series) -> dict[str, str]:
    return {column: source_value(row, column) for column in PROPERTY_SOURCE_COLUMNS}


def has_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def classify_property(row: pd.Series) -> tuple[str, str, bool, str]:
    texts = property_texts(row)
    combined = " | ".join(compact_text(value) for value in texts.values() if value)

    if not combined:
        return "", "", False, "no property name"

    if has_any(
        combined,
        [
            "power factor",
            "carrier concentration",
            "hall coefficient",
            "mobility",
            "specific heat",
            "density",
            "thermal diffusivity",
            "lorenz number",
            "/ relaxation time",
            "relaxation time",
        ],
    ):
        return "", exclusion_source(texts), False, "excluded non-target property"

    if has_any(
        combined,
        [
            "lattice thermal conductivity",
            "electronic thermal conductivity",
            "electron thermal conductivity",
            "carrier thermal conductivity",
            "lattice contribution",
            "latice contribution",
            "lattice+bipolar thermal conductivity",
        ],
    ):
        return "", exclusion_source(texts), False, "excluded partial thermal conductivity"

    for column in PROPERTY_SOURCE_COLUMNS:
        text = compact_text(texts[column])
        if not text:
            continue
        classified = classify_property_text(text)
        if classified:
            return classified, column, True, "target property"

    return "", exclusion_source(texts), False, "excluded non-target property"


def exclusion_source(texts: dict[str, str]) -> str:
    for column in PROPERTY_SOURCE_COLUMNS:
        if texts.get(column):
            return column
    return ""


def classify_property_text(text: str) -> str:
    text = text.casefold()
    if "power factor" in text:
        return ""

    if text == "zt" or "dimensionless figure of merit" in text or "figure of merit" in text:
        return "ZT"

    if "seebeck" in text or "thermopower" in text or "thermoelectric power" in text:
        return "Seebeck coefficient"

    if (
        "electrical_resistivity" in text
        or "electrical resistivity" in text
        or "electric resistivity" in text
        or re.search(r"(^|[^a-z])rho([^a-z]|$)", text)
        or "ρ" in text
    ):
        return "Electrical resistivity"
    if "resistivity" in text and "thermal" not in text:
        return "Electrical resistivity"

    if (
        "thermal_conductivity" in text
        or "thermal conductivity" in text
        or "total thermal conductivity" in text
    ):
        return "Thermal conductivity"

    if (
        "electrical_conductivity" in text
        or "electrical conductivity" in text
        or "electric conductivity" in text
        or re.search(r"(^|[^a-z])sigma([^a-z]|$)", text)
        or "σ" in text
    ):
        return "Electrical conductivity"
    if "conductivity" in text and "thermal" not in text:
        return "Electrical conductivity"

    return ""


def add_step5_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    classifications = output.apply(classify_property, axis=1, result_type="expand")
    classifications.columns = [
        "property_step5",
        "property_step5_source",
        "is_target_property_step5",
        "property_filter_reason",
    ]
    output = pd.concat([output, classifications], axis=1)
    output["is_target_property_step5"] = output["is_target_property_step5"].astype(bool)

    xy_checks = output.apply(check_xy_lengths, axis=1, result_type="expand")
    xy_checks.columns = ["xy_length_check", "n_points_step5"]
    output = pd.concat([output, xy_checks], axis=1)
    output["unit_check_note_step5"] = output.apply(step5_unit_note, axis=1)
    return output


def check_xy_lengths(row: pd.Series) -> tuple[str, int]:
    x_values, x_failed = parse_numeric_list(row.get("x_values_json", ""))
    y_values, y_failed = parse_numeric_list(row.get("y_values_json", ""))
    if x_failed or y_failed:
        numeric = pd.to_numeric(row.get("n_points", ""), errors="coerce")
        return "parse_failed", int(numeric) if pd.notna(numeric) else 0
    if len(x_values) != len(y_values):
        return "x_y_length_mismatch", len(x_values)
    numeric = pd.to_numeric(row.get("n_points", ""), errors="coerce")
    if pd.notna(numeric):
        return "ok", int(numeric)
    return "ok", len(x_values)


def step5_unit_note(row: pd.Series) -> str:
    existing = normalize_text(row.get("unit_check_note", ""))
    if row.get("property_step5") != "ZT":
        return existing
    unit = normalize_text(row.get("unit_y", row.get("unit", ""))).casefold()
    if unit in DIMENSIONLESS_ZT_UNITS:
        return existing
    note = "ZT unit is not dimensionless; check later"
    if existing and existing != note:
        return f"{existing} | {note}"
    return note


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in OPTIONAL_KEEP_COLUMNS if column in df.columns]
    remaining = [
        column
        for column in df.columns
        if column not in columns and column.startswith(("zt_", "is_", "has_"))
    ]
    return df.loc[:, columns + remaining]


def build_candidate_core(
    candidate_path: Path | None,
    property_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    if candidate_path is not None and candidate_path.exists():
        candidate_df = read_csv_text(candidate_path)
        validate_input(candidate_df, "candidate input")
        candidate_df = add_step5_quality_columns(candidate_df)
        return candidate_df[candidate_df["is_target_property_step5"]].copy(), str(candidate_path)

    if "is_candidate_sample" in property_df.columns:
        mask = property_df["is_candidate_sample"].map(normalize_bool)
        return property_df[mask & property_df["is_target_property_step5"]].copy(), "is_candidate_sample"

    if "is_learning_candidate" in property_df.columns:
        mask = property_df["is_learning_candidate"].map(normalize_bool)
        return property_df[mask & property_df["is_target_property_step5"]].copy(), "is_learning_candidate"

    raise KeyError(
        "candidate_property_curves.csv not found and input has no is_candidate_sample or is_learning_candidate"
    )


def first_value(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    for value in group[column]:
        text = normalize_text(value)
        if text:
            return text
    return ""


def build_sample_availability(core_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in core_df.groupby("sample_key", sort=True):
        counts = {prop: int((group["property_step5"] == prop).sum()) for prop in PROPERTY_ORDER}
        points = {
            prop: int(group.loc[group["property_step5"] == prop, "n_points_step5"].sum())
            for prop in PROPERTY_ORDER
        }

        has_ec = counts["Electrical conductivity"] > 0
        has_er = counts["Electrical resistivity"] > 0
        has_s = counts["Seebeck coefficient"] > 0
        has_k = counts["Thermal conductivity"] > 0
        has_zt = counts["ZT"] > 0
        sigma_or_rho_points = points["Electrical conductivity"] + points["Electrical resistivity"]
        kappa_or_zt_points = points["Thermal conductivity"] + points["ZT"]
        is_learning_candidate = (
            (has_ec or has_er) and has_s and (has_k or has_zt) and sigma_or_rho_points >= 5
        )

        records.append(
            {
                "sample_key": sample_key,
                "SID": first_value(group, "SID"),
                "DOI": first_value(group, "DOI"),
                "sample_id": first_value(group, "sample_id"),
                "paper_title": first_value(group, "paper_title"),
                "year": first_value(group, "year"),
                "composition": first_value(group, "composition"),
                "material_system": first_value(group, "material_system"),
                "n_or_p": first_value(group, "n_or_p") or "unknown",
                "has_electrical_conductivity": has_ec,
                "has_electrical_resistivity": has_er,
                "has_sigma_or_rho": has_ec or has_er,
                "has_seebeck": has_s,
                "has_thermal_conductivity": has_k,
                "has_zt": has_zt,
                "has_kappa_or_zt": has_k or has_zt,
                "electrical_conductivity_curve_count": counts["Electrical conductivity"],
                "electrical_resistivity_curve_count": counts["Electrical resistivity"],
                "seebeck_curve_count": counts["Seebeck coefficient"],
                "thermal_conductivity_curve_count": counts["Thermal conductivity"],
                "zt_curve_count": counts["ZT"],
                "electrical_conductivity_point_count": points["Electrical conductivity"],
                "electrical_resistivity_point_count": points["Electrical resistivity"],
                "sigma_or_rho_point_count": sigma_or_rho_points,
                "seebeck_point_count": points["Seebeck coefficient"],
                "thermal_conductivity_point_count": points["Thermal conductivity"],
                "zt_point_count": points["ZT"],
                "kappa_or_zt_point_count": kappa_or_zt_points,
                "is_learning_candidate_step5": is_learning_candidate,
                "learning_candidate_reason_step5": learning_candidate_reason(
                    has_ec or has_er, has_s, has_k or has_zt, sigma_or_rho_points
                ),
            }
        )
    return pd.DataFrame.from_records(records, columns=SAMPLE_AVAILABILITY_COLUMNS)


def learning_candidate_reason(
    has_sigma_or_rho: bool,
    has_seebeck: bool,
    has_kappa_or_zt: bool,
    sigma_or_rho_points: int,
) -> str:
    if not has_sigma_or_rho:
        return "missing sigma/rho"
    if not has_seebeck:
        return "missing Seebeck"
    if not has_kappa_or_zt:
        return "missing kappa/ZT"
    if sigma_or_rho_points < 5:
        return "insufficient sigma/rho points"
    return "ok: sigma/rho, Seebeck, and kappa/ZT available"


def summarize_excluded(excluded_df: pd.DataFrame) -> pd.DataFrame:
    if excluded_df.empty:
        return pd.DataFrame(
            columns=[
                "property_family",
                "property",
                "prop_y_canonical",
                "prop_y_raw",
                "property_filter_reason",
                "curve_count",
            ]
        )
    columns = [
        column
        for column in [
            "property_family",
            "property",
            "prop_y_canonical",
            "prop_y_raw",
            "property_filter_reason",
        ]
        if column in excluded_df.columns
    ]
    return (
        excluded_df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="curve_count")
        .sort_values("curve_count", ascending=False, kind="stable")
        .head(50)
    )


def summarize_property_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["property_step5", "curve_count", "point_count"])
    return (
        df.groupby("property_step5", sort=True)
        .agg(curve_count=("sample_key", "count"), point_count=("n_points_step5", "sum"))
        .reset_index()
    )


def build_report(
    input_rows: int,
    core_df: pd.DataFrame,
    candidate_core_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    sample_availability_df: pd.DataFrame,
    excluded_summary_df: pd.DataFrame,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    report_rows: list[tuple[str, str]] = [
        ("input_rows", str(input_rows)),
        ("property_core_curves_step5_rows", str(len(core_df))),
        ("candidate_core_curves_step5_rows", str(len(candidate_core_df))),
        ("excluded_property_curves_step5_rows", str(len(excluded_df))),
        ("sample_property_availability_step5_rows", str(len(sample_availability_df))),
    ]

    for label, frame in (
        ("property_core_curves_step5", core_df),
        ("candidate_core_curves_step5", candidate_core_df),
    ):
        summary = summarize_property_counts(frame)
        for row in summary.itertuples(index=False):
            report_rows.append((f"{label}_{row.property_step5}_curve_count", str(int(row.curve_count))))
            report_rows.append((f"{label}_{row.property_step5}_point_count", str(int(row.point_count))))

    for row in excluded_summary_df.itertuples(index=False):
        key = "excluded_top50"
        value = " | ".join(f"{name}={getattr(row, name)}" for name in excluded_summary_df.columns)
        report_rows.append((key, value))

    for column in [
        "has_sigma_or_rho",
        "has_seebeck",
        "has_kappa_or_zt",
        "is_learning_candidate_step5",
    ]:
        report_rows.append((f"{column}_sample_count", str(int(sample_availability_df[column].sum()))))

    for n_or_p in ["n", "p", "mixed", "unknown"]:
        report_rows.append(
            (
                f"{n_or_p}_type_candidate_sample_count",
                str(
                    int(
                        sample_availability_df[
                            sample_availability_df["is_learning_candidate_step5"]
                        ]["n_or_p"].eq(n_or_p).sum()
                    )
                ),
            )
        )

    x_missing = core_df["x_values_json"].map(normalize_text).eq("")
    y_missing = core_df["y_values_json"].map(normalize_text).eq("")
    report_rows.extend(
        [
            ("x_values_json_missing_curve_count", str(int(x_missing.sum()))),
            ("y_values_json_missing_curve_count", str(int(y_missing.sum()))),
            ("xy_length_ok_curve_count", str(int(core_df["xy_length_check"].eq("ok").sum()))),
            (
                "xy_length_mismatch_curve_count",
                str(int(core_df["xy_length_check"].eq("x_y_length_mismatch").sum())),
            ),
            (
                "xy_parse_failed_curve_count",
                str(int(core_df["xy_length_check"].eq("parse_failed").sum())),
            ),
        ]
    )

    zt_abnormal = core_df["unit_check_note_step5"].str.contains(
        "ZT unit is not dimensionless", na=False
    )
    report_rows.append(("zt_unit_abnormal_curve_count", str(int(zt_abnormal.sum()))))
    report_rows.append(
        ("zt_unit_abnormal_sample_count", str(core_df.loc[zt_abnormal, "sample_key"].nunique()))
    )

    bad_core = sorted(set(core_df["property_step5"]) - TARGET_PROPERTIES)
    bad_candidate = sorted(set(candidate_core_df["property_step5"]) - TARGET_PROPERTIES)
    if bad_core:
        report_rows.append(("error_core_has_non_target_properties", ", ".join(bad_core)))
    if bad_candidate:
        report_rows.append(("error_candidate_has_non_target_properties", ", ".join(bad_candidate)))
    for note in excel_notes:
        report_rows.append(("excel_note", note))

    report_df = pd.DataFrame(report_rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in report_rows) + "\n"
    return report_text, report_df


def assert_acceptance(
    core_df: pd.DataFrame,
    candidate_core_df: pd.DataFrame,
    sample_availability_df: pd.DataFrame,
) -> None:
    for label, frame in (
        ("property_core_curves_step5.csv", core_df),
        ("candidate_core_curves_step5.csv", candidate_core_df),
    ):
        missing = [
            column
            for column in [
                "sample_key",
                "x_values_json",
                "y_values_json",
                "property_step5",
                "is_target_property_step5",
            ]
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"{label} missing columns: {missing}")
        bad = sorted(set(frame["property_step5"]) - TARGET_PROPERTIES)
        if bad:
            raise ValueError(f"{label} has non-target properties: {bad}")

    missing_availability = [
        column
        for column in [
            "has_sigma_or_rho",
            "has_seebeck",
            "has_kappa_or_zt",
            "sigma_or_rho_point_count",
            "is_learning_candidate_step5",
        ]
        if column not in sample_availability_df.columns
    ]
    if missing_availability:
        raise KeyError(
            f"sample_property_availability_step5.csv missing columns: {missing_availability}"
        )
    if sample_availability_df["sample_key"].duplicated().any():
        raise ValueError("sample_property_availability_step5.csv has duplicate sample_key rows")


def write_csv_outputs(
    output_dir: Path,
    core_df: pd.DataFrame,
    candidate_core_df: pd.DataFrame,
    sample_availability_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    core_df.to_csv(output_dir / "property_core_curves_step5.csv", index=False)
    candidate_core_df.to_csv(output_dir / "candidate_core_curves_step5.csv", index=False)
    sample_availability_df.to_csv(output_dir / "sample_property_availability_step5.csv", index=False)
    excluded_df.to_csv(output_dir / "excluded_property_curves_step5.csv", index=False)
    (output_dir / "step5_property_filter_report.txt").write_text(report_text, encoding="utf-8")


def excel_frame(df: pd.DataFrame, sheet_name: str, excel_notes: list[str]) -> pd.DataFrame:
    if len(df) <= EXCEL_MAX_ROWS - 1:
        return df
    excel_notes.append(
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
    for index, column in enumerate(df.columns, start=1):
        max_length = len(str(column))
        if not preview.empty:
            max_length = max(max_length, int(preview[column].astype(str).map(len).max()))
        worksheet.column_dimensions[worksheet.cell(row=1, column=index).column_letter].width = min(
            max(max_length + 2, 12), 60
        )


def write_excel_output(
    output_dir: Path,
    core_df: pd.DataFrame,
    candidate_core_df: pd.DataFrame,
    sample_availability_df: pd.DataFrame,
    excluded_summary_df: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    path = output_dir / "starrydata2_step5_core_properties.xlsx"
    sheets = {
        "property_core_curves": excel_frame(core_df, "property_core_curves", excel_notes),
        "candidate_core_curves": excel_frame(candidate_core_df, "candidate_core_curves", excel_notes),
        "sample_property_availability": excel_frame(
            sample_availability_df, "sample_property_availability", excel_notes
        ),
        "excluded_property_summary": excluded_summary_df,
        "filter_report": report_df,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def main() -> None:
    args = parse_args()
    input_path = resolve_input_path(args.input)
    candidate_path = resolve_candidate_path(args.candidate_input, input_path)

    raw_df = read_csv_text(input_path)
    validate_input(raw_df, "input")
    enriched_df = add_step5_quality_columns(raw_df)

    core_df = enriched_df[enriched_df["is_target_property_step5"]].copy()
    excluded_df = enriched_df[~enriched_df["is_target_property_step5"]].copy()
    candidate_core_df, candidate_source = build_candidate_core(candidate_path, enriched_df)

    core_df = select_output_columns(core_df)
    candidate_core_df = select_output_columns(candidate_core_df)
    excluded_df = select_output_columns(excluded_df)

    sample_availability_df = build_sample_availability(core_df)
    excluded_summary_df = summarize_excluded(excluded_df)

    assert_acceptance(core_df, candidate_core_df, sample_availability_df)

    excel_notes: list[str] = []
    report_text, report_df = build_report(
        input_rows=len(raw_df),
        core_df=core_df,
        candidate_core_df=candidate_core_df,
        excluded_df=excluded_df,
        sample_availability_df=sample_availability_df,
        excluded_summary_df=excluded_summary_df,
        excel_notes=[f"candidate_source: {candidate_source}"],
    )

    write_csv_outputs(
        args.output_dir,
        core_df,
        candidate_core_df,
        sample_availability_df,
        excluded_df,
        report_text,
    )
    write_excel_output(
        args.output_dir,
        core_df,
        candidate_core_df,
        sample_availability_df,
        excluded_summary_df,
        report_df,
        excel_notes,
    )
    if excel_notes:
        report_text, report_df = build_report(
            input_rows=len(raw_df),
            core_df=core_df,
            candidate_core_df=candidate_core_df,
            excluded_df=excluded_df,
            sample_availability_df=sample_availability_df,
            excluded_summary_df=excluded_summary_df,
            excel_notes=[f"candidate_source: {candidate_source}", *excel_notes],
        )
        (args.output_dir / "step5_property_filter_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    counts = core_df["property_step5"].value_counts()
    zt_abnormal = core_df["unit_check_note_step5"].str.contains(
        "ZT unit is not dimensionless", na=False
    )
    print("Done.")
    print("Created:")
    print("- property_core_curves_step5.csv")
    print("- candidate_core_curves_step5.csv")
    print("- sample_property_availability_step5.csv")
    print("- excluded_property_curves_step5.csv")
    print("- step5_property_filter_report.txt")
    print("- starrydata2_step5_core_properties.xlsx")
    print("")
    print("Summary:")
    print(f"input rows: {len(raw_df)}")
    print(f"property_core_curves_step5 rows: {len(core_df)}")
    print(f"candidate_core_curves_step5 rows: {len(candidate_core_df)}")
    print(f"excluded rows: {len(excluded_df)}")
    print(
        "learning candidate samples step5: "
        f"{int(sample_availability_df['is_learning_candidate_step5'].sum())}"
    )
    for prop in PROPERTY_ORDER:
        print(f"{prop} curves: {int(counts.get(prop, 0))}")
    print(f"ZT unit abnormal curves: {int(zt_abnormal.sum())}")
    print(
        "x/y mismatch curves: "
        f"{int(core_df['xy_length_check'].eq('x_y_length_mismatch').sum())}"
    )
    print(
        "x/y parse failed curves: "
        f"{int(core_df['xy_length_check'].eq('parse_failed').sum())}"
    )


if __name__ == "__main__":
    main()
