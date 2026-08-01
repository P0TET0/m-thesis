import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "raw" / "starrydata2"
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed"
DEFAULT_REPORT = EXP_DIR / "reports" / "step0_dataset_report.md"

TABLE_EXTENSIONS = {".csv", ".json", ".jsonl", ".xlsx", ".xls", ".parquet"}
EMPTY_TEXT = {"", "nan", "none", "null", "na", "n/a"}

OUTPUT_COLUMNS = [
    "row_id",
    "paper_id",
    "doi",
    "sample_id",
    "sample_key",
    "sample_label",
    "formula_raw",
    "material_name_raw",
    "material_family_raw",
    "T_K",
    "T_S_K",
    "T_sigma_K",
    "T_delta_K",
    "S_V_per_K",
    "S_uV_per_K",
    "S_sign",
    "sigma_S_per_m",
    "rho_ohm_m",
    "sigma_source",
    "match_method",
    "source_file_S",
    "source_file_sigma",
    "source_property_label_S",
    "source_property_label_sigma",
    "source_unit_S",
    "source_unit_sigma_or_rho",
    "source_curve_id_S",
    "source_curve_id_sigma",
    "source_notes",
]

REJECT_COLUMNS = [
    "source_file",
    "source_row_or_id",
    "reject_reason",
    "property_label",
    "raw_value",
    "raw_unit",
    "paper_id",
    "sample_id",
    "T_K",
]

PROPERTY_LABEL_COLUMNS = [
    "property",
    "property_family",
    "prop_y",
    "prop_y_raw",
    "prop_y_canonical",
    "property_step5",
    "label",
    "property_label",
    "quantity",
    "name",
]
UNIT_COLUMNS = ["unit", "unit_y", "y_unit", "unit_value", "units"]
TEMPERATURE_COLUMNS = [
    "T",
    "temperature",
    "Temperature",
    "temp",
    "T_K",
    "temperature_K",
    "Temperature_K",
]
VALUE_COLUMNS = ["value", "Value", "y", "Y", "raw_value", "measurement", "data_value"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build step0 thermoelectric analysis table from Starrydata2 tables."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--match-tol-k", type=float, default=1.0)
    parser.add_argument("--allow-interpolation", action="store_true")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in EMPTY_TEXT:
        return ""
    return text


def compact(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_text(value).casefold())


def numeric(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def parse_numeric_list(value: Any) -> tuple[list[float], str | None]:
    text = clean_text(value)
    if not text:
        return [], None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError) as exc:
            return [], f"cannot parse numeric list: {exc}"
    if not isinstance(parsed, (list, tuple)):
        return [], "parsed value is not a list"
    values: list[float] = []
    for item in parsed:
        val = numeric(item)
        if math.isfinite(val):
            values.append(val)
        else:
            return [], f"non-numeric list item: {item!r}"
    return values, None


def list_table_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        return [input_dir] if input_dir.suffix.casefold() in TABLE_EXTENSIONS else []
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in TABLE_EXTENSIONS
    )


def read_table_file(path: Path) -> list[tuple[str, pd.DataFrame, str | None]]:
    try:
        suffix = path.suffix.casefold()
        if suffix == ".csv":
            return [(str(path), pd.read_csv(path, dtype=str, keep_default_na=False), None)]
        if suffix == ".jsonl":
            return [(str(path), pd.read_json(path, lines=True, dtype=False), None)]
        if suffix == ".json":
            return [(str(path), pd.read_json(path, dtype=False), None)]
        if suffix == ".parquet":
            return [(str(path), pd.read_parquet(path), None)]
        if suffix in {".xlsx", ".xls"}:
            sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False)
            return [(f"{path}::{sheet}", df, None) for sheet, df in sheets.items()]
    except Exception as exc:  # noqa: BLE001
        return [(str(path), pd.DataFrame(), f"{type(exc).__name__}: {exc}")]
    return [(str(path), pd.DataFrame(), "unsupported extension")]


def missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    missing = series.map(lambda value: clean_text(value) == "").sum()
    return float(missing / len(series))


def representative_values(series: pd.Series, limit: int = 5) -> list[str]:
    values: list[str] = []
    for value in series:
        text = clean_text(value)
        if text and text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return values


def column_matches(columns: list[str], patterns: list[str]) -> list[str]:
    hits = []
    for column in columns:
        text = compact(column)
        if any(pattern in text for pattern in patterns):
            hits.append(column)
    return hits


def infer_schema(table_name: str, df: pd.DataFrame, error: str | None) -> dict[str, Any]:
    columns = [str(col) for col in df.columns]
    column_info = {}
    for column in columns:
        series = df[column]
        column_info[column] = {
            "missing_rate": missing_rate(series),
            "representative_values": representative_values(series),
        }
    property_like = column_matches(
        columns,
        ["property", "prop", "seebeck", "conductivity", "resistivity", "thermopower", "sigma", "rho"],
    )
    return {
        "file": table_name,
        "read_error": error,
        "row_count": int(len(df)),
        "column_count": int(len(columns)),
        "columns": columns,
        "column_info": column_info,
        "inferred_columns": {
            "property_like": property_like,
            "temperature_like": column_matches(columns, ["temperature", "temp", "t_k"]),
            "unit_like": column_matches(columns, ["unit"]),
            "sample_id_like": column_matches(columns, ["sample", "sid"]),
            "paper_id_or_doi_like": column_matches(columns, ["paper", "doi", "sid"]),
            "formula_or_material_like": column_matches(
                columns, ["formula", "composition", "material", "compound"]
            ),
        },
        "unit_candidates": sorted(
            {
                value
                for column in columns
                if "unit" in compact(column)
                for value in representative_values(df[column], limit=10)
            }
        ),
        "property_candidates": sorted(
            {
                value
                for column in property_like
                for value in representative_values(df[column], limit=20)
            }
        ),
    }


def unit_key(unit: Any) -> str:
    text = clean_text(unit)
    text = text.replace("µ", "u").replace("μ", "u")
    text = text.replace("Ω", "ohm").replace("σ", "sigma").replace("ρ", "rho")
    text = text.replace("·", "*").replace("−", "-")
    return re.sub(r"\s+", "", text.casefold())


def seebeck_factor(unit: Any) -> float | None:
    key = unit_key(unit)
    if not key:
        return None
    if key in {"v/k", "v*k^(-1)", "v*k-1", "vk-1", "vperkelvin"}:
        return 1.0
    if key in {"mv/k", "mv*k^(-1)", "mv*k-1", "mvk-1"}:
        return 1e-3
    if key in {"uv/k", "uv*k^(-1)", "uv*k-1", "uvk-1"}:
        return 1e-6
    return None


def sigma_factor(unit: Any) -> float | None:
    key = unit_key(unit)
    if not key:
        return None
    direct = {
        "s/m": 1.0,
        "s*m^(-1)": 1.0,
        "s*m-1": 1.0,
        "ohm^(-1)*m^(-1)": 1.0,
        "ohm-1*m-1": 1.0,
        "1/ohm/m": 1.0,
        "s/cm": 100.0,
        "scm-1": 100.0,
        "s*cm^(-1)": 100.0,
        "s*cm-1": 100.0,
        "ohm^(-1)*cm^(-1)": 100.0,
        "ohm-1*cm-1": 100.0,
        "1/ohm/cm": 100.0,
    }
    return direct.get(key)


def rho_factor(unit: Any) -> float | None:
    key = unit_key(unit)
    if not key:
        return None
    direct = {
        "ohmm": 1.0,
        "ohm*m": 1.0,
        "ohm-m": 1.0,
        "ohmcm": 0.01,
        "ohm*cm": 0.01,
        "ohm-cm": 0.01,
        "mohmcm": 1e-5,
        "mohm*cm": 1e-5,
        "mohm-cm": 1e-5,
        "uohmm": 1e-6,
        "uohm*m": 1e-6,
        "uohm-m": 1e-6,
    }
    return direct.get(key)


def classify_property(label: Any, unit: Any, column_name: Any = "") -> str | None:
    label_text = compact(f"{label} {column_name}")
    s_unit = seebeck_factor(unit) is not None
    sigma_unit = sigma_factor(unit) is not None
    rho_unit = rho_factor(unit) is not None
    if any(word in label_text for word in ["seebeck", "thermopower"]):
        return "seebeck" if s_unit or "coefficient" in label_text else None
    if re.search(r"(^|[^a-z])s([^a-z]|$)", label_text) and s_unit:
        return "seebeck"
    if any(word in label_text for word in ["resistivity", "rho"]) or "ρ" in str(label):
        return "rho" if rho_unit or "resistivity" in label_text else None
    if any(word in label_text for word in ["electrical conductivity", "electric conductivity"]):
        return "sigma" if sigma_unit or "conductivity" in label_text else None
    if "conductivity" in label_text and not any(word in label_text for word in ["thermal", "heat"]):
        return "sigma" if sigma_unit or "electrical" in label_text or "electric" in label_text else None
    if any(word in label_text for word in ["sigma", "σ"]) and sigma_unit:
        return "sigma"
    return None


def convert_temperature(value: Any, unit: Any = "K") -> float:
    temp = numeric(value)
    if not math.isfinite(temp):
        return math.nan
    key = unit_key(unit)
    if key in {"", "k", "kelvin"}:
        return temp
    if key in {"c", "degc", "celsius", "°c"}:
        return temp + 273.15
    return temp


def is_temperature_axis(row: pd.Series) -> bool:
    label = compact(first_value(row, ["prop_x", "property_x", "x_label", "x_name"]) or "temperature")
    unit = unit_key(first_value(row, ["unit_x", "x_unit", "temperature_unit"]) or "K")
    label_ok = any(word in label for word in ["temperature", "temp"]) or label in {"t", "t_k"}
    unit_ok = unit in {"", "k", "kelvin", "c", "degc", "celsius", "°c"}
    return label_ok and unit_ok


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(col).casefold(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.casefold() in lower_map:
            return lower_map[candidate.casefold()]
    return None


def first_value(row: pd.Series, candidates: list[str]) -> str:
    for col in candidates:
        if col in row.index:
            text = clean_text(row.get(col))
            if text:
                return text
    return ""


def metadata_from_row(row: pd.Series, source_file: str) -> dict[str, str]:
    doi = first_value(row, ["DOI", "doi", "DOI_curve", "DOI_sample"])
    paper_id = first_value(row, ["paper_id", "paper", "SID", "sid", "paper_title"])
    sample_id = first_value(row, ["sample_id", "sample_id_curve", "sample_id_sample", "SID"])
    sample_label = first_value(row, ["sample_label", "sample_name", "legend_label", "label"])
    formula = first_value(row, ["formula", "formula_raw", "composition", "composition_sample"])
    material = first_value(row, ["material_name", "material_name_raw", "compound", "composition"])
    family = first_value(row, ["material_family", "material_family_raw", "material_system"])
    synthesis = first_value(row, ["synthesis_condition", "sintering_method"])
    sample_key = first_value(row, ["sample_key", "sample_key_curve", "sample_key_sample"])
    if not sample_key:
        sample_key = "|".join(
            part
            for part in [doi or paper_id, sample_id, sample_label, formula, synthesis]
            if part
        )
    if not paper_id:
        paper_id = doi
    return {
        "paper_id": paper_id,
        "doi": doi,
        "sample_id": sample_id,
        "sample_key": sample_key,
        "sample_label": sample_label,
        "formula_raw": formula,
        "material_name_raw": material,
        "material_family_raw": family or "unknown",
        "source_file": source_file,
    }


def property_label_from_row(row: pd.Series) -> str:
    labels = [first_value(row, PROPERTY_LABEL_COLUMNS)]
    labels.extend(clean_text(row.get(col)) for col in PROPERTY_LABEL_COLUMNS if col in row.index)
    labels = [label for label in labels if label]
    return " | ".join(dict.fromkeys(labels))


def unit_from_row(row: pd.Series) -> str:
    return first_value(row, UNIT_COLUMNS)


def curve_id_from_row(row: pd.Series) -> str:
    return first_value(row, ["curve_id", "curve_key", "figure_id", "id"])


def append_reject(
    rejects: list[dict[str, Any]],
    source_file: str,
    row_id: Any,
    reason: str,
    label: Any = "",
    raw_value: Any = "",
    raw_unit: Any = "",
    paper_id: Any = "",
    sample_id: Any = "",
    t_k: Any = "",
) -> None:
    rejects.append(
        {
            "source_file": source_file,
            "source_row_or_id": row_id,
            "reject_reason": reason,
            "property_label": clean_text(label),
            "raw_value": clean_text(raw_value),
            "raw_unit": clean_text(raw_unit),
            "paper_id": clean_text(paper_id),
            "sample_id": clean_text(sample_id),
            "T_K": t_k,
        }
    )


def make_point(
    row: pd.Series,
    source_file: str,
    source_row: Any,
    prop_type: str,
    t_raw: Any,
    y_raw: Any,
    label: str,
    unit: str,
) -> dict[str, Any] | None:
    meta = metadata_from_row(row, source_file)
    value = numeric(y_raw)
    t_unit = first_value(row, ["unit_x", "x_unit", "temperature_unit"]) or "K"
    t_k = convert_temperature(t_raw, t_unit)
    point = {
        **meta,
        "source_row_or_id": source_row,
        "property_type": prop_type,
        "property_label": label,
        "unit": unit,
        "curve_id": curve_id_from_row(row),
        "T_K": t_k,
        "raw_value": value,
        "source_notes": "",
    }
    if prop_type == "seebeck":
        factor = seebeck_factor(unit)
        point["S_V_per_K"] = value * factor if factor is not None else math.nan
        point["sigma_S_per_m"] = math.nan
        point["rho_ohm_m"] = math.nan
    elif prop_type == "sigma":
        factor = sigma_factor(unit)
        point["S_V_per_K"] = math.nan
        point["sigma_S_per_m"] = value * factor if factor is not None else math.nan
        point["rho_ohm_m"] = math.nan
    elif prop_type == "rho":
        factor = rho_factor(unit)
        rho = value * factor if factor is not None else math.nan
        point["S_V_per_K"] = math.nan
        point["rho_ohm_m"] = rho
        point["sigma_S_per_m"] = 1.0 / rho if math.isfinite(rho) and rho > 0 else math.nan
    else:
        return None
    return point


def extract_curve_points(
    table_name: str, df: pd.DataFrame, rejects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if "x_values_json" not in df.columns or "y_values_json" not in df.columns:
        return []
    points: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        label = property_label_from_row(row)
        unit = unit_from_row(row)
        prop_type = classify_property(label, unit)
        if prop_type is None:
            continue
        if not is_temperature_axis(row):
            append_reject(
                rejects,
                table_name,
                curve_id_from_row(row) or idx,
                "x_axis_is_not_temperature",
                label,
                "",
                unit,
                first_value(row, ["paper_id", "SID", "DOI"]),
                first_value(row, ["sample_id", "SID"]),
            )
            continue
        x_values, x_error = parse_numeric_list(row.get("x_values_json"))
        y_values, y_error = parse_numeric_list(row.get("y_values_json"))
        if x_error or y_error:
            append_reject(
                rejects,
                table_name,
                curve_id_from_row(row) or idx,
                f"curve_parse_error: {x_error or y_error}",
                label,
                "",
                unit,
                first_value(row, ["paper_id", "SID", "DOI"]),
                first_value(row, ["sample_id", "SID"]),
            )
            continue
        if len(x_values) != len(y_values):
            append_reject(
                rejects,
                table_name,
                curve_id_from_row(row) or idx,
                "x_y_length_mismatch",
                label,
                "",
                unit,
                first_value(row, ["paper_id", "SID", "DOI"]),
                first_value(row, ["sample_id", "SID"]),
            )
            continue
        for point_idx, (t_raw, y_raw) in enumerate(zip(x_values, y_values)):
            point = make_point(
                row,
                table_name,
                f"{curve_id_from_row(row) or idx}:{point_idx}",
                prop_type,
                t_raw,
                y_raw,
                label,
                unit,
            )
            if point:
                points.append(point)
    return points


def extract_long_points(
    table_name: str, df: pd.DataFrame, rejects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if "x_values_json" in df.columns and "y_values_json" in df.columns:
        return []
    temp_col = first_existing(df, TEMPERATURE_COLUMNS)
    value_col = first_existing(df, VALUE_COLUMNS)
    if not temp_col or not value_col:
        return []
    points: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        label = property_label_from_row(row)
        unit = unit_from_row(row)
        prop_type = classify_property(label, unit)
        if prop_type is None:
            continue
        point = make_point(row, table_name, idx, prop_type, row.get(temp_col), row.get(value_col), label, unit)
        if point:
            points.append(point)
    return points


def extract_wide_points(
    table_name: str, df: pd.DataFrame, rejects: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    temp_col = first_existing(df, TEMPERATURE_COLUMNS)
    if not temp_col:
        return []
    points: list[dict[str, Any]] = []
    for column in df.columns:
        if column == temp_col:
            continue
        prop_type = classify_property(column, "")
        if prop_type is None:
            continue
        unit = ""
        unit_col = first_existing(df, [f"{column}_unit", f"unit_{column}"])
        if unit_col:
            unit = clean_text(df[unit_col].iloc[0])
        if prop_type == "seebeck" and seebeck_factor(unit) is None:
            continue
        if prop_type == "sigma" and sigma_factor(unit) is None:
            continue
        if prop_type == "rho" and rho_factor(unit) is None:
            continue
        for idx, row in df.iterrows():
            point = make_point(row, table_name, idx, prop_type, row.get(temp_col), row.get(column), str(column), unit)
            if point:
                points.append(point)
    return points


def validate_points(points: list[dict[str, Any]], rejects: list[dict[str, Any]]) -> pd.DataFrame:
    valid: list[dict[str, Any]] = []
    for point in points:
        prop = point["property_type"]
        t_k = point["T_K"]
        if not math.isfinite(t_k) or t_k <= 0:
            append_reject(
                rejects,
                point["source_file"],
                point["source_row_or_id"],
                "invalid_temperature",
                point["property_label"],
                point["raw_value"],
                point["unit"],
                point["paper_id"],
                point["sample_id"],
                t_k,
            )
            continue
        if prop == "seebeck":
            if not math.isfinite(point["S_V_per_K"]):
                append_reject(
                    rejects,
                    point["source_file"],
                    point["source_row_or_id"],
                    "invalid_seebeck_value_or_unit",
                    point["property_label"],
                    point["raw_value"],
                    point["unit"],
                    point["paper_id"],
                    point["sample_id"],
                    t_k,
                )
                continue
        else:
            sigma = point["sigma_S_per_m"]
            rho = point["rho_ohm_m"]
            if prop == "rho" and (not math.isfinite(rho) or rho <= 0):
                append_reject(
                    rejects,
                    point["source_file"],
                    point["source_row_or_id"],
                    "nonpositive_or_invalid_resistivity",
                    point["property_label"],
                    point["raw_value"],
                    point["unit"],
                    point["paper_id"],
                    point["sample_id"],
                    t_k,
                )
                continue
            if not math.isfinite(sigma) or sigma <= 0:
                append_reject(
                    rejects,
                    point["source_file"],
                    point["source_row_or_id"],
                    "nonpositive_or_invalid_conductivity",
                    point["property_label"],
                    point["raw_value"],
                    point["unit"],
                    point["paper_id"],
                    point["sample_id"],
                    t_k,
                )
                continue
        valid.append(point)
    return pd.DataFrame(valid)


def point_value_column(prop_type: str) -> str:
    return "S_V_per_K" if prop_type == "seebeck" else "sigma_S_per_m"


def remove_conflicting_point_duplicates(
    points: pd.DataFrame, rejects: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if points.empty:
        return points, pd.DataFrame()
    dup_rows: list[pd.DataFrame] = []
    keep_indices: list[int] = []
    reject_indices: set[int] = set()
    group_cols = ["paper_id", "sample_key", "property_type", "T_K"]
    for _, group in points.groupby(group_cols, dropna=False, sort=False):
        if len(group) == 1:
            keep_indices.extend(group.index.tolist())
            continue
        dup_rows.append(group.copy())
        value_col = point_value_column(str(group["property_type"].iloc[0]))
        values = pd.to_numeric(group[value_col], errors="coerce").dropna().to_numpy()
        if len(values) and np.allclose(values, values[0], rtol=1e-10, atol=1e-14):
            keep_indices.append(group.index[0])
        else:
            reject_indices.update(group.index.tolist())
            for _, row in group.iterrows():
                append_reject(
                    rejects,
                    row["source_file"],
                    row["source_row_or_id"],
                    "conflicting_duplicate_property_point",
                    row["property_label"],
                    row["raw_value"],
                    row["unit"],
                    row["paper_id"],
                    row["sample_id"],
                    row["T_K"],
                )
    duplicate_df = pd.concat(dup_rows, ignore_index=True) if dup_rows else pd.DataFrame()
    keep = points.loc[[idx for idx in keep_indices if idx not in reject_indices]].copy()
    return keep.reset_index(drop=True), duplicate_df


def make_output_row(s_row: pd.Series, sigma_row: pd.Series, method: str, t_sigma: float) -> dict[str, Any]:
    sigma_source = "resistivity_converted" if sigma_row["property_type"] == "rho" else "conductivity_direct"
    s_value = float(s_row["S_V_per_K"])
    sigma_value = float(sigma_row["sigma_S_per_m"])
    rho_value = (
        float(sigma_row["rho_ohm_m"])
        if math.isfinite(float(sigma_row.get("rho_ohm_m", math.nan)))
        else (1.0 / sigma_value if sigma_source == "conductivity_direct" and sigma_value > 0 else math.nan)
    )
    return {
        "row_id": "",
        "paper_id": s_row["paper_id"] or sigma_row["paper_id"],
        "doi": s_row["doi"] or sigma_row["doi"],
        "sample_id": s_row["sample_id"] or sigma_row["sample_id"],
        "sample_key": s_row["sample_key"] or sigma_row["sample_key"],
        "sample_label": s_row["sample_label"] or sigma_row["sample_label"],
        "formula_raw": s_row["formula_raw"] or sigma_row["formula_raw"],
        "material_name_raw": s_row["material_name_raw"] or sigma_row["material_name_raw"],
        "material_family_raw": s_row["material_family_raw"] or sigma_row["material_family_raw"],
        "T_K": float(s_row["T_K"]),
        "T_S_K": float(s_row["T_K"]),
        "T_sigma_K": float(t_sigma),
        "T_delta_K": abs(float(s_row["T_K"]) - float(t_sigma)),
        "S_V_per_K": s_value,
        "S_uV_per_K": s_value * 1e6,
        "S_sign": "positive" if s_value > 0 else ("negative" if s_value < 0 else "zero"),
        "sigma_S_per_m": sigma_value,
        "rho_ohm_m": rho_value,
        "sigma_source": sigma_source,
        "match_method": method,
        "source_file_S": s_row["source_file"],
        "source_file_sigma": sigma_row["source_file"],
        "source_property_label_S": s_row["property_label"],
        "source_property_label_sigma": sigma_row["property_label"],
        "source_unit_S": s_row["unit"],
        "source_unit_sigma_or_rho": sigma_row["unit"],
        "source_curve_id_S": s_row["curve_id"],
        "source_curve_id_sigma": sigma_row["curve_id"],
        "source_notes": "; ".join(
            note for note in [s_row.get("source_notes", ""), sigma_row.get("source_notes", "")] if note
        ),
    }


def match_points(points: pd.DataFrame, match_tol_k: float, allow_interpolation: bool) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    s_points = points[points["property_type"] == "seebeck"].copy()
    sigma_points = points[points["property_type"].isin(["sigma", "rho"])].copy()
    sigma_groups = {
        key: group.copy()
        for key, group in sigma_points.groupby(["paper_id", "sample_key"], dropna=False, sort=False)
    }
    for group in sigma_groups.values():
        group["_T"] = pd.to_numeric(group["T_K"], errors="coerce")
    rows: list[dict[str, Any]] = []
    for group_key, s_group in s_points.groupby(["paper_id", "sample_key"], dropna=False, sort=False):
        sigma_group = sigma_groups.get(group_key)
        if sigma_group is None:
            continue
        if sigma_group.empty:
            continue
        for _, s_row in s_group.iterrows():
            t_s = float(s_row["T_K"])
            exact = sigma_group[np.isclose(sigma_group["_T"], t_s, rtol=0.0, atol=1e-12)]
            if not exact.empty:
                for _, sigma_row in exact.iterrows():
                    rows.append(make_output_row(s_row, sigma_row, "exact", float(sigma_row["T_K"])))
                continue
            sigma_group["_delta"] = (sigma_group["_T"] - t_s).abs()
            nearest = sigma_group[sigma_group["_delta"] <= match_tol_k]
            if not nearest.empty:
                min_delta = nearest["_delta"].min()
                nearest = nearest[np.isclose(nearest["_delta"], min_delta, rtol=0.0, atol=1e-12)]
                for _, sigma_row in nearest.iterrows():
                    rows.append(make_output_row(s_row, sigma_row, "nearest", float(sigma_row["T_K"])))
                continue
            if allow_interpolation:
                for prop_type, type_group in sigma_group.groupby("property_type", sort=False):
                    curve_groups = type_group.groupby("curve_id", dropna=False, sort=False)
                    for _, curve_group in curve_groups:
                        curve_group = curve_group.sort_values("_T")
                        temps = curve_group["_T"].to_numpy(dtype=float)
                        sigmas = pd.to_numeric(curve_group["sigma_S_per_m"], errors="coerce").to_numpy(dtype=float)
                        if len(temps) < 2 or not (temps.min() <= t_s <= temps.max()):
                            continue
                        interp_sigma = float(np.interp(t_s, temps, sigmas))
                        if not math.isfinite(interp_sigma) or interp_sigma <= 0:
                            continue
                        base = curve_group.iloc[0].copy()
                        base["sigma_S_per_m"] = interp_sigma
                        base["rho_ohm_m"] = 1.0 / interp_sigma if prop_type == "rho" else math.nan
                        base["source_notes"] = f"linear interpolation from curve_id={base['curve_id']}"
                        rows.append(make_output_row(s_row, base, "interpolated", t_s))
                        break
                    if rows and rows[-1]["T_S_K"] == t_s and rows[-1]["match_method"] == "interpolated":
                        break
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return out[OUTPUT_COLUMNS]


def remove_conflicting_output_duplicates(
    output: pd.DataFrame, rejects: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if output.empty:
        return output, pd.DataFrame()
    dup_rows: list[pd.DataFrame] = []
    keep_indices: list[int] = []
    reject_indices: set[int] = set()
    for _, group in output.groupby(["paper_id", "sample_key", "T_K"], dropna=False, sort=False):
        if len(group) == 1:
            keep_indices.extend(group.index.tolist())
            continue
        dup_rows.append(group.copy())
        same_s = np.allclose(group["S_V_per_K"].astype(float), float(group["S_V_per_K"].iloc[0]), rtol=1e-10, atol=1e-14)
        same_sigma = np.allclose(
            group["sigma_S_per_m"].astype(float), float(group["sigma_S_per_m"].iloc[0]), rtol=1e-10, atol=1e-8
        )
        if same_s and same_sigma:
            keep_indices.append(group.index[0])
        else:
            reject_indices.update(group.index.tolist())
            for _, row in group.iterrows():
                append_reject(
                    rejects,
                    row["source_file_S"],
                    row["row_id"] or row.name,
                    "conflicting_duplicate_matched_row",
                    row["source_property_label_S"],
                    row["S_V_per_K"],
                    row["source_unit_S"],
                    row["paper_id"],
                    row["sample_id"],
                    row["T_K"],
                )
    duplicate_df = pd.concat(dup_rows, ignore_index=True) if dup_rows else pd.DataFrame()
    keep = output.loc[[idx for idx in keep_indices if idx not in reject_indices]].copy().reset_index(drop=True)
    keep["row_id"] = [f"step0_{idx + 1:08d}" for idx in range(len(keep))]
    return keep[OUTPUT_COLUMNS], duplicate_df


def sanity_checks(output: pd.DataFrame, match_tol_k: float) -> dict[str, Any]:
    required = set(OUTPUT_COLUMNS)
    checks: dict[str, Any] = {}
    checks["required_columns_present"] = sorted(required - set(output.columns)) == []
    checks["missing_required_columns"] = sorted(required - set(output.columns))
    if output.empty:
        checks.update(
            {
                "T_K_positive_finite": True,
                "S_V_per_K_finite": True,
                "sigma_S_per_m_positive_finite": True,
                "rho_ohm_m_positive_when_present": True,
                "S_uV_per_K_consistent": True,
                "T_delta_within_tolerance_for_non_interpolated": True,
                "row_id_unique": True,
            }
        )
        return checks
    checks["T_K_positive_finite"] = bool(np.isfinite(output["T_K"].astype(float)).all() and (output["T_K"].astype(float) > 0).all())
    checks["S_V_per_K_finite"] = bool(np.isfinite(output["S_V_per_K"].astype(float)).all())
    checks["sigma_S_per_m_positive_finite"] = bool(
        np.isfinite(output["sigma_S_per_m"].astype(float)).all() and (output["sigma_S_per_m"].astype(float) > 0).all()
    )
    rho = pd.to_numeric(output["rho_ohm_m"], errors="coerce")
    checks["rho_ohm_m_positive_when_present"] = bool((rho.dropna() > 0).all())
    checks["S_uV_per_K_consistent"] = bool(
        np.allclose(output["S_uV_per_K"].astype(float), output["S_V_per_K"].astype(float) * 1e6)
    )
    non_interp = output[output["match_method"] != "interpolated"]
    checks["T_delta_within_tolerance_for_non_interpolated"] = bool(
        non_interp.empty or (non_interp["T_delta_K"].astype(float) <= match_tol_k + 1e-12).all()
    )
    checks["row_id_unique"] = bool(output["row_id"].is_unique and output["row_id"].notna().all())
    checks["row_id_duplicate_count"] = int(output["row_id"].duplicated().sum())
    direct = output[output["sigma_source"] == "conductivity_direct"]
    converted = output[output["sigma_source"] == "resistivity_converted"]
    checks["conductivity_direct_positive"] = bool(direct.empty or (direct["sigma_S_per_m"].astype(float) > 0).all())
    checks["resistivity_converted_consistent"] = bool(
        converted.empty
        or np.allclose(
            converted["sigma_S_per_m"].astype(float),
            1.0 / converted["rho_ohm_m"].astype(float),
            rtol=1e-10,
            atol=1e-12,
        )
    )
    return checks


def summarize_range(series: pd.Series) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    return f"{values.min():.6g} to {values.max():.6g}"


def make_questions(schema: list[dict[str, Any]], points: pd.DataFrame, output: pd.DataFrame) -> list[str]:
    questions: list[str] = []
    if not schema:
        questions.append("入力ディレクトリに読み込める表データがありません。Starrydata2 の配置場所を確認してください。")
    if points.empty:
        questions.append("Seebeck と electrical conductivity/resistivity の候補を安全に抽出できませんでした。物性名列、単位列、温度列、値列を確認してください。")
    elif output.empty:
        questions.append("S と sigma/rho は見つかりましたが、同一 paper/sample/温度で対応づく行がありません。sample_id と温度対応の扱いを確認してください。")
    if not points.empty and points["sample_id"].map(clean_text).eq("").any():
        questions.append("sample_id が欠損している行があります。sample_key 生成に使う列が妥当か確認してください。")
    ambiguous = [
        item["file"]
        for item in schema
        if item.get("row_count", 0) > 0
        and item.get("inferred_columns", {}).get("property_like")
        and not item.get("inferred_columns", {}).get("unit_like")
    ]
    if ambiguous:
        questions.append("物性名らしい列はあるが単位列が見つからない表があります: " + ", ".join(ambiguous[:5]))
    return questions


def write_outputs(
    output: pd.DataFrame,
    rejects: pd.DataFrame,
    duplicates: pd.DataFrame,
    schema: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "step0_te_analysis_base.csv", index=False, encoding="utf-8-sig")
    rejects.to_csv(output_dir / "step0_rejected_rows.csv", index=False, encoding="utf-8-sig")
    duplicates.to_csv(output_dir / "step0_duplicate_candidates.csv", index=False, encoding="utf-8-sig")
    (output_dir / "step0_schema_detected.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        output.to_parquet(output_dir / "step0_te_analysis_base.parquet", index=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def write_report(
    report_path: Path,
    schema: list[dict[str, Any]],
    points: pd.DataFrame,
    output: pd.DataFrame,
    rejects: pd.DataFrame,
    duplicates: pd.DataFrame,
    checks: dict[str, Any],
    questions: list[str],
    parquet_saved: bool,
    parquet_error: str,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    loaded_files = [item for item in schema if not item.get("read_error")]
    total_rows = sum(int(item.get("row_count", 0)) for item in loaded_files)
    prop_counts = Counter(points["property_type"]) if not points.empty else Counter()
    lines = [
        "# Step0 Dataset Report",
        "",
        "## Summary",
        "",
        f"- 読み込んだファイル数: {len(loaded_files)}",
        f"- 読み込んだ総行数: {total_rows}",
        f"- Seebeck係数として認識したデータ数: {prop_counts.get('seebeck', 0)}",
        f"- electrical conductivityとして認識したデータ数: {prop_counts.get('sigma', 0)}",
        f"- electrical resistivityとして認識したデータ数: {prop_counts.get('rho', 0)}",
        f"- 最終的に S と sigma が対応づいた行数: {len(output)}",
        f"- exact / nearest / interpolated の行数: {dict(Counter(output['match_method'])) if not output.empty else {}}",
        f"- sigma_source の内訳: {dict(Counter(output['sigma_source'])) if not output.empty else {}}",
        f"- 温度範囲: {summarize_range(output['T_K']) if not output.empty else 'n/a'}",
        f"- S の範囲: {summarize_range(output['S_V_per_K']) if not output.empty else 'n/a'} V/K",
        f"- sigma の範囲: {summarize_range(output['sigma_S_per_m']) if not output.empty else 'n/a'} S/m",
        f"- paper_id 数: {output['paper_id'].nunique() if not output.empty else 0}",
        f"- sample_id 数: {output['sample_id'].nunique() if not output.empty else 0}",
        f"- formula_raw 数: {output['formula_raw'].nunique() if not output.empty else 0}",
        f"- material_family_raw 数: {output['material_family_raw'].nunique() if not output.empty else 0}",
        f"- parquet 保存: {'成功' if parquet_saved else '不可'}",
    ]
    if parquet_error:
        lines.append(f"- parquet 保存不可理由: {parquet_error}")
    lines.extend(["", "## 欠損値の概要", ""])
    if output.empty:
        lines.append("- 解析用データ表は空です。")
    else:
        missing = output.isna().mean().sort_values(ascending=False)
        for column, rate in missing.items():
            if rate > 0:
                lines.append(f"- {column}: {rate:.3f}")
        if output.isna().sum().sum() == 0:
            lines.append("- 欠損値はありません。")
    lines.extend(["", "## 除外理由", ""])
    if rejects.empty:
        lines.append("- 除外行はありません。")
    else:
        for reason, count in rejects["reject_reason"].value_counts().items():
            lines.append(f"- {reason}: {count}")
    lines.extend(["", "## 重複候補", "", f"- 重複候補の件数: {len(duplicates)}"])
    lines.extend(["", "## Sanity Check", ""])
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 読み込んだ表と推定スキーマ", ""])
    for item in schema:
        lines.append(f"### {item['file']}")
        if item.get("read_error"):
            lines.append(f"- read_error: {item['read_error']}")
        lines.append(f"- rows: {item.get('row_count', 0)}")
        lines.append(f"- columns: {item.get('column_count', 0)}")
        lines.append("- column_names: " + ", ".join(item.get("columns", [])))
        inferred = item.get("inferred_columns", {})
        for label, cols in inferred.items():
            lines.append(f"- {label}: {', '.join(cols) if cols else 'n/a'}")
        unit_candidates = item.get("unit_candidates", [])
        property_candidates = item.get("property_candidates", [])
        lines.append("- unit_candidates: " + (", ".join(unit_candidates[:20]) if unit_candidates else "n/a"))
        lines.append("- property_candidates: " + (", ".join(property_candidates[:20]) if property_candidates else "n/a"))
        column_info = item.get("column_info", {})
        if column_info:
            lines.append("")
            lines.append("| column | missing_rate | representative_values |")
            lines.append("| --- | ---: | --- |")
            for column, info in column_info.items():
                reps = "; ".join(info.get("representative_values", []))
                lines.append(f"| {column} | {info.get('missing_rate', 0):.3f} | {reps} |")
        lines.append("")
    lines.extend(["", "## 人間に確認すべき事項", ""])
    if questions:
        for question in questions:
            lines.append(f"- {question}")
    else:
        lines.append("- 現時点で必須の確認事項はありません。")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input
    output_dir = args.output
    rejects: list[dict[str, Any]] = []
    schema: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []

    for path in list_table_files(input_dir):
        for table_name, df, error in read_table_file(path):
            schema.append(infer_schema(table_name, df, error))
            if error:
                append_reject(rejects, table_name, "", f"table_read_error: {error}")
                continue
            try:
                all_points.extend(extract_curve_points(table_name, df, rejects))
                all_points.extend(extract_long_points(table_name, df, rejects))
                all_points.extend(extract_wide_points(table_name, df, rejects))
            except Exception as exc:  # noqa: BLE001
                append_reject(rejects, table_name, "", f"table_extract_error: {type(exc).__name__}: {exc}")

    raw_points = pd.DataFrame(all_points)
    valid_points = validate_points(all_points, rejects)
    valid_points, point_duplicates = remove_conflicting_point_duplicates(valid_points, rejects)
    output = match_points(valid_points, args.match_tol_k, args.allow_interpolation)
    output, output_duplicates = remove_conflicting_output_duplicates(output, rejects)
    duplicates = pd.concat(
        [frame for frame in [point_duplicates, output_duplicates] if not frame.empty],
        ignore_index=True,
    ) if (not point_duplicates.empty or not output_duplicates.empty) else pd.DataFrame()
    rejects_df = pd.DataFrame(rejects, columns=REJECT_COLUMNS)
    checks = sanity_checks(output, args.match_tol_k)
    questions = make_questions(schema, raw_points, output)
    parquet_saved, parquet_error = write_outputs(output, rejects_df, duplicates, schema, output_dir)
    write_report(
        args.report,
        schema,
        raw_points,
        output,
        rejects_df,
        duplicates,
        checks,
        questions,
        parquet_saved,
        parquet_error,
    )
    print(f"wrote {len(output)} matched rows to {output_dir / 'step0_te_analysis_base.csv'}")
    print(f"wrote report to {args.report}")


if __name__ == "__main__":
    main()
