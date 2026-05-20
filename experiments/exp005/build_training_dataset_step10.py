import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP9_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step9_literature_annotations"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step10_training_dataset"

INPUT_FILES = {
    "learning": "learning_candidates_step9.csv",
    "initial": "initial_tau_fit_candidates_step9.csv",
    "review": "review_needed_candidates_step9.csv",
    "candidate_curves": "candidate_core_curves_step9.csv",
    "sigma_rho_curves": "sigma_rho_curves_for_fitting_step9.csv",
    "sample_annotations": "sample_literature_annotations_step9.csv",
}

PROPERTY_SOURCE_COLUMNS = [
    "property_step8",
    "property_step5",
    "property",
    "property_family",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
]
TARGET_PROPERTIES = [
    "Electrical conductivity",
    "Electrical resistivity",
    "Seebeck coefficient",
    "Thermal conductivity",
    "ZT",
]
SIGMA_PROPERTY = "Electrical conductivity"
RHO_PROPERTY = "Electrical resistivity"
SEEBECK_PROPERTY = "Seebeck coefficient"
KAPPA_PROPERTY = "Thermal conductivity"
ZT_PROPERTY = "ZT"

POINT_METADATA_COLUMNS = [
    "sample_key",
    "curve_id",
    "curve_key",
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "sintering_status_step7",
    "sintering_status_step9",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "figure_id",
    "caption",
    "comments",
    "prop_x",
    "unit_x",
    "unit_y",
    "unit",
    "prop_y_raw",
    "n_points",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "fitting_source_preference_step8",
    "candidate_priority_tier_step8",
    "is_initial_tau_fit_candidate_step8",
    "is_full_learning_candidate_step8",
    "is_tau_fitting_candidate_step8",
    "learning_candidate_reason_step8",
]

WIDE_METADATA_COLUMNS = [
    "SID",
    "DOI",
    "doi_url",
    "sample_id",
    "paper_title",
    "year",
    "composition",
    "material_system",
    "n_or_p",
    "n_or_p_basis",
    "n_or_p_step6",
    "n_or_p_basis_step6",
    "n_or_p_confidence_step6",
    "sintering_method",
    "sintering_checked",
    "record_checked",
    "sintering_status_step7",
    "sintering_status_step9",
    "additive_auto_step9",
    "additive_manual_step9",
    "structure_auto_step9",
    "structure_manual_step9",
    "nanocarbon_keyword_detected_step9",
    "nanocarbon_type_auto_step9",
    "rare_metal_flag_auto_step9",
    "toxicity_flag_auto_step9",
    "fitting_source_preference_step8",
    "is_initial_tau_fit_candidate_step8",
    "is_full_learning_candidate_step8",
    "is_tau_fitting_candidate_step8",
    "candidate_priority_tier_step8",
    "learning_candidate_reason_step8",
]

NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
EXCEL_PREVIEW_ROWS = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Step10 1-sample x 1-temperature training dataset."
    )
    parser.add_argument("--step9_dir", type=Path, default=DEFAULT_STEP9_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature_tolerance_K", type=float, default=2.0)
    parser.add_argument("--temperature_round_decimals", type=int, default=3)
    return parser.parse_args()


def read_csv_text(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_bool(value: Any) -> bool:
    return normalize_text(value).casefold() in {"true", "1", "yes", "y"}


def compact_text(value: Any) -> str:
    return re.sub(r"\s+", " ", normalize_text(value).casefold())


def finite_float(value: Any) -> float | None:
    text = normalize_text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def validate_curve_input(df: pd.DataFrame, filename: str) -> None:
    missing = [
        column
        for column in ["sample_key", "x_values_json", "y_values_json"]
        if column not in df.columns
    ]
    if missing:
        raise KeyError(f"{filename} missing required columns: {missing}")
    if not any(column in df.columns for column in PROPERTY_SOURCE_COLUMNS):
        raise KeyError(
            f"{filename} needs at least one property source column: {PROPERTY_SOURCE_COLUMNS}"
        )


def validate_inputs(inputs: dict[str, pd.DataFrame]) -> None:
    for label, filename in INPUT_FILES.items():
        if "sample_key" not in inputs[label].columns:
            raise KeyError(f"{filename} missing required column: sample_key")
    validate_curve_input(inputs["candidate_curves"], INPUT_FILES["candidate_curves"])
    validate_curve_input(inputs["sigma_rho_curves"], INPUT_FILES["sigma_rho_curves"])


def values_from_iterable(parsed: Any) -> tuple[list[float], bool]:
    if isinstance(parsed, (str, bytes)):
        return [], True
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        return [], True
    values: list[float] = []
    saw_unparseable = False
    for item in parsed:
        number = finite_float(item)
        if number is None:
            if normalize_text(item):
                saw_unparseable = True
            continue
        values.append(number)
    return values, saw_unparseable


def parse_numeric_values(raw_value: Any) -> tuple[list[float], str]:
    text = normalize_text(raw_value)
    if not text:
        return [], "parse_failed"

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue
        values, saw_unparseable = values_from_iterable(parsed)
        if values or not saw_unparseable:
            return values, "ok"

    for separator in (",", None):
        if separator == "," and "," not in text:
            continue
        tokens = text.split(separator) if separator else text.split()
        if len(tokens) <= 1:
            continue
        values = [number for token in tokens if (number := finite_float(token)) is not None]
        if values:
            return values, "ok"

    values = [float(match.group(0)) for match in NUMERIC_RE.finditer(text)]
    values = [value for value in values if math.isfinite(value)]
    if values:
        return values, "ok"
    return [], "parse_failed"


def classify_property_text(value: Any) -> str:
    text = compact_text(value)
    if not text or "power factor" in text:
        return ""
    if text == "zt" or "dimensionless figure of merit" in text or "figure of merit" in text:
        return ZT_PROPERTY
    if "seebeck" in text or "thermopower" in text or "thermoelectric power" in text:
        return SEEBECK_PROPERTY
    if (
        "electrical_resistivity" in text
        or "electrical resistivity" in text
        or "electric resistivity" in text
        or re.search(r"(^|[^a-z])rho([^a-z]|$)", text)
        or "\u03c1" in text
    ):
        return RHO_PROPERTY
    if "resistivity" in text and "thermal" not in text:
        return RHO_PROPERTY
    if (
        "thermal_conductivity" in text
        or "thermal conductivity" in text
        or "total thermal conductivity" in text
    ):
        return KAPPA_PROPERTY
    if (
        "electrical_conductivity" in text
        or "electrical conductivity" in text
        or "electric conductivity" in text
        or re.search(r"(^|[^a-z])sigma([^a-z]|$)", text)
        or "\u03c3" in text
    ):
        return SIGMA_PROPERTY
    if "conductivity" in text and "thermal" not in text:
        return SIGMA_PROPERTY
    return ""


def classify_property(row: pd.Series | dict[str, Any]) -> str:
    for column in PROPERTY_SOURCE_COLUMNS:
        if column in row:
            property_name = classify_property_text(row[column])
            if property_name:
                return property_name
    return ""


def parse_curve_status(x_values: list[float], x_status: str, y_values: list[float], y_status: str) -> str:
    if x_status != "ok" and y_status != "ok":
        return "parse_failed"
    if x_status != "ok":
        return "x_parse_failed"
    if y_status != "ok":
        return "y_parse_failed"
    if len(x_values) != len(y_values):
        return "x_y_length_mismatch"
    return "ok"


def convert_temperature(value: float | None, unit_x: Any) -> tuple[float | None, str]:
    if value is None or not math.isfinite(value):
        return None, "temperature invalid"
    unit = normalize_text(unit_x).casefold().replace(" ", "")
    if unit in {"k", "kelvin"}:
        return value, "ok: unit_x is K"
    if unit in {"c", "celsius", "degc", "degreec", "degreesc", "\u00b0c"}:
        return value + 273.15, "converted Celsius to K"
    if not unit:
        return value, "unit_x missing; temperature values used as-is"
    return value, f"unit_x={normalize_text(unit_x)}; temperature values used as-is"


def point_invalid_reason(
    xy_check: str,
    temperature_k: float | None,
    temperature_outlier: bool,
    value_raw: float | None,
) -> str:
    if xy_check == "x_parse_failed":
        return "x_parse_failed"
    if xy_check == "y_parse_failed":
        return "y_parse_failed"
    if xy_check == "parse_failed":
        return "x_parse_failed;y_parse_failed"
    if xy_check == "x_y_length_mismatch":
        return "x_y_length_mismatch"
    if temperature_k is None:
        return "temperature_invalid"
    if temperature_outlier:
        return "temperature_outlier"
    if value_raw is None:
        return "value_invalid"
    return ""


def get_value(values: list[float], index: int) -> float | None:
    if index < 0 or index >= len(values):
        return None
    value = values[index]
    return value if math.isfinite(value) else None


def expand_curves_to_points(curves: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(curves.columns)
    point_columns = [
        *POINT_METADATA_COLUMNS,
        "point_index",
        "property_step10",
        "temperature_K",
        "temperature_original",
        "temperature_unit_original",
        "temperature_unit_note_step10",
        "temperature_outlier_flag_step10",
        "value_raw",
        "x_parse_status_step10",
        "y_parse_status_step10",
        "xy_length_check_step10",
        "x_point_count_step10",
        "y_point_count_step10",
        "valid_xy_point_count_step10",
        "point_valid_for_step10",
        "point_invalid_reason_step10",
    ]
    rows: list[list[Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for raw in curves.itertuples(index=False, name=None):
        row = dict(zip(columns, raw))
        property_name = classify_property(row)
        x_values, x_status = parse_numeric_values(row.get("x_values_json", ""))
        y_values, y_status = parse_numeric_values(row.get("y_values_json", ""))
        xy_check = parse_curve_status(x_values, x_status, y_values, y_status)
        valid_xy_count = len(x_values) if xy_check == "ok" else 0
        curve_rows.append(
            {
                "sample_key": row.get("sample_key", ""),
                "curve_id": row.get("curve_id", ""),
                "curve_key": row.get("curve_key", ""),
                "property_step10": property_name,
                "x_parse_status_step10": x_status,
                "y_parse_status_step10": y_status,
                "xy_length_check_step10": xy_check,
                "x_point_count_step10": len(x_values),
                "y_point_count_step10": len(y_values),
                "valid_xy_point_count_step10": valid_xy_count,
            }
        )
        if property_name not in TARGET_PROPERTIES:
            continue

        n_points = max(len(x_values), len(y_values))
        if n_points == 0:
            n_points = 1
        metadata = [row.get(column, "") for column in POINT_METADATA_COLUMNS]
        for point_index in range(n_points):
            original_temp = get_value(x_values, point_index)
            value_raw = get_value(y_values, point_index)
            temperature_k, unit_note = convert_temperature(original_temp, row.get("unit_x", ""))
            temperature_outlier = (
                temperature_k is not None and (temperature_k < 0 or temperature_k > 2000)
            )
            invalid_reason = point_invalid_reason(
                xy_check, temperature_k, temperature_outlier, value_raw
            )
            point_valid = invalid_reason == ""
            rows.append(
                [
                    *metadata,
                    point_index,
                    property_name,
                    temperature_k if temperature_k is not None else "",
                    original_temp if original_temp is not None else "",
                    row.get("unit_x", ""),
                    unit_note,
                    temperature_outlier,
                    value_raw if value_raw is not None else "",
                    x_status,
                    y_status,
                    xy_check,
                    len(x_values),
                    len(y_values),
                    valid_xy_count,
                    point_valid,
                    invalid_reason,
                ]
            )

    return pd.DataFrame(rows, columns=point_columns), pd.DataFrame(curve_rows)


def unique_join(values: pd.Series) -> str:
    seen: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.append(text)
    return ";".join(seen)


def aggregate_points(points: pd.DataFrame, temperature_round_decimals: int) -> pd.DataFrame:
    if points.empty:
        return pd.DataFrame(
            columns=[
                "sample_key",
                "property_step10",
                "temperature_K_rounded",
                "value_raw_median",
                "value_raw_mean",
                "value_raw_std",
                "value_raw_min",
                "value_raw_max",
                "source_point_count_step10",
                "source_curve_count_step10",
                "curve_ids_step10",
                "units_y_step10",
            ]
        )
    valid = points[
        points["point_valid_for_step10"].map(normalize_bool)
        & points["property_step10"].isin(TARGET_PROPERTIES)
    ].copy()
    valid["temperature_K"] = pd.to_numeric(valid["temperature_K"], errors="coerce")
    valid["value_raw"] = pd.to_numeric(valid["value_raw"], errors="coerce")
    valid = valid.dropna(subset=["temperature_K", "value_raw"])
    valid["temperature_K_rounded"] = valid["temperature_K"].round(temperature_round_decimals)
    grouped = valid.groupby(
        ["sample_key", "property_step10", "temperature_K_rounded"], sort=True
    )
    output = grouped.agg(
        value_raw_median=("value_raw", "median"),
        value_raw_mean=("value_raw", "mean"),
        value_raw_std=("value_raw", "std"),
        value_raw_min=("value_raw", "min"),
        value_raw_max=("value_raw", "max"),
        source_point_count_step10=("value_raw", "count"),
        source_curve_count_step10=("curve_id", pd.Series.nunique),
        curve_ids_step10=("curve_id", unique_join),
        units_y_step10=("unit_y", unique_join),
    ).reset_index()
    output["value_raw_std"] = output["value_raw_std"].fillna(0)
    return output


def add_sigma_rho_flags(points: pd.DataFrame) -> pd.DataFrame:
    output = points[points["property_step10"].isin([SIGMA_PROPERTY, RHO_PROPERTY])].copy()
    output["is_sigma_point_step10"] = output["property_step10"].eq(SIGMA_PROPERTY)
    output["is_rho_point_step10"] = output["property_step10"].eq(RHO_PROPERTY)
    output["is_sigma_or_rho_point_step10"] = True
    value_numeric = pd.to_numeric(output["value_raw"], errors="coerce")
    output["positive_value_for_log_fit_step10"] = value_numeric > 0
    output["usable_for_tau_fit_step10"] = (
        output["point_valid_for_step10"].map(normalize_bool)
        & output["positive_value_for_log_fit_step10"]
        & ~output["temperature_outlier_flag_step10"].map(normalize_bool)
    )
    output["not_usable_for_tau_fit_reason_step10"] = output.apply(
        lambda row: ""
        if normalize_bool(row["usable_for_tau_fit_step10"])
        else (
            normalize_text(row.get("point_invalid_reason_step10", ""))
            or ("nonpositive_value" if not normalize_bool(row["positive_value_for_log_fit_step10"]) else "")
            or "not_usable"
        ),
        axis=1,
    )
    return output


def determine_actual_source(group: pd.DataFrame) -> tuple[str, str]:
    usable = group[group["usable_for_tau_fit_step10"].map(normalize_bool)]
    sigma_count = int(usable["property_step10"].eq(SIGMA_PROPERTY).sum())
    rho_count = int(usable["property_step10"].eq(RHO_PROPERTY).sum())
    preference = first_nonempty(group["fitting_source_preference_step8"])

    if preference == SIGMA_PROPERTY and sigma_count >= 5:
        return SIGMA_PROPERTY, "use preferred Electrical conductivity"
    if preference == RHO_PROPERTY and rho_count >= 5:
        return RHO_PROPERTY, "use preferred Electrical resistivity"
    if preference == SIGMA_PROPERTY and rho_count >= 5:
        return RHO_PROPERTY, "fallback to Electrical resistivity"
    if preference == RHO_PROPERTY and sigma_count >= 5:
        return SIGMA_PROPERTY, "fallback to Electrical conductivity"
    if sigma_count >= 5:
        return SIGMA_PROPERTY, "fallback to Electrical conductivity"
    if rho_count >= 5:
        return RHO_PROPERTY, "fallback to Electrical resistivity"
    return "none", "not usable: no sigma/rho source with >=5 usable points"


def first_nonempty(values: pd.Series) -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return ""


def apply_actual_sources(sigma_rho_points: pd.DataFrame) -> pd.DataFrame:
    output = sigma_rho_points.copy()
    records: list[dict[str, str]] = []
    for sample_key, group in output.groupby("sample_key", sort=True):
        actual, reason = determine_actual_source(group)
        records.append(
            {
                "sample_key": sample_key,
                "fitting_source_actual_step10": actual,
                "fitting_source_actual_reason_step10": reason,
            }
        )
    source_df = pd.DataFrame(records)
    output = output.merge(source_df, on="sample_key", how="left")
    output["selected_for_tau_fit_step10"] = (
        output["usable_for_tau_fit_step10"].map(normalize_bool)
        & output["property_step10"].eq(output["fitting_source_actual_step10"])
    )
    return output


def build_sample_info(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        inputs["sample_annotations"],
        inputs["learning"],
        inputs["initial"],
        inputs["review"],
    ]
    sample_info = pd.concat(frames, ignore_index=True, sort=False)
    sample_info = sample_info.drop_duplicates("sample_key", keep="first").copy()
    for column in WIDE_METADATA_COLUMNS:
        if column not in sample_info.columns:
            sample_info[column] = ""
    return sample_info.set_index("sample_key")


def aggregate_sigma_rho_points_for_obs(
    sigma_rho_points: pd.DataFrame,
    temperature_round_decimals: int,
) -> pd.DataFrame:
    usable = sigma_rho_points[sigma_rho_points["usable_for_tau_fit_step10"].map(normalize_bool)].copy()
    return aggregate_points(usable, temperature_round_decimals)


def build_lookup(aggregated: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for (sample_key, property_name), group in aggregated.groupby(
        ["sample_key", "property_step10"], sort=False
    ):
        sorted_group = group.sort_values("temperature_K_rounded").reset_index(drop=True)
        lookup[(sample_key, property_name)] = sorted_group
    return lookup


def nearest_property_row(
    lookup: dict[tuple[str, str], pd.DataFrame],
    sample_key: str,
    property_name: str,
    anchor_temperature: float,
    tolerance: float,
) -> tuple[pd.Series | None, float | None]:
    frame = lookup.get((sample_key, property_name))
    if frame is None or frame.empty:
        return None, None
    temps = pd.to_numeric(frame["temperature_K_rounded"], errors="coerce").to_numpy(dtype=float)
    if temps.size == 0:
        return None, None
    deltas = np.abs(temps - anchor_temperature)
    index = int(np.nanargmin(deltas))
    delta = float(deltas[index])
    if not math.isfinite(delta) or delta > tolerance:
        return None, None
    return frame.iloc[index], delta


def get_sample_value(sample_info: pd.DataFrame, sample_key: str, column: str) -> str:
    if sample_key not in sample_info.index or column not in sample_info.columns:
        return ""
    value = sample_info.at[sample_key, column]
    if isinstance(value, pd.Series):
        return normalize_text(value.iloc[0])
    return normalize_text(value)


def build_anchor_points(
    sigma_rho_points: pd.DataFrame,
    temperature_round_decimals: int,
) -> pd.DataFrame:
    selected = sigma_rho_points[
        sigma_rho_points["selected_for_tau_fit_step10"].map(normalize_bool)
        & sigma_rho_points["usable_for_tau_fit_step10"].map(normalize_bool)
    ].copy()
    selected["temperature_K"] = pd.to_numeric(selected["temperature_K"], errors="coerce")
    selected["value_raw"] = pd.to_numeric(selected["value_raw"], errors="coerce")
    selected = selected.dropna(subset=["temperature_K", "value_raw"])
    selected["temperature_K_rounded"] = selected["temperature_K"].round(temperature_round_decimals)
    grouped = selected.groupby(
        ["sample_key", "property_step10", "temperature_K_rounded"], sort=True
    )
    anchors = grouped.agg(
        temperature_anchor_curve_ids_step10=("curve_id", unique_join),
        anchor_point_count_step10=("value_raw", "count"),
        fitting_source_actual_step10=("fitting_source_actual_step10", first_nonempty),
        fitting_source_actual_reason_step10=("fitting_source_actual_reason_step10", first_nonempty),
    ).reset_index()
    anchors = anchors.rename(
        columns={
            "property_step10": "temperature_anchor_property_step10",
            "temperature_K_rounded": "temperature_K",
        }
    )
    return anchors


def property_prefix(property_name: str) -> str:
    return {
        SIGMA_PROPERTY: "sigma",
        RHO_PROPERTY: "rho",
        SEEBECK_PROPERTY: "seebeck",
        KAPPA_PROPERTY: "kappa",
        ZT_PROPERTY: "zt",
    }[property_name]


def build_training_wide(
    anchors: pd.DataFrame,
    property_agg: pd.DataFrame,
    sigma_rho_agg: pd.DataFrame,
    sample_info: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    eval_lookup = build_lookup(property_agg)
    sigma_rho_lookup = build_lookup(sigma_rho_agg)
    records: list[dict[str, Any]] = []
    for row in anchors.itertuples(index=False):
        sample_key = getattr(row, "sample_key")
        anchor_temp = float(getattr(row, "temperature_K"))
        record: dict[str, Any] = {
            "sample_key": sample_key,
            "temperature_K": anchor_temp,
            "temperature_anchor_property_step10": getattr(
                row, "temperature_anchor_property_step10"
            ),
            "temperature_anchor_curve_ids_step10": getattr(
                row, "temperature_anchor_curve_ids_step10"
            ),
            "fitting_source_actual_step10": getattr(row, "fitting_source_actual_step10"),
            "fitting_source_actual_reason_step10": getattr(
                row, "fitting_source_actual_reason_step10"
            ),
        }
        for column in WIDE_METADATA_COLUMNS:
            if column in {
                "fitting_source_actual_step10",
                "fitting_source_actual_reason_step10",
            }:
                continue
            record[column] = get_sample_value(sample_info, sample_key, column)

        for property_name in TARGET_PROPERTIES:
            prefix = property_prefix(property_name)
            lookup = sigma_rho_lookup if property_name in {SIGMA_PROPERTY, RHO_PROPERTY} else eval_lookup
            nearest, delta = nearest_property_row(
                lookup, sample_key, property_name, anchor_temp, tolerance
            )
            if nearest is None:
                record[f"{prefix}_obs_raw"] = ""
                record[f"{prefix}_unit_raw"] = ""
                record[f"{prefix}_source_curve_ids_step10"] = ""
                record[f"{prefix}_temperature_delta_K_step10"] = ""
            else:
                record[f"{prefix}_obs_raw"] = nearest["value_raw_median"]
                record[f"{prefix}_unit_raw"] = nearest["units_y_step10"]
                record[f"{prefix}_source_curve_ids_step10"] = nearest["curve_ids_step10"]
                record[f"{prefix}_temperature_delta_K_step10"] = delta

        sigma_value = finite_float(record["sigma_obs_raw"])
        rho_value = finite_float(record["rho_obs_raw"])
        record["has_sigma_obs_step10"] = sigma_value is not None
        record["has_rho_obs_step10"] = rho_value is not None
        record["has_sigma_or_rho_obs_step10"] = (
            (sigma_value is not None and sigma_value > 0)
            or (rho_value is not None and rho_value > 0)
        )
        record["has_seebeck_obs_step10"] = finite_float(record["seebeck_obs_raw"]) is not None
        record["has_kappa_obs_step10"] = finite_float(record["kappa_obs_raw"]) is not None
        record["has_zt_obs_step10"] = finite_float(record["zt_obs_raw"]) is not None
        record["has_kappa_or_zt_obs_step10"] = (
            record["has_kappa_obs_step10"] or record["has_zt_obs_step10"]
        )
        record["usable_for_tau_fit_step10"] = (
            math.isfinite(anchor_temp) and record["has_sigma_or_rho_obs_step10"]
        )
        record["usable_for_pf_eval_step10"] = (
            record["usable_for_tau_fit_step10"] and record["has_seebeck_obs_step10"]
        )
        record["usable_for_zt_eval_step10"] = (
            record["usable_for_tau_fit_step10"]
            and record["has_seebeck_obs_step10"]
            and record["has_kappa_or_zt_obs_step10"]
        )
        if record["usable_for_zt_eval_step10"]:
            record["training_row_type_step10"] = "tau_fit_and_zt_eval"
        elif record["usable_for_pf_eval_step10"]:
            record["training_row_type_step10"] = "tau_fit_and_pf_eval"
        elif record["usable_for_tau_fit_step10"]:
            record["training_row_type_step10"] = "tau_fit_only"
        else:
            record["training_row_type_step10"] = "not_usable"
        records.append(record)
    return pd.DataFrame(records)


def mean_delta(group: pd.DataFrame, column: str) -> Any:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return "" if values.empty else float(values.mean())


def max_delta(group: pd.DataFrame, column: str) -> Any:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return "" if values.empty else float(values.max())


def alignment_quality(record: dict[str, Any]) -> tuple[str, str]:
    if record["n_zt_eval_rows_step10"] >= 5:
        return "good", "ZT evaluation rows >=5"
    if record["n_pf_eval_rows_step10"] > 0 or record["kappa_matched_rows_step10"] > 0 or record["zt_matched_rows_step10"] > 0:
        return "partial", "some evaluation properties matched"
    if (
        record["seebeck_matched_rows_step10"] == 0
        and record["kappa_matched_rows_step10"] == 0
        and record["zt_matched_rows_step10"] == 0
    ):
        return "no_eval_property", "sigma/rho only at anchor temperatures"
    return "poor", "few aligned evaluation properties"


def build_alignment_summary(wide: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in wide.groupby("sample_key", sort=True):
        n_rows = len(group)
        record = {
            "sample_key": sample_key,
            "n_training_rows_step10": n_rows,
            "n_tau_fit_rows_step10": int(group["usable_for_tau_fit_step10"].map(normalize_bool).sum()),
            "n_pf_eval_rows_step10": int(group["usable_for_pf_eval_step10"].map(normalize_bool).sum()),
            "n_zt_eval_rows_step10": int(group["usable_for_zt_eval_step10"].map(normalize_bool).sum()),
            "seebeck_matched_rows_step10": int(group["has_seebeck_obs_step10"].map(normalize_bool).sum()),
            "kappa_matched_rows_step10": int(group["has_kappa_obs_step10"].map(normalize_bool).sum()),
            "zt_matched_rows_step10": int(group["has_zt_obs_step10"].map(normalize_bool).sum()),
            "seebeck_missing_rows_step10": int(n_rows - group["has_seebeck_obs_step10"].map(normalize_bool).sum()),
            "kappa_missing_rows_step10": int(n_rows - group["has_kappa_obs_step10"].map(normalize_bool).sum()),
            "zt_missing_rows_step10": int(n_rows - group["has_zt_obs_step10"].map(normalize_bool).sum()),
            "seebeck_mean_temperature_delta_K_step10": mean_delta(
                group, "seebeck_temperature_delta_K_step10"
            ),
            "kappa_mean_temperature_delta_K_step10": mean_delta(
                group, "kappa_temperature_delta_K_step10"
            ),
            "zt_mean_temperature_delta_K_step10": mean_delta(
                group, "zt_temperature_delta_K_step10"
            ),
            "seebeck_max_temperature_delta_K_step10": max_delta(
                group, "seebeck_temperature_delta_K_step10"
            ),
            "kappa_max_temperature_delta_K_step10": max_delta(
                group, "kappa_temperature_delta_K_step10"
            ),
            "zt_max_temperature_delta_K_step10": max_delta(
                group, "zt_temperature_delta_K_step10"
            ),
            "temperature_tolerance_K_step10": tolerance,
        }
        quality, note = alignment_quality(record)
        record["alignment_quality_step10"] = quality
        record["alignment_note_step10"] = note
        records.append(record)
    return pd.DataFrame(records)


def sintering_invalid_rows(df: pd.DataFrame) -> int:
    total = 0
    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if column not in df.columns:
            total += len(df)
        else:
            total += int(df[column].map(lambda value: normalize_text(value).casefold()).ne(expected).sum())
    return total


def value_counts_rows(prefix: str, series: pd.Series) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for value, count in series.fillna("").astype(str).value_counts(dropna=False).sort_index().items():
        rows.append((f"{prefix}_{value}_count", str(int(count))))
    return rows


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(normalize_bool).sum())


def build_report(
    inputs: dict[str, pd.DataFrame],
    curve_summary: pd.DataFrame,
    property_points_long: pd.DataFrame,
    property_points_aggregated: pd.DataFrame,
    duplicate_points: pd.DataFrame,
    sigma_rho_points: pd.DataFrame,
    wide: pd.DataFrame,
    initial_training: pd.DataFrame,
    review_training: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    tolerance: float,
    round_decimals: int,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = [
        ("input_candidate_core_curves_step9_rows", str(len(inputs["candidate_curves"]))),
        ("input_sigma_rho_curves_for_fitting_step9_rows", str(len(inputs["sigma_rho_curves"]))),
        ("input_learning_candidates_step9_rows", str(len(inputs["learning"]))),
        ("input_initial_tau_fit_candidates_step9_rows", str(len(inputs["initial"]))),
        ("input_review_needed_candidates_step9_rows", str(len(inputs["review"]))),
        ("property_points_long_step10_rows", str(len(property_points_long))),
        ("property_points_aggregated_step10_rows", str(len(property_points_aggregated))),
        ("duplicate_property_points_step10_rows", str(len(duplicate_points))),
        ("sigma_rho_points_for_fitting_step10_rows", str(len(sigma_rho_points))),
        ("training_dataset_wide_step10_rows", str(len(wide))),
        ("initial_tau_fit_training_dataset_step10_rows", str(len(initial_training))),
        ("review_training_dataset_step10_rows", str(len(review_training))),
        ("temperature_alignment_summary_step10_rows", str(len(alignment_summary))),
    ]

    if not property_points_long.empty:
        point_summary = (
            property_points_long.groupby("property_step10", sort=True)
            .agg(
                point_count=("sample_key", "count"),
                valid_point_count=(
                    "point_valid_for_step10",
                    lambda values: int(pd.Series(values).map(normalize_bool).sum()),
                ),
            )
            .reset_index()
        )
        curve_counts = curve_summary["property_step10"].value_counts(dropna=False)
        for row in point_summary.itertuples(index=False):
            rows.append(
                (
                    f"property_step10_{row.property_step10}_curve_count",
                    str(int(curve_counts.get(row.property_step10, 0))),
                )
            )
            rows.append((f"property_step10_{row.property_step10}_point_count", str(int(row.point_count))))
            rows.append(
                (
                    f"property_step10_{row.property_step10}_valid_point_count",
                    str(int(row.valid_point_count)),
                )
            )

    unclassified_curve_count = int(curve_summary["property_step10"].fillna("").astype(str).eq("").sum())
    rows.append(("unclassified_property_curve_count", str(unclassified_curve_count)))
    rows.extend(value_counts_rows("x_parse_status_step10", curve_summary["x_parse_status_step10"]))
    rows.extend(value_counts_rows("y_parse_status_step10", curve_summary["y_parse_status_step10"]))
    rows.extend(value_counts_rows("xy_length_check_step10", curve_summary["xy_length_check_step10"]))
    rows.extend(
        value_counts_rows(
            "temperature_unit_note_step10", property_points_long["temperature_unit_note_step10"]
        )
    )
    rows.extend(
        [
            (
                "temperature_outlier_flag_step10_true_point_count",
                str(bool_count(property_points_long, "temperature_outlier_flag_step10")),
            ),
            (
                "point_valid_for_step10_true_point_count",
                str(bool_count(property_points_long, "point_valid_for_step10")),
            ),
            (
                "point_valid_for_step10_false_point_count",
                str(len(property_points_long) - bool_count(property_points_long, "point_valid_for_step10")),
            ),
            ("sigma_point_count", str(int(sigma_rho_points["is_sigma_point_step10"].sum()))),
            ("rho_point_count", str(int(sigma_rho_points["is_rho_point_step10"].sum()))),
            (
                "usable_for_tau_fit_step10_true_point_count",
                str(bool_count(sigma_rho_points, "usable_for_tau_fit_step10")),
            ),
            (
                "selected_for_tau_fit_step10_true_point_count",
                str(bool_count(sigma_rho_points, "selected_for_tau_fit_step10")),
            ),
            (
                "training_dataset_wide_step10_usable_for_tau_fit_true_rows",
                str(bool_count(wide, "usable_for_tau_fit_step10")),
            ),
            (
                "training_dataset_wide_step10_usable_for_pf_eval_true_rows",
                str(bool_count(wide, "usable_for_pf_eval_step10")),
            ),
            (
                "training_dataset_wide_step10_usable_for_zt_eval_true_rows",
                str(bool_count(wide, "usable_for_zt_eval_step10")),
            ),
            (
                "initial_tau_fit_training_dataset_step10_sample_key_count",
                str(initial_training["sample_key"].nunique() if "sample_key" in initial_training else 0),
            ),
            ("initial_tau_fit_training_dataset_step10_rows", str(len(initial_training))),
            (
                "review_training_dataset_step10_sample_key_count",
                str(review_training["sample_key"].nunique() if "sample_key" in review_training else 0),
            ),
            ("review_training_dataset_step10_rows", str(len(review_training))),
        ]
    )

    if "n_or_p" in wide.columns:
        sample_np = wide.drop_duplicates("sample_key")
        rows.extend(value_counts_rows("training_sample_n_or_p", sample_np["n_or_p"]))
        rows.extend(value_counts_rows("training_row_n_or_p", wide["n_or_p"]))
    if "fitting_source_actual_step10" in wide.columns:
        source_sample = wide.drop_duplicates("sample_key")
        rows.extend(
            value_counts_rows(
                "fitting_source_actual_step10_sample", source_sample["fitting_source_actual_step10"]
            )
        )
        rows.extend(
            value_counts_rows("fitting_source_actual_step10_row", wide["fitting_source_actual_step10"])
        )

    rows.extend(
        [
            ("seebeck_matched_rows", str(bool_count(wide, "has_seebeck_obs_step10"))),
            ("thermal_conductivity_matched_rows", str(bool_count(wide, "has_kappa_obs_step10"))),
            ("zt_matched_rows", str(bool_count(wide, "has_zt_obs_step10"))),
            ("temperature_tolerance_K", str(tolerance)),
            ("temperature_round_decimals", str(round_decimals)),
            (
                "sintering_method_unknown_rows",
                str(int(wide["sintering_method"].map(lambda value: normalize_text(value).casefold()).eq("unknown").sum()))
                if "sintering_method" in wide.columns
                else "0",
            ),
            (
                "sintering_checked_no_rows",
                str(int(wide["sintering_checked"].map(lambda value: normalize_text(value).casefold()).eq("no").sum()))
                if "sintering_checked" in wide.columns
                else "0",
            ),
            (
                "record_checked_no_rows",
                str(int(wide["record_checked"].map(lambda value: normalize_text(value).casefold()).eq("no").sum()))
                if "record_checked" in wide.columns
                else "0",
            ),
        ]
    )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    return report_text, report_df


def add_excel_preview_note(sheet_name: str, row_count: int, excel_notes: list[str]) -> None:
    if row_count <= EXCEL_PREVIEW_ROWS:
        return
    note = (
        f"{sheet_name} has {row_count} rows; wrote first {EXCEL_PREVIEW_ROWS} "
        "rows to workbook; full data is in CSV"
    )
    if note not in excel_notes:
        excel_notes.append(note)


def excel_frame(df: pd.DataFrame, sheet_name: str, excel_notes: list[str]) -> pd.DataFrame:
    add_excel_preview_note(sheet_name, len(df), excel_notes)
    if len(df) <= EXCEL_PREVIEW_ROWS:
        return df
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
    wide: pd.DataFrame,
    initial_training: pd.DataFrame,
    sigma_rho_points: pd.DataFrame,
    property_points_aggregated: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    duplicate_points: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    sheets = {
        "training_dataset_wide": excel_frame(wide, "training_dataset_wide", excel_notes),
        "initial_tau_fit_training": excel_frame(
            initial_training, "initial_tau_fit_training", excel_notes
        ),
        "sigma_rho_points_for_fitting": excel_frame(
            sigma_rho_points, "sigma_rho_points_for_fitting", excel_notes
        ),
        "property_points_aggregated": excel_frame(
            property_points_aggregated, "property_points_aggregated", excel_notes
        ),
        "temperature_alignment_summary": excel_frame(
            alignment_summary, "temperature_alignment_summary", excel_notes
        ),
        "duplicate_property_points": excel_frame(
            duplicate_points, "duplicate_property_points", excel_notes
        ),
        "dataset_report": report_df,
    }
    path = output_dir / "starrydata2_step10_training_dataset.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def assert_acceptance(
    property_points_long: pd.DataFrame,
    property_points_aggregated: pd.DataFrame,
    sigma_rho_points: pd.DataFrame,
    wide: pd.DataFrame,
    initial_training: pd.DataFrame,
) -> None:
    for column in [
        "sample_key",
        "property_step10",
        "temperature_K",
        "value_raw",
        "point_valid_for_step10",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in property_points_long.columns:
            raise KeyError(f"property_points_long_step10 missing {column}")
    if sintering_invalid_rows(property_points_long):
        raise ValueError("property_points_long_step10 has non-standard sintering values")

    duplicated_agg = property_points_aggregated.duplicated(
        ["sample_key", "property_step10", "temperature_K_rounded"]
    )
    if duplicated_agg.any():
        raise ValueError("property_points_aggregated_step10 has duplicate aggregation keys")
    if "value_raw_median" not in property_points_aggregated.columns:
        raise KeyError("property_points_aggregated_step10 missing value_raw_median")

    if not sigma_rho_points["property_step10"].isin([SIGMA_PROPERTY, RHO_PROPERTY]).all():
        raise ValueError("sigma_rho_points_for_fitting_step10 has non sigma/rho properties")
    for column in ["usable_for_tau_fit_step10", "selected_for_tau_fit_step10"]:
        if column not in sigma_rho_points.columns:
            raise KeyError(f"sigma_rho_points_for_fitting_step10 missing {column}")

    duplicated_wide = wide.duplicated(["sample_key", "temperature_K"])
    if duplicated_wide.any():
        raise ValueError("training_dataset_wide_step10 is not one row per sample_key x temperature")
    for column in [
        "temperature_K",
        "usable_for_tau_fit_step10",
        "usable_for_pf_eval_step10",
        "usable_for_zt_eval_step10",
        "n_or_p",
        "additive_auto_step9",
        "structure_auto_step9",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in wide.columns:
            raise KeyError(f"training_dataset_wide_step10 missing {column}")
    if sintering_invalid_rows(wide):
        raise ValueError("training_dataset_wide_step10 has non-standard sintering values")
    if not initial_training.empty:
        if not initial_training["is_initial_tau_fit_candidate_step8"].map(normalize_bool).all():
            raise ValueError("initial_tau_fit_training_dataset_step10 has non-initial samples")
        if not initial_training["usable_for_tau_fit_step10"].map(normalize_bool).all():
            raise ValueError("initial_tau_fit_training_dataset_step10 has unusable rows")


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def write_csv_outputs(
    output_dir: Path,
    property_points_long: pd.DataFrame,
    property_points_aggregated: pd.DataFrame,
    sigma_rho_points: pd.DataFrame,
    wide: pd.DataFrame,
    initial_training: pd.DataFrame,
    review_training: pd.DataFrame,
    duplicate_points: pd.DataFrame,
    alignment_summary: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_frame(property_points_long).to_csv(output_dir / "property_points_long_step10.csv", index=False)
    csv_frame(property_points_aggregated).to_csv(
        output_dir / "property_points_aggregated_step10.csv", index=False
    )
    csv_frame(sigma_rho_points).to_csv(
        output_dir / "sigma_rho_points_for_fitting_step10.csv", index=False
    )
    csv_frame(wide).to_csv(output_dir / "training_dataset_wide_step10.csv", index=False)
    csv_frame(initial_training).to_csv(
        output_dir / "initial_tau_fit_training_dataset_step10.csv", index=False
    )
    csv_frame(review_training).to_csv(output_dir / "review_training_dataset_step10.csv", index=False)
    csv_frame(duplicate_points).to_csv(
        output_dir / "duplicate_property_points_step10.csv", index=False
    )
    csv_frame(alignment_summary).to_csv(
        output_dir / "temperature_alignment_summary_step10.csv", index=False
    )
    (output_dir / "step10_training_dataset_report.txt").write_text(
        report_text, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    inputs = {
        label: read_csv_text(args.step9_dir / filename) for label, filename in INPUT_FILES.items()
    }
    validate_inputs(inputs)

    property_points_long, curve_summary = expand_curves_to_points(inputs["candidate_curves"])
    property_points_aggregated = aggregate_points(
        property_points_long, args.temperature_round_decimals
    )
    duplicate_points = property_points_aggregated[
        pd.to_numeric(property_points_aggregated["source_point_count_step10"], errors="coerce").fillna(0)
        > 1
    ].copy()

    sigma_rho_points_raw, sigma_rho_curve_summary = expand_curves_to_points(
        inputs["sigma_rho_curves"]
    )
    sigma_rho_points = add_sigma_rho_flags(sigma_rho_points_raw)
    sigma_rho_points = apply_actual_sources(sigma_rho_points)

    sigma_rho_agg = aggregate_sigma_rho_points_for_obs(
        sigma_rho_points, args.temperature_round_decimals
    )
    anchors = build_anchor_points(sigma_rho_points, args.temperature_round_decimals)
    sample_info = build_sample_info(inputs)
    wide = build_training_wide(
        anchors,
        property_points_aggregated,
        sigma_rho_agg,
        sample_info,
        args.temperature_tolerance_K,
    )

    review_keys = set(inputs["review"]["sample_key"])
    initial_training = wide[
        wide["is_initial_tau_fit_candidate_step8"].map(normalize_bool)
        & wide["usable_for_tau_fit_step10"].map(normalize_bool)
    ].copy()
    review_training = wide[
        wide["sample_key"].isin(review_keys)
        | (
            wide["is_tau_fitting_candidate_step8"].map(normalize_bool)
            & ~wide["is_initial_tau_fit_candidate_step8"].map(normalize_bool)
        )
    ].copy()
    alignment_summary = build_alignment_summary(wide, args.temperature_tolerance_K)

    # Include sigma/rho parse outcomes in the report as well as candidate-core outcomes.
    combined_curve_summary = pd.concat([curve_summary, sigma_rho_curve_summary], ignore_index=True)

    excel_notes: list[str] = []
    for sheet_name, frame in [
        ("training_dataset_wide", wide),
        ("initial_tau_fit_training", initial_training),
        ("sigma_rho_points_for_fitting", sigma_rho_points),
        ("property_points_aggregated", property_points_aggregated),
        ("temperature_alignment_summary", alignment_summary),
        ("duplicate_property_points", duplicate_points),
    ]:
        add_excel_preview_note(sheet_name, len(frame), excel_notes)

    report_text, report_df = build_report(
        inputs,
        combined_curve_summary,
        property_points_long,
        property_points_aggregated,
        duplicate_points,
        sigma_rho_points,
        wide,
        initial_training,
        review_training,
        alignment_summary,
        args.temperature_tolerance_K,
        args.temperature_round_decimals,
        excel_notes,
    )

    assert_acceptance(
        property_points_long,
        property_points_aggregated,
        sigma_rho_points,
        wide,
        initial_training,
    )

    write_csv_outputs(
        args.output_dir,
        property_points_long,
        property_points_aggregated,
        sigma_rho_points,
        wide,
        initial_training,
        review_training,
        duplicate_points,
        alignment_summary,
        report_text,
    )
    write_excel_output(
        args.output_dir,
        wide,
        initial_training,
        sigma_rho_points,
        property_points_aggregated,
        alignment_summary,
        duplicate_points,
        report_df,
        excel_notes,
    )
    if excel_notes:
        report_text, report_df = build_report(
            inputs,
            combined_curve_summary,
            property_points_long,
            property_points_aggregated,
            duplicate_points,
            sigma_rho_points,
            wide,
            initial_training,
            review_training,
            alignment_summary,
            args.temperature_tolerance_K,
            args.temperature_round_decimals,
            excel_notes,
        )
        (args.output_dir / "step10_training_dataset_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    parse_failed_curves = int(
        combined_curve_summary["xy_length_check_step10"].isin(
            ["parse_failed", "x_parse_failed", "y_parse_failed"]
        ).sum()
    )
    xy_mismatch_curves = int(
        combined_curve_summary["xy_length_check_step10"].eq("x_y_length_mismatch").sum()
    )

    print("Done.")
    print("Created:")
    print("- property_points_long_step10.csv")
    print("- property_points_aggregated_step10.csv")
    print("- sigma_rho_points_for_fitting_step10.csv")
    print("- training_dataset_wide_step10.csv")
    print("- initial_tau_fit_training_dataset_step10.csv")
    print("- review_training_dataset_step10.csv")
    print("- duplicate_property_points_step10.csv")
    print("- temperature_alignment_summary_step10.csv")
    print("- step10_training_dataset_report.txt")
    print("- starrydata2_step10_training_dataset.xlsx")
    print("")
    print("Summary:")
    print(f"property points long rows: {len(property_points_long)}")
    print(f"property points aggregated rows: {len(property_points_aggregated)}")
    print(f"training wide rows: {len(wide)}")
    print(f"initial tau fit training rows: {len(initial_training)}")
    print(f"review training rows: {len(review_training)}")
    print(f"usable tau fit rows: {bool_count(wide, 'usable_for_tau_fit_step10')}")
    print(f"usable PF eval rows: {bool_count(wide, 'usable_for_pf_eval_step10')}")
    print(f"usable ZT eval rows: {bool_count(wide, 'usable_for_zt_eval_step10')}")
    print(f"sigma points: {int(sigma_rho_points['is_sigma_point_step10'].sum())}")
    print(f"rho points: {int(sigma_rho_points['is_rho_point_step10'].sum())}")
    print(f"selected tau fit points: {bool_count(sigma_rho_points, 'selected_for_tau_fit_step10')}")
    print(f"samples in training wide: {wide['sample_key'].nunique() if 'sample_key' in wide else 0}")
    print(f"temperature tolerance K: {args.temperature_tolerance_K}")
    print(f"x/y parse failed curves: {parse_failed_curves}")
    print(f"x/y mismatch curves: {xy_mismatch_curves}")


if __name__ == "__main__":
    main()
