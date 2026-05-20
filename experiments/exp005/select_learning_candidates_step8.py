import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP7_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step7_sintering_unknown"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step8_learning_candidates"

SAMPLE_AVAILABILITY_FILE = "sample_property_availability_step7.csv"
SAMPLE_CLASSIFICATION_FILE = "sample_np_classification_step7.csv"
CANDIDATE_SAMPLES_FILE = "candidate_samples_np_step7.csv"
PROPERTY_CORE_FILE = "property_core_curves_step7.csv"
CANDIDATE_CORE_FILE = "candidate_core_curves_step7.csv"

PROPERTY_SOURCE_COLUMNS = [
    "property_step5",
    "property",
    "property_family",
    "prop_y_canonical",
    "prop_y",
    "prop_y_raw",
]
TARGET_PROPERTIES = {
    "Electrical conductivity",
    "Electrical resistivity",
    "Seebeck coefficient",
    "Thermal conductivity",
    "ZT",
}
SIGMA_PROPERTY = "Electrical conductivity"
RHO_PROPERTY = "Electrical resistivity"
SEEBECK_PROPERTY = "Seebeck coefficient"
KAPPA_PROPERTY = "Thermal conductivity"
ZT_PROPERTY = "ZT"

SAMPLE_CANDIDATE_COLUMNS = [
    "has_valid_sigma_step8",
    "has_valid_rho_step8",
    "has_valid_sigma_or_rho_step8",
    "sigma_curve_count_step8",
    "rho_curve_count_step8",
    "sigma_or_rho_curve_count_step8",
    "sigma_point_count_step8",
    "rho_point_count_step8",
    "sigma_or_rho_point_count_step8",
    "valid_sigma_point_count_step8",
    "valid_rho_point_count_step8",
    "valid_sigma_or_rho_point_count_step8",
    "nonpositive_sigma_point_count_step8",
    "nonpositive_rho_point_count_step8",
    "nonpositive_sigma_or_rho_point_count_step8",
    "sigma_temperature_min_step8",
    "sigma_temperature_max_step8",
    "rho_temperature_min_step8",
    "rho_temperature_max_step8",
    "sigma_or_rho_temperature_min_step8",
    "sigma_or_rho_temperature_max_step8",
    "sigma_or_rho_temperature_span_step8",
    "sigma_rho_parse_failed_curve_count_step8",
    "sigma_rho_xy_mismatch_curve_count_step8",
    "has_seebeck_step8",
    "has_thermal_conductivity_step8",
    "has_zt_step8",
    "has_kappa_or_zt_step8",
    "seebeck_curve_count_step8",
    "thermal_conductivity_curve_count_step8",
    "zt_curve_count_step8",
    "seebeck_point_count_step8",
    "thermal_conductivity_point_count_step8",
    "zt_point_count_step8",
    "kappa_or_zt_point_count_step8",
    "is_tau_fitting_candidate_step8",
    "is_full_learning_candidate_step8",
    "is_initial_tau_fit_candidate_step8",
    "candidate_priority_tier_step8",
    "fitting_source_preference_step8",
    "fitting_source_reason_step8",
    "learning_candidate_reason_step8",
]

EXCEL_MAX_ROWS = 1_048_576
EXCEL_PREVIEW_ROWS = 100_000
NUMERIC_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select Step8 learning candidates from Step7 Starrydata2 outputs."
    )
    parser.add_argument("--step7_dir", type=Path, default=DEFAULT_STEP7_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def validate_inputs(sample_df: pd.DataFrame, property_df: pd.DataFrame) -> None:
    if "sample_key" not in sample_df.columns:
        raise KeyError(f"{SAMPLE_AVAILABILITY_FILE} missing required column: sample_key")

    missing_property = [
        column
        for column in ["sample_key", "x_values_json", "y_values_json"]
        if column not in property_df.columns
    ]
    if missing_property:
        raise KeyError(f"{PROPERTY_CORE_FILE} missing required columns: {missing_property}")
    if not any(column in property_df.columns for column in PROPERTY_SOURCE_COLUMNS):
        raise KeyError(
            f"{PROPERTY_CORE_FILE} needs at least one property source column: "
            f"{PROPERTY_SOURCE_COLUMNS}"
        )


def validate_sample_key(df: pd.DataFrame, filename: str) -> None:
    if "sample_key" not in df.columns:
        raise KeyError(f"{filename} missing required column: sample_key")


def classify_property_text(text: str) -> str:
    normalized = compact_text(text)
    if not normalized:
        return ""
    if "power factor" in normalized:
        return ""

    if (
        normalized == "zt"
        or "dimensionless figure of merit" in normalized
        or "figure of merit" in normalized
    ):
        return ZT_PROPERTY

    if (
        "seebeck" in normalized
        or "thermopower" in normalized
        or "thermoelectric power" in normalized
    ):
        return SEEBECK_PROPERTY

    if (
        "electrical_resistivity" in normalized
        or "electrical resistivity" in normalized
        or "electric resistivity" in normalized
        or re.search(r"(^|[^a-z])rho([^a-z]|$)", normalized)
        or "\u03c1" in normalized
    ):
        return RHO_PROPERTY
    if "resistivity" in normalized and "thermal" not in normalized:
        return RHO_PROPERTY

    if (
        "thermal_conductivity" in normalized
        or "thermal conductivity" in normalized
        or "total thermal conductivity" in normalized
    ):
        return KAPPA_PROPERTY

    if (
        "electrical_conductivity" in normalized
        or "electrical conductivity" in normalized
        or "electric conductivity" in normalized
        or re.search(r"(^|[^a-z])sigma([^a-z]|$)", normalized)
        or "\u03c3" in normalized
    ):
        return SIGMA_PROPERTY
    if "conductivity" in normalized and "thermal" not in normalized:
        return SIGMA_PROPERTY

    return ""


def classify_property_row(row: pd.Series) -> str:
    for column in PROPERTY_SOURCE_COLUMNS:
        if column not in row.index:
            continue
        property_name = classify_property_text(row[column])
        if property_name:
            return property_name
    return ""


def temperature_values_to_kelvin(values: list[float], unit_x: Any) -> tuple[list[float], str]:
    unit = normalize_text(unit_x).casefold().replace(" ", "")
    if unit in {"k", "kelvin"}:
        return values, "ok: unit_x is K"
    if unit in {"c", "celsius", "degc", "degreec", "degreesc", "\u00b0c"}:
        return [value + 273.15 for value in values], "converted Celsius to K"
    if not unit:
        return values, "unit_x missing; temperature values used as-is"
    return values, f"unit_x={normalize_text(unit_x)}; temperature values used as-is"


def parse_curve_metrics(row: pd.Series) -> dict[str, Any]:
    x_values_raw, x_status = parse_numeric_values(row.get("x_values_json", ""))
    y_values, y_status = parse_numeric_values(row.get("y_values_json", ""))
    x_values, unit_note = temperature_values_to_kelvin(x_values_raw, row.get("unit_x", ""))

    if x_status != "ok" and y_status != "ok":
        xy_check = "parse_failed"
    elif x_status != "ok":
        xy_check = "x_parse_failed"
    elif y_status != "ok":
        xy_check = "y_parse_failed"
    elif len(x_values) != len(y_values):
        xy_check = "x_y_length_mismatch"
    else:
        xy_check = "ok"

    valid_xy_count = 0
    valid_positive_count = 0
    nonpositive_count = 0
    if xy_check == "ok":
        for x_value, y_value in zip(x_values, y_values):
            if math.isfinite(x_value) and math.isfinite(y_value):
                valid_xy_count += 1
                if y_value > 0:
                    valid_positive_count += 1
                else:
                    nonpositive_count += 1

    finite_x = [value for value in x_values if math.isfinite(value)]
    if finite_x:
        temp_min: Any = min(finite_x)
        temp_max: Any = max(finite_x)
        temp_span: Any = temp_max - temp_min
    else:
        temp_min = ""
        temp_max = ""
        temp_span = ""

    return {
        "x_parse_status_step8": x_status,
        "y_parse_status_step8": y_status,
        "xy_length_check_step8": xy_check,
        "x_point_count_step8": len(x_values),
        "y_point_count_step8": len(y_values),
        "valid_xy_point_count_step8": valid_xy_count,
        "valid_positive_y_point_count_step8": valid_positive_count,
        "nonpositive_y_point_count_step8": nonpositive_count,
        "temperature_min_step8": temp_min,
        "temperature_max_step8": temp_max,
        "temperature_span_step8": temp_span,
        "temperature_unit_note_step8": unit_note,
        "point_count_for_summary_step8": valid_xy_count
        if valid_xy_count > 0
        else max(len(x_values), len(y_values)),
    }


def add_curve_step8_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["property_step8"] = output.apply(classify_property_row, axis=1)
    output["is_sigma_curve_step8"] = output["property_step8"].eq(SIGMA_PROPERTY)
    output["is_rho_curve_step8"] = output["property_step8"].eq(RHO_PROPERTY)
    output["is_sigma_or_rho_curve_step8"] = output["property_step8"].isin(
        [SIGMA_PROPERTY, RHO_PROPERTY]
    )
    output["is_seebeck_curve_step8"] = output["property_step8"].eq(SEEBECK_PROPERTY)
    output["is_kappa_or_zt_curve_step8"] = output["property_step8"].isin(
        [KAPPA_PROPERTY, ZT_PROPERTY]
    )

    metrics = output.apply(parse_curve_metrics, axis=1, result_type="expand")
    return pd.concat([output, metrics], axis=1)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(0)


def min_or_blank(values: pd.Series) -> Any:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return float(numeric.min())


def max_or_blank(values: pd.Series) -> Any:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return ""
    return float(numeric.max())


def span_or_blank(min_value: Any, max_value: Any) -> Any:
    min_numeric = finite_float(min_value)
    max_numeric = finite_float(max_value)
    if min_numeric is None or max_numeric is None:
        return ""
    return max_numeric - min_numeric


def count_parse_failed(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame["xy_length_check_step8"].isin(["parse_failed", "x_parse_failed", "y_parse_failed"]).sum())


def count_xy_mismatch(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return int(frame["xy_length_check_step8"].eq("x_y_length_mismatch").sum())


def build_empty_sample_summary(sample_keys: pd.Series) -> pd.DataFrame:
    output = pd.DataFrame({"sample_key": sample_keys})
    numeric_columns = [
        column
        for column in SAMPLE_CANDIDATE_COLUMNS
        if column.endswith("_count_step8")
        or column.endswith("_min_step8")
        or column.endswith("_max_step8")
        or column.endswith("_span_step8")
    ]
    bool_columns = [column for column in SAMPLE_CANDIDATE_COLUMNS if column.startswith("has_")]
    for column in numeric_columns:
        output[column] = 0
    for column in bool_columns:
        output[column] = False
    return output


def build_sample_summary(property_curves: pd.DataFrame, sample_keys: pd.Series) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for sample_key, group in property_curves.groupby("sample_key", sort=True):
        sigma = group[group["is_sigma_curve_step8"]]
        rho = group[group["is_rho_curve_step8"]]
        sigma_rho = group[group["is_sigma_or_rho_curve_step8"]]
        seebeck = group[group["is_seebeck_curve_step8"]]
        kappa = group[group["property_step8"].eq(KAPPA_PROPERTY)]
        zt = group[group["property_step8"].eq(ZT_PROPERTY)]

        sigma_min = min_or_blank(sigma["temperature_min_step8"])
        sigma_max = max_or_blank(sigma["temperature_max_step8"])
        rho_min = min_or_blank(rho["temperature_min_step8"])
        rho_max = max_or_blank(rho["temperature_max_step8"])
        sigma_rho_min = min_or_blank(sigma_rho["temperature_min_step8"])
        sigma_rho_max = max_or_blank(sigma_rho["temperature_max_step8"])

        valid_sigma = int(numeric_series(sigma, "valid_positive_y_point_count_step8").sum())
        valid_rho = int(numeric_series(rho, "valid_positive_y_point_count_step8").sum())
        valid_sigma_rho = valid_sigma + valid_rho

        record = {
            "sample_key": sample_key,
            "sigma_curve_count_step8": len(sigma),
            "rho_curve_count_step8": len(rho),
            "sigma_or_rho_curve_count_step8": len(sigma_rho),
            "sigma_point_count_step8": int(numeric_series(sigma, "point_count_for_summary_step8").sum()),
            "rho_point_count_step8": int(numeric_series(rho, "point_count_for_summary_step8").sum()),
            "sigma_or_rho_point_count_step8": int(
                numeric_series(sigma_rho, "point_count_for_summary_step8").sum()
            ),
            "valid_sigma_point_count_step8": valid_sigma,
            "valid_rho_point_count_step8": valid_rho,
            "valid_sigma_or_rho_point_count_step8": valid_sigma_rho,
            "nonpositive_sigma_point_count_step8": int(
                numeric_series(sigma, "nonpositive_y_point_count_step8").sum()
            ),
            "nonpositive_rho_point_count_step8": int(
                numeric_series(rho, "nonpositive_y_point_count_step8").sum()
            ),
            "nonpositive_sigma_or_rho_point_count_step8": int(
                numeric_series(sigma_rho, "nonpositive_y_point_count_step8").sum()
            ),
            "sigma_temperature_min_step8": sigma_min,
            "sigma_temperature_max_step8": sigma_max,
            "rho_temperature_min_step8": rho_min,
            "rho_temperature_max_step8": rho_max,
            "sigma_or_rho_temperature_min_step8": sigma_rho_min,
            "sigma_or_rho_temperature_max_step8": sigma_rho_max,
            "sigma_or_rho_temperature_span_step8": span_or_blank(sigma_rho_min, sigma_rho_max),
            "sigma_rho_parse_failed_curve_count_step8": count_parse_failed(sigma_rho),
            "sigma_rho_xy_mismatch_curve_count_step8": count_xy_mismatch(sigma_rho),
            "has_valid_sigma_step8": valid_sigma > 0,
            "has_valid_rho_step8": valid_rho > 0,
            "has_valid_sigma_or_rho_step8": valid_sigma_rho > 0,
            "has_seebeck_step8": len(seebeck) > 0,
            "has_thermal_conductivity_step8": len(kappa) > 0,
            "has_zt_step8": len(zt) > 0,
            "has_kappa_or_zt_step8": len(kappa) > 0 or len(zt) > 0,
            "seebeck_curve_count_step8": len(seebeck),
            "thermal_conductivity_curve_count_step8": len(kappa),
            "zt_curve_count_step8": len(zt),
            "seebeck_point_count_step8": int(
                numeric_series(seebeck, "point_count_for_summary_step8").sum()
            ),
            "thermal_conductivity_point_count_step8": int(
                numeric_series(kappa, "point_count_for_summary_step8").sum()
            ),
            "zt_point_count_step8": int(numeric_series(zt, "point_count_for_summary_step8").sum()),
            "kappa_or_zt_point_count_step8": int(
                numeric_series(group[group["property_step8"].isin([KAPPA_PROPERTY, ZT_PROPERTY])], "point_count_for_summary_step8").sum()
            ),
        }
        records.append(record)

    summary = build_empty_sample_summary(sample_keys)
    if records:
        summary = summary.drop(columns=[column for column in summary.columns if column != "sample_key"])
        summary = summary.merge(pd.DataFrame(records), on="sample_key", how="left")
    return summary


def fill_sample_summary_defaults(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    count_columns = [
        column
        for column in output.columns
        if column.endswith("_count_step8")
        or column.endswith("_min_step8")
        or column.endswith("_max_step8")
        or column.endswith("_span_step8")
    ]
    for column in count_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0)
    bool_columns = [column for column in output.columns if column.startswith("has_") and column.endswith("_step8")]
    for column in bool_columns:
        output[column] = output[column].map(normalize_bool)
    return output


def fitting_source_preference(row: pd.Series) -> tuple[str, str]:
    valid_sigma = int(row.get("valid_sigma_point_count_step8", 0))
    valid_rho = int(row.get("valid_rho_point_count_step8", 0))
    if valid_sigma >= 5:
        return (
            SIGMA_PROPERTY,
            "use Electrical conductivity: valid sigma points >= 5",
        )
    if valid_rho >= 5:
        return (
            RHO_PROPERTY,
            "use Electrical resistivity: valid rho points >= 5 and sigma insufficient",
        )
    return "none", "not usable: valid sigma/rho points < 5"


def candidate_priority(row: pd.Series) -> str:
    if normalize_bool(row.get("is_initial_tau_fit_candidate_step8", False)):
        valid_points = float(row.get("valid_sigma_or_rho_point_count_step8", 0))
        span = float(row.get("sigma_or_rho_temperature_span_step8", 0))
        if valid_points >= 10 and span >= 100:
            return "A"
        if valid_points >= 5:
            return "B"
    if normalize_bool(row.get("is_full_learning_candidate_step8", False)):
        confidence = normalize_text(row.get("n_or_p_confidence_step6", "")).casefold()
        span = float(row.get("sigma_or_rho_temperature_span_step8", 0))
        if confidence == "low" or span < 100:
            return "C"
    if normalize_bool(row.get("is_tau_fitting_candidate_step8", False)):
        return "review"
    return "not_candidate"


def learning_candidate_reason(row: pd.Series) -> str:
    if not normalize_bool(row.get("has_valid_sigma_or_rho_step8", False)):
        return "not candidate: missing sigma/rho"
    if int(row.get("valid_sigma_or_rho_point_count_step8", 0)) < 5:
        return "not candidate: valid sigma/rho points <5"
    if int(row.get("sigma_rho_parse_failed_curve_count_step8", 0)) > 0:
        return "review: sigma/rho parse failed"
    if int(row.get("sigma_rho_xy_mismatch_curve_count_step8", 0)) > 0:
        return "review: sigma/rho x-y mismatch"

    n_or_p = normalize_text(row.get("n_or_p", "")).casefold()
    confidence = normalize_text(row.get("n_or_p_confidence_step6", "")).casefold()
    if n_or_p == "mixed":
        return "review: n_or_p is mixed"
    if n_or_p == "unknown":
        return "review: n_or_p is unknown"
    if confidence == "low":
        return "review: n_or_p confidence is low"

    if not normalize_bool(row.get("has_kappa_or_zt_step8", False)):
        return "ok for tau fitting only: valid sigma/rho >=5 but missing kappa/ZT"
    if not normalize_bool(row.get("has_seebeck_step8", False)):
        return "ok for tau fitting only: valid sigma/rho >=5 but missing Seebeck"
    return "ok: valid sigma/rho >=5, Seebeck available, kappa/ZT available"


def add_candidate_flags(availability: pd.DataFrame) -> pd.DataFrame:
    output = availability.copy()
    output = fill_sample_summary_defaults(output)

    output["is_tau_fitting_candidate_step8"] = (
        output["has_valid_sigma_or_rho_step8"]
        & (pd.to_numeric(output["valid_sigma_or_rho_point_count_step8"], errors="coerce").fillna(0) >= 5)
    )
    output["is_full_learning_candidate_step8"] = (
        output["is_tau_fitting_candidate_step8"]
        & output["has_seebeck_step8"]
        & output["has_kappa_or_zt_step8"]
    )

    n_or_p = output["n_or_p"].map(lambda value: normalize_text(value).casefold()) if "n_or_p" in output.columns else ""
    confidence = (
        output["n_or_p_confidence_step6"].map(lambda value: normalize_text(value).casefold())
        if "n_or_p_confidence_step6" in output.columns
        else ""
    )
    output["is_initial_tau_fit_candidate_step8"] = (
        output["is_full_learning_candidate_step8"]
        & n_or_p.isin(["n", "p"])
        & confidence.isin(["high", "medium"])
        & (pd.to_numeric(output["valid_sigma_or_rho_point_count_step8"], errors="coerce").fillna(0) >= 5)
        & (pd.to_numeric(output["sigma_rho_parse_failed_curve_count_step8"], errors="coerce").fillna(0) == 0)
        & (pd.to_numeric(output["sigma_rho_xy_mismatch_curve_count_step8"], errors="coerce").fillna(0) == 0)
    )

    preferences = output.apply(fitting_source_preference, axis=1, result_type="expand")
    preferences.columns = ["fitting_source_preference_step8", "fitting_source_reason_step8"]
    output = pd.concat([output, preferences], axis=1)
    output["candidate_priority_tier_step8"] = output.apply(candidate_priority, axis=1)
    output["learning_candidate_reason_step8"] = output.apply(learning_candidate_reason, axis=1)
    return output


def merge_sample_flags(curves: pd.DataFrame, availability_step8: pd.DataFrame) -> pd.DataFrame:
    flag_columns = ["sample_key"] + [column for column in SAMPLE_CANDIDATE_COLUMNS if column in availability_step8.columns]
    flags = availability_step8.loc[:, flag_columns].drop_duplicates("sample_key")
    return curves.merge(flags, on="sample_key", how="left")


def value_counts_rows(prefix: str, series: pd.Series) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for value, count in series.fillna("").astype(str).value_counts(dropna=False).sort_index().items():
        rows.append((f"{prefix}_{value}_count", str(int(count))))
    return rows


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(normalize_bool).sum())


def nonpositive_sigma_rho_curve_count(sigma_rho: pd.DataFrame) -> int:
    if sigma_rho.empty:
        return 0
    counts = pd.to_numeric(sigma_rho["nonpositive_y_point_count_step8"], errors="coerce").fillna(0)
    return int((counts > 0).sum())


def build_report(
    sample_step7: pd.DataFrame,
    property_step7: pd.DataFrame,
    candidate_step7: pd.DataFrame,
    sample_step8: pd.DataFrame,
    learning_candidates: pd.DataFrame,
    initial_candidates: pd.DataFrame,
    review_candidates: pd.DataFrame,
    non_learning_samples: pd.DataFrame,
    property_step8: pd.DataFrame,
    candidate_step8: pd.DataFrame,
    sigma_rho_for_fitting: pd.DataFrame,
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    sigma_rho = property_step8[property_step8["is_sigma_or_rho_curve_step8"]]
    rows: list[tuple[str, str]] = [
        ("input_sample_property_availability_step7_rows", str(len(sample_step7))),
        ("input_property_core_curves_step7_rows", str(len(property_step7))),
        ("input_candidate_core_curves_step7_rows", str(len(candidate_step7))),
        ("output_sample_property_availability_step8_rows", str(len(sample_step8))),
        ("output_learning_candidates_step8_rows", str(len(learning_candidates))),
        ("output_initial_tau_fit_candidates_step8_rows", str(len(initial_candidates))),
        ("output_review_needed_candidates_step8_rows", str(len(review_candidates))),
        ("output_non_learning_samples_step8_rows", str(len(non_learning_samples))),
        ("output_property_core_curves_step8_rows", str(len(property_step8))),
        ("output_candidate_core_curves_step8_rows", str(len(candidate_step8))),
        ("output_sigma_rho_curves_for_fitting_step8_rows", str(len(sigma_rho_for_fitting))),
        ("sigma_rho_curve_count", str(len(sigma_rho))),
        ("Electrical conductivity_curve_count", str(int(property_step8["is_sigma_curve_step8"].sum()))),
        ("Electrical resistivity_curve_count", str(int(property_step8["is_rho_curve_step8"].sum()))),
        (
            "valid_sigma_rho_point_ge5_sample_count",
            str(int((pd.to_numeric(sample_step8["valid_sigma_or_rho_point_count_step8"], errors="coerce").fillna(0) >= 5).sum())),
        ),
        (
            "is_tau_fitting_candidate_step8_true_sample_count",
            str(bool_count(sample_step8, "is_tau_fitting_candidate_step8")),
        ),
        (
            "is_full_learning_candidate_step8_true_sample_count",
            str(bool_count(sample_step8, "is_full_learning_candidate_step8")),
        ),
        (
            "is_initial_tau_fit_candidate_step8_true_sample_count",
            str(bool_count(sample_step8, "is_initial_tau_fit_candidate_step8")),
        ),
        (
            "property_core_curves_step8_row_count_changed",
            str(len(property_step7) != len(property_step8)),
        ),
        (
            "candidate_core_curves_step8_row_count_changed",
            str(len(candidate_step7) != len(candidate_step8)),
        ),
    ]

    rows.extend(value_counts_rows("candidate_priority_tier_step8", sample_step8["candidate_priority_tier_step8"]))
    rows.extend(
        value_counts_rows("fitting_source_preference_step8", sample_step8["fitting_source_preference_step8"])
    )
    if "n_or_p" in learning_candidates.columns:
        rows.extend(value_counts_rows("learning_candidates_step8_n_or_p", learning_candidates["n_or_p"]))
    if "n_or_p" in initial_candidates.columns:
        rows.extend(value_counts_rows("initial_tau_fit_candidates_step8_n_or_p", initial_candidates["n_or_p"]))
    if "n_or_p" in review_candidates.columns:
        rows.extend(value_counts_rows("review_needed_candidates_step8_n_or_p", review_candidates["n_or_p"]))

    rows.extend(value_counts_rows("sigma_rho_x_parse_status_step8", sigma_rho["x_parse_status_step8"]))
    rows.extend(value_counts_rows("sigma_rho_y_parse_status_step8", sigma_rho["y_parse_status_step8"]))
    rows.extend(value_counts_rows("sigma_rho_xy_length_check_step8", sigma_rho["xy_length_check_step8"]))
    rows.extend(
        [
            (
                "sigma_rho_nonpositive_value_curve_count",
                str(nonpositive_sigma_rho_curve_count(sigma_rho)),
            ),
            (
                "sigma_rho_nonpositive_value_point_count",
                str(int(pd.to_numeric(sigma_rho["nonpositive_y_point_count_step8"], errors="coerce").fillna(0).sum())),
            ),
            (
                "sample_property_availability_step8_sintering_method_unknown_rows",
                str(int(sample_step8["sintering_method"].map(lambda value: normalize_text(value).casefold()).eq("unknown").sum()))
                if "sintering_method" in sample_step8.columns
                else "0",
            ),
            (
                "sample_property_availability_step8_sintering_checked_no_rows",
                str(int(sample_step8["sintering_checked"].map(lambda value: normalize_text(value).casefold()).eq("no").sum()))
                if "sintering_checked" in sample_step8.columns
                else "0",
            ),
            (
                "sample_property_availability_step8_record_checked_no_rows",
                str(int(sample_step8["record_checked"].map(lambda value: normalize_text(value).casefold()).eq("no").sum()))
                if "record_checked" in sample_step8.columns
                else "0",
            ),
        ]
    )

    for note in excel_notes:
        rows.append(("excel_note", note))

    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    return report_text, report_df


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
    sample_step8: pd.DataFrame,
    learning_candidates: pd.DataFrame,
    initial_candidates: pd.DataFrame,
    review_candidates: pd.DataFrame,
    non_learning_samples: pd.DataFrame,
    sigma_rho_for_fitting: pd.DataFrame,
    candidate_step8: pd.DataFrame,
    report_df: pd.DataFrame,
    excel_notes: list[str],
) -> None:
    path = output_dir / "starrydata2_step8_learning_candidates.xlsx"
    sheets = {
        "sample_property_availability": excel_frame(
            sample_step8, "sample_property_availability", excel_notes
        ),
        "learning_candidates": excel_frame(learning_candidates, "learning_candidates", excel_notes),
        "initial_tau_fit_candidates": excel_frame(
            initial_candidates, "initial_tau_fit_candidates", excel_notes
        ),
        "review_needed_candidates": excel_frame(
            review_candidates, "review_needed_candidates", excel_notes
        ),
        "non_learning_samples": excel_frame(non_learning_samples, "non_learning_samples", excel_notes),
        "sigma_rho_curves_for_fitting": excel_frame(
            sigma_rho_for_fitting, "sigma_rho_curves_for_fitting", excel_notes
        ),
        "candidate_core_curves": excel_frame(candidate_step8, "candidate_core_curves", excel_notes),
        "candidate_report": report_df,
    }
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)


def assert_acceptance(
    sample_step7: pd.DataFrame,
    property_step7: pd.DataFrame,
    candidate_step7: pd.DataFrame,
    sample_step8: pd.DataFrame,
    learning_candidates: pd.DataFrame,
    initial_candidates: pd.DataFrame,
    property_step8: pd.DataFrame,
    candidate_step8: pd.DataFrame,
    sigma_rho_for_fitting: pd.DataFrame,
) -> None:
    if sample_step8["sample_key"].duplicated().any():
        raise ValueError("sample_property_availability_step8.csv is not one row per sample_key")
    if len(property_step7) != len(property_step8):
        raise ValueError("property_core_curves_step8 row count changed from Step7")
    if len(candidate_step7) != len(candidate_step8):
        raise ValueError("candidate_core_curves_step8 row count changed from Step7")

    for column in [
        "valid_sigma_or_rho_point_count_step8",
        "is_tau_fitting_candidate_step8",
        "is_full_learning_candidate_step8",
        "is_initial_tau_fit_candidate_step8",
        "fitting_source_preference_step8",
        "learning_candidate_reason_step8",
    ]:
        if column not in sample_step8.columns:
            raise KeyError(f"sample_property_availability_step8 missing {column}")

    if not learning_candidates.empty:
        if not learning_candidates["is_full_learning_candidate_step8"].map(normalize_bool).all():
            raise ValueError("learning_candidates_step8 contains non-full-learning candidates")
        if (pd.to_numeric(learning_candidates["valid_sigma_or_rho_point_count_step8"], errors="coerce").fillna(0) < 5).any():
            raise ValueError("learning_candidates_step8 contains samples with valid sigma/rho points < 5")
        if not learning_candidates["has_seebeck_step8"].map(normalize_bool).all():
            raise ValueError("learning_candidates_step8 contains samples without Seebeck")
        if not learning_candidates["has_kappa_or_zt_step8"].map(normalize_bool).all():
            raise ValueError("learning_candidates_step8 contains samples without kappa/ZT")

    if not initial_candidates.empty:
        if not initial_candidates["is_initial_tau_fit_candidate_step8"].map(normalize_bool).all():
            raise ValueError("initial_tau_fit_candidates_step8 contains non-initial candidates")
        if not initial_candidates["n_or_p"].map(lambda value: normalize_text(value).casefold()).isin(["n", "p"]).all():
            raise ValueError("initial_tau_fit_candidates_step8 contains non n/p samples")
        if initial_candidates["fitting_source_preference_step8"].eq("none").any():
            raise ValueError("initial_tau_fit_candidates_step8 contains samples with no fitting source")

    for frame_name, frame in [
        ("property_core_curves_step8", property_step8),
        ("candidate_core_curves_step8", candidate_step8),
        ("sigma_rho_curves_for_fitting_step8", sigma_rho_for_fitting),
    ]:
        for column in ["x_values_json", "y_values_json"]:
            if column not in frame.columns:
                raise KeyError(f"{frame_name} missing {column}")

    if not sigma_rho_for_fitting.empty:
        if not sigma_rho_for_fitting["is_sigma_or_rho_curve_step8"].map(normalize_bool).all():
            raise ValueError("sigma_rho_curves_for_fitting_step8 contains non sigma/rho curves")
        if not sigma_rho_for_fitting["is_tau_fitting_candidate_step8"].map(normalize_bool).all():
            raise ValueError("sigma_rho_curves_for_fitting_step8 contains non tau candidate samples")

    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if column in sample_step8.columns:
            bad = sample_step8[column].map(lambda value: normalize_text(value).casefold()).ne(expected)
            if bad.any():
                raise ValueError(f"sample_property_availability_step8 has non-standard {column}")


def write_csv_outputs(
    output_dir: Path,
    sample_step8: pd.DataFrame,
    learning_candidates: pd.DataFrame,
    initial_candidates: pd.DataFrame,
    review_candidates: pd.DataFrame,
    non_learning_samples: pd.DataFrame,
    property_step8: pd.DataFrame,
    candidate_step8: pd.DataFrame,
    sigma_rho_for_fitting: pd.DataFrame,
    report_text: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_step8.to_csv(output_dir / "sample_property_availability_step8.csv", index=False)
    learning_candidates.to_csv(output_dir / "learning_candidates_step8.csv", index=False)
    initial_candidates.to_csv(output_dir / "initial_tau_fit_candidates_step8.csv", index=False)
    review_candidates.to_csv(output_dir / "review_needed_candidates_step8.csv", index=False)
    non_learning_samples.to_csv(output_dir / "non_learning_samples_step8.csv", index=False)
    property_step8.to_csv(output_dir / "property_core_curves_step8.csv", index=False)
    candidate_step8.to_csv(output_dir / "candidate_core_curves_step8.csv", index=False)
    sigma_rho_for_fitting.to_csv(output_dir / "sigma_rho_curves_for_fitting_step8.csv", index=False)
    (output_dir / "step8_learning_candidate_report.txt").write_text(report_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_classification = read_csv_text(args.step7_dir / SAMPLE_CLASSIFICATION_FILE)
    candidate_samples = read_csv_text(args.step7_dir / CANDIDATE_SAMPLES_FILE)
    sample_step7 = read_csv_text(args.step7_dir / SAMPLE_AVAILABILITY_FILE)
    property_step7 = read_csv_text(args.step7_dir / PROPERTY_CORE_FILE)
    candidate_step7 = read_csv_text(args.step7_dir / CANDIDATE_CORE_FILE)

    validate_sample_key(sample_classification, SAMPLE_CLASSIFICATION_FILE)
    validate_sample_key(candidate_samples, CANDIDATE_SAMPLES_FILE)
    validate_sample_key(candidate_step7, CANDIDATE_CORE_FILE)
    validate_inputs(sample_step7, property_step7)

    property_parsed = add_curve_step8_columns(property_step7)
    candidate_parsed = add_curve_step8_columns(candidate_step7)

    sample_summary = build_sample_summary(property_parsed, sample_step7["sample_key"])
    sample_step8 = sample_step7.merge(sample_summary, on="sample_key", how="left")
    sample_step8 = add_candidate_flags(sample_step8)

    property_step8 = merge_sample_flags(property_parsed, sample_step8)
    candidate_step8 = merge_sample_flags(candidate_parsed, sample_step8)

    learning_candidates = sample_step8[
        sample_step8["is_full_learning_candidate_step8"].map(normalize_bool)
    ].copy()
    initial_candidates = sample_step8[
        sample_step8["is_initial_tau_fit_candidate_step8"].map(normalize_bool)
    ].copy()
    review_candidates = sample_step8[
        sample_step8["is_tau_fitting_candidate_step8"].map(normalize_bool)
        & ~sample_step8["is_initial_tau_fit_candidate_step8"].map(normalize_bool)
    ].copy()
    non_learning_samples = sample_step8[
        ~sample_step8["is_tau_fitting_candidate_step8"].map(normalize_bool)
    ].copy()

    sigma_rho_for_fitting = property_step8[
        property_step8["is_sigma_or_rho_curve_step8"].map(normalize_bool)
        & property_step8["is_tau_fitting_candidate_step8"].map(normalize_bool)
    ].copy()

    assert_acceptance(
        sample_step7,
        property_step7,
        candidate_step7,
        sample_step8,
        learning_candidates,
        initial_candidates,
        property_step8,
        candidate_step8,
        sigma_rho_for_fitting,
    )

    excel_notes: list[str] = []
    report_text, report_df = build_report(
        sample_step7,
        property_step7,
        candidate_step7,
        sample_step8,
        learning_candidates,
        initial_candidates,
        review_candidates,
        non_learning_samples,
        property_step8,
        candidate_step8,
        sigma_rho_for_fitting,
        excel_notes,
    )

    write_csv_outputs(
        args.output_dir,
        sample_step8,
        learning_candidates,
        initial_candidates,
        review_candidates,
        non_learning_samples,
        property_step8,
        candidate_step8,
        sigma_rho_for_fitting,
        report_text,
    )
    write_excel_output(
        args.output_dir,
        sample_step8,
        learning_candidates,
        initial_candidates,
        review_candidates,
        non_learning_samples,
        sigma_rho_for_fitting,
        candidate_step8,
        report_df,
        excel_notes,
    )
    if excel_notes:
        report_text, report_df = build_report(
            sample_step7,
            property_step7,
            candidate_step7,
            sample_step8,
            learning_candidates,
            initial_candidates,
            review_candidates,
            non_learning_samples,
            property_step8,
            candidate_step8,
            sigma_rho_for_fitting,
            excel_notes,
        )
        (args.output_dir / "step8_learning_candidate_report.txt").write_text(
            report_text, encoding="utf-8"
        )

    full_n_or_p = (
        learning_candidates["n_or_p"].map(lambda value: normalize_text(value).casefold())
        if "n_or_p" in learning_candidates.columns
        else pd.Series(dtype=str)
    )
    sigma_rho = property_step8[property_step8["is_sigma_or_rho_curve_step8"].map(normalize_bool)]
    print("Done.")
    print("Created:")
    print("- sample_property_availability_step8.csv")
    print("- learning_candidates_step8.csv")
    print("- initial_tau_fit_candidates_step8.csv")
    print("- review_needed_candidates_step8.csv")
    print("- non_learning_samples_step8.csv")
    print("- property_core_curves_step8.csv")
    print("- candidate_core_curves_step8.csv")
    print("- sigma_rho_curves_for_fitting_step8.csv")
    print("- step8_learning_candidate_report.txt")
    print("- starrydata2_step8_learning_candidates.xlsx")
    print("")
    print("Summary:")
    print(f"samples total: {len(sample_step8)}")
    print(f"tau fitting candidates: {bool_count(sample_step8, 'is_tau_fitting_candidate_step8')}")
    print(f"full learning candidates: {len(learning_candidates)}")
    print(f"initial tau fit candidates: {len(initial_candidates)}")
    print(f"review needed candidates: {len(review_candidates)}")
    print(f"non-learning samples: {len(non_learning_samples)}")
    print(f"sigma/rho curves for fitting: {len(sigma_rho_for_fitting)}")
    print(
        "Electrical conductivity preferred samples: "
        f"{int(sample_step8['fitting_source_preference_step8'].eq(SIGMA_PROPERTY).sum())}"
    )
    print(
        "Electrical resistivity preferred samples: "
        f"{int(sample_step8['fitting_source_preference_step8'].eq(RHO_PROPERTY).sum())}"
    )
    print(f"p candidates: {int(full_n_or_p.eq('p').sum())}")
    print(f"n candidates: {int(full_n_or_p.eq('n').sum())}")
    print(f"mixed candidates: {int(full_n_or_p.eq('mixed').sum())}")
    print(f"unknown candidates: {int(full_n_or_p.eq('unknown').sum())}")
    print(
        "sigma/rho parse failed curves: "
        f"{count_parse_failed(sigma_rho)}"
    )
    print(
        "sigma/rho x-y mismatch curves: "
        f"{count_xy_mismatch(sigma_rho)}"
    )


if __name__ == "__main__":
    main()
