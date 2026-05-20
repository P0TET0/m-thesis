import argparse
import math
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl.styles import Font

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STEP10_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step10_training_dataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "starrydata2_step11_unit_normalized"

REQUIRED_INPUT_FILES = {
    "training_wide": "training_dataset_wide_step10.csv",
    "initial_training": "initial_tau_fit_training_dataset_step10.csv",
    "review_training": "review_training_dataset_step10.csv",
    "sigma_rho_points": "sigma_rho_points_for_fitting_step10.csv",
}
OPTIONAL_INPUT_FILES = {
    "property_points_aggregated": "property_points_aggregated_step10.csv",
    "temperature_alignment": "temperature_alignment_summary_step10.csv",
}

SIGMA_PROPERTY = "Electrical conductivity"
RHO_PROPERTY = "Electrical resistivity"
EXCEL_PREVIEW_ROWS = 100_000
MISSING_UNIT_VALUES = {"", "nan", "none", "null", "unknown", "-"}
BAD_CONVERSION_STATUSES = {"unit_unknown", "invalid_value", "unit_abnormal"}
SCALE_RE = re.compile(r"(?i)(?:[x*]\s*)?(?:1(?:\.0+)?e|10\s*\^)\s*\(?\s*([+-]?\d+)\s*\)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Step10 units for Step11.")
    parser.add_argument("--step10_dir", type=Path, default=DEFAULT_STEP10_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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


def finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return np.isfinite(numeric) & (numeric > 0)


def read_csv_text(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False, nrows=nrows)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def input_paths(step10_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for label, filename in REQUIRED_INPUT_FILES.items():
        path = step10_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required Step10 input file not found: {path}")
        paths[label] = path
    for label, filename in OPTIONAL_INPUT_FILES.items():
        path = step10_dir / filename
        if path.exists():
            paths[label] = path
    return paths


def validate_training_wide(df: pd.DataFrame) -> None:
    required = [
        "sample_key",
        "temperature_K",
        "has_sigma_or_rho_obs_step10",
        "usable_for_tau_fit_step10",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"training_dataset_wide_step10.csv missing required columns: {missing}")
    if "sigma_obs_raw" not in df.columns and "rho_obs_raw" not in df.columns:
        raise KeyError("training_dataset_wide_step10.csv needs sigma_obs_raw or rho_obs_raw")


def validate_sigma_rho_points(df: pd.DataFrame) -> None:
    required = [
        "sample_key",
        "property_step10",
        "temperature_K",
        "value_raw",
        "unit_y",
        "usable_for_tau_fit_step10",
        "selected_for_tau_fit_step10",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(
            f"sigma_rho_points_for_fitting_step10.csv missing required columns: {missing}"
        )


@lru_cache(maxsize=None)
def normalize_unit(raw_unit: Any) -> str:
    text = normalize_text(raw_unit)
    if not text:
        return "missing"
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "Ω": "ohm",
        "Ω": "ohm",
        "ω": "ohm",
        "μ": "u",
        "µ": "u",
        "−": "-",
        "–": "-",
        "—": "-",
        "·": "*",
        "・": "*",
        "×": "x",
        "⁻¹": "^-1",
        "⁻²": "^-2",
        "¹": "1",
        "²": "2",
        "³": "3",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.strip().casefold()
    if text in MISSING_UNIT_VALUES:
        return "missing"
    text = text.replace("**", "^")
    text = re.sub(r"\^\s*\(\s*([+-]?\d+)\s*\)", r"^\1", text)
    text = re.sub(r"(?<=[a-z)])-\s*1\b", "^-1", text)
    text = re.sub(r"(?<=[a-z)])-\s*2\b", "^-2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text not in MISSING_UNIT_VALUES else "missing"


def extract_scale_factor(unit_norm: str) -> tuple[float, str]:
    if unit_norm == "missing":
        return 1.0, unit_norm
    scale = 1.0
    cleaned = unit_norm
    for match in list(SCALE_RE.finditer(unit_norm)):
        exponent = int(match.group(1))
        scale *= 10.0**exponent
    cleaned = SCALE_RE.sub("", cleaned)
    cleaned = re.sub(r"^[*/x\s]+|[*/x\s]+$", "", cleaned).strip()
    return scale, cleaned


def compact_unit(unit_norm: str) -> str:
    unit = unit_norm.casefold()
    unit = unit.replace(" per ", "/")
    unit = unit.replace("ohms", "ohm")
    unit = unit.replace("siemens", "s")
    unit = unit.replace("volts", "v").replace("volt", "v")
    unit = unit.replace("watts", "w").replace("watt", "w")
    unit = unit.replace("kelvin", "k")
    unit = unit.replace("metres", "m").replace("metre", "m")
    unit = unit.replace("meters", "m").replace("meter", "m")
    unit = unit.replace("centimetres", "cm").replace("centimetre", "cm")
    unit = unit.replace("centimeters", "cm").replace("centimeter", "cm")
    unit = unit.replace("milliohm", "mohm")
    unit = unit.replace("microohm", "uohm")
    unit = unit.replace("micro", "u")
    unit = unit.replace(" ", "")
    unit = unit.replace("(", "").replace(")", "")
    unit = unit.replace("**", "^")
    unit = re.sub(r"\^\s*([+-]?\d+)", r"^\1", unit)
    unit = re.sub(r"(?<=[a-z)])-1\b", "^-1", unit)
    unit = re.sub(r"(?<=[a-z)])-2\b", "^-2", unit)
    return unit


def has_cm_inverse(unit: str) -> bool:
    return "/cm" in unit or "cm^-1" in unit or "cm-1" in unit


def has_m_inverse(unit: str) -> bool:
    if has_cm_inverse(unit):
        return False
    return "/m" in unit or "m^-1" in unit or "m-1" in unit


def conductivity_base_factor(unit_without_scale: str) -> tuple[str, float, str]:
    unit = compact_unit(unit_without_scale)
    if not unit or unit == "missing":
        return "unit_unknown", math.nan, "unit missing"

    if "ohm^-1" in unit or "ohm-1" in unit:
        if has_cm_inverse(unit):
            return "ok", 100.0, "conductivity unit interpreted as S/cm"
        if has_m_inverse(unit):
            return "ok", 1.0, "conductivity unit interpreted as S/m"
    if unit.startswith("1/") and "ohm" in unit:
        if "cm" in unit:
            return "ok", 100.0, "conductivity unit interpreted as 1/(ohm cm)"
        if "m" in unit:
            return "ok", 1.0, "conductivity unit interpreted as 1/(ohm m)"

    for prefix, multiplier, label in [
        ("us", 1e-6, "uS"),
        ("ms", 1e-3, "mS"),
        ("ks", 1e3, "kS"),
        ("s", 1.0, "S"),
    ]:
        if not unit.startswith(prefix):
            continue
        rest = unit[len(prefix) :]
        if rest.startswith("/cm") or rest.startswith("*cm^-1") or rest == "cm^-1":
            return "ok", multiplier * 100.0, f"conductivity unit interpreted as {label}/cm"
        if rest.startswith("/m") or rest.startswith("*m^-1") or rest == "m^-1":
            return "ok", multiplier, f"conductivity unit interpreted as {label}/m"
    return "unit_unknown", math.nan, f"unsupported conductivity unit: {unit_without_scale}"


def resistivity_base_factor(unit_without_scale: str) -> tuple[str, float, str]:
    unit = compact_unit(unit_without_scale)
    if not unit or unit == "missing":
        return "unit_unknown", math.nan, "unit missing"
    if unit in {"kg*m^3/a^2/s^3", "kg*m^3/s^3/a^2"}:
        return "ok", 1.0, "SI base unit interpreted as ohm m"
    if "/m" in unit or "m^-1" in unit or "m^2" in unit or "cm^-1" in unit:
        return "unit_unknown", math.nan, f"unsupported resistivity unit: {unit_without_scale}"

    for prefix, multiplier, label in [
        ("uohm", 1e-6, "uohm"),
        ("mohm", 1e-3, "mohm"),
        ("ohm", 1.0, "ohm"),
    ]:
        if not unit.startswith(prefix):
            continue
        rest = unit[len(prefix) :].lstrip("*")
        if rest == "m":
            return "ok", multiplier, f"resistivity unit interpreted as {label} m"
        if rest == "cm":
            return "ok", multiplier * 0.01, f"resistivity unit interpreted as {label} cm"
    return "unit_unknown", math.nan, f"unsupported resistivity unit: {unit_without_scale}"


def seebeck_base_factor(unit_without_scale: str) -> tuple[str, float, str]:
    unit = compact_unit(unit_without_scale)
    if not unit or unit == "missing":
        return "unit_unknown", math.nan, "unit missing"
    for prefix, multiplier, label in [
        ("nv", 1e-9, "nV"),
        ("uv", 1e-6, "uV"),
        ("mv", 1e-3, "mV"),
        ("v", 1.0, "V"),
    ]:
        if not unit.startswith(prefix):
            continue
        rest = unit[len(prefix) :]
        if rest in {"/k", "*k^-1", "k^-1"}:
            return "ok", multiplier, f"Seebeck unit interpreted as {label}/K"
    return "unit_unknown", math.nan, f"unsupported Seebeck unit: {unit_without_scale}"


def kappa_base_factor(unit_without_scale: str) -> tuple[str, float, str]:
    unit = compact_unit(unit_without_scale)
    if not unit or unit == "missing":
        return "unit_unknown", math.nan, "unit missing"
    if "k" not in unit:
        return "unit_unknown", math.nan, f"unsupported thermal conductivity unit: {unit_without_scale}"
    for prefix, multiplier, label in [("mw", 1e-3, "mW"), ("w", 1.0, "W")]:
        if not unit.startswith(prefix):
            continue
        if has_cm_inverse(unit) or "/cm" in unit:
            return "ok", multiplier * 100.0, f"thermal conductivity unit interpreted as {label}/(cm K)"
        if has_m_inverse(unit) or "/m" in unit:
            return "ok", multiplier, f"thermal conductivity unit interpreted as {label}/(m K)"
    return "unit_unknown", math.nan, f"unsupported thermal conductivity unit: {unit_without_scale}"


def zt_base_factor(unit_without_scale: str) -> tuple[str, float, str, bool]:
    unit = compact_unit(unit_without_scale)
    if unit in {"", "missing", "1", "dimensionless"}:
        return "ok", 1.0, "ZT treated as dimensionless", False
    if "k^-1" in unit or "/k" in unit or "perk" in unit or unit.startswith("w") or "s/m" in unit:
        return "unit_abnormal", math.nan, f"abnormal ZT unit: {unit_without_scale}", True
    return "unit_unknown", math.nan, f"unsupported ZT unit: {unit_without_scale}", False


@lru_cache(maxsize=None)
def unit_conversion_info(quantity: str, raw_unit: Any) -> tuple[str, float, float, str, str, bool]:
    unit_norm = normalize_unit(raw_unit)
    scale, unit_without_scale = extract_scale_factor(unit_norm)
    if quantity == "sigma":
        status, base_factor, note = conductivity_base_factor(unit_without_scale)
        abnormal = False
    elif quantity == "rho":
        status, base_factor, note = resistivity_base_factor(unit_without_scale)
        abnormal = False
    elif quantity == "seebeck":
        status, base_factor, note = seebeck_base_factor(unit_without_scale)
        abnormal = False
    elif quantity == "kappa":
        status, base_factor, note = kappa_base_factor(unit_without_scale)
        abnormal = False
    elif quantity == "zt":
        status, base_factor, note, abnormal = zt_base_factor(unit_without_scale)
    else:
        raise ValueError(f"Unknown quantity: {quantity}")
    factor = scale * base_factor if status == "ok" and math.isfinite(base_factor) else math.nan
    if status == "ok" and scale != 1.0:
        note = f"{note}; applied unit scale factor {scale:g}"
    return unit_norm, scale, factor, status, note, abnormal


def raw_present(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(lambda value: normalize_text(value) != "")


def convert_quantity(
    raw_values: pd.Series,
    raw_units: pd.Series,
    quantity: str,
    require_positive: bool,
) -> pd.DataFrame:
    info = pd.DataFrame(
        raw_units.map(lambda unit: unit_conversion_info(quantity, unit)).tolist(),
        index=raw_units.index,
        columns=[
            "unit_norm",
            "unit_scale_factor",
            "conversion_factor",
            "unit_status",
            "conversion_note",
            "unit_abnormal_flag",
        ],
    )
    numeric = pd.to_numeric(raw_values, errors="coerce")
    present = raw_present(raw_values)
    finite = np.isfinite(numeric)
    invalid_value = present & (~finite | (numeric <= 0 if require_positive else False))
    status = info["unit_status"].astype(object).copy()
    status.loc[~present] = "missing"
    status.loc[invalid_value] = "invalid_value"
    ok = present & ~invalid_value & status.eq("ok")
    converted = numeric * pd.to_numeric(info["conversion_factor"], errors="coerce")
    converted.loc[~ok] = np.nan
    note = info["conversion_note"].astype(object).copy()
    note.loc[~present] = "raw value missing"
    note.loc[invalid_value] = "invalid or nonpositive raw value" if require_positive else "invalid raw value"
    return pd.DataFrame(
        {
            "unit_norm": info["unit_norm"],
            "unit_scale_factor": info["unit_scale_factor"],
            "conversion_factor": info["conversion_factor"],
            "conversion_status": status,
            "conversion_note": note,
            "unit_abnormal_flag": info["unit_abnormal_flag"],
            "converted_value": converted,
        },
        index=raw_values.index,
    )


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in df.columns:
            df[column] = ""


def add_unit_normalization_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    ensure_columns(
        output,
        [
            "sigma_obs_raw",
            "sigma_unit_raw",
            "rho_obs_raw",
            "rho_unit_raw",
            "seebeck_obs_raw",
            "seebeck_unit_raw",
            "kappa_obs_raw",
            "kappa_unit_raw",
            "zt_obs_raw",
            "zt_unit_raw",
            "fitting_source_actual_step10",
            "usable_for_tau_fit_step10",
            "is_initial_tau_fit_candidate_step8",
            "is_tau_fitting_candidate_step8",
            "n_or_p",
        ],
    )

    sigma = convert_quantity(output["sigma_obs_raw"], output["sigma_unit_raw"], "sigma", True)
    output["sigma_unit_norm_step11"] = sigma["unit_norm"]
    output["sigma_unit_scale_factor_step11"] = sigma["unit_scale_factor"]
    output["sigma_obs_S_per_m_from_sigma_step11"] = sigma["converted_value"]
    output["sigma_conversion_factor_from_sigma_step11"] = sigma["conversion_factor"]
    output["sigma_conversion_status_from_sigma_step11"] = sigma["conversion_status"]
    output["sigma_conversion_note_from_sigma_step11"] = sigma["conversion_note"]

    rho = convert_quantity(output["rho_obs_raw"], output["rho_unit_raw"], "rho", True)
    output["rho_unit_norm_step11"] = rho["unit_norm"]
    output["rho_unit_scale_factor_step11"] = rho["unit_scale_factor"]
    output["rho_obs_ohm_m_step11"] = rho["converted_value"]
    output["rho_conversion_factor_step11"] = rho["conversion_factor"]
    output["rho_conversion_status_step11"] = rho["conversion_status"]
    output["rho_conversion_note_step11"] = rho["conversion_note"]
    rho_sigma = 1.0 / output["rho_obs_ohm_m_step11"]
    rho_sigma.loc[~np.isfinite(rho_sigma) | (rho_sigma <= 0)] = np.nan
    output["sigma_obs_S_per_m_from_rho_step11"] = rho_sigma

    seebeck = convert_quantity(output["seebeck_obs_raw"], output["seebeck_unit_raw"], "seebeck", False)
    output["seebeck_unit_norm_step11"] = seebeck["unit_norm"]
    output["seebeck_unit_scale_factor_step11"] = seebeck["unit_scale_factor"]
    output["seebeck_obs_V_per_K_step11"] = seebeck["converted_value"]
    output["seebeck_conversion_factor_step11"] = seebeck["conversion_factor"]
    output["seebeck_conversion_status_step11"] = seebeck["conversion_status"]
    output["seebeck_conversion_note_step11"] = seebeck["conversion_note"]

    kappa = convert_quantity(output["kappa_obs_raw"], output["kappa_unit_raw"], "kappa", True)
    output["kappa_unit_norm_step11"] = kappa["unit_norm"]
    output["kappa_unit_scale_factor_step11"] = kappa["unit_scale_factor"]
    output["kappa_obs_W_per_mK_step11"] = kappa["converted_value"]
    output["kappa_conversion_factor_step11"] = kappa["conversion_factor"]
    output["kappa_conversion_status_step11"] = kappa["conversion_status"]
    output["kappa_conversion_note_step11"] = kappa["conversion_note"]

    zt = convert_quantity(output["zt_obs_raw"], output["zt_unit_raw"], "zt", False)
    output["zt_unit_norm_step11"] = zt["unit_norm"]
    output["zt_unit_scale_factor_step11"] = zt["unit_scale_factor"]
    output["zt_obs_dimensionless_step11"] = zt["converted_value"]
    output["zt_conversion_factor_step11"] = zt["conversion_factor"]
    output["zt_conversion_status_step11"] = zt["conversion_status"]
    output["zt_conversion_note_step11"] = zt["conversion_note"]
    output["zt_unit_abnormal_flag_step11"] = zt["unit_abnormal_flag"]

    choose_final_sigma(output)
    add_derived_observation_columns(output)
    add_quality_columns(output)
    return output


def choose_final_sigma(df: pd.DataFrame) -> None:
    sigma_from_sigma = pd.to_numeric(df["sigma_obs_S_per_m_from_sigma_step11"], errors="coerce")
    sigma_from_rho = pd.to_numeric(df["sigma_obs_S_per_m_from_rho_step11"], errors="coerce")
    sigma_ok = np.isfinite(sigma_from_sigma) & (sigma_from_sigma > 0)
    rho_ok = np.isfinite(sigma_from_rho) & (sigma_from_rho > 0)
    actual = df["fitting_source_actual_step10"].fillna("").astype(str)

    final = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series("unavailable", index=df.index, dtype="object")
    note = pd.Series("no usable sigma/rho conversion", index=df.index, dtype="object")
    scale = pd.Series(np.nan, index=df.index, dtype="float64")

    use_sigma = actual.eq(SIGMA_PROPERTY) & sigma_ok
    use_rho = actual.eq(RHO_PROPERTY) & rho_ok
    fallback_sigma = ~actual.isin([SIGMA_PROPERTY, RHO_PROPERTY]) & sigma_ok
    fallback_rho = ~actual.isin([SIGMA_PROPERTY, RHO_PROPERTY]) & ~sigma_ok & rho_ok

    final.loc[use_sigma | fallback_sigma] = sigma_from_sigma.loc[use_sigma | fallback_sigma]
    source.loc[use_sigma | fallback_sigma] = "from_sigma"
    note.loc[use_sigma | fallback_sigma] = df.loc[
        use_sigma | fallback_sigma, "sigma_conversion_note_from_sigma_step11"
    ]
    scale.loc[use_sigma | fallback_sigma] = pd.to_numeric(
        df.loc[use_sigma | fallback_sigma, "sigma_unit_scale_factor_step11"], errors="coerce"
    )

    final.loc[use_rho | fallback_rho] = sigma_from_rho.loc[use_rho | fallback_rho]
    source.loc[use_rho | fallback_rho] = "from_rho"
    note.loc[use_rho | fallback_rho] = df.loc[use_rho | fallback_rho, "rho_conversion_note_step11"]
    scale.loc[use_rho | fallback_rho] = pd.to_numeric(
        df.loc[use_rho | fallback_rho, "rho_unit_scale_factor_step11"], errors="coerce"
    )

    actual_sigma_unavailable = actual.eq(SIGMA_PROPERTY) & ~sigma_ok
    actual_rho_unavailable = actual.eq(RHO_PROPERTY) & ~rho_ok
    note.loc[actual_sigma_unavailable] = (
        "preferred Electrical conductivity unavailable: "
        + df.loc[actual_sigma_unavailable, "sigma_conversion_status_from_sigma_step11"].astype(str)
    )
    note.loc[actual_rho_unavailable] = (
        "preferred Electrical resistivity unavailable: "
        + df.loc[actual_rho_unavailable, "rho_conversion_status_step11"].astype(str)
    )

    df["sigma_obs_S_per_m_step11"] = final
    df["sigma_obs_source_step11"] = source
    df["sigma_obs_unit_status_step11"] = np.where(source.eq("unavailable"), "unavailable", "ok")
    df["sigma_obs_conversion_note_step11"] = note
    df["unit_scale_factor_step11"] = scale


def add_derived_observation_columns(df: pd.DataFrame) -> None:
    sigma = pd.to_numeric(df["sigma_obs_S_per_m_step11"], errors="coerce")
    seebeck = pd.to_numeric(df["seebeck_obs_V_per_K_step11"], errors="coerce")
    kappa = pd.to_numeric(df["kappa_obs_W_per_mK_step11"], errors="coerce")
    temperature = pd.to_numeric(df["temperature_K"], errors="coerce")
    zt_obs = pd.to_numeric(df["zt_obs_dimensionless_step11"], errors="coerce")

    pf_ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(seebeck)
    pf = seebeck.pow(2) * sigma
    pf.loc[~pf_ok] = np.nan
    df["power_factor_obs_W_per_mK2_step11"] = pf
    df["power_factor_obs_uW_per_cmK2_step11"] = pf * 10000.0
    df["power_factor_calc_status_step11"] = np.where(pf_ok, "ok", "missing_sigma_or_seebeck")

    zt_calc_ok = pf_ok & np.isfinite(temperature) & np.isfinite(kappa) & (kappa > 0)
    zt_calc = pf * temperature / kappa
    zt_calc.loc[~zt_calc_ok] = np.nan
    df["zt_calc_from_obs_step11"] = zt_calc
    df["zt_calc_from_obs_status_step11"] = np.where(zt_calc_ok, "ok", "missing_required_observation")

    compare_ok = zt_calc_ok & np.isfinite(zt_obs)
    abs_error = (zt_calc - zt_obs).abs()
    rel_error = abs_error / np.maximum(zt_obs.abs(), 1e-12)
    abs_error.loc[~compare_ok] = np.nan
    rel_error.loc[~compare_ok] = np.nan
    df["zt_obs_vs_calc_abs_error_step11"] = abs_error
    df["zt_obs_vs_calc_relative_error_step11"] = rel_error
    status = pd.Series("not_available", index=df.index, dtype="object")
    status.loc[compare_ok & (rel_error <= 0.2)] = "ok"
    status.loc[compare_ok & (rel_error > 0.2) & (rel_error <= 1.0)] = "warning"
    status.loc[compare_ok & (rel_error > 1.0)] = "large_mismatch"
    df["zt_consistency_status_step11"] = status


def add_quality_columns(df: pd.DataFrame) -> None:
    sigma = pd.to_numeric(df["sigma_obs_S_per_m_step11"], errors="coerce")
    temperature = pd.to_numeric(df["temperature_K"], errors="coerce")
    seebeck = pd.to_numeric(df["seebeck_obs_V_per_K_step11"], errors="coerce")
    kappa = pd.to_numeric(df["kappa_obs_W_per_mK_step11"], errors="coerce")
    zt_obs = pd.to_numeric(df["zt_obs_dimensionless_step11"], errors="coerce")

    can_fit = (
        df["usable_for_tau_fit_step10"].map(normalize_bool)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(temperature)
    )
    can_pf = can_fit & np.isfinite(seebeck)
    can_zt = can_pf & np.isfinite(kappa) & (kappa > 0)
    can_compare = can_zt & np.isfinite(zt_obs)
    can_initial = (
        can_fit
        & df["is_initial_tau_fit_candidate_step8"].map(normalize_bool)
        & df["n_or_p"].fillna("").astype(str).str.casefold().isin(["n", "p"])
    )
    df["can_fit_tau_step11"] = can_fit
    df["can_eval_power_factor_step11"] = can_pf
    df["can_calc_zt_from_obs_step11"] = can_zt
    df["can_compare_zt_obs_step11"] = can_compare
    df["can_use_for_initial_tau_fit_step11"] = can_initial

    conversion_problem = df["sigma_obs_unit_status_step11"].eq("unavailable")
    for column in [
        "sigma_conversion_status_from_sigma_step11",
        "rho_conversion_status_step11",
        "seebeck_conversion_status_step11",
        "kappa_conversion_status_step11",
        "zt_conversion_status_step11",
    ]:
        conversion_problem |= df[column].isin(BAD_CONVERSION_STATUSES)

    quality = pd.Series("not_usable", index=df.index, dtype="object")
    quality.loc[conversion_problem & ~can_fit] = "unit_problem"
    quality.loc[can_fit] = "fit_only"
    quality.loc[can_pf] = "fit_and_pf_eval"
    quality.loc[can_zt] = "fit_and_zt_eval"
    df["row_quality_step11"] = quality

    note = pd.Series("not usable for tau fitting", index=df.index, dtype="object")
    note.loc[conversion_problem] = "unit conversion problem"
    note.loc[can_fit] = "usable for tau fitting"
    note.loc[can_pf] = "usable for tau fitting and PF evaluation"
    note.loc[can_zt] = "usable for tau fitting and ZT calculation"
    df["row_quality_note_step11"] = note


def normalize_sigma_rho_points(points: pd.DataFrame) -> pd.DataFrame:
    output = points[points["property_step10"].isin([SIGMA_PROPERTY, RHO_PROPERTY])].copy()
    unit_source = output["unit_y"]
    missing_unit_y = unit_source.map(lambda value: normalize_text(value) == "")
    if "unit" in output.columns:
        unit_source = unit_source.mask(missing_unit_y, output["unit"])

    sigma_mask = output["property_step10"].eq(SIGMA_PROPERTY)
    rho_mask = output["property_step10"].eq(RHO_PROPERTY)
    sigma_conv = convert_quantity(output["value_raw"], unit_source, "sigma", True)
    rho_conv = convert_quantity(output["value_raw"], unit_source, "rho", True)

    output["value_raw_unit_norm_step11"] = np.where(
        sigma_mask, sigma_conv["unit_norm"], rho_conv["unit_norm"]
    )
    output["unit_scale_factor_step11"] = np.where(
        sigma_mask, sigma_conv["unit_scale_factor"], rho_conv["unit_scale_factor"]
    )
    output["value_converted_step11"] = np.where(
        sigma_mask, sigma_conv["converted_value"], rho_conv["converted_value"]
    )
    output["value_converted_unit_step11"] = np.where(sigma_mask, "S/m", "ohm m")
    output["sigma_point_S_per_m_step11"] = np.nan
    output.loc[sigma_mask, "sigma_point_S_per_m_step11"] = sigma_conv.loc[
        sigma_mask, "converted_value"
    ]
    rho_value = rho_conv["converted_value"]
    rho_sigma = 1.0 / rho_value
    rho_sigma.loc[~np.isfinite(rho_sigma) | (rho_sigma <= 0)] = np.nan
    output.loc[rho_mask, "sigma_point_S_per_m_step11"] = rho_sigma.loc[rho_mask]
    output["rho_point_ohm_m_step11"] = np.nan
    output.loc[rho_mask, "rho_point_ohm_m_step11"] = rho_conv.loc[rho_mask, "converted_value"]
    output["sigma_point_source_step11"] = np.where(sigma_mask, "from_sigma", "from_rho")
    output["point_conversion_status_step11"] = np.where(
        sigma_mask, sigma_conv["conversion_status"], rho_conv["conversion_status"]
    )
    output["point_conversion_note_step11"] = np.where(
        sigma_mask, sigma_conv["conversion_note"], rho_conv["conversion_note"]
    )
    point_sigma = pd.to_numeric(output["sigma_point_S_per_m_step11"], errors="coerce")
    output["can_use_point_for_tau_fit_step11"] = (
        output["usable_for_tau_fit_step10"].map(normalize_bool)
        & output["selected_for_tau_fit_step10"].map(normalize_bool)
        & np.isfinite(point_sigma)
        & (point_sigma > 0)
    )
    return output


def first_examples(series: pd.Series) -> str:
    return ";".join(series.dropna().astype(str).drop_duplicates().head(5))


def build_unit_conversion_audit(df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "sigma",
            "sigma_obs_raw",
            "sigma_unit_raw",
            "sigma_unit_norm_step11",
            "sigma_conversion_status_from_sigma_step11",
            "sigma_conversion_factor_from_sigma_step11",
            "sigma_obs_S_per_m_from_sigma_step11",
        ),
        (
            "rho",
            "rho_obs_raw",
            "rho_unit_raw",
            "rho_unit_norm_step11",
            "rho_conversion_status_step11",
            "rho_conversion_factor_step11",
            "rho_obs_ohm_m_step11",
        ),
        (
            "seebeck",
            "seebeck_obs_raw",
            "seebeck_unit_raw",
            "seebeck_unit_norm_step11",
            "seebeck_conversion_status_step11",
            "seebeck_conversion_factor_step11",
            "seebeck_obs_V_per_K_step11",
        ),
        (
            "kappa",
            "kappa_obs_raw",
            "kappa_unit_raw",
            "kappa_unit_norm_step11",
            "kappa_conversion_status_step11",
            "kappa_conversion_factor_step11",
            "kappa_obs_W_per_mK_step11",
        ),
        (
            "zt",
            "zt_obs_raw",
            "zt_unit_raw",
            "zt_unit_norm_step11",
            "zt_conversion_status_step11",
            "zt_conversion_factor_step11",
            "zt_obs_dimensionless_step11",
        ),
    ]
    rows: list[pd.DataFrame] = []
    for quantity, raw_value, raw_unit, unit_norm, status, factor, converted in specs:
        frame = pd.DataFrame(
            {
                "quantity_step11": quantity,
                "raw_unit": df[raw_unit] if raw_unit in df else "",
                "normalized_unit": df[unit_norm] if unit_norm in df else "",
                "conversion_status": df[status] if status in df else "",
                "conversion_factor": df[factor] if factor in df else "",
                "sample_key": df["sample_key"],
                "raw_numeric": pd.to_numeric(df[raw_value], errors="coerce") if raw_value in df else np.nan,
                "converted_numeric": pd.to_numeric(df[converted], errors="coerce") if converted in df else np.nan,
            }
        )
        rows.append(frame)
    audit_source = pd.concat(rows, ignore_index=True)
    grouped = audit_source.groupby(
        ["quantity_step11", "raw_unit", "normalized_unit", "conversion_status", "conversion_factor"],
        dropna=False,
        sort=True,
    )
    return grouped.agg(
        row_count=("sample_key", "count"),
        sample_count=("sample_key", pd.Series.nunique),
        min_raw_value=("raw_numeric", "min"),
        max_raw_value=("raw_numeric", "max"),
        min_converted_value=("converted_numeric", "min"),
        max_converted_value=("converted_numeric", "max"),
        example_sample_keys=("sample_key", first_examples),
    ).reset_index()


def build_problematic_rows(df: pd.DataFrame) -> pd.DataFrame:
    problem = (
        df["sigma_conversion_status_from_sigma_step11"].isin(BAD_CONVERSION_STATUSES)
        | df["rho_conversion_status_step11"].isin(BAD_CONVERSION_STATUSES)
        | df["seebeck_conversion_status_step11"].isin(BAD_CONVERSION_STATUSES)
        | df["kappa_conversion_status_step11"].isin(BAD_CONVERSION_STATUSES)
        | df["zt_conversion_status_step11"].isin(BAD_CONVERSION_STATUSES)
        | df["sigma_obs_unit_status_step11"].eq("unavailable")
        | ~df["can_fit_tau_step11"].map(normalize_bool)
        | df["row_quality_step11"].eq("unit_problem")
        | df["zt_consistency_status_step11"].eq("large_mismatch")
    )
    columns = [
        "sample_key",
        "DOI",
        "paper_title",
        "sample_id",
        "composition",
        "temperature_K",
        "n_or_p",
        "sigma_obs_raw",
        "sigma_unit_raw",
        "rho_obs_raw",
        "rho_unit_raw",
        "seebeck_obs_raw",
        "seebeck_unit_raw",
        "kappa_obs_raw",
        "kappa_unit_raw",
        "zt_obs_raw",
        "zt_unit_raw",
        "sigma_obs_S_per_m_step11",
        "seebeck_obs_V_per_K_step11",
        "kappa_obs_W_per_mK_step11",
        "zt_obs_dimensionless_step11",
        "row_quality_step11",
        "row_quality_note_step11",
        "zt_consistency_status_step11",
    ]
    ensure_columns(df, columns)
    output = df.loc[problem, columns].copy()
    output["problem_reason_step11"] = output.apply(problem_reason, axis=1)
    return output


def problem_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if normalize_text(row.get("row_quality_step11")) == "unit_problem":
        reasons.append("row_quality unit_problem")
    if normalize_text(row.get("zt_consistency_status_step11")) == "large_mismatch":
        reasons.append("ZT large mismatch")
    if normalize_text(row.get("sigma_obs_S_per_m_step11")) == "":
        reasons.append("sigma unavailable")
    return "; ".join(reasons) if reasons else "conversion or usability problem"


def build_zt_consistency_check(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sample_key",
        "temperature_K",
        "DOI",
        "paper_title",
        "sample_id",
        "composition",
        "n_or_p",
        "sigma_obs_S_per_m_step11",
        "seebeck_obs_V_per_K_step11",
        "kappa_obs_W_per_mK_step11",
        "zt_obs_dimensionless_step11",
        "zt_calc_from_obs_step11",
        "zt_obs_vs_calc_abs_error_step11",
        "zt_obs_vs_calc_relative_error_step11",
        "zt_consistency_status_step11",
        "sigma_unit_raw",
        "seebeck_unit_raw",
        "kappa_unit_raw",
        "zt_unit_raw",
    ]
    ensure_columns(df, columns)
    return df.loc[df["can_compare_zt_obs_step11"].map(normalize_bool), columns].copy()


def value_counts_dict(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.fillna("").astype(str).value_counts().sort_index().items()}


def bool_count(df: pd.DataFrame, column: str) -> int:
    if column not in df.columns:
        return 0
    return int(df[column].map(normalize_bool).sum())


def build_report(
    input_counts: dict[str, int],
    output_counts: dict[str, int],
    stats: dict[str, Any],
    row_quality_counts: dict[str, int],
    sigma_source_counts: dict[str, int],
    n_or_p_can_fit_counts: dict[str, int],
    n_or_p_initial_counts: dict[str, int],
    zt_status_counts: dict[str, int],
    excel_notes: list[str],
) -> tuple[str, pd.DataFrame]:
    rows: list[tuple[str, str]] = [
        ("input_training_dataset_wide_step10_rows", str(input_counts["training_wide"])),
        ("input_initial_tau_fit_training_dataset_step10_rows", str(input_counts["initial_training"])),
        ("input_review_training_dataset_step10_rows", str(input_counts["review_training"])),
        ("input_sigma_rho_points_for_fitting_step10_rows", str(input_counts["sigma_rho_points"])),
        ("output_training_dataset_normalized_step11_rows", str(output_counts["training_normalized"])),
        ("output_initial_tau_fit_training_normalized_step11_rows", str(output_counts["initial_normalized"])),
        ("output_review_training_dataset_normalized_step11_rows", str(output_counts["review_normalized"])),
        ("output_sigma_rho_points_normalized_step11_rows", str(output_counts["sigma_rho_normalized"])),
        ("output_unit_conversion_audit_step11_rows", str(output_counts["unit_audit"])),
        ("output_problematic_unit_rows_step11_rows", str(output_counts["problematic_rows"])),
        ("output_zt_consistency_check_step11_rows", str(output_counts["zt_consistency"])),
        ("sigma_from_sigma_success_rows", str(stats["sigma_from_sigma_success"])),
        ("rho_conversion_success_rows", str(stats["rho_success"])),
        ("rho_to_sigma_success_rows", str(stats["rho_to_sigma_success"])),
        ("final_sigma_obs_S_per_m_success_rows", str(stats["final_sigma_success"])),
        ("seebeck_conversion_success_rows", str(stats["seebeck_success"])),
        ("kappa_conversion_success_rows", str(stats["kappa_success"])),
        ("zt_normal_unit_rows", str(stats["zt_normal"])),
        ("zt_unit_abnormal_rows", str(stats["zt_abnormal"])),
        ("can_fit_tau_step11_true_rows", str(stats["can_fit_tau"])),
        ("can_eval_power_factor_step11_true_rows", str(stats["can_eval_pf"])),
        ("can_calc_zt_from_obs_step11_true_rows", str(stats["can_calc_zt"])),
        ("can_compare_zt_obs_step11_true_rows", str(stats["can_compare_zt"])),
        ("can_use_for_initial_tau_fit_step11_true_rows", str(stats["can_initial"])),
    ]
    for key, value in row_quality_counts.items():
        rows.append((f"row_quality_step11_{key}_rows", str(value)))
    for key, value in sigma_source_counts.items():
        rows.append((f"sigma_obs_source_step11_{key}_rows", str(value)))
    for key, value in n_or_p_can_fit_counts.items():
        rows.append((f"n_or_p_{key}_can_fit_tau_step11_rows", str(value)))
    for key, value in n_or_p_initial_counts.items():
        rows.append((f"n_or_p_{key}_can_use_for_initial_tau_fit_step11_rows", str(value)))
    for key, value in zt_status_counts.items():
        rows.append((f"zt_consistency_status_step11_{key}_rows", str(value)))
    rows.extend(
        [
            ("zt_relative_error_le_0.2_rows", str(stats["zt_relative_le_02"])),
            ("zt_relative_error_gt_1.0_rows", str(stats["zt_relative_gt_10"])),
            ("sintering_method_unknown_rows", str(stats["sintering_method_unknown"])),
            ("sintering_checked_no_rows", str(stats["sintering_checked_no"])),
            ("record_checked_no_rows", str(stats["record_checked_no"])),
            ("n_or_p_columns_preserved_from_step10", "yes"),
            ("note", "Step11 did not perform tau_eff fitting."),
            ("note", "Step11 performed unit conversion and standardization only."),
        ]
    )
    for note in excel_notes:
        rows.append(("excel_note", note))
    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    return "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n", report_df


def collect_wide_stats(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, int], dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    zt_rel = pd.to_numeric(df["zt_obs_vs_calc_relative_error_step11"], errors="coerce")
    stats = {
        "sigma_from_sigma_success": int(df["sigma_conversion_status_from_sigma_step11"].eq("ok").sum()),
        "rho_success": int(df["rho_conversion_status_step11"].eq("ok").sum()),
        "rho_to_sigma_success": int(finite_positive(df["sigma_obs_S_per_m_from_rho_step11"]).sum()),
        "final_sigma_success": int(finite_positive(df["sigma_obs_S_per_m_step11"]).sum()),
        "seebeck_success": int(df["seebeck_conversion_status_step11"].eq("ok").sum()),
        "kappa_success": int(df["kappa_conversion_status_step11"].eq("ok").sum()),
        "zt_normal": int(df["zt_conversion_status_step11"].eq("ok").sum()),
        "zt_abnormal": bool_count(df, "zt_unit_abnormal_flag_step11"),
        "can_fit_tau": bool_count(df, "can_fit_tau_step11"),
        "can_eval_pf": bool_count(df, "can_eval_power_factor_step11"),
        "can_calc_zt": bool_count(df, "can_calc_zt_from_obs_step11"),
        "can_compare_zt": bool_count(df, "can_compare_zt_obs_step11"),
        "can_initial": bool_count(df, "can_use_for_initial_tau_fit_step11"),
        "zt_relative_le_02": int((zt_rel <= 0.2).sum()),
        "zt_relative_gt_10": int((zt_rel > 1.0).sum()),
        "sintering_method_unknown": int(df["sintering_method"].fillna("").astype(str).str.casefold().eq("unknown").sum())
        if "sintering_method" in df.columns
        else 0,
        "sintering_checked_no": int(df["sintering_checked"].fillna("").astype(str).str.casefold().eq("no").sum())
        if "sintering_checked" in df.columns
        else 0,
        "record_checked_no": int(df["record_checked"].fillna("").astype(str).str.casefold().eq("no").sum())
        if "record_checked" in df.columns
        else 0,
    }
    row_quality_counts = value_counts_dict(df["row_quality_step11"])
    sigma_source_counts = value_counts_dict(df["sigma_obs_source_step11"])
    n_or_p_can_fit_counts = value_counts_dict(df.loc[df["can_fit_tau_step11"].map(normalize_bool), "n_or_p"])
    n_or_p_initial_counts = value_counts_dict(
        df.loc[df["can_use_for_initial_tau_fit_step11"].map(normalize_bool), "n_or_p"]
    )
    zt_status_counts = value_counts_dict(df["zt_consistency_status_step11"])
    return (
        stats,
        row_quality_counts,
        sigma_source_counts,
        n_or_p_can_fit_counts,
        n_or_p_initial_counts,
        zt_status_counts,
    )


def csv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "doi_url" not in df.columns:
        return df
    columns = [column for column in df.columns if column != "doi_url"] + ["doi_url"]
    return df.loc[:, columns]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    csv_frame(df).to_csv(path, index=False)


def add_excel_preview_note(sheet_name: str, row_count: int, excel_notes: list[str]) -> None:
    if row_count <= EXCEL_PREVIEW_ROWS:
        return
    excel_notes.append(
        f"{sheet_name} has {row_count} rows; wrote first {EXCEL_PREVIEW_ROWS} rows to workbook; full data is in CSV"
    )


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


def write_excel_output(output_dir: Path, report_df: pd.DataFrame) -> None:
    sheet_files = {
        "training_dataset_normalized": "training_dataset_normalized_step11.csv",
        "initial_tau_fit_training": "initial_tau_fit_training_normalized_step11.csv",
        "review_training_dataset": "review_training_dataset_normalized_step11.csv",
        "sigma_rho_points_normalized": "sigma_rho_points_normalized_step11.csv",
        "unit_conversion_audit": "unit_conversion_audit_step11.csv",
        "problematic_unit_rows": "problematic_unit_rows_step11.csv",
        "zt_consistency_check": "zt_consistency_check_step11.csv",
    }
    xlsx_path = output_dir / "starrydata2_step11_unit_normalized.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sheet_name, filename in sheet_files.items():
            frame = read_csv_text(output_dir / filename, nrows=EXCEL_PREVIEW_ROWS)
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            fit_worksheet(writer, sheet_name, frame)
        report_df.to_excel(writer, sheet_name="unit_report", index=False)
        fit_worksheet(writer, "unit_report", report_df)


def assert_acceptance(training: pd.DataFrame, initial: pd.DataFrame, points: pd.DataFrame) -> None:
    for column in [
        "sample_key",
        "temperature_K",
        "sigma_obs_S_per_m_step11",
        "sigma_obs_source_step11",
        "seebeck_obs_V_per_K_step11",
        "kappa_obs_W_per_mK_step11",
        "zt_obs_dimensionless_step11",
        "power_factor_obs_W_per_mK2_step11",
        "zt_calc_from_obs_step11",
        "can_fit_tau_step11",
        "can_use_for_initial_tau_fit_step11",
        "n_or_p",
        "sintering_method",
        "sintering_checked",
        "record_checked",
    ]:
        if column not in training.columns:
            raise KeyError(f"training_dataset_normalized_step11.csv missing {column}")
    if not initial.empty:
        if not initial["can_use_for_initial_tau_fit_step11"].map(normalize_bool).all():
            raise ValueError("initial_tau_fit_training_normalized_step11 has non-initial rows")
        if not finite_positive(initial["sigma_obs_S_per_m_step11"]).all():
            raise ValueError("initial_tau_fit_training_normalized_step11 has nonpositive sigma")
    if not points["property_step10"].isin([SIGMA_PROPERTY, RHO_PROPERTY]).all():
        raise ValueError("sigma_rho_points_normalized_step11 contains non sigma/rho properties")
    for column in ["sigma_point_S_per_m_step11", "can_use_point_for_tau_fit_step11"]:
        if column not in points.columns:
            raise KeyError(f"sigma_rho_points_normalized_step11 missing {column}")
    for column, expected in [
        ("sintering_method", "unknown"),
        ("sintering_checked", "no"),
        ("record_checked", "no"),
    ]:
        if column in training.columns and not training[column].fillna("").astype(str).str.casefold().eq(expected).all():
            raise ValueError(f"{column} is not preserved as {expected}")


def main() -> None:
    args = parse_args()
    paths = input_paths(args.step10_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_counts = {
        label: count_csv_rows(path)
        for label, path in paths.items()
        if label in REQUIRED_INPUT_FILES or label in OPTIONAL_INPUT_FILES
    }

    training = read_csv_text(paths["training_wide"])
    validate_training_wide(training)
    training_norm = add_unit_normalization_columns(training)
    del training

    initial_norm = training_norm[
        training_norm["can_use_for_initial_tau_fit_step11"].map(normalize_bool)
    ].copy()
    review_norm = training_norm[
        training_norm["is_tau_fitting_candidate_step8"].map(normalize_bool)
        & ~training_norm["can_use_for_initial_tau_fit_step11"].map(normalize_bool)
    ].copy()
    unit_audit = build_unit_conversion_audit(training_norm)
    problematic = build_problematic_rows(training_norm)
    zt_check = build_zt_consistency_check(training_norm)
    (
        stats,
        row_quality_counts,
        sigma_source_counts,
        n_or_p_can_fit_counts,
        n_or_p_initial_counts,
        zt_status_counts,
    ) = collect_wide_stats(training_norm)

    write_csv(training_norm, args.output_dir / "training_dataset_normalized_step11.csv")
    write_csv(initial_norm, args.output_dir / "initial_tau_fit_training_normalized_step11.csv")
    write_csv(review_norm, args.output_dir / "review_training_dataset_normalized_step11.csv")
    write_csv(unit_audit, args.output_dir / "unit_conversion_audit_step11.csv")
    write_csv(problematic, args.output_dir / "problematic_unit_rows_step11.csv")
    write_csv(zt_check, args.output_dir / "zt_consistency_check_step11.csv")

    output_counts = {
        "training_normalized": len(training_norm),
        "initial_normalized": len(initial_norm),
        "review_normalized": len(review_norm),
        "unit_audit": len(unit_audit),
        "problematic_rows": len(problematic),
        "zt_consistency": len(zt_check),
    }

    sigma_rho_points = read_csv_text(paths["sigma_rho_points"])
    validate_sigma_rho_points(sigma_rho_points)
    points_norm = normalize_sigma_rho_points(sigma_rho_points)
    del sigma_rho_points
    write_csv(points_norm, args.output_dir / "sigma_rho_points_normalized_step11.csv")
    output_counts["sigma_rho_normalized"] = len(points_norm)

    assert_acceptance(training_norm, initial_norm, points_norm)

    excel_notes: list[str] = []
    sheet_counts = {
        "training_dataset_normalized": output_counts["training_normalized"],
        "initial_tau_fit_training": output_counts["initial_normalized"],
        "review_training_dataset": output_counts["review_normalized"],
        "sigma_rho_points_normalized": output_counts["sigma_rho_normalized"],
        "unit_conversion_audit": output_counts["unit_audit"],
        "problematic_unit_rows": output_counts["problematic_rows"],
        "zt_consistency_check": output_counts["zt_consistency"],
    }
    for sheet_name, row_count in sheet_counts.items():
        add_excel_preview_note(sheet_name, row_count, excel_notes)

    report_text, report_df = build_report(
        input_counts,
        output_counts,
        stats,
        row_quality_counts,
        sigma_source_counts,
        n_or_p_can_fit_counts,
        n_or_p_initial_counts,
        zt_status_counts,
        excel_notes,
    )
    (args.output_dir / "step11_unit_normalization_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    write_excel_output(args.output_dir, report_df)

    zt_large = zt_status_counts.get("large_mismatch", 0)
    print("Done.")
    print("Created:")
    print("- training_dataset_normalized_step11.csv")
    print("- initial_tau_fit_training_normalized_step11.csv")
    print("- review_training_dataset_normalized_step11.csv")
    print("- sigma_rho_points_normalized_step11.csv")
    print("- unit_conversion_audit_step11.csv")
    print("- problematic_unit_rows_step11.csv")
    print("- zt_consistency_check_step11.csv")
    print("- step11_unit_normalization_report.txt")
    print("- starrydata2_step11_unit_normalized.xlsx")
    print("")
    print("Summary:")
    print(f"training normalized rows: {output_counts['training_normalized']}")
    print(f"initial tau fit normalized rows: {output_counts['initial_normalized']}")
    print(f"review normalized rows: {output_counts['review_normalized']}")
    print(f"sigma/rho normalized points: {output_counts['sigma_rho_normalized']}")
    print(f"can_fit_tau rows: {stats['can_fit_tau']}")
    print(f"can_eval_power_factor rows: {stats['can_eval_pf']}")
    print(f"can_calc_zt_from_obs rows: {stats['can_calc_zt']}")
    print(f"can_compare_zt_obs rows: {stats['can_compare_zt']}")
    print(f"sigma from sigma success rows: {stats['sigma_from_sigma_success']}")
    print(f"rho to sigma success rows: {stats['rho_to_sigma_success']}")
    print(f"Seebeck conversion success rows: {stats['seebeck_success']}")
    print(f"kappa conversion success rows: {stats['kappa_success']}")
    print(f"ZT unit abnormal rows: {stats['zt_abnormal']}")
    print(f"ZT consistency large mismatch rows: {zt_large}")


if __name__ == "__main__":
    main()
