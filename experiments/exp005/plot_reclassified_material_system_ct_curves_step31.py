import argparse
import ast
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from material_group_classifier_step31 import (
        add_material_group_keys,
        build_material_group_counts,
    )
except ModuleNotFoundError:
    from .material_group_classifier_step31 import (
        add_material_group_keys,
        build_material_group_counts,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "starrydata2_step31_reclassified_ct_curves"
)
BIN_WIDTH_K = 25.0
MIN_ROWS_PER_PRIMARY_BIN = 3
SEEBECK_ZERO_THRESHOLD_V_PER_K = 1e-8
PLOT_REQUIRED_TEMPERATURE_MIN_K = 0.0
PLOT_REQUIRED_TEMPERATURE_MAX_K = 1300.0
NP_DISPLAY_ORDER = ["n", "p", "unknown"]
NP_STYLES = {
    "n": {"color": "#1769aa", "linestyle": "-", "label": "n-type"},
    "p": {"color": "#c62828", "linestyle": "--", "label": "p-type"},
    "unknown": {"color": "#666666", "linestyle": ":", "label": "unknown"},
}

POINT_ALIASES = {
    "sample_key": ["sample_key", "sample_id"],
    "sample_id": ["sample_id", "sample_id_raw"],
    "paper_id": ["paper_id", "DOI", "doi", "SID"],
    "composition": ["formula_raw", "composition", "formula"],
    "material_name": ["material_name_raw", "material_name", "material"],
    "material_system": ["material_family_raw", "material_system", "material_family"],
    "temperature": ["T_K", "temperature_K", "temperature_K_step11"],
    "sigma": [
        "sigma_exp_S_per_m",
        "sigma_obs_S_per_m_step11",
        "sigma_obs_S_per_m",
        "sigma_exp",
        "sigma_obs",
    ],
    "seebeck": [
        "seebeck_obs_V_per_K_step11",
        "seebeck_obs_V_per_K",
        "seebeck_obs",
    ],
    "n_or_p": ["carrier_type", "n_or_p", "n_or_p_final_step17", "n_or_p_step6"],
}
SOURCE_ALIASES = [
    "source",
    "fitting_source",
    "fitting_source_actual_step10",
    "sigma_obs_source_step11",
    "property_step5_source",
]
CURVE_REQUIRED = [
    "sample_key",
    "composition",
    "material_system",
    "n_or_p",
    "prop_x",
    "property_step5",
    "unit_x",
    "unit_y",
    "x_values_json",
    "y_values_json",
]

SIGMA_PROPERTY = "Electrical conductivity"
RHO_PROPERTY = "Electrical resistivity"
SEEBECK_PROPERTY = "Seebeck coefficient"

MATERIAL_SYSTEM_ORDER = [
    "host::BiSbTe",
    "host::PbTe",
    "host::SnTe",
    "host::GeTe",
    "host::BiTe",
    "host::SbTe",
    "host::SiGe",
    "host::Mg2SiSn",
    "host::CoSb_skutterudite",
    "host::oxide",
    "host::sulfide",
    "host::selenide",
    "host::telluride",
    "host::other_formula_system",
    "unknown_material_group",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Step31 empirical C(T) curves as median measured electrical "
            "conductivity in 25 K bins for reclassified material systems."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional point-table or curve-table CSV. If omitted, candidates are audited.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument("--dopant-threshold", type=float, default=0.05)
    parser.add_argument("--min-rows-per-group", type=int, default=30)
    parser.add_argument("--min-samples-per-group", type=int, default=3)
    parser.add_argument("--min-papers-per-group", type=int, default=1)
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Atomically replace an existing Step31 output directory after validation.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def first_present(columns: Iterable[str], aliases: Iterable[str]) -> str:
    column_set = set(columns)
    return next((name for name in aliases if name in column_set), "")


def point_mapping(columns: Iterable[str]) -> dict[str, str]:
    return {key: first_present(columns, aliases) for key, aliases in POINT_ALIASES.items()}


def candidate_paths() -> list[Path]:
    output = PROJECT_ROOT / "data" / "output"
    explicit = [
        output / "starrydata2_step18_tau_eff_ml_dataset" / "tau_eff_ml_dataset_recommended_step18.csv",
        output / "starrydata2_step18_tau_eff_ml_dataset" / "tau_eff_ml_metadata_step18.csv",
        output / "starrydata2_step16_result_summary" / "step16_material_np_summary.csv",
        output / "starrydata2_step15_pf_zt_error_analysis" / "pf_zt_error_samples_step15.csv",
    ]
    glob_patterns = [
        "starrydata2_step14_pf_zt_prediction/**/*.csv",
        "starrydata2_step11*/**/*.csv",
        "starrydata2_step12*/**/*.csv",
    ]
    fallbacks = [
        output / "starrydata2_step6_np_classification" / "candidate_core_curves_step6.csv",
        output / "starrydata2_step5_core_properties" / "candidate_core_curves_step5.csv",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for path in explicit:
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    for pattern in glob_patterns:
        for path in sorted(output.glob(pattern)):
            resolved = path.resolve()
            if resolved not in seen:
                paths.append(path)
                seen.add(resolved)
    for path in fallbacks:
        resolved = path.resolve()
        if resolved not in seen:
            paths.append(path)
            seen.add(resolved)
    return paths


def inspect_and_select_input(explicit_input: Path | None) -> tuple[Path, str, list[dict[str, Any]], dict[str, str]]:
    paths = [explicit_input] if explicit_input is not None else candidate_paths()
    audit: list[dict[str, Any]] = []
    selected: tuple[Path, str, dict[str, str]] | None = None
    for path in paths:
        assert path is not None
        record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if not path.exists():
            record.update(status="missing", columns=[])
            audit.append(record)
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as exc:
            record.update(status=f"header_read_error: {exc}", columns=[])
            audit.append(record)
            continue
        mapping = point_mapping(columns)
        required_point = ["sample_key", "temperature", "sigma"]
        missing_point = [key for key in required_point if not mapping.get(key)]
        missing_curve = [column for column in CURVE_REQUIRED if column not in columns]
        if not missing_point:
            mode = "point_table"
            status = "usable point table"
        elif not missing_curve:
            mode = "curve_table"
            status = "usable curve table"
        else:
            mode = ""
            status = (
                "not usable; point fields missing=" + ",".join(missing_point)
                + "; curve fields missing=" + ",".join(missing_curve)
            )
        record.update(
            status=status,
            n_columns=len(columns),
            point_mapping=mapping,
            source_columns=[column for column in SOURCE_ALIASES if column in columns],
        )
        audit.append(record)
        if selected is None and mode:
            selected = (path, mode, mapping)
            if explicit_input is not None:
                break
            if mode == "point_table":
                break
            # Curve tables are fallbacks at the end of the candidate list.
            if path.name in {"candidate_core_curves_step6.csv", "candidate_core_curves_step5.csv"}:
                break
    if selected is None:
        raise FileNotFoundError("No usable Step31 point-table or curve-table input was found")
    return (*selected[:2], audit, selected[2])


def parse_numeric_list(raw_value: Any) -> list[float]:
    if isinstance(raw_value, (list, tuple)):
        values = raw_value
    else:
        text = normalize_text(raw_value)
        if not text:
            return []
        try:
            values = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []
    if not isinstance(values, (list, tuple)):
        return []
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        parsed.append(number)
    return parsed


def normalized_unit(value: Any) -> str:
    return (
        normalize_text(value)
        .casefold()
        .replace(" ", "")
        .replace("µ", "u")
        .replace("μ", "u")
    )


def temperature_to_kelvin(value: float, unit: Any) -> float:
    normalized = normalized_unit(unit)
    if normalized in {"k", "kelvin"}:
        return value
    if normalized in {"c", "°c", "degc", "celsius"}:
        return value + 273.15
    return math.nan


def conductivity_factor(unit: Any) -> float:
    normalized = normalized_unit(unit)
    if normalized in {"ohm^(-1)*m^(-1)", "s*m^(-1)", "s/m", "a**2*s**2/kg/m**2"}:
        return 1.0
    if normalized in {"ohm^(-1)*cm^(-1)", "s*cm^(-1)", "s/cm"}:
        return 100.0
    return math.nan


def resistivity_factor(unit: Any) -> float:
    normalized = normalized_unit(unit)
    if normalized in {"ohm*m", "kg*m**3/a**2/s**3"}:
        return 1.0
    if normalized == "ohm*cm":
        return 0.01
    return math.nan


def seebeck_factor(unit: Any) -> float:
    normalized = normalized_unit(unit)
    if normalized in {"v*k^(-1)", "v/k", "kg*m**2/a/s**3"}:
        return 1.0
    if normalized in {"uv*k^(-1)", "uv/k"}:
        return 1e-6
    if normalized in {"mv*k^(-1)", "mv/k"}:
        return 1e-3
    return math.nan


def clear_np_label(value: Any) -> str:
    text = normalize_text(value).casefold().replace("_", "-").strip()
    if text in {"n", "n-type", "n type", "ntype"}:
        return "n"
    if text in {"p", "p-type", "p type", "ptype"}:
        return "p"
    return ""


def choose_sample_text(group: pd.Series) -> str:
    values = [normalize_text(value) for value in group if normalize_text(value)]
    if not values:
        return ""
    counts = Counter(values)
    return sorted(counts, key=lambda value: (-counts[value], values.index(value)))[0]


def infer_seebeck_factor_from_column(column: str, values: pd.Series) -> tuple[float, str]:
    if "V_per_K" in column or "V/K" in column:
        return 1.0, f"{column} explicitly indicates V/K"
    numeric = pd.to_numeric(values, errors="coerce")
    finite_abs = numeric[np.isfinite(numeric)].abs()
    median_abs = float(finite_abs.median()) if len(finite_abs) else math.nan
    if math.isfinite(median_abs) and median_abs > 0.05:
        return 1e-6, f"unit inferred as microV/K because median abs(S)={median_abs:.6g}"
    return 1.0, f"unit inferred as V/K because median abs(S)={median_abs:.6g}"


def load_point_table(
    path: Path, mapping: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str], dict[str, Any]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    source_columns = [column for column in SOURCE_ALIASES if column in header]
    usecols = sorted({value for value in mapping.values() if value} | set(source_columns))
    raw = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False)
    stats: Counter[str] = Counter(input_rows=len(raw))

    def mapped_text(key: str) -> pd.Series:
        column = mapping.get(key, "")
        if column:
            return raw[column].map(normalize_text)
        return pd.Series("", index=raw.index, dtype=object)

    if mapping.get("seebeck"):
        seebeck_numeric = pd.to_numeric(raw[mapping["seebeck"]], errors="coerce")
        factor, inference = infer_seebeck_factor_from_column(mapping["seebeck"], seebeck_numeric)
    else:
        seebeck_numeric = pd.Series(np.nan, index=raw.index, dtype=float)
        factor, inference = 1.0, "Seebeck column not found; unavailable values retained as missing"
    raw["_seebeck_V_per_K"] = seebeck_numeric * factor
    sample_seebeck = raw.groupby(mapping["sample_key"], sort=False)["_seebeck_V_per_K"].apply(list)
    records = pd.DataFrame(
        {
            "sample_key": mapped_text("sample_key"),
            "sample_id": mapped_text("sample_id"),
            "paper_id": mapped_text("paper_id"),
            "formula_raw": mapped_text("composition"),
            "material_name_raw": mapped_text("material_name"),
            "material_family_raw": mapped_text("material_system"),
            "carrier_type": mapped_text("n_or_p"),
            "composition": mapped_text("composition"),
            "material_system_original": mapped_text("material_system"),
            "n_or_p_original": mapped_text("n_or_p"),
            "T_K": pd.to_numeric(raw[mapping["temperature"]], errors="coerce"),
            "temperature_K": pd.to_numeric(raw[mapping["temperature"]], errors="coerce"),
            "sigma_exp_S_per_m": pd.to_numeric(raw[mapping["sigma"]], errors="coerce"),
        }
    )
    for column in source_columns:
        records[column] = raw[column]
    missing_sample = records["sample_key"].eq("")
    invalid_temperature = ~np.isfinite(records["temperature_K"])
    invalid_sigma = ~np.isfinite(records["sigma_exp_S_per_m"])
    nonpositive_sigma = records["sigma_exp_S_per_m"].le(0) & ~invalid_sigma
    stats["excluded_missing_sample_key_points"] = int(missing_sample.sum())
    stats["excluded_missing_or_nonfinite_temperature_points"] = int(invalid_temperature.sum())
    stats["excluded_nonfinite_sigma_points"] = int(invalid_sigma.sum())
    stats["excluded_nonpositive_sigma_points"] = int(nonpositive_sigma.sum())
    keep = ~(missing_sample | invalid_temperature | invalid_sigma | nonpositive_sigma)
    records = records.loc[keep].reset_index(drop=True)
    stats["valid_sigma_points"] = len(records)
    np_source = pd.DataFrame(
        {
            "sample_key": mapped_text("sample_key"),
            "n_or_p_original": mapped_text("n_or_p"),
        }
    )
    metadata = {
        "seebeck_unit_inference": inference,
        "source_columns": source_columns,
        "input_column_mapping": mapping,
    }
    return records, np_source, stats, {**metadata, "sample_seebeck": sample_seebeck.to_dict()}


def load_curve_table(
    path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str], dict[str, Any]]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    optional = [
        "curve_key",
        "sample_id",
        "DOI",
        "SID",
        "property_step5_source",
        "fitting_source_actual_step10",
        "sigma_obs_source_step11",
    ]
    usecols = CURVE_REQUIRED + [column for column in optional if column in header]
    raw = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=False, low_memory=False)
    stats: Counter[str] = Counter(input_rows=len(raw))
    sigma_records: list[dict[str, Any]] = []
    np_records: list[dict[str, str]] = []
    seebeck_by_sample: defaultdict[str, list[float]] = defaultdict(list)
    sigma_units: Counter[str] = Counter()
    rho_units: Counter[str] = Counter()
    seebeck_units: Counter[str] = Counter()
    source_columns = [column for column in SOURCE_ALIASES if column in raw.columns]
    if "curve_key" in raw.columns:
        source_columns = ["curve_key", *source_columns]

    for row in raw.itertuples(index=False):
        sample_key = normalize_text(getattr(row, "sample_key"))
        np_records.append({"sample_key": sample_key, "n_or_p_original": normalize_text(getattr(row, "n_or_p"))})
        property_name = normalize_text(getattr(row, "property_step5"))
        if normalize_text(getattr(row, "prop_x")).casefold() != "temperature":
            continue
        if property_name not in {SIGMA_PROPERTY, RHO_PROPERTY, SEEBECK_PROPERTY}:
            continue
        x_values = parse_numeric_list(getattr(row, "x_values_json"))
        y_values = parse_numeric_list(getattr(row, "y_values_json"))
        if not x_values or not y_values:
            stats["excluded_parse_failed_curves"] += 1
            continue
        if len(x_values) != len(y_values):
            stats["xy_length_mismatch_curves"] += 1
        pairs = min(len(x_values), len(y_values))
        unit_y = getattr(row, "unit_y")
        unit_x = getattr(row, "unit_x")
        if property_name == SIGMA_PROPERTY:
            sigma_units[normalize_text(unit_y)] += 1
            property_factor = conductivity_factor(unit_y)
        elif property_name == RHO_PROPERTY:
            rho_units[normalize_text(unit_y)] += 1
            property_factor = resistivity_factor(unit_y)
        else:
            seebeck_units[normalize_text(unit_y)] += 1
            property_factor = seebeck_factor(unit_y)
        if not math.isfinite(property_factor):
            stats["excluded_unsupported_unit_curves"] += 1
            stats["excluded_unsupported_unit_points"] += pairs
            continue
        for index in range(pairs):
            temperature = temperature_to_kelvin(x_values[index], unit_x)
            value = y_values[index]
            if not math.isfinite(temperature):
                stats["excluded_missing_or_nonfinite_temperature_points"] += 1
                continue
            if not math.isfinite(value):
                stats["excluded_nonfinite_property_points"] += 1
                continue
            if property_name == SEEBECK_PROPERTY:
                if sample_key:
                    seebeck_by_sample[sample_key].append(value * property_factor)
                continue
            if not sample_key:
                stats["excluded_missing_sample_key_points"] += 1
                continue
            if value <= 0:
                stats["excluded_nonpositive_sigma_points"] += 1
                continue
            sigma_value = value * property_factor if property_name == SIGMA_PROPERTY else 1.0 / (value * property_factor)
            if not math.isfinite(sigma_value) or sigma_value <= 0:
                stats["excluded_nonpositive_sigma_points"] += 1
                continue
            record = {
                "sample_key": sample_key,
                "sample_id": normalize_text(getattr(row, "sample_id", "")),
                "paper_id": normalize_text(getattr(row, "DOI", ""))
                or normalize_text(getattr(row, "SID", "")),
                "formula_raw": normalize_text(getattr(row, "composition")),
                "material_name_raw": "",
                "material_family_raw": normalize_text(getattr(row, "material_system")),
                "carrier_type": normalize_text(getattr(row, "n_or_p")),
                "composition": normalize_text(getattr(row, "composition")),
                "material_system_original": normalize_text(getattr(row, "material_system")),
                "n_or_p_original": normalize_text(getattr(row, "n_or_p")),
                "T_K": temperature,
                "temperature_K": temperature,
                "sigma_exp_S_per_m": sigma_value,
            }
            for column in source_columns:
                record[column] = normalize_text(getattr(row, column))
            sigma_records.append(record)
    records = pd.DataFrame(sigma_records)
    stats["valid_sigma_points"] = len(records)
    stats["samples_with_seebeck"] = len(seebeck_by_sample)
    metadata = {
        "sample_seebeck": dict(seebeck_by_sample),
        "seebeck_unit_inference": "curve units read explicitly and converted to V/K",
        "sigma_curve_units": dict(sigma_units),
        "rho_curve_units": dict(rho_units),
        "seebeck_curve_units": dict(seebeck_units),
        "source_columns": source_columns,
        "input_column_mapping": {
            "sample_key": "sample_key",
            "composition": "composition",
            "material_system": "material_system",
            "temperature": "x_values_json where prop_x=Temperature",
            "sigma": "y_values_json for conductivity or inverse resistivity",
            "seebeck": "y_values_json for Seebeck coefficient",
            "n_or_p": "n_or_p",
        },
    }
    return records, pd.DataFrame(np_records), stats, metadata


def sample_np_reclassification(
    points: pd.DataFrame,
    np_source: pd.DataFrame,
    sample_seebeck: dict[str, list[float]],
) -> pd.DataFrame:
    all_samples = points["sample_key"].drop_duplicates().tolist()
    existing_by_sample = defaultdict(list)
    for row in np_source.itertuples(index=False):
        sample_key = normalize_text(getattr(row, "sample_key"))
        if sample_key:
            existing_by_sample[sample_key].append(getattr(row, "n_or_p_original"))
    rows: list[dict[str, Any]] = []
    for sample_key in all_samples:
        seebeck_values = np.asarray(sample_seebeck.get(sample_key, []), dtype=float)
        seebeck_values = seebeck_values[np.isfinite(seebeck_values)]
        median_s = float(np.median(seebeck_values)) if len(seebeck_values) else math.nan
        has_positive = bool(np.any(seebeck_values > SEEBECK_ZERO_THRESHOLD_V_PER_K))
        has_negative = bool(np.any(seebeck_values < -SEEBECK_ZERO_THRESHOLD_V_PER_K))
        sign_mixed = has_positive and has_negative
        clear_values = {label for value in existing_by_sample[sample_key] if (label := clear_np_label(value))}
        if len(clear_values) == 1:
            reclassified = next(iter(clear_values))
            basis = "existing clear n_or_p"
        elif math.isfinite(median_s) and abs(median_s) >= SEEBECK_ZERO_THRESHOLD_V_PER_K:
            reclassified = "p" if median_s > 0 else "n"
            basis = "sample median Seebeck sign"
        else:
            reclassified = "unknown"
            basis = "no clear existing n_or_p and no nonzero median Seebeck"
        rows.append(
            {
                "sample_key": sample_key,
                "n_or_p_reclass_step31": reclassified,
                "np_reclassification_basis_step31": basis,
                "sign_mixed_step31": sign_mixed,
                "seebeck_obs_V_per_K": median_s,
            }
        )
    return pd.DataFrame(rows)


def prepare_points(
    points: pd.DataFrame,
    np_source: pd.DataFrame,
    sample_seebeck: dict[str, list[float]],
    source_columns: list[str],
    dopant_threshold: float,
    min_rows_per_group: int,
    min_samples_per_group: int,
    min_papers_per_group: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    np_by_sample = sample_np_reclassification(points, np_source, sample_seebeck)
    output = points.merge(np_by_sample, on="sample_key", how="left", validate="many_to_one")
    output, system_counts = add_material_group_keys(
        output,
        dopant_threshold=dopant_threshold,
        min_rows_per_group=min_rows_per_group,
        min_samples_per_group=min_samples_per_group,
        min_papers_per_group=min_papers_per_group,
    )
    # Step31 C(T) remains a stable mother-phase baseline. The finer
    # final_material_group_key is exported for downstream analysis.
    output["material_system_reclass_step31"] = output["host_family_key"]
    output["temp_bin_center_K"] = (
        np.floor(output["temperature_K"].astype(float) / BIN_WIDTH_K + 0.5) * BIN_WIDTH_K
    )
    ordered = [
        "sample_key",
        "sample_id",
        "paper_id",
        "formula_raw",
        "material_name_raw",
        "material_family_raw",
        "carrier_type",
        "composition",
        "material_system_original",
        "material_system_reclass_step31",
        "parsed_elements",
        "element_fraction",
        "major_elements",
        "dopant_elements",
        "major_element_system_key",
        "host_family_key",
        "final_material_group_key",
        "group_backoff_level",
        "multiple_host_candidate",
        "host_candidate_keys",
        "parse_status",
        "parse_source",
        "classification_reason",
        "n_or_p_original",
        "n_or_p_reclass_step31",
        "sign_mixed_step31",
        "T_K",
        "temperature_K",
        "temp_bin_center_K",
        "sigma_exp_S_per_m",
        "seebeck_obs_V_per_K",
        "paper_key_step31",
    ]
    ordered.extend(column for column in source_columns if column in output.columns and column not in ordered)
    output = output[ordered].sort_values(
        ["material_system_reclass_step31", "sample_key", "temperature_K"], kind="mergesort"
    ).reset_index(drop=True)
    return output, np_by_sample, system_counts


def aggregate_ct(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = (
        points.groupby(["material_system_reclass_step31", "temp_bin_center_K"], sort=True)
        .agg(
            C_T_sigma_median_S_per_m_step31=("sigma_exp_S_per_m", "median"),
            sigma_q25_S_per_m=("sigma_exp_S_per_m", lambda values: values.quantile(0.25)),
            sigma_q75_S_per_m=("sigma_exp_S_per_m", lambda values: values.quantile(0.75)),
            n_rows=("sigma_exp_S_per_m", "size"),
            n_samples=("sample_key", "nunique"),
        )
        .reset_index()
    )
    primary["plot_in_primary"] = primary["n_rows"].ge(MIN_ROWS_PER_PRIMARY_BIN)
    np_split = (
        points.groupby(
            ["material_system_reclass_step31", "n_or_p_reclass_step31", "temp_bin_center_K"],
            sort=True,
        )
        .agg(
            C_T_sigma_median_S_per_m_step31=("sigma_exp_S_per_m", "median"),
            n_rows=("sigma_exp_S_per_m", "size"),
            n_samples=("sample_key", "nunique"),
        )
        .reset_index()
    )
    np_split["plot_in_np_split"] = np_split["n_rows"].ge(MIN_ROWS_PER_PRIMARY_BIN)
    return primary, np_split


def make_summaries(
    points: pd.DataFrame, primary: pd.DataFrame, np_by_sample: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    material = (
        points.groupby("material_system_reclass_step31", sort=False)
        .agg(
            n_rows=("sigma_exp_S_per_m", "size"),
            n_samples=("sample_key", "nunique"),
            temperature_min_K=("temperature_K", "min"),
            temperature_max_K=("temperature_K", "max"),
            sigma_min_S_per_m=("sigma_exp_S_per_m", "min"),
            sigma_max_S_per_m=("sigma_exp_S_per_m", "max"),
        )
        .reset_index()
    )
    bin_summary = (
        primary.groupby("material_system_reclass_step31", sort=False)
        .agg(
            n_temperature_bins=("temp_bin_center_K", "size"),
            n_primary_temperature_bins=("plot_in_primary", "sum"),
        )
        .reset_index()
    )
    material = material.merge(bin_summary, on="material_system_reclass_step31", how="left")
    material["_order"] = material["material_system_reclass_step31"].map(
        {name: index for index, name in enumerate(MATERIAL_SYSTEM_ORDER)}
    )
    material = material.sort_values(["_order", "material_system_reclass_step31"]).drop(columns="_order")

    originals = points[["sample_key", "n_or_p_original"]].copy()
    originals["n_or_p_original_normalized_step31"] = originals["n_or_p_original"].map(
        lambda value: clear_np_label(value) or normalize_text(value).casefold() or "blank"
    )
    sample_original = (
        originals.groupby("sample_key", sort=False)["n_or_p_original_normalized_step31"]
        .agg(choose_sample_text)
        .reset_index()
    )
    np_summary_source = np_by_sample.merge(sample_original, on="sample_key", how="left")
    np_summary = (
        np_summary_source.groupby(
            [
                "n_or_p_original_normalized_step31",
                "n_or_p_reclass_step31",
                "np_reclassification_basis_step31",
                "sign_mixed_step31",
            ],
            dropna=False,
            sort=True,
        )
        .agg(n_samples=("sample_key", "nunique"))
        .reset_index()
    )
    return material, np_summary


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return slug or "unclassified"


def set_temperature_axis_range(ax: plt.Axes, temperatures: Iterable[float]) -> None:
    finite = np.asarray(list(temperatures), dtype=float)
    finite = finite[np.isfinite(finite)]
    observed_max = float(finite.max()) if len(finite) else PLOT_REQUIRED_TEMPERATURE_MAX_K
    upper = max(PLOT_REQUIRED_TEMPERATURE_MAX_K, math.ceil(observed_max / 100.0) * 100.0)
    ax.set_xlim(PLOT_REQUIRED_TEMPERATURE_MIN_K, upper)


def style_axis(ax: plt.Axes, temperatures: Iterable[float]) -> None:
    ax.set_xlabel("Temperature [K]", fontsize=14)
    ax.set_ylabel(r"$C(T)$, median $\sigma_{\mathrm{exp}}$ [S/m]", fontsize=14)
    ax.set_yscale("log")
    set_temperature_axis_range(ax, temperatures)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
    ax.grid(True, which="minor", axis="y", alpha=0.10, linewidth=0.5)


def plotted_systems(material_summary: pd.DataFrame) -> list[str]:
    eligible = material_summary[material_summary["n_primary_temperature_bins"].gt(0)].copy()
    return eligible.sort_values(["n_samples", "n_rows"], ascending=False)[
        "material_system_reclass_step31"
    ].tolist()


def plot_overlay(
    primary: pd.DataFrame,
    systems: list[str],
    path: Path,
    title: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(systems), 1)))
    plotted_temperatures: list[float] = []
    for color, system in zip(colors, systems):
        curve = primary[
            primary["material_system_reclass_step31"].eq(system)
            & primary["plot_in_primary"]
        ].sort_values("temp_bin_center_K")
        ax.plot(
            curve["temp_bin_center_K"],
            curve["C_T_sigma_median_S_per_m_step31"],
            linestyle="-",
            linewidth=2.3,
            color=color,
            label=system,
        )
        plotted_temperatures.extend(curve["temp_bin_center_K"].tolist())
    style_axis(ax, plotted_temperatures)
    ax.set_title(title, fontsize=16, pad=12)
    ax.legend(fontsize=10.5, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_figures(
    primary: pd.DataFrame,
    material_summary: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> tuple[list[Path], list[str], list[str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    systems = plotted_systems(material_summary)
    all_plot_systems = systems[:10] if len(systems) > 10 else systems
    grid_systems = systems[:12]
    figure_paths: list[Path] = []
    main_overlay = figures_dir / "figure_step31_all_material_system_ct_curves.png"
    plot_overlay(
        primary,
        all_plot_systems,
        main_overlay,
        "Step31 empirical C(T): reclassified material-system medians",
        dpi,
    )
    figure_paths.append(main_overlay)
    if len(systems) > len(all_plot_systems):
        complete_overlay = figures_dir / "figure_step31_all_material_system_ct_curves_complete.png"
        plot_overlay(
            primary,
            systems,
            complete_overlay,
            "Step31 empirical C(T): all reclassified material systems",
            dpi,
        )
        figure_paths.append(complete_overlay)

    summary_index = material_summary.set_index("material_system_reclass_step31")
    for system in systems:
        curve = primary[
            primary["material_system_reclass_step31"].eq(system)
            & primary["plot_in_primary"]
        ].sort_values("temp_bin_center_K")
        sample_count = int(summary_index.loc[system, "n_samples"])
        bin_count = int(curve["temp_bin_center_K"].nunique())
        fig, ax = plt.subplots(figsize=(9.0, 6.3))
        ax.plot(
            curve["temp_bin_center_K"],
            curve["C_T_sigma_median_S_per_m_step31"],
            color="#1769aa",
            linewidth=3.0,
            linestyle="-",
            label=r"Step31 $C(T)$",
        )
        style_axis(ax, curve["temp_bin_center_K"])
        ax.set_title(f"{system}\n{sample_count:,} samples; {bin_count:,} temperature bins", fontsize=16, pad=12)
        ax.legend(fontsize=12, frameon=False)
        fig.tight_layout()
        path = figures_dir / f"figure_step31_ct_curve_{safe_slug(system)}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(path)

    if grid_systems:
        ncols = 3 if len(grid_systems) > 4 else 2
        nrows = math.ceil(len(grid_systems) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False)
        for ax, system in zip(axes.flat, grid_systems):
            curve = primary[
                primary["material_system_reclass_step31"].eq(system)
                & primary["plot_in_primary"]
            ].sort_values("temp_bin_center_K")
            ax.plot(
                curve["temp_bin_center_K"],
                curve["C_T_sigma_median_S_per_m_step31"],
                color="#1769aa",
                linewidth=2.7,
                linestyle="-",
                label=r"$C(T)$",
            )
            ax.set_title(system, fontsize=13)
            ax.set_yscale("log")
            set_temperature_axis_range(ax, curve["temp_bin_center_K"])
            ax.set_xlabel("Temperature [K]", fontsize=11)
            ax.set_ylabel(r"$C(T)$ [S/m]", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
        for ax in axes.flat[len(grid_systems):]:
            ax.set_visible(False)
        fig.suptitle("Step31 empirical C(T) by reclassified material system", fontsize=17, y=1.01)
        fig.tight_layout()
        grid_path = figures_dir / "figure_step31_ct_curves_grid.png"
        fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(grid_path)
    return figure_paths, all_plot_systems, grid_systems


def plotted_np_split_systems(np_split: pd.DataFrame) -> list[str]:
    available = set(
        np_split.loc[np_split["plot_in_np_split"], "material_system_reclass_step31"]
    )
    return [system for system in MATERIAL_SYSTEM_ORDER if system in available]


def plot_np_carrier_overlay(
    np_split: pd.DataFrame,
    carrier: str,
    systems: list[str],
    path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 8.0))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(systems), 1)))
    plotted_temperatures: list[float] = []
    for color, system in zip(colors, systems):
        curve = np_split[
            np_split["material_system_reclass_step31"].eq(system)
            & np_split["n_or_p_reclass_step31"].eq(carrier)
            & np_split["plot_in_np_split"]
        ].sort_values("temp_bin_center_K")
        if curve.empty:
            continue
        ax.plot(
            curve["temp_bin_center_K"],
            curve["C_T_sigma_median_S_per_m_step31"],
            linewidth=2.3,
            linestyle="-",
            color=color,
            label=system,
        )
        plotted_temperatures.extend(curve["temp_bin_center_K"].tolist())
    style_axis(ax, plotted_temperatures)
    carrier_label = NP_STYLES.get(carrier, NP_STYLES["unknown"])["label"]
    ax.set_title(f"Step31 empirical C(T): {carrier_label} material systems", fontsize=16, pad=12)
    ax.legend(fontsize=10.5, ncol=2, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def create_single_carrier_system_figures(
    np_split: pd.DataFrame,
    sample_counts: dict[tuple[str, str], int],
    systems: list[str],
    carrier: str,
    np_figures_dir: Path,
    dpi: int,
) -> list[Path]:
    style = NP_STYLES[carrier]
    carrier_dir = np_figures_dir / f"{carrier}_only"
    carrier_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []
    carrier_systems: list[str] = []

    for system in systems:
        curve = np_split[
            np_split["material_system_reclass_step31"].eq(system)
            & np_split["n_or_p_reclass_step31"].eq(carrier)
            & np_split["plot_in_np_split"]
        ].sort_values("temp_bin_center_K")
        if curve.empty:
            continue
        carrier_systems.append(system)
        sample_count = int(sample_counts.get((system, carrier), 0))
        bin_count = int(curve["temp_bin_center_K"].nunique())
        fig, ax = plt.subplots(figsize=(9.0, 6.3))
        ax.plot(
            curve["temp_bin_center_K"],
            curve["C_T_sigma_median_S_per_m_step31"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=3.0,
            label=style["label"],
        )
        style_axis(ax, curve["temp_bin_center_K"])
        ax.set_title(
            f"{system}: {style['label']} C(T)\n"
            f"{sample_count:,} samples; {bin_count:,} temperature bins",
            fontsize=16,
            pad=12,
        )
        ax.legend(fontsize=12, frameon=False)
        fig.tight_layout()
        path = carrier_dir / f"figure_step31_ct_curve_{carrier}_{safe_slug(system)}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(path)

    if carrier_systems:
        ncols = 3 if len(carrier_systems) > 4 else 2
        nrows = math.ceil(len(carrier_systems) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False)
        for ax, system in zip(axes.flat, carrier_systems):
            curve = np_split[
                np_split["material_system_reclass_step31"].eq(system)
                & np_split["n_or_p_reclass_step31"].eq(carrier)
                & np_split["plot_in_np_split"]
            ].sort_values("temp_bin_center_K")
            ax.plot(
                curve["temp_bin_center_K"],
                curve["C_T_sigma_median_S_per_m_step31"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.5,
            )
            ax.set_title(system, fontsize=13)
            ax.set_yscale("log")
            set_temperature_axis_range(ax, curve["temp_bin_center_K"])
            ax.set_xlabel("Temperature [K]", fontsize=11)
            ax.set_ylabel(r"$C(T)$ [S/m]", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
        for ax in axes.flat[len(carrier_systems):]:
            ax.set_visible(False)
        fig.suptitle(
            f"Step31 {style['label']} empirical C(T) by material system",
            fontsize=17,
            y=1.01,
        )
        fig.tight_layout()
        grid_path = np_figures_dir / f"figure_step31_ct_curves_{carrier}_only_grid.png"
        fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(grid_path)
    return figure_paths


def create_np_split_figures(
    np_split: pd.DataFrame,
    points: pd.DataFrame,
    figures_dir: Path,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    np_figures_dir = figures_dir / "np_split"
    np_figures_dir.mkdir(parents=True, exist_ok=True)
    systems = plotted_np_split_systems(np_split)
    figure_paths: list[Path] = []
    sample_counts = (
        points[
            ["sample_key", "material_system_reclass_step31", "n_or_p_reclass_step31"]
        ]
        .drop_duplicates()
        .groupby(["material_system_reclass_step31", "n_or_p_reclass_step31"])["sample_key"]
        .nunique()
        .to_dict()
    )

    for system in systems:
        system_curves = np_split[
            np_split["material_system_reclass_step31"].eq(system)
            & np_split["plot_in_np_split"]
        ].copy()
        fig, ax = plt.subplots(figsize=(9.0, 6.3))
        plotted_temperatures: list[float] = []
        for carrier in NP_DISPLAY_ORDER:
            curve = system_curves[
                system_curves["n_or_p_reclass_step31"].eq(carrier)
            ].sort_values("temp_bin_center_K")
            if curve.empty:
                continue
            style = NP_STYLES[carrier]
            count = int(sample_counts.get((system, carrier), 0))
            ax.plot(
                curve["temp_bin_center_K"],
                curve["C_T_sigma_median_S_per_m_step31"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=3.0,
                label=f"{style['label']} ({count:,} samples)",
            )
            plotted_temperatures.extend(curve["temp_bin_center_K"].tolist())
        style_axis(ax, plotted_temperatures)
        ax.set_title(f"{system}: n/p-split C(T)", fontsize=16, pad=12)
        ax.legend(fontsize=12, frameon=False)
        fig.tight_layout()
        path = np_figures_dir / f"figure_step31_ct_curve_np_split_{safe_slug(system)}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(path)

    if systems:
        ncols = 3 if len(systems) > 4 else 2
        nrows = math.ceil(len(systems) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.9 * nrows), squeeze=False)
        legend_handles: dict[str, Any] = {}
        for ax, system in zip(axes.flat, systems):
            system_curves = np_split[
                np_split["material_system_reclass_step31"].eq(system)
                & np_split["plot_in_np_split"]
            ]
            plotted_temperatures: list[float] = []
            for carrier in NP_DISPLAY_ORDER:
                curve = system_curves[
                    system_curves["n_or_p_reclass_step31"].eq(carrier)
                ].sort_values("temp_bin_center_K")
                if curve.empty:
                    continue
                style = NP_STYLES[carrier]
                handle, = ax.plot(
                    curve["temp_bin_center_K"],
                    curve["C_T_sigma_median_S_per_m_step31"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.5,
                    label=style["label"],
                )
                legend_handles.setdefault(carrier, handle)
                plotted_temperatures.extend(curve["temp_bin_center_K"].tolist())
            ax.set_title(system, fontsize=13)
            ax.set_yscale("log")
            set_temperature_axis_range(ax, plotted_temperatures)
            ax.set_xlabel("Temperature [K]", fontsize=11)
            ax.set_ylabel(r"$C(T)$ [S/m]", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(True, which="major", alpha=0.20, linewidth=0.6)
        for ax in axes.flat[len(systems):]:
            ax.set_visible(False)
        ordered_handles = [legend_handles[key] for key in NP_DISPLAY_ORDER if key in legend_handles]
        ordered_labels = [NP_STYLES[key]["label"] for key in NP_DISPLAY_ORDER if key in legend_handles]
        if ordered_handles:
            fig.legend(
                ordered_handles,
                ordered_labels,
                loc="upper center",
                ncol=len(ordered_handles),
                frameon=False,
                fontsize=12,
                bbox_to_anchor=(0.5, 1.005),
            )
        fig.suptitle("Step31 n/p-split empirical C(T) by material system", fontsize=17, y=1.025)
        fig.tight_layout()
        grid_path = np_figures_dir / "figure_step31_ct_curves_np_split_grid.png"
        fig.savefig(grid_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(grid_path)

    for carrier in ["n", "p"]:
        carrier_systems = [
            system
            for system in systems
            if np_split[
                np_split["material_system_reclass_step31"].eq(system)
                & np_split["n_or_p_reclass_step31"].eq(carrier)
                & np_split["plot_in_np_split"]
            ].shape[0]
            > 0
        ]
        if not carrier_systems:
            continue
        overlay_path = np_figures_dir / f"figure_step31_all_material_system_ct_curves_{carrier}.png"
        plot_np_carrier_overlay(np_split, carrier, carrier_systems, overlay_path, dpi)
        figure_paths.append(overlay_path)
        figure_paths.extend(
            create_single_carrier_system_figures(
                np_split,
                sample_counts,
                carrier_systems,
                carrier,
                np_figures_dir,
                dpi,
            )
        )
    return figure_paths, systems


def write_notes(path: Path) -> None:
    path.write_text(
        "# Step31 C(T) の定義\n\n"
        "Step31 では、組成から新たに再分類した材料系ごとに、25 K 幅の温度 bin 内にある "
        "Starrydata2 由来の実測電気伝導率 $\\sigma_{\\mathrm{exp}}(T)$ の中央値を求め、"
        "その温度依存曲線を経験的基準曲線 $C(T)$ と定義した。primary の $C(T)$ は n/p 型で"
        "分割せず、各 bin に3点以上の実測値がある区間だけを図示した。\n\n"
        "この $C(T)$ は Step31 で新たに導入した論文・説明用の基準曲線であり、Step12 の "
        "tau_eff fitting で用いた経験的 prefactor $C(T)$ とは定義が異なる。Step12 の出力は"
        "変更していない。今後の tau_eff は、必要に応じてこの Step31 の $C(T)$ に対する"
        "相対倍率として解釈する想定である。\n\n"
        "組成式から元素比を推定し、既定では 5 at% 以上を主成分、5 at% 未満をドーパント候補とした。"
        "母相は `host_family_key`、主成分元素系は `major_element_system_key` として保存し、後者が"
        "行数・試料数・論文数の基準を満たす場合だけ `final_material_group_key` に採用した。基準未満"
        "では母相へ戻すため、微量置換による過分割を抑えている。Ge–Te は `host::GeTe` として独立"
        "分類した。元素抽出可能だが既知母相に該当しないものは `host::other_formula_system`、抽出"
        "不能なものは `unknown_material_group` とした。Step31 の C(T) は安定した母相比較のため "
        "`host_family_key` ごとに集計し、細分類キーは解析用CSVに保持した。\n\n"
        "すべての図の横軸は少なくとも 0–1300 K を含み、1300 K を超える実測 bin がある場合は"
        "その最大温度を切り捨てない範囲まで拡張した。\n",
        encoding="utf-8",
    )


def format_mapping(mapping: dict[str, str]) -> str:
    return ", ".join(f"{key}={value or '[not found]'}" for key, value in mapping.items())


def write_report(
    path: Path,
    input_path: Path,
    input_mode: str,
    audit: list[dict[str, Any]],
    metadata: dict[str, Any],
    stats: Counter[str],
    points: pd.DataFrame,
    material_summary: pd.DataFrame,
    np_summary: pd.DataFrame,
    primary: pd.DataFrame,
    material_group_counts: pd.DataFrame,
    parse_failure_examples: pd.DataFrame,
    figure_paths: list[Path],
    all_plot_systems: list[str],
    grid_systems: list[str],
    replaced_existing_step31: bool,
    dopant_threshold: float,
    min_rows_per_group: int,
    min_samples_per_group: int,
    min_papers_per_group: int,
) -> None:
    lines = [
        "Step31 reclassified material-system C(T) report",
        "=" * 55,
        "",
        "Definition",
        "----------",
        "C(T) is newly defined in Step31 as the median measured sigma_exp in each 25 K bin",
        "for each host_family_key. The primary C(T) is not split by n/p.",
        "Finer final_material_group_key values are exported for analysis but do not fragment the host-level C(T).",
        "This definition is separate from the empirical prefactor C(T) used by Step12; no Step12 file was modified.",
        "",
        "Selected input",
        "--------------",
        f"path: {input_path}",
        f"mode: {input_mode}",
        f"column mapping: {format_mapping(metadata['input_column_mapping'])}",
        f"preserved source columns: {metadata.get('source_columns', [])}",
        f"Seebeck unit handling: {metadata.get('seebeck_unit_inference', '')}",
    ]
    if not metadata["input_column_mapping"].get("composition") and not metadata["input_column_mapping"].get("material_name"):
        lines.append(
            "CONFIRMATION REQUIRED: no formula/composition or material-name column was found; material groups are left unknown."
        )
    for key in ["sigma_curve_units", "rho_curve_units", "seebeck_curve_units"]:
        if metadata.get(key):
            lines.append(f"{key}: {metadata[key]}")
    lines.extend(["", "Candidate input audit", "---------------------"])
    for record in audit:
        lines.append(f"- {record['path']}: {record['status']}")
        if record.get("point_mapping"):
            lines.append(f"  point aliases: {format_mapping(record['point_mapping'])}")
    lines.extend(["", "Filtering and binning", "---------------------"])
    for key, value in sorted(stats.items()):
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            f"temperature_bin_width_K: {BIN_WIDTH_K:g}",
            "temperature bin convention: floor(T / 25 + 0.5) * 25 (nearest center; half-up)",
            f"minimum rows per plotted primary bin: {MIN_ROWS_PER_PRIMARY_BIN}",
            "sigma condition: finite and strictly positive",
            "temperature condition: finite after conversion to kelvin",
            "bins with fewer than 3 rows are retained in CSV and excluded from all primary plots",
            "outlier handling: no positive finite sigma value was clipped; extreme input values remain in C(T) and can affect plot limits",
            f"dopant threshold: {dopant_threshold:.6g} (fraction >= threshold is major; fraction < threshold is dopant candidate)",
            f"final-group thresholds: min_rows={min_rows_per_group}, min_samples={min_samples_per_group}, min_papers={min_papers_per_group}",
            "catch-all split: host::other_formula_system=valid elements extracted but no named rule matched; unknown_material_group=no valid elements extracted from formula/name/family",
            "GeTe handling: Ge and Te as major elements are classified as host::GeTe after higher-priority BiSbTe/PbTe/SnTe rules",
            f"plot x-axis policy: every panel includes at least {PLOT_REQUIRED_TEMPERATURE_MIN_K:g}-{PLOT_REQUIRED_TEMPERATURE_MAX_K:g} K; data above 1300 K are retained",
            "",
            "Hierarchical material grouping",
            "------------------------------",
            f"parse_status counts: {points['parse_status'].value_counts(dropna=False).to_dict()}",
            f"group_backoff_level counts: {points['group_backoff_level'].value_counts(dropna=False).to_dict()}",
            f"host_family_key groups: {points['host_family_key'].nunique()}",
            f"major_element_system_key groups: {points.loc[points['major_element_system_key'].ne(''), 'major_element_system_key'].nunique()}",
            f"final_material_group_key groups: {points['final_material_group_key'].nunique()}",
            f"low-confidence/parse-failure rows exported: {len(parse_failure_examples)}",
            f"material_group_counts rows: {len(material_group_counts)}",
            "",
            "Material-system classification counts",
            "-------------------------------------",
        ]
    )
    for row in material_summary.itertuples(index=False):
        lines.append(
            f"- {row.material_system_reclass_step31}: rows={row.n_rows}, samples={row.n_samples}, "
            f"bins={row.n_temperature_bins}, plotted_bins={row.n_primary_temperature_bins}"
        )
    ge_te_mask = points["parsed_elements"].map(
        lambda value: {"Ge", "Te"}.issubset(set(normalize_text(value).split(";")))
    )
    ge_te_points = points.loc[ge_te_mask]
    ge_te_by_group = (
        ge_te_points.groupby("material_system_reclass_step31")["sample_key"]
        .nunique()
        .sort_values(ascending=False)
        .to_dict()
    )
    lines.extend(
        [
            "",
            "Ge-Te-containing records",
            "------------------------",
            f"rows containing both Ge and Te: {len(ge_te_points)}",
            f"samples containing both Ge and Te: {ge_te_points['sample_key'].nunique()}",
            f"sample counts by assigned Step31 group: {ge_te_by_group}",
            "GeTe is now an explicit mother phase: host::GeTe.",
        ]
    )
    lines.extend(["", "n/p reclassification summary", "-----------------------------"])
    for row in np_summary.itertuples(index=False):
        lines.append(
            f"- original={row.n_or_p_original_normalized_step31}, "
            f"reclassified={row.n_or_p_reclass_step31}, basis={row.np_reclassification_basis_step31}, "
            f"sign_mixed={row.sign_mixed_step31}, samples={row.n_samples}"
        )
    np_counts = points.drop_duplicates("sample_key")["n_or_p_reclass_step31"].value_counts()
    lines.extend(
        [
            f"overall n/p sample counts: {np_counts.to_dict()}",
            "S≈0 threshold: abs(sample median S) < 1e-8 V/K -> unknown",
            "sign_mixed_step31 is True when a sample contains both S > +1e-8 and S < -1e-8 V/K.",
            "",
            "C(T) outputs and plots",
            "----------------------",
            f"material systems with at least one plotted C(T) bin: {primary.loc[primary.plot_in_primary, 'material_system_reclass_step31'].nunique()}",
            f"systems in requested all-material overlay: {all_plot_systems}",
            f"systems in paper grid: {grid_systems}",
            "n/p supplementary figures include paired comparison plots plus separate n-only and p-only plots for every eligible material system.",
            "figure files:",
        ]
    )
    lines.extend(f"- {figure.relative_to(path.parent).as_posix()}" for figure in figure_paths)
    lines.extend(
        [
            "",
            "Output safety and interpretation",
            "--------------------------------",
            f"existing Step31 output replaced after staging validation: {replaced_existing_step31}",
            "All files were first generated and validated in a staging directory, then installed into the Step31 directory.",
            "Only the previously generated Step31 directory was replaced; no Step12 output was modified.",
            "Original material_system and n_or_p values remain preserved in dedicated original columns.",
            "The n/p-split C(T) CSV and figures are supplementary; the unsplit host-family C(T) remains the primary Step31 baseline.",
            "",
            "Sanity checks",
            "-------------",
            f"input/output row count equal: {int(stats.get('input_rows', -1)) == len(points)} ({stats.get('input_rows', -1)} vs {len(points)})",
            f"final_material_group_key has no blanks: {points['final_material_group_key'].map(normalize_text).ne('').all()}",
            f"parsed_elements column exists: {'parsed_elements' in points.columns}",
            f"host_family_key column exists: {'host_family_key' in points.columns}",
            f"major_element_system_key column exists: {'major_element_system_key' in points.columns}",
            f"group_backoff_level column exists: {'group_backoff_level' in points.columns}",
            "material_group_counts.csv exists: True",
            "report exists: True",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_outputs(
    points: pd.DataFrame,
    primary: pd.DataFrame,
    np_split: pd.DataFrame,
    output_dir: Path,
    expected_input_rows: int,
) -> None:
    required_point_columns = {
        "sample_key", "composition", "material_system_original",
        "material_system_reclass_step31", "n_or_p_original", "n_or_p_reclass_step31",
        "sign_mixed_step31", "temperature_K", "temp_bin_center_K",
        "sigma_exp_S_per_m", "seebeck_obs_V_per_K",
        "parsed_elements", "element_fraction", "major_elements", "dopant_elements",
        "major_element_system_key", "host_family_key", "final_material_group_key",
        "group_backoff_level", "classification_reason", "multiple_host_candidate",
    }
    missing = required_point_columns - set(points.columns)
    if missing:
        raise AssertionError(f"reclassified point columns missing: {sorted(missing)}")
    if points.empty:
        raise AssertionError("no valid sigma points")
    if len(points) != expected_input_rows:
        raise AssertionError(
            f"classification output row count differs from input: {len(points)} != {expected_input_rows}"
        )
    if not points["final_material_group_key"].map(normalize_text).ne("").all():
        raise AssertionError("blank final_material_group_key remains")
    if not np.isfinite(points["temperature_K"]).all():
        raise AssertionError("nonfinite temperatures remain")
    if not points["sigma_exp_S_per_m"].gt(0).all():
        raise AssertionError("nonpositive sigma remains")
    if not set(points["material_system_reclass_step31"]).issubset(MATERIAL_SYSTEM_ORDER):
        raise AssertionError("unexpected material-system label")
    expected_plot_flag = primary["n_rows"].ge(MIN_ROWS_PER_PRIMARY_BIN)
    if not primary["plot_in_primary"].eq(expected_plot_flag).all():
        raise AssertionError("plot_in_primary does not match the three-row rule")
    expected_np_plot_flag = np_split["n_rows"].ge(MIN_ROWS_PER_PRIMARY_BIN)
    if "plot_in_np_split" not in np_split.columns:
        raise AssertionError("plot_in_np_split column missing")
    if not np_split["plot_in_np_split"].eq(expected_np_plot_flag).all():
        raise AssertionError("plot_in_np_split does not match the three-row rule")
    required_files = [
        "reclassified_sigma_points_step31.csv",
        "ct_curves_material_system_step31.csv",
        "ct_curves_material_system_np_split_step31.csv",
        "material_system_reclassification_summary_step31.csv",
        "np_reclassification_summary_step31.csv",
        "material_classification_keys_step31.csv",
        "material_group_counts.csv",
        "parse_failure_examples.csv",
        "step31_ct_curve_report.txt",
        "step31_ct_curve_notes.md",
        "figures/figure_step31_all_material_system_ct_curves.png",
        "figures/figure_step31_ct_curves_grid.png",
        "figures/np_split/figure_step31_ct_curves_np_split_grid.png",
    ]
    for system in plotted_np_split_systems(np_split):
        required_files.append(
            f"figures/np_split/figure_step31_ct_curve_np_split_{safe_slug(system)}.png"
        )
    for carrier in ["n", "p"]:
        carrier_systems = [
            system
            for system in plotted_np_split_systems(np_split)
            if (
                np_split["material_system_reclass_step31"].eq(system)
                & np_split["n_or_p_reclass_step31"].eq(carrier)
                & np_split["plot_in_np_split"]
            ).any()
        ]
        if not carrier_systems:
            continue
        required_files.extend(
            [
                f"figures/np_split/figure_step31_all_material_system_ct_curves_{carrier}.png",
                f"figures/np_split/figure_step31_ct_curves_{carrier}_only_grid.png",
            ]
        )
        for system in carrier_systems:
            required_files.append(
                f"figures/np_split/{carrier}_only/"
                f"figure_step31_ct_curve_{carrier}_{safe_slug(system)}.png"
            )
    missing_files = [name for name in required_files if not (output_dir / name).exists()]
    if missing_files:
        raise AssertionError(f"required outputs missing: {missing_files}")


def install_validated_staging(staging: Path, output_dir: Path) -> None:
    """Install validated Step31 files without renaming the directory on Windows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    staged_files = [path for path in staging.rglob("*") if path.is_file()]
    staged_relatives = {path.relative_to(staging) for path in staged_files}
    existing_files = [path for path in output_dir.rglob("*") if path.is_file()]
    for source in staged_files:
        relative = source.relative_to(staging)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    for existing in existing_files:
        if existing.relative_to(output_dir) not in staged_relatives:
            existing.unlink()
    for directory in sorted(
        [path for path in output_dir.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    shutil.rmtree(staging)


def main() -> None:
    args = parse_args()
    if not 0 < args.dopant_threshold < 1:
        raise ValueError("--dopant-threshold must be between 0 and 1")
    if min(args.min_rows_per_group, args.min_samples_per_group, args.min_papers_per_group) < 1:
        raise ValueError("all minimum group-count arguments must be at least 1")
    output_dir = args.output_dir.resolve()
    output_existed = output_dir.exists()
    if output_existed and not args.replace_output:
        raise FileExistsError(
            f"Refusing to overwrite existing Step31 output directory without --replace-output: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    input_path, input_mode, audit, mapping = inspect_and_select_input(args.input)
    print(f"Selected input: {input_path}")
    print(f"Input mode: {input_mode}")

    if input_mode == "point_table":
        raw_points, np_source, stats, metadata = load_point_table(input_path, mapping)
    else:
        raw_points, np_source, stats, metadata = load_curve_table(input_path)
    points, np_by_sample, system_threshold_counts = prepare_points(
        raw_points,
        np_source,
        metadata.pop("sample_seebeck"),
        metadata.get("source_columns", []),
        args.dopant_threshold,
        args.min_rows_per_group,
        args.min_samples_per_group,
        args.min_papers_per_group,
    )
    primary, np_split = aggregate_ct(points)
    material_summary, np_summary = make_summaries(points, primary, np_by_sample)
    material_group_counts = build_material_group_counts(points)
    parse_failure_examples = points[points["parse_status"].ne("stoichiometry_parsed")].copy()

    staging = Path(tempfile.mkdtemp(prefix=".step31_staging_", dir=output_dir.parent))
    try:
        points.to_csv(staging / "reclassified_sigma_points_step31.csv", index=False)
        points.to_csv(staging / "material_classification_keys_step31.csv", index=False)
        material_group_counts.to_csv(staging / "material_group_counts.csv", index=False)
        parse_failure_examples.to_csv(staging / "parse_failure_examples.csv", index=False)
        system_threshold_counts.to_csv(
            staging / "major_element_system_threshold_audit_step31.csv", index=False
        )
        primary.to_csv(staging / "ct_curves_material_system_step31.csv", index=False)
        np_split.to_csv(staging / "ct_curves_material_system_np_split_step31.csv", index=False)
        material_summary.to_csv(
            staging / "material_system_reclassification_summary_step31.csv", index=False
        )
        np_summary.to_csv(staging / "np_reclassification_summary_step31.csv", index=False)
        write_notes(staging / "step31_ct_curve_notes.md")
        figures, all_plot_systems, grid_systems = create_figures(
            primary, material_summary, staging / "figures", args.dpi
        )
        np_figures, np_plot_systems = create_np_split_figures(
            np_split, points, staging / "figures", args.dpi
        )
        figures.extend(np_figures)
        write_report(
            staging / "step31_ct_curve_report.txt",
            input_path,
            input_mode,
            audit,
            metadata,
            stats,
            points,
            material_summary,
            np_summary,
            primary,
            material_group_counts,
            parse_failure_examples,
            figures,
            all_plot_systems,
            grid_systems,
            output_existed,
            args.dopant_threshold,
            args.min_rows_per_group,
            args.min_samples_per_group,
            args.min_papers_per_group,
        )
        validate_outputs(points, primary, np_split, staging, int(stats["input_rows"]))
        install_validated_staging(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    np_counts = points.drop_duplicates("sample_key")["n_or_p_reclass_step31"].value_counts().to_dict()
    curve_system_count = int(primary.loc[primary["plot_in_primary"], "material_system_reclass_step31"].nunique())
    print(f"Valid sigma points: {len(points):,}")
    print(f"Samples: {points['sample_key'].nunique():,}")
    print(f"n/p sample counts: {np_counts}")
    print(f"Material systems with plotted C(T): {curve_system_count}")
    print(f"Material systems with n/p-split C(T) figures: {len(np_plot_systems)}")
    print(f"Host families: {points['host_family_key'].nunique():,}")
    print(f"Final material groups: {points['final_material_group_key'].nunique():,}")
    print(f"Low-confidence/parse-failure rows: {len(parse_failure_examples):,}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
