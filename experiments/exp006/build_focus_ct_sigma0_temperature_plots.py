import argparse
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
DEFAULT_CURRENT_ROWS = [
    EXP_DIR / "data" / "processed" / "step6a_validation_rows_with_splits_key_broad_family.parquet",
    EXP_DIR / "data" / "processed" / "step6a_validation_rows_with_splits_key_broad_family.csv",
    EXP_DIR / "data" / "processed" / "step3_sigma0_valid.parquet",
    EXP_DIR / "data" / "processed" / "step3_sigma0_valid.csv",
]
DEFAULT_CURRENT_REF = [
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_train_reference_curve_bins.parquet",
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_train_reference_curve_bins.csv",
]
DEFAULT_OLD_CT_CANDIDATES = [
    PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit" / "sigma_predictions_step12.csv",
    PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit" / "initial_tau_fit_predictions_step12.csv",
    PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit" / "prefactor_baseline_audit_step12.csv",
]
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "focus_ct_sigma0_temperature"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_ct_sigma0_temperature"
DEFAULT_REPORT = EXP_DIR / "reports" / "focus_ct_sigma0_temperature" / "focus_ct_sigma0_temperature_report.md"
DEFAULT_TARGET_GROUPS = [
    "broad::SnTe_like",
    "broad::PbTe_like",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::SiGe_like",
    "broad::oxide",
    "broad::sulfide",
]
TARGET_CONFIG = "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
OLD_CT_COLUMNS = [
    "prefactor_C_S_per_m_step12",
    "median_prefactor_C_S_per_m_step12",
    "old_C_T_S_per_m",
    "C_T",
    "c_t",
    "sigma_ref",
    "reference_sigma",
]
OLD_CT_TEMPERATURE_COLUMNS = ["temperature_bin_K_step12", "temperature_K", "T_K", "T_bin_center_K"]
OLD_CT_MATERIAL_COLUMNS = ["material_system", "composition", "formula_raw", "material_name_raw", "prefactor_group_key_step12"]
OLD_CT_CARRIER_COLUMNS = ["n_or_p", "carrier_type"]


def log(message: str) -> None:
    print(f"[focus_ct] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Step12 old empirical C(T) with broad-family Seebeck-derived sigma0 reference curves."
    )
    parser.add_argument("--current-rows", type=Path, default=None)
    parser.add_argument("--current-sigma0-ref", type=Path, default=None)
    parser.add_argument("--old-ct-input", type=Path, default=None)
    parser.add_argument("--old-ct-script", type=Path, default=None)
    parser.add_argument("--old-ct-column", default=None)
    parser.add_argument("--target-groups", nargs="+", default=DEFAULT_TARGET_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def resolve_first(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"No {label} input found. Tried: {paths}")


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.casefold() == ".csv":
        if columns is None:
            return pd.read_csv(path, low_memory=False)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [column for column in columns if column in header]
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    raise ValueError(f"Unsupported input file: {path}")


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    if path.suffix.casefold() == ".parquet":
        return len(pd.read_parquet(path, columns=[]))
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def output_path(directory: Path, stem: str, suffix: str, extension: str = ".csv") -> Path:
    return directory / f"{stem}{suffix}{extension}"


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().isin({"true", "1", "yes", "y"})


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & (values > 0)


def log10_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.log10(values.where(np.isfinite(values) & (values > 0), np.nan))


def contains_element(text: str, symbol: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(symbol)}(?![a-z])", text))


def map_old_material_label(value: Any) -> str:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    lowered = text.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if compact in {"", "unknown", "nan", "none", "all", "alldata"}:
        return "unmatched"
    if "bisbte" in compact or "bisbte" in lowered or "tetradymite" in lowered:
        return "broad::BiSbTe_tetradymite_like"
    if "bi2te3" in compact or "bite" in compact or "bite" in lowered or "bi-te" in lowered:
        return "broad::BiTe_like"
    if "sb2te3" in compact or "sbte" in compact or "sb-te" in lowered:
        return "broad::SbTe_like"
    if "snte" in compact or "sn-te" in lowered:
        return "broad::SnTe_like"
    if "pbte" in compact or "pb-te" in lowered:
        return "broad::PbTe_like"
    if "gete" in compact or "ge-te" in lowered:
        return "broad::GeTe_like"
    if "sige" in compact or "si-ge" in lowered:
        return "broad::SiGe_like"
    if "mg2si" in compact or "mg2sn" in compact or "mg-si" in lowered or "mg-sn" in lowered:
        return "broad::Mg2SiSn_like"
    if "cosb3" in compact or "cosb" in compact or "skutterudite" in lowered:
        return "broad::CoSb_skutterudite_like"
    if "oxide" in lowered or "o-containing" in lowered or contains_element(text, "O"):
        return "broad::oxide"
    if "sulfide" in lowered or "s-containing" in lowered or contains_element(text, "S"):
        return "broad::sulfide"
    if "selenide" in lowered or "se-containing" in lowered or contains_element(text, "Se"):
        return "broad::selenide"
    if "telluride" in lowered or "te-containing" in lowered or contains_element(text, "Te"):
        return "broad::telluride"
    return "unmatched"


def old_material_label(df: pd.DataFrame) -> pd.Series:
    label = pd.Series("", index=df.index, dtype="object")
    for column in ["material_system", "composition", "formula_raw", "material_name_raw", "prefactor_group_key_step12"]:
        if column not in df.columns:
            continue
        values = df[column].fillna("").astype(str).str.strip()
        usable = label.str.strip().eq("") | label.str.casefold().isin({"unknown", "nan", "none", "all_data"})
        label = label.where(~usable, values)
    return label.replace("", "unknown")


def guess_old_ct_columns(header: list[str], forced_ct_column: str | None = None) -> dict[str, Any]:
    temp_guess = next((c for c in OLD_CT_TEMPERATURE_COLUMNS if c in header), "")
    ct_candidates = [c for c in OLD_CT_COLUMNS if c in header]
    if forced_ct_column:
        ct_guess = forced_ct_column if forced_ct_column in header else ""
    else:
        ct_guess = ct_candidates[0] if ct_candidates else ""
    material_guess = next((c for c in OLD_CT_MATERIAL_COLUMNS if c in header), "")
    carrier_guess = next((c for c in OLD_CT_CARRIER_COLUMNS if c in header), "")
    return {
        "temperature_column_guess": temp_guess,
        "ct_column_guess": ct_guess,
        "ct_column_candidates": ";".join(ct_candidates),
        "material_column_guess": material_guess,
        "carrier_column_guess": carrier_guess,
    }


def infer_default_output_dir_from_script(script_path: Path) -> Path | None:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"DEFAULT_OUTPUT_DIR\s*=\s*PROJECT_ROOT\s*/\s*\"data\"\s*/\s*\"output\"\s*/\s*\"([^\"]+)\"", text)
    if match:
        return PROJECT_ROOT / "data" / "output" / match.group(1)
    if "starrydata2_step12_tau_fit" in text:
        return PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"
    return None


def output_csv_names_from_script(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r"args\.output_dir\s*/\s*\"([^\"]+\.csv)\"", text)
    names.extend(re.findall(r"\"([^\"]+_step12\.csv)\"", text))
    return list(dict.fromkeys(names))


def script_old_ct_candidates(script_path: Path) -> list[Path]:
    if not script_path.exists():
        raise FileNotFoundError(f"old C(T) source script not found: {script_path}")
    output_dir = infer_default_output_dir_from_script(script_path)
    if output_dir is None:
        return []
    names = output_csv_names_from_script(script_path)
    return [output_dir / name for name in names if (output_dir / name).exists()]


def add_old_ct_candidate_row(
    rows: list[dict[str, Any]],
    path: Path,
    source: str,
    selected: bool,
    comment: str,
    forced_ct_column: str | None = None,
) -> None:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist() if path.suffix.casefold() == ".csv" else list(pd.read_parquet(path, columns=[]).columns)
    except Exception as exc:
        rows.append(
            {
                "candidate_file": str(path),
                "row_count": "",
                "columns": "",
                "temperature_column_guess": "",
                "ct_column_guess": "",
                "ct_column_candidates": "",
                "material_column_guess": "",
                "carrier_column_guess": "",
                "selected_as_old_ct": False,
                "candidate_source": source,
                "comment": f"could not read header: {exc}",
            }
        )
        return
    guesses = guess_old_ct_columns(header, forced_ct_column)
    rows.append(
        {
            "candidate_file": str(path),
            "row_count": row_count(path),
            "columns": ";".join(header),
            **guesses,
            "selected_as_old_ct": selected,
            "candidate_source": source,
            "comment": comment,
        }
    )


def select_old_ct_from_script(script_path: Path, forced_ct_column: str | None) -> tuple[Path | None, str | None, list[dict[str, Any]], dict[str, Any]]:
    log("reading old C(T) source script...")
    script_candidates = script_old_ct_candidates(script_path)
    rows: list[dict[str, Any]] = []
    selected_path: Path | None = None
    selected_column: str | None = None
    selected_reason = ""
    rejected: list[str] = []

    preferred_names = ["sigma_predictions_step12.csv", "initial_tau_fit_predictions_step12.csv", "prefactor_baseline_audit_step12.csv"]
    ordered = sorted(
        script_candidates,
        key=lambda p: preferred_names.index(p.name) if p.name in preferred_names else len(preferred_names),
    )
    for path in ordered:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        guesses = guess_old_ct_columns(header, forced_ct_column)
        has_core = bool(guesses["temperature_column_guess"] and guesses["ct_column_guess"])
        if forced_ct_column and forced_ct_column not in header:
            rejected.append(f"{path}: requested --old-ct-column {forced_ct_column} not found")
            continue
        if not has_core:
            rejected.append(f"{path}: missing temperature or old C(T) column")
            continue
        if selected_path is None and path.name == "sigma_predictions_step12.csv" and guesses["ct_column_guess"] == "prefactor_C_S_per_m_step12":
            selected_path = path
            selected_column = guesses["ct_column_guess"]
            selected_reason = (
                "fit_tau_eff_step12.py writes sigma_predictions_step12.csv with temperature_K, "
                "temperature_bin_K_step12, material_system/composition, n_or_p, and prefactor_C_S_per_m_step12; "
                "assert_acceptance also requires prefactor_C_S_per_m_step12 in sigma_predictions_step12."
            )
        elif selected_path is None and has_core:
            selected_path = path
            selected_column = guesses["ct_column_guess"]
            selected_reason = "first script-derived output with temperature and old C(T)-like columns"

    for path in ordered:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        guesses = guess_old_ct_columns(header, forced_ct_column)
        is_selected = selected_path is not None and path.resolve() == selected_path.resolve()
        if is_selected:
            comment = selected_reason
        elif not guesses["ct_column_guess"]:
            comment = "not adopted: no C(T)-like column found"
        elif path.name == "prefactor_baseline_audit_step12.csv":
            comment = "not adopted: audit summary of prefactors, not row-level Step12 C(T) data"
        elif path.name == "initial_tau_fit_predictions_step12.csv":
            comment = "not adopted: filtered fit-ok subset; sigma_predictions_step12.csv is the primary full Step12 prediction output"
        else:
            comment = "not adopted: lower priority than selected script-derived C(T) output"
        add_old_ct_candidate_row(rows, path, "old_ct_script", is_selected, comment, forced_ct_column)

    meta = {
        "old_ct_source_mode": "script",
        "old_ct_source_script": str(script_path),
        "old_ct_script_output_dir": str(infer_default_output_dir_from_script(script_path) or ""),
        "old_ct_selected_reason": selected_reason,
        "old_ct_rejected_candidates": rejected,
    }
    return selected_path, selected_column, rows, meta


def discover_old_ct_files(
    selected: Path | None,
    selected_column: str | None,
    script_path: Path | None,
    output_dir: Path,
    suffix: str,
) -> tuple[pd.DataFrame, Path | None, str | None, dict[str, Any]]:
    log("discovering old C(T) files...")
    rows: list[dict[str, Any]] = []
    best_path: Path | None = None
    best_column: str | None = selected_column
    meta: dict[str, Any] = {
        "old_ct_source_mode": "candidate_search",
        "old_ct_source_script": "",
        "old_ct_script_output_dir": "",
        "old_ct_selected_reason": "",
        "old_ct_rejected_candidates": [],
    }

    if selected is not None and selected_column is not None:
        if not selected.exists():
            raise FileNotFoundError(f"--old-ct-input not found: {selected}")
        header = pd.read_csv(selected, nrows=0).columns.tolist()
        if selected_column not in header:
            raise KeyError(f"--old-ct-column not found in --old-ct-input: {selected_column}")
        best_path = selected
        best_column = selected_column
        meta.update(
            {
                "old_ct_source_mode": "explicit_input_and_column",
                "old_ct_selected_reason": "--old-ct-input and --old-ct-column were both specified explicitly",
            }
        )
        add_old_ct_candidate_row(rows, selected, "explicit", True, meta["old_ct_selected_reason"], selected_column)
    elif script_path is not None:
        best_path, best_column, script_rows, meta = select_old_ct_from_script(script_path, selected_column)
        rows.extend(script_rows)
    else:
        candidates = list(dict.fromkeys([*(DEFAULT_OLD_CT_CANDIDATES), *([selected] if selected else [])]))
        for path in candidates:
            if path is None or not path.exists():
                continue
            header = pd.read_csv(path, nrows=0).columns.tolist()
            guesses = guess_old_ct_columns(header, selected_column)
            if best_path is None and guesses["temperature_column_guess"] and guesses["ct_column_guess"]:
                best_path = path
                best_column = guesses["ct_column_guess"]
            comment = (
                "specified by --old-ct-input"
                if selected and path.resolve() == selected.resolve()
                else ("usable Step12 C(T) candidate" if guesses["temperature_column_guess"] and guesses["ct_column_guess"] else "missing required old C(T) columns")
            )
            add_old_ct_candidate_row(rows, path, "candidate_search", False, comment, selected_column)

    if best_path is None:
        frame = pd.DataFrame(rows)
        frame.to_csv(output_path(output_dir, "old_ct_candidate_files", suffix), index=False)
        return frame, None, None, meta

    for row in rows:
        row["selected_as_old_ct"] = Path(row["candidate_file"]).resolve() == best_path.resolve()
        if row["selected_as_old_ct"] and best_column:
            row["ct_column_guess"] = best_column

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path(output_dir, "old_ct_candidate_files", suffix), index=False)
    if not meta.get("old_ct_selected_reason"):
        meta["old_ct_selected_reason"] = "selected first candidate with temperature and old C(T)-like columns"
    return frame, best_path, best_column, meta


def load_current_rows(path: Path) -> pd.DataFrame:
    log("loading current broad_family rows...")
    columns = [
        "material_group_key",
        "carrier_type",
        "T_K",
        "sigma_S_per_m",
        "log10_sigma_S_per_m",
        "sigma0_S_per_m",
        "log10_sigma0_S_per_m",
        "is_valid_sigma0",
        "row_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "formula_raw",
        "material_name_raw",
    ]
    df = read_table(path, columns)
    if "material_group_key" not in df.columns:
        raise KeyError("current rows must contain material_group_key; use step6a broad_family output")
    for column in ["T_K", "sigma_S_per_m", "sigma0_S_per_m"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "log10_sigma_S_per_m" not in df.columns:
        df["log10_sigma_S_per_m"] = log10_positive(df["sigma_S_per_m"])
    if "log10_sigma0_S_per_m" not in df.columns:
        df["log10_sigma0_S_per_m"] = log10_positive(df["sigma0_S_per_m"])
    valid = (
        as_bool(df["is_valid_sigma0"])
        & finite_positive(df["sigma_S_per_m"])
        & finite_positive(df["sigma0_S_per_m"])
        & df["carrier_type"].astype(str).isin(["p", "n"])
        & np.isfinite(df["T_K"])
    )
    return df.loc[valid, [c for c in columns if c in df.columns]].copy()


def load_current_sigma0_ref(path: Path) -> pd.DataFrame:
    log("loading current sigma0 reference curves...")
    columns = [
        "config_id",
        "split_scheme",
        "reference_source_subset",
        "eval_target_subset",
        "group_scheme",
        "curve_method",
        "material_group_key",
        "carrier_type",
        "T_bin_center_K",
        "log10_sigma0_ref_S_per_m",
        "sigma0_ref_S_per_m",
        "train_row_count",
        "train_sample_count",
        "train_paper_count",
        "is_reference_bin_candidate",
        "reliability_level",
    ]
    df = read_table(path, columns)
    mask = (
        df["config_id"].astype(str).eq(TARGET_CONFIG)
        & df["split_scheme"].astype(str).eq("sample_holdout")
        & df["reference_source_subset"].astype(str).eq("conservative_valid")
        & df["eval_target_subset"].astype(str).eq("all_valid")
        & df["group_scheme"].astype(str).eq("material_family")
        & df["curve_method"].astype(str).eq("sample_median")
        & as_bool(df["is_reference_bin_candidate"])
    )
    ref = df.loc[mask].copy()
    for column in ["T_bin_center_K", "sigma0_ref_S_per_m", "log10_sigma0_ref_S_per_m"]:
        ref[column] = pd.to_numeric(ref[column], errors="coerce")
    ref = ref[np.isfinite(ref["T_bin_center_K"]) & finite_positive(ref["sigma0_ref_S_per_m"])].copy()
    ref["log10_sigma0_ref_S_per_m"] = ref["log10_sigma0_ref_S_per_m"].where(
        np.isfinite(ref["log10_sigma0_ref_S_per_m"]), np.log10(ref["sigma0_ref_S_per_m"])
    )
    return ref[
        [
            "material_group_key",
            "carrier_type",
            "T_bin_center_K",
            "sigma0_ref_S_per_m",
            "log10_sigma0_ref_S_per_m",
            "train_row_count",
            "train_sample_count",
            "train_paper_count",
            "reliability_level",
        ]
    ].copy()


def load_old_ct(path: Path, ct_column: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    log("loading old C(T) curves...")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    temp_col = next((c for c in OLD_CT_TEMPERATURE_COLUMNS if c in header), None)
    if ct_column is not None:
        if ct_column not in header:
            raise KeyError(f"old C(T) column not found in {path}: {ct_column}")
        ct_col = ct_column
    else:
        ct_col = next((c for c in OLD_CT_COLUMNS if c in header), None)
    if temp_col is None or ct_col is None:
        raise KeyError(f"old C(T) input missing temperature/C(T) columns: {path}")
    keep = [
        temp_col,
        ct_col,
        *[c for c in ["material_system", "composition", "formula_raw", "material_name_raw", "prefactor_group_key_step12", "n_or_p", "carrier_type"] if c in header],
    ]
    raw = pd.read_csv(path, usecols=list(dict.fromkeys(keep)), low_memory=False)
    raw["source_file"] = str(path)
    raw["old_material_label"] = old_material_label(raw)
    raw["material_group_key_mapped"] = raw["old_material_label"].map(map_old_material_label)
    if "carrier_type" in raw.columns:
        raw["carrier_type"] = raw["carrier_type"].astype(str).str.strip().str.lower()
    elif "n_or_p" in raw.columns:
        raw["carrier_type"] = raw["n_or_p"].astype(str).str.strip().str.lower()
    else:
        raw["carrier_type"] = ""
    raw["T_K"] = pd.to_numeric(raw[temp_col], errors="coerce")
    raw["old_C_T_S_per_m"] = pd.to_numeric(raw[ct_col], errors="coerce")
    raw["old_ct_parse_status"] = np.where(
        raw["material_group_key_mapped"].eq("unmatched"),
        "unmatched_material",
        np.where(raw["carrier_type"].isin(["p", "n"]) & np.isfinite(raw["T_K"]) & finite_positive(raw["old_C_T_S_per_m"]), "ok", "invalid_numeric_or_carrier"),
    )
    mapping = (
        raw[["old_material_label", "material_group_key_mapped", "old_ct_parse_status"]]
        .drop_duplicates()
        .sort_values(["material_group_key_mapped", "old_material_label"])
    )
    ok = raw[raw["old_ct_parse_status"].eq("ok")].copy()
    grouped = (
        ok.groupby(["source_file", "old_material_label", "material_group_key_mapped", "carrier_type", "T_K"], dropna=False, sort=True)
        .agg(old_C_T_S_per_m=("old_C_T_S_per_m", "median"))
        .reset_index()
    )
    grouped["log10_old_C_T_S_per_m"] = np.log10(grouped["old_C_T_S_per_m"])
    grouped["old_ct_parse_status"] = "ok"
    column_meta = {
        "old_ct_selected_file": str(path),
        "old_ct_selected_column": str(ct_col),
        "old_ct_temperature_column": str(temp_col),
        "old_ct_material_columns": ";".join([c for c in OLD_CT_MATERIAL_COLUMNS if c in header]),
        "old_ct_carrier_column": next((c for c in OLD_CT_CARRIER_COLUMNS if c in header), ""),
    }
    return grouped, mapping, column_meta


def nearest_comparison(old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            old_sub = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))].copy()
            ref_sub = sigma0_ref[(sigma0_ref["material_group_key"].eq(group)) & (sigma0_ref["carrier_type"].eq(carrier))].copy()
            if old_sub.empty or ref_sub.empty:
                continue
            ref_t = ref_sub["T_bin_center_K"].to_numpy(dtype=float)
            for _, old_row in old_sub.sort_values("T_K").iterrows():
                t = float(old_row["T_K"])
                pos = int(np.nanargmin(np.abs(ref_t - t)))
                ref_row = ref_sub.iloc[pos]
                old_c = float(old_row["old_C_T_S_per_m"])
                sigma0 = float(ref_row["sigma0_ref_S_per_m"])
                rows.append(
                    {
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "T_K": t,
                        "old_C_T_S_per_m": old_c,
                        "log10_old_C_T_S_per_m": math.log10(old_c),
                        "sigma0_ref_S_per_m": sigma0,
                        "log10_sigma0_ref_S_per_m": float(ref_row["log10_sigma0_ref_S_per_m"]),
                        "log10_sigma0ref_over_oldCT": math.log10(sigma0 / old_c),
                        "match_method": "nearest_T_bin_center",
                        "T_delta_K": float(ref_row["T_bin_center_K"] - t),
                    }
                )
    return pd.DataFrame(rows)


def plot_two_panel(group: str, carrier: str, sigma: pd.DataFrame, old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating two-panel figure...")
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle(f"{group}, {carrier}: old C(T) and Seebeck-derived coefficient")
    axes[0].scatter(
        sigma["T_K"],
        sigma["sigma_S_per_m"],
        s=12,
        alpha=0.45,
        color="#1f77b4",
        edgecolors="none",
        label="sigma_exp(T)",
    )
    if not old_ct.empty:
        axes[0].plot(
            old_ct["T_K"],
            old_ct["old_C_T_S_per_m"],
            linewidth=2.4,
            color="#d62728",
            label="old empirical C(T) from Step12",
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Electrical conductivity sigma [S/m]")
    axes[0].set_title(f"{group}, {carrier}-type")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].scatter(
        sigma["T_K"],
        sigma["sigma0_S_per_m"],
        s=12,
        alpha=0.45,
        color="#2ca02c",
        edgecolors="none",
        label="Seebeck-derived coefficient points",
    )
    if not sigma0_ref.empty:
        axes[1].plot(
            sigma0_ref["T_bin_center_K"],
            sigma0_ref["sigma0_ref_S_per_m"],
            linewidth=2.4,
            color="#ff7f0e",
            label="Seebeck-derived coefficient reference",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Temperature T [K]")
    axes[1].set_ylabel("Reference conductivity coefficient [S/m]")
    axes[1].set_title("Seebeck-derived coefficient")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_overlay(group: str, carrier: str, sigma: pd.DataFrame, old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating overlay figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sigma["T_K"], sigma["sigma_S_per_m"], s=10, alpha=0.3, label="sigma_exp(T)")
    if not old_ct.empty:
        ax.plot(old_ct["T_K"], old_ct["old_C_T_S_per_m"], linewidth=2, label="old empirical C(T), sigma baseline")
    if not sigma0_ref.empty:
        ax.plot(sigma0_ref["T_bin_center_K"], sigma0_ref["sigma0_ref_S_per_m"], linewidth=2, label="Seebeck-derived sigma0_ref")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("S/m")
    ax.set_title(f"{group}, {carrier}: same-unit overlay, different physical meanings")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_log_ratio(group: str, carrier: str, comp: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating log-ratio figure...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(comp["T_K"], comp["log10_sigma0ref_over_oldCT"], marker="o", linewidth=1.5, label="log10(sigma0_ref / old C(T))")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("log10(sigma0_ref / old_C_T)")
    ax.set_title(f"{group}, {carrier}: Seebeck-derived coefficient relative to old C(T)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    sigma_rows: pd.DataFrame,
    old_ct: pd.DataFrame,
    sigma0_ref: pd.DataFrame,
    comparison: pd.DataFrame,
    target_groups: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            sig = sigma_rows[(sigma_rows["material_group_key"].eq(group)) & (sigma_rows["carrier_type"].eq(carrier))]
            old = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))]
            ref = sigma0_ref[(sigma0_ref["material_group_key"].eq(group)) & (sigma0_ref["carrier_type"].eq(carrier))]
            comp = comparison[(comparison["material_group_key"].eq(group)) & (comparison["carrier_type"].eq(carrier))]
            warnings = []
            if sig.empty:
                warnings.append("no_sigma_rows")
            if old.empty:
                warnings.append("no_old_ct")
            if ref.empty:
                warnings.append("no_current_sigma0_ref")
            if comp.empty:
                warnings.append("no_curve_comparison")
            rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "sigma_row_count": len(sig),
                    "sigma_sample_count": sig["sample_key"].nunique(dropna=True) if not sig.empty and "sample_key" in sig else 0,
                    "old_ct_points": len(old),
                    "current_sigma0_ref_points": len(ref),
                    "comparison_points": len(comp),
                    "T_min_K": sig["T_K"].min() if not sig.empty else np.nan,
                    "T_max_K": sig["T_K"].max() if not sig.empty else np.nan,
                    "sigma_median_S_per_m": sig["sigma_S_per_m"].median() if not sig.empty else np.nan,
                    "sigma0_median_S_per_m": sig["sigma0_S_per_m"].median() if not sig.empty else np.nan,
                    "old_C_T_median_S_per_m": old["old_C_T_S_per_m"].median() if not old.empty else np.nan,
                    "sigma0_ref_median_S_per_m": ref["sigma0_ref_S_per_m"].median() if not ref.empty else np.nan,
                    "median_log10_sigma0ref_over_oldCT": comp["log10_sigma0ref_over_oldCT"].median() if not comp.empty else np.nan,
                    "warning": ";".join(warnings),
                }
            )
    return pd.DataFrame(rows)


def make_report(
    path: Path,
    inputs: dict[str, Path],
    old_ct_meta: dict[str, Any],
    old_ct_candidates: pd.DataFrame,
    target_groups: list[str],
    summary: pd.DataFrame,
    figure_index: pd.DataFrame,
    mapping: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    old_missing = summary[summary["old_ct_points"].eq(0)][["material_group_key", "carrier_type"]]
    ref_missing = summary[summary["current_sigma0_ref_points"].eq(0)][["material_group_key", "carrier_type"]]
    ratio = pd.to_numeric(comparison.get("log10_sigma0ref_over_oldCT", pd.Series(dtype=float)), errors="coerce")
    lines = [
        "# Focus C(T) vs Sigma0 Temperature Comparison",
        "",
        "## Inputs",
    ]
    for label, input_path in inputs.items():
        lines.append(f"- {label}: `{input_path}`")
    lines.extend(
        [
            "",
            "## Old C(T) File Selection",
            "- Candidate Step12 C(T) files are scanned for temperature and C(T)-like columns.",
            f"- old C(T) source script: `{old_ct_meta.get('old_ct_source_script', '')}`",
            f"- source mode: `{old_ct_meta.get('old_ct_source_mode', '')}`",
            f"- script output directory: `{old_ct_meta.get('old_ct_script_output_dir', '')}`",
            f"- selected old C(T) file: `{inputs['old_ct']}`",
            f"- selected old C(T) column: `{old_ct_meta.get('old_ct_selected_column', '')}`",
            f"- temperature column: `{old_ct_meta.get('old_ct_temperature_column', '')}`",
            f"- material columns: `{old_ct_meta.get('old_ct_material_columns', '')}`",
            f"- carrier type column: `{old_ct_meta.get('old_ct_carrier_column', '')}`",
            f"- adoption reason: {old_ct_meta.get('old_ct_selected_reason', '')}",
            "",
            "### Old C(T) Candidates",
        ]
    )
    if old_ct_candidates.empty:
        lines.append("- none")
    else:
        for _, row in old_ct_candidates.iterrows():
            selected = "selected" if bool(row.get("selected_as_old_ct", False)) else "not selected"
            lines.append(
                f"- {selected}: `{row.get('candidate_file', '')}`; ct_candidates=`{row.get('ct_column_candidates', '')}`; "
                f"temperature=`{row.get('temperature_column_guess', '')}`; material=`{row.get('material_column_guess', '')}`; "
                f"carrier=`{row.get('carrier_column_guess', '')}`; reason={row.get('comment', '')}"
            )
    rejected = old_ct_meta.get("old_ct_rejected_candidates", [])
    if rejected:
        lines.append("")
        lines.append("### Rejected Script-Derived Candidates")
        for item in rejected:
            lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Targets",
            f"- material groups: {', '.join(target_groups)}",
            "- carrier types: p, n",
            "",
            "## Physical Difference",
            "- Old C(T) is the empirical electrical-conductivity baseline from Step12 tau_eff fitting.",
            "- The current sigma0_ref is a Seebeck-derived coefficient corrected using Fermi-level information.",
            "- They share units of S/m, but they are not the same physical quantity.",
            "- The two-panel figure is the main figure so measured sigma and Seebeck-derived coefficients are not visually conflated.",
            "",
            "## Figure List",
        ]
    )
    if figure_index.empty:
        lines.append("- none")
    else:
        for _, row in figure_index.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']} / {row['figure_type']}: `{row['figure_path_png']}`")
    lines.extend(["", "## Data Availability"])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['material_group_key']} / {row['carrier_type']}: sigma_rows={row['sigma_row_count']}, old_ct={row['old_ct_points']}, sigma0_ref={row['current_sigma0_ref_points']}, warning={row['warning']}"
        )
    lines.extend(["", "## Missing Old C(T)"])
    if old_missing.empty:
        lines.append("- none")
    else:
        for _, row in old_missing.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']}")
    lines.extend(["", "## Missing Current Sigma0 Ref"])
    if ref_missing.empty:
        lines.append("- none")
    else:
        for _, row in ref_missing.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']}")
    unmatched = mapping[mapping["material_group_key_mapped"].eq("unmatched")]
    lines.extend(["", "## Unmatched Old Material Labels"])
    lines.append(f"- unmatched unique labels: {len(unmatched)}")
    for label in unmatched["old_material_label"].astype(str).head(30):
        lines.append(f"- {label}")
    lines.extend(["", "## Ratio Summary"])
    if ratio.dropna().empty:
        lines.append("- no matched old C(T) and current sigma0_ref points")
    else:
        lines.append(f"- count: {int(ratio.count())}")
        lines.append(f"- median log10(sigma0_ref / old_C_T): {float(ratio.median()):.6g}")
        lines.append(f"- min/max log10 ratio: {float(ratio.min()):.6g} / {float(ratio.max()):.6g}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Old C(T) is an observed-sigma empirical baseline.",
            "- The current coefficient is corrected using Seebeck-derived Fermi-level information.",
            "- Both have units of S/m but different meanings.",
            "- No new sigma_pred is calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data are not read.",
            "",
            "## Next Checks",
            "- Compare temperature trends for SnTe_like, PbTe_like, BiTe_like, and SiGe_like.",
            "- Check whether p/n carrier type changes the offset or slope.",
            "- Identify material groups where sigma0_ref deviates strongly from old C(T).",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    current_rows: pd.DataFrame,
    sigma0_ref: pd.DataFrame,
    old_ct: pd.DataFrame,
    target_groups: list[str],
    figure_index: pd.DataFrame,
    report_path: Path,
    inputs: dict[str, Path],
) -> None:
    log("running sanity checks...")
    if current_rows.empty:
        raise RuntimeError("current broad_family data could not be loaded or is empty")
    if sigma0_ref.empty:
        raise RuntimeError("current sigma0_ref curve data could not be loaded or is empty")
    if inputs["old_ct"] is None or not inputs["old_ct"].exists():
        raise RuntimeError("old C(T) file was not identified")
    if not target_groups:
        raise RuntimeError("no target groups")
    if not finite_positive(current_rows["sigma_S_per_m"]).all():
        raise RuntimeError("sigma_S_per_m contains invalid values after filtering")
    if not finite_positive(current_rows["sigma0_S_per_m"]).all():
        raise RuntimeError("sigma0_S_per_m contains invalid values after filtering")
    if not finite_positive(sigma0_ref["sigma0_ref_S_per_m"]).all():
        raise RuntimeError("sigma0_ref_S_per_m contains invalid values")
    if not old_ct.empty and not finite_positive(old_ct["old_C_T_S_per_m"]).all():
        raise RuntimeError("old_C_T_S_per_m contains invalid values")
    if figure_index.empty:
        raise RuntimeError("figure index was not created")
    if not figure_index["figure_type"].eq("two_panel").any():
        raise RuntimeError("no two-panel figure was created")
    if not report_path.exists():
        raise RuntimeError("report was not created")
    for label, input_path in inputs.items():
        if label == "old_ct":
            continue
        text = str(input_path).replace("\\", "/")
        if "step4_sigma0_reference_curve" in text:
            raise RuntimeError("Step4 full-data reference curve was used unexpectedly")
        if "starrydata2/raw" in text.casefold() or "raw" in Path(text).parts:
            raise RuntimeError("raw Starrydata2 data path was used unexpectedly")


def main() -> None:
    start = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    current_rows_path = args.current_rows or resolve_first(DEFAULT_CURRENT_ROWS, "current broad_family")
    current_ref_path = args.current_sigma0_ref or resolve_first(DEFAULT_CURRENT_REF, "current sigma0 reference")

    candidates, old_ct_path, old_ct_column, old_ct_meta = discover_old_ct_files(
        args.old_ct_input,
        args.old_ct_column,
        args.old_ct_script,
        args.output,
        args.output_suffix,
    )
    if old_ct_path is None:
        raise RuntimeError(f"No usable old C(T) file found. Candidate list written to {output_path(args.output, 'old_ct_candidate_files', args.output_suffix)}")

    current_rows = load_current_rows(current_rows_path)
    current_ref = load_current_sigma0_ref(current_ref_path)
    old_ct, mapping, old_ct_column_meta = load_old_ct(old_ct_path, old_ct_column)
    old_ct_meta.update(old_ct_column_meta)

    log("normalizing old C(T) columns...")
    log("mapping old material labels to broad_family groups...")
    target_groups = list(args.target_groups)
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]
    log("filtering target groups...")
    current_rows = current_rows[current_rows["material_group_key"].isin(target_groups)].copy()
    current_ref = current_ref[current_ref["material_group_key"].isin(target_groups)].copy()
    old_ct = old_ct[old_ct["material_group_key_mapped"].isin(target_groups)].copy()
    mapping.to_csv(output_path(args.output, "focus_ct_sigma0_material_mapping", args.output_suffix), index=False)

    comparison = nearest_comparison(old_ct, current_ref, target_groups)
    summary = build_summary(current_rows, old_ct, current_ref, comparison, target_groups)
    figure_rows: list[dict[str, Any]] = []
    figure_id = 1

    for group in target_groups:
        for carrier in ["p", "n"]:
            log(f"processing group/carrier: {group} / {carrier}")
            sig = current_rows[(current_rows["material_group_key"].eq(group)) & (current_rows["carrier_type"].eq(carrier))].sort_values("T_K")
            old = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))].sort_values("T_K")
            ref = current_ref[(current_ref["material_group_key"].eq(group)) & (current_ref["carrier_type"].eq(carrier))].sort_values("T_bin_center_K")
            comp = comparison[(comparison["material_group_key"].eq(group)) & (comparison["carrier_type"].eq(carrier))].sort_values("T_K")
            if sig.empty:
                log(f"warning: no sigma rows for {group} / {carrier}; skipping figures")
                continue
            safe = f"{safe_name(group)}_{carrier}"
            title = f"{group}, {carrier}: old C(T) and Seebeck-derived coefficient"
            two_png = args.figures / f"{safe}_sigma_C_and_sigma0_two_panel{args.output_suffix}.png"
            two_pdf = args.figures / f"{safe}_sigma_C_and_sigma0_two_panel{args.output_suffix}.pdf"
            plot_two_panel(group, carrier, sig, old, ref, two_png, two_pdf)
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": "two_panel",
                    "figure_path_png": str(two_png),
                    "figure_path_pdf": str(two_pdf),
                    "title": title,
                    "n_sigma_points": len(sig),
                    "n_old_ct_points": len(old),
                    "n_sigma0_ref_points": len(ref),
                    "description": "Main two-panel figure separating sigma_exp/C(T) from Seebeck-derived coefficient.",
                }
            )
            figure_id += 1
            overlay_png = args.figures / f"{safe}_overlay_sigma_C_sigma0{args.output_suffix}.png"
            overlay_pdf = args.figures / f"{safe}_overlay_sigma_C_sigma0{args.output_suffix}.pdf"
            plot_overlay(group, carrier, sig, old, ref, overlay_png, overlay_pdf)
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": "overlay",
                    "figure_path_png": str(overlay_png),
                    "figure_path_pdf": str(overlay_pdf),
                    "title": f"{group}, {carrier}: overlay of sigma_exp, old C(T), and sigma0_ref",
                    "n_sigma_points": len(sig),
                    "n_old_ct_points": len(old),
                    "n_sigma0_ref_points": len(ref),
                    "description": "Same-unit overlay; old C(T) and sigma0_ref have different meanings.",
                }
            )
            figure_id += 1
            if not comp.empty:
                ratio_png = args.figures / f"{safe}_log_ratio_sigma0ref_over_oldCT{args.output_suffix}.png"
                ratio_pdf = args.figures / f"{safe}_log_ratio_sigma0ref_over_oldCT{args.output_suffix}.pdf"
                plot_log_ratio(group, carrier, comp, ratio_png, ratio_pdf)
                figure_rows.append(
                    {
                        "figure_id": figure_id,
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "figure_type": "log_ratio",
                        "figure_path_png": str(ratio_png),
                        "figure_path_pdf": str(ratio_pdf),
                        "title": f"{group}, {carrier}: log10 sigma0_ref over old C(T)",
                        "n_sigma_points": len(sig),
                        "n_old_ct_points": len(old),
                        "n_sigma0_ref_points": len(ref),
                        "description": "Nearest-temperature comparison of current sigma0_ref against old C(T).",
                    }
                )
                figure_id += 1

    figure_index = pd.DataFrame(figure_rows)

    log("writing CSV outputs...")
    current_rows.to_csv(output_path(args.output, "focus_ct_sigma0_input_sigma_rows", args.output_suffix), index=False)
    old_ct.to_csv(output_path(args.output, "focus_ct_sigma0_old_ct_curves_normalized", args.output_suffix), index=False)
    current_ref.to_csv(output_path(args.output, "focus_ct_sigma0_current_sigma0_ref_curves", args.output_suffix), index=False)
    comparison.to_csv(output_path(args.output, "focus_ct_sigma0_curve_comparison_table", args.output_suffix), index=False)
    summary.to_csv(output_path(args.output, "focus_ct_sigma0_summary_by_group_carrier", args.output_suffix), index=False)
    figure_index.to_csv(output_path(args.output, "focus_ct_sigma0_figure_index", args.output_suffix), index=False)

    inputs = {"current_rows": current_rows_path, "current_sigma0_ref": current_ref_path, "old_ct": old_ct_path}
    log("writing report...")
    make_report(args.report, inputs, old_ct_meta, candidates, target_groups, summary, figure_index, mapping, comparison)

    run_sanity_checks(current_rows, current_ref, old_ct, target_groups, figure_index, args.report, inputs)
    elapsed = time.time() - start
    log(f"done. elapsed_seconds={elapsed:.2f}")
    print(f"old_ct_input_used: {old_ct_path}")
    print(f"target_groups: {', '.join(target_groups)}")
    print(f"group_carrier_rows: {len(summary)}")
    print(f"old_ct_found_combinations: {int((summary['old_ct_points'] > 0).sum())}")
    print(f"current_sigma0_ref_found_combinations: {int((summary['current_sigma0_ref_points'] > 0).sum())}")
    print(f"two_panel_figures: {int(figure_index['figure_type'].eq('two_panel').sum())}")
    print(f"overlay_figures: {int(figure_index['figure_type'].eq('overlay').sum())}")
    print(f"log_ratio_figures: {int(figure_index['figure_type'].eq('log_ratio').sum())}")
    if not comparison.empty:
        print(f"median_log10_sigma0ref_over_oldCT: {comparison['log10_sigma0ref_over_oldCT'].median()}")
    print(f"output_dir: {args.output}")
    print(f"figure_dir: {args.figures}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
