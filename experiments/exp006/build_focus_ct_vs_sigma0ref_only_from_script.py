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
DEFAULT_SIGMA0_REF = [
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_train_reference_curve_bins.parquet",
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_train_reference_curve_bins.csv",
]
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "focus_ct_vs_sigma0ref_only_from_script"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_ct_vs_sigma0ref_only_from_script"
DEFAULT_REPORT = EXP_DIR / "reports" / "focus_ct_vs_sigma0ref_only_from_script" / "focus_ct_vs_sigma0ref_only_from_script_report.md"
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
    "C_T",
    "CT",
    "C(T)",
    "C_value",
    "C_S_per_m",
    "C_ref",
    "sigma_ref",
    "sigma_median",
    "sigma_median_S_per_m",
    "conductivity_ref",
    "log10_C_T",
    "log10_C",
    "log10_sigma_ref",
]
OLD_CT_LOG_COLUMNS = {"log10_C_T", "log10_C", "log10_sigma_ref"}
OLD_TEMPERATURE_COLUMNS = [
    "temperature_bin_K_step12",
    "T",
    "T_K",
    "temperature",
    "temperature_K",
    "temp_K",
    "T_bin_center_K",
    "bin_center_K",
]
OLD_MATERIAL_COLUMNS = [
    "material_system",
    "material_family",
    "material_family_raw",
    "material_group_key",
    "material_group",
    "family",
    "system",
    "composition_group",
    "composition",
    "prefactor_group_key_step12",
]
OLD_CARRIER_COLUMNS = ["n_or_p", "carrier_type", "pn_type", "type", "conduction_type", "np_type"]


def log(message: str) -> None:
    print(f"[ct_vs_sigma0ref] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay old SS2026 C(T) and broad-family sigma0_ref(T) curves only.")
    parser.add_argument("--old-ct-script", type=Path, required=True)
    parser.add_argument("--current-sigma0-ref", type=Path, default=None)
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
    raise FileNotFoundError(f"No {label} file found. Tried: {paths}")


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.casefold() == ".csv":
        if columns is None:
            return pd.read_csv(path, low_memory=False)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [column for column in columns if column in header]
        return pd.read_csv(path, usecols=usecols, low_memory=False)
    raise ValueError(f"Unsupported table file: {path}")


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


def contains_element(text: str, symbol: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(symbol)}(?![a-z])", text))


def map_old_material_label(value: Any) -> tuple[str, str, str]:
    text = "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)
    lowered = text.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    if compact in {"", "unknown", "nan", "none", "all", "alldata"}:
        return "unmatched", "unmatched", "empty_or_unknown"
    rules = [
        ("broad::BiSbTe_tetradymite_like", "BiSbTe/tetradymite", lambda: "bisbte" in compact or "tetradymite" in lowered),
        ("broad::BiTe_like", "BiTe/Bi2Te3", lambda: "bi2te3" in compact or "bite" in compact or "bi-te" in lowered),
        ("broad::SbTe_like", "SbTe/Sb2Te3", lambda: "sb2te3" in compact or "sbte" in compact or "sb-te" in lowered),
        ("broad::SnTe_like", "SnTe", lambda: "snte" in compact or "sn-te" in lowered),
        ("broad::PbTe_like", "PbTe", lambda: "pbte" in compact or "pb-te" in lowered),
        ("broad::GeTe_like", "GeTe", lambda: "gete" in compact or "ge-te" in lowered),
        ("broad::SiGe_like", "SiGe", lambda: "sige" in compact or "si-ge" in lowered),
        ("broad::Mg2SiSn_like", "Mg2Si/Mg2Sn", lambda: "mg2si" in compact or "mg2sn" in compact or "mg-si" in lowered or "mg-sn" in lowered),
        ("broad::CoSb_skutterudite_like", "CoSb/skutterudite", lambda: "cosb3" in compact or "cosb" in compact or "skutterudite" in lowered),
        ("broad::oxide", "oxide/O-containing", lambda: "oxide" in lowered or "o-containing" in lowered or contains_element(text, "O")),
        ("broad::sulfide", "sulfide/S-containing", lambda: "sulfide" in lowered or "s-containing" in lowered or contains_element(text, "S")),
        ("broad::selenide", "selenide/Se-containing", lambda: "selenide" in lowered or "se-containing" in lowered or contains_element(text, "Se")),
        ("broad::telluride", "telluride/Te-containing", lambda: "telluride" in lowered or "te-containing" in lowered or contains_element(text, "Te")),
    ]
    for group, rule, predicate in rules:
        if predicate():
            return group, "matched", rule
    return "unmatched", "unmatched", "no_rule_matched"


def infer_default_output_dir(script_path: Path) -> Path | None:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"DEFAULT_OUTPUT_DIR\s*=\s*PROJECT_ROOT\s*/\s*\"data\"\s*/\s*\"output\"\s*/\s*\"([^\"]+)\"", text)
    if match:
        return PROJECT_ROOT / "data" / "output" / match.group(1)
    if "starrydata2_step12_tau_fit" in text:
        return PROJECT_ROOT / "data" / "output" / "starrydata2_step12_tau_fit"
    return None


def script_csv_names(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8", errors="replace")
    names = re.findall(r"args\.output_dir\s*/\s*\"([^\"]+\.csv)\"", text)
    names.extend(re.findall(r"\"([^\"]+_step12\.csv)\"", text))
    return list(dict.fromkeys(names))


def header_columns(path: Path) -> list[str]:
    if path.suffix.casefold() == ".parquet":
        return list(pd.read_parquet(path, columns=[]).columns)
    return pd.read_csv(path, nrows=0).columns.tolist()


def detect_columns(header: list[str]) -> tuple[str, str, str, str, str]:
    ct_col = next((column for column in OLD_CT_COLUMNS if column in header), "")
    temp_col = next((column for column in OLD_TEMPERATURE_COLUMNS if column in header), "")
    material_col = next((column for column in OLD_MATERIAL_COLUMNS if column in header), "")
    carrier_col = next((column for column in OLD_CARRIER_COLUMNS if column in header), "")
    status = "ok" if ct_col and temp_col and material_col else "missing_required_column"
    return ct_col, temp_col, material_col, carrier_col, status


def detect_old_ct_from_script(script_path: Path, output_dir: Path, suffix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    log("reading old C(T) script...")
    log("detecting old C(T) output file...")
    rows: list[dict[str, Any]] = []
    if not script_path.exists():
        rows.append(
            {
                "old_ct_script_path": str(script_path),
                "detected_output_file": "",
                "detected_old_ct_column": "",
                "detected_temperature_column": "",
                "detected_material_column": "",
                "detected_carrier_column": "",
                "detection_status": "script_missing",
                "notes": "old C(T) source script does not exist",
            }
        )
        return pd.DataFrame(rows), {}

    step_output_dir = infer_default_output_dir(script_path)
    names = script_csv_names(script_path)
    candidate_paths = [step_output_dir / name for name in names] if step_output_dir else []
    preferred = ["sigma_predictions_step12.csv", "initial_tau_fit_predictions_step12.csv", "prefactor_baseline_audit_step12.csv"]
    candidate_paths = sorted(
        [path for path in candidate_paths if path.exists()],
        key=lambda p: preferred.index(p.name) if p.name in preferred else len(preferred),
    )

    selected: dict[str, Any] = {}
    for path in candidate_paths:
        header = header_columns(path)
        ct_col, temp_col, material_col, carrier_col, status = detect_columns(header)
        notes = "candidate from fit_tau_eff_step12.py output writes"
        if path.name == "sigma_predictions_step12.csv" and ct_col == "prefactor_C_S_per_m_step12":
            notes = "selected: primary Step12 prediction output contains prefactor_C_S_per_m_step12 required by fit_tau_eff_step12.py"
            selected = {
                "path": path,
                "ct_col": ct_col,
                "temp_col": temp_col,
                "material_col": material_col,
                "carrier_col": carrier_col,
                "notes": notes,
            }
        elif not selected and status == "ok":
            notes = "fallback selected candidate with old C(T)-like and temperature columns"
            selected = {
                "path": path,
                "ct_col": ct_col,
                "temp_col": temp_col,
                "material_col": material_col,
                "carrier_col": carrier_col,
                "notes": notes,
            }
        rows.append(
            {
                "old_ct_script_path": str(script_path),
                "detected_output_file": str(path),
                "detected_old_ct_column": ct_col,
                "detected_temperature_column": temp_col,
                "detected_material_column": material_col,
                "detected_carrier_column": carrier_col,
                "detection_status": status,
                "notes": notes if path == selected.get("path") else f"not selected: {status}",
            }
        )

    if not rows:
        rows.append(
            {
                "old_ct_script_path": str(script_path),
                "detected_output_file": "",
                "detected_old_ct_column": "",
                "detected_temperature_column": "",
                "detected_material_column": "",
                "detected_carrier_column": "",
                "detection_status": "no_existing_output_csv",
                "notes": f"script output directory {step_output_dir} or expected CSV files are missing",
            }
        )
    parse_summary = pd.DataFrame(rows)
    if selected:
        selected_file = str(selected["path"])
        parse_summary["detection_status"] = np.where(
            parse_summary["detected_output_file"].eq(selected_file), "selected", parse_summary["detection_status"]
        )
    parse_summary.to_csv(output_path(output_dir, "focus_ct_vs_sigma0ref_fit_tau_script_parse_summary", suffix), index=False)
    return parse_summary, selected


def old_material_label(df: pd.DataFrame, material_col: str) -> pd.Series:
    label = pd.Series("", index=df.index, dtype="object")
    for column in [material_col, "material_system", "composition", "material_name_raw", "prefactor_group_key_step12"]:
        if column and column in df.columns:
            values = df[column].fillna("").astype(str).str.strip()
            usable = label.str.strip().eq("") | label.str.casefold().isin({"unknown", "nan", "none", "all_data"})
            label = label.where(~usable, values)
    return label.replace("", "unknown")


def load_old_ct(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("loading old C(T) curves...")
    log("detecting old C(T) column...")
    path = selected["path"]
    ct_col = selected["ct_col"]
    temp_col = selected["temp_col"]
    material_col = selected["material_col"]
    carrier_col = selected.get("carrier_col", "")
    header = header_columns(path)
    keep = list(dict.fromkeys([c for c in [ct_col, temp_col, material_col, carrier_col, "material_system", "composition", "prefactor_group_key_step12"] if c and c in header]))
    raw = pd.read_csv(path, usecols=keep, low_memory=False)
    raw["source_file"] = str(path)
    raw["old_material_label"] = old_material_label(raw, material_col)
    mapped = raw["old_material_label"].map(map_old_material_label)
    raw["material_group_key_mapped"] = mapped.map(lambda x: x[0])
    raw["mapping_status"] = mapped.map(lambda x: x[1])
    raw["mapping_rule"] = mapped.map(lambda x: x[2])
    if carrier_col and carrier_col in raw.columns:
        raw["carrier_type"] = raw[carrier_col].fillna("").astype(str).str.strip().str.lower()
    else:
        raw["carrier_type"] = "all"
    raw["T_K"] = pd.to_numeric(raw[temp_col], errors="coerce")
    values = pd.to_numeric(raw[ct_col], errors="coerce")
    raw["old_C_T_S_per_m"] = 10.0 ** values if ct_col in OLD_CT_LOG_COLUMNS else values
    raw["old_ct_parse_status"] = np.where(
        raw["material_group_key_mapped"].eq("unmatched"),
        "unmatched_material",
        np.where(np.isfinite(raw["T_K"]) & finite_positive(raw["old_C_T_S_per_m"]), "ok", "invalid_numeric"),
    )
    mapping = (
        raw[["old_material_label", "material_group_key_mapped", "mapping_status", "mapping_rule"]]
        .drop_duplicates()
        .sort_values(["material_group_key_mapped", "old_material_label"])
    )
    unmatched = mapping[mapping["mapping_status"].eq("unmatched")].copy()
    ok = raw[raw["old_ct_parse_status"].eq("ok")].copy()
    carrier_values = ["p", "n"] if ok["carrier_type"].eq("all").any() else []
    if carrier_values:
        common = ok[ok["carrier_type"].eq("all")].copy()
        ok = ok[~ok["carrier_type"].eq("all")].copy()
        expanded = []
        for carrier in carrier_values:
            sub = common.copy()
            sub["carrier_type"] = carrier
            expanded.append(sub)
        if expanded:
            ok = pd.concat([ok, *expanded], ignore_index=True)
    ok = ok[ok["carrier_type"].isin(["p", "n"])].copy()
    old_ct = (
        ok.groupby(["source_file", "old_material_label", "material_group_key_mapped", "carrier_type", "T_K"], dropna=False, sort=True)
        .agg(old_C_T_S_per_m=("old_C_T_S_per_m", "median"))
        .reset_index()
    )
    old_ct["log10_old_C_T_S_per_m"] = np.log10(old_ct["old_C_T_S_per_m"])
    old_ct["old_ct_parse_status"] = "ok"
    return old_ct, mapping, unmatched


def load_sigma0_ref(path: Path) -> pd.DataFrame:
    log("loading sigma0_ref curves...")
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
        "sigma0_ref_S_per_m",
        "log10_sigma0_ref_S_per_m",
        "train_row_count",
        "train_sample_count",
        "train_paper_count",
        "is_reference_bin_candidate",
        "reliability_level",
    ]
    df = read_table(path, columns)
    base = (
        df["config_id"].astype(str).eq(TARGET_CONFIG)
        & df["split_scheme"].astype(str).eq("sample_holdout")
        & df["reference_source_subset"].astype(str).eq("conservative_valid")
        & df["eval_target_subset"].astype(str).eq("all_valid")
        & df["group_scheme"].astype(str).eq("material_family")
        & df["curve_method"].astype(str).eq("sample_median")
    )
    filtered = df[base & as_bool(df["is_reference_bin_candidate"])].copy()
    if filtered.empty:
        filtered = df[base].copy()
    for column in ["T_bin_center_K", "sigma0_ref_S_per_m", "log10_sigma0_ref_S_per_m"]:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    filtered = filtered[np.isfinite(filtered["T_bin_center_K"]) & finite_positive(filtered["sigma0_ref_S_per_m"])].copy()
    filtered["log10_sigma0_ref_S_per_m"] = filtered["log10_sigma0_ref_S_per_m"].where(
        np.isfinite(filtered["log10_sigma0_ref_S_per_m"]), np.log10(filtered["sigma0_ref_S_per_m"])
    )
    return filtered[
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


def nearest_comparison(old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            old_sub = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))].sort_values("T_K")
            ref_sub = sigma0_ref[(sigma0_ref["material_group_key"].eq(group)) & (sigma0_ref["carrier_type"].eq(carrier))].sort_values("T_bin_center_K")
            if old_sub.empty or ref_sub.empty:
                continue
            ref_t = ref_sub["T_bin_center_K"].to_numpy(dtype=float)
            for _, old_row in old_sub.iterrows():
                t_old = float(old_row["T_K"])
                pos = int(np.nanargmin(np.abs(ref_t - t_old)))
                ref_row = ref_sub.iloc[pos]
                old_c = float(old_row["old_C_T_S_per_m"])
                sigma0 = float(ref_row["sigma0_ref_S_per_m"])
                rows.append(
                    {
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "T_K_old_ct": t_old,
                        "old_C_T_S_per_m": old_c,
                        "log10_old_C_T_S_per_m": math.log10(old_c),
                        "T_K_sigma0_ref": float(ref_row["T_bin_center_K"]),
                        "sigma0_ref_S_per_m": sigma0,
                        "log10_sigma0_ref_S_per_m": float(ref_row["log10_sigma0_ref_S_per_m"]),
                        "T_delta_K": float(ref_row["T_bin_center_K"] - t_old),
                        "log10_sigma0ref_over_oldCT": math.log10(sigma0 / old_c),
                        "match_method": "nearest_T_bin_center",
                    }
                )
    return pd.DataFrame(rows)


def plot_overlay(group: str, carrier: str, old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating overlay figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        old_ct["T_K"],
        old_ct["old_C_T_S_per_m"],
        marker="o",
        markersize=3,
        linewidth=2.2,
        color="#d62728",
        label="Old C(T) from SS2026",
    )
    ax.plot(
        sigma0_ref["T_bin_center_K"],
        sigma0_ref["sigma0_ref_S_per_m"],
        marker="s",
        markersize=3,
        linewidth=2.2,
        color="#ff7f0e",
        label="Seebeck-derived sigma0_ref(T)",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("S/m")
    ax.set_title(f"{group}, {carrier}: old C(T) vs sigma0_ref(T)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_ratio(group: str, carrier: str, comp: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating log-ratio figure...")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0.0, color="black", linewidth=1)
    ax.plot(comp["T_K_old_ct"], comp["log10_sigma0ref_over_oldCT"], marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("log10(sigma0_ref / old_C_T)")
    ax.set_title(f"{group}, {carrier}: sigma0_ref relative to old C(T)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary(old_ct: pd.DataFrame, sigma0_ref: pd.DataFrame, comparison: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            old = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))]
            ref = sigma0_ref[(sigma0_ref["material_group_key"].eq(group)) & (sigma0_ref["carrier_type"].eq(carrier))]
            comp = comparison[(comparison["material_group_key"].eq(group)) & (comparison["carrier_type"].eq(carrier))]
            warnings = []
            if old.empty:
                warnings.append("no_old_ct")
            if ref.empty:
                warnings.append("no_sigma0_ref")
            if comp.empty:
                warnings.append("no_comparison")
            t_min = np.nan
            t_max = np.nan
            if not old.empty or not ref.empty:
                temps = []
                if not old.empty:
                    temps.append(old["T_K"])
                if not ref.empty:
                    temps.append(ref["T_bin_center_K"])
                all_t = pd.concat(temps)
                t_min = all_t.min()
                t_max = all_t.max()
            rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "old_ct_points": len(old),
                    "sigma0_ref_points": len(ref),
                    "comparison_points": len(comp),
                    "T_min_K": t_min,
                    "T_max_K": t_max,
                    "old_C_T_median_S_per_m": old["old_C_T_S_per_m"].median() if not old.empty else np.nan,
                    "sigma0_ref_median_S_per_m": ref["sigma0_ref_S_per_m"].median() if not ref.empty else np.nan,
                    "median_log10_sigma0ref_over_oldCT": comp["log10_sigma0ref_over_oldCT"].median() if not comp.empty else np.nan,
                    "warning": ";".join(warnings),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    report_path: Path,
    script_path: Path,
    selected: dict[str, Any],
    sigma0_ref_path: Path,
    target_groups: list[str],
    summary: pd.DataFrame,
    figure_index: pd.DataFrame,
    unmatched: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = [
        "# Old C(T) vs Sigma0_ref(T) Only From Script",
        "",
        "## Old C(T) Source",
        f"- old C(T) source script: `{script_path}`",
        f"- detected old C(T) CSV: `{selected.get('path', '')}`",
        f"- adopted old C(T) column: `{selected.get('ct_col', '')}`",
        f"- temperature column: `{selected.get('temp_col', '')}`",
        f"- material column: `{selected.get('material_col', '')}`",
        f"- carrier_type column: `{selected.get('carrier_col', '')}`",
        f"- detection note: {selected.get('notes', '')}",
        "",
        "## Current Sigma0_ref(T)",
        f"- sigma0_ref file: `{sigma0_ref_path}`",
        "- filter: sample_holdout / conservative_valid / all_valid / material_family / sample_median; reference-bin candidates preferred",
        "",
        "## Targets",
        f"- material groups: {', '.join(target_groups)}",
        "- carrier types: p, n",
        "",
        "## Physical Interpretation",
        "- Old C(T) is the SS2026 empirical baseline against measured electrical conductivity.",
        "- sigma0_ref(T) is the Seebeck-derived coefficient after Fermi-level correction.",
        "- They both have units of S/m, but they are not the same physical quantity.",
        "- This overlay is for comparing temperature-dependence shape and scale, not for treating the curves as identical observables.",
        "",
        "## Summary By Group And Carrier",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['material_group_key']} / {row['carrier_type']}: old_ct={row['old_ct_points']}, "
            f"sigma0_ref={row['sigma0_ref_points']}, comparison={row['comparison_points']}, "
            f"median_log10_ratio={row['median_log10_sigma0ref_over_oldCT']}, warning={row['warning']}"
        )
    ratio = pd.to_numeric(comparison.get("log10_sigma0ref_over_oldCT", pd.Series(dtype=float)), errors="coerce").dropna()
    lines.extend(["", "## Overall Ratio Summary"])
    if ratio.empty:
        lines.append("- no matched points")
    else:
        lines.append(f"- count: {int(ratio.count())}")
        lines.append(f"- median log10(sigma0_ref / old_C_T): {float(ratio.median()):.6g}")
        lines.append(f"- min/max: {float(ratio.min()):.6g} / {float(ratio.max()):.6g}")
    lines.extend(["", "## Figures"])
    if figure_index.empty:
        lines.append("- none")
    else:
        for _, row in figure_index.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']} / {row['figure_type']}: `{row['figure_path_png']}`")
    warnings = summary[summary["warning"].fillna("").astype(str).str.len() > 0]
    lines.extend(["", "## Warnings"])
    if warnings.empty:
        lines.append("- none")
    else:
        for _, row in warnings.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']}: {row['warning']}")
    lines.extend(["", "## Unmatched Old Material Labels"])
    lines.append(f"- unmatched unique labels: {len(unmatched)}")
    for label in unmatched["old_material_label"].astype(str).head(50):
        lines.append(f"- {label}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Old C(T) and sigma0_ref(T) have different meanings.",
            "- No new sigma_pred is calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data are not read.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    script_path: Path,
    selected: dict[str, Any],
    old_ct: pd.DataFrame,
    sigma0_ref: pd.DataFrame,
    target_groups: list[str],
    comparison_path: Path,
    summary_path: Path,
    figure_index: pd.DataFrame,
    report_path: Path,
    sigma0_ref_path: Path,
) -> None:
    log("running sanity checks...")
    if not script_path.exists():
        raise RuntimeError("fit_tau_eff_step12.py is missing")
    if not selected:
        raise RuntimeError("old C(T) output CSV could not be detected")
    if not Path(selected["path"]).exists():
        raise RuntimeError("old C(T) output CSV is missing")
    if not selected.get("ct_col"):
        raise RuntimeError("old C(T) column could not be detected")
    if sigma0_ref.empty:
        raise RuntimeError("sigma0_ref(T) file could not be loaded")
    if not target_groups:
        raise RuntimeError("no target groups")
    if old_ct.empty or not finite_positive(old_ct["old_C_T_S_per_m"]).all():
        raise RuntimeError("old_C_T_S_per_m is empty or invalid")
    if not finite_positive(sigma0_ref["sigma0_ref_S_per_m"]).all():
        raise RuntimeError("sigma0_ref_S_per_m contains invalid values")
    if figure_index.empty or not figure_index["figure_type"].eq("overlay").any():
        raise RuntimeError("no overlay figure was created")
    if not comparison_path.exists():
        raise RuntimeError("comparison table was not created")
    if not summary_path.exists():
        raise RuntimeError("summary CSV was not created")
    if figure_index.empty:
        raise RuntimeError("figure index was not created")
    if not report_path.exists():
        raise RuntimeError("report was not created")
    for path in [sigma0_ref_path, Path(selected["path"])]:
        text = str(path).replace("\\", "/").casefold()
        if "step4_sigma0_reference_curve" in text:
            raise RuntimeError("Step4 full-data reference curve was used")
        if "/raw/" in text or "starrydata2/raw" in text:
            raise RuntimeError("Starrydata2 raw data was used")


def main() -> None:
    start = time.time()
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    target_groups = list(args.target_groups)
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]

    parse_summary, selected = detect_old_ct_from_script(args.old_ct_script, args.output, args.output_suffix)
    if not selected:
        write_report(args.report, args.old_ct_script, {}, Path(""), target_groups, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        raise RuntimeError("old C(T) output CSV could not be detected; parse summary and report were written")

    old_ct, mapping, unmatched = load_old_ct(selected)
    sigma0_ref_path = args.current_sigma0_ref or resolve_first(DEFAULT_SIGMA0_REF, "sigma0_ref")
    sigma0_ref = load_sigma0_ref(sigma0_ref_path)

    log("normalizing old C(T) columns...")
    log("mapping material labels...")
    log("filtering target groups...")
    old_ct = old_ct[old_ct["material_group_key_mapped"].isin(target_groups)].copy()
    sigma0_ref = sigma0_ref[sigma0_ref["material_group_key"].isin(target_groups)].copy()

    comparison = nearest_comparison(old_ct, sigma0_ref, target_groups)
    summary = build_summary(old_ct, sigma0_ref, comparison, target_groups)

    figure_rows: list[dict[str, Any]] = []
    figure_id = 1
    for group in target_groups:
        for carrier in ["p", "n"]:
            log(f"processing group/carrier {group} / {carrier}")
            old = old_ct[(old_ct["material_group_key_mapped"].eq(group)) & (old_ct["carrier_type"].eq(carrier))].sort_values("T_K")
            ref = sigma0_ref[(sigma0_ref["material_group_key"].eq(group)) & (sigma0_ref["carrier_type"].eq(carrier))].sort_values("T_bin_center_K")
            comp = comparison[(comparison["material_group_key"].eq(group)) & (comparison["carrier_type"].eq(carrier))].sort_values("T_K_old_ct")
            if old.empty or ref.empty:
                log(f"warning: missing curve for {group} / {carrier}; skipping figures")
                continue
            safe = f"{safe_name(group)}_{carrier}"
            overlay_png = args.figures / f"{safe}_oldCT_vs_sigma0ref_overlay{args.output_suffix}.png"
            overlay_pdf = args.figures / f"{safe}_oldCT_vs_sigma0ref_overlay{args.output_suffix}.pdf"
            plot_overlay(group, carrier, old, ref, overlay_png, overlay_pdf)
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": "overlay",
                    "figure_path_png": str(overlay_png),
                    "figure_path_pdf": str(overlay_pdf),
                    "title": f"{group}, {carrier}: old C(T) vs sigma0_ref(T)",
                    "n_old_ct_points": len(old),
                    "n_sigma0_ref_points": len(ref),
                    "description": "Curve-only overlay of old SS2026 C(T) and Seebeck-derived sigma0_ref(T).",
                }
            )
            figure_id += 1
            if not comp.empty:
                ratio_png = args.figures / f"{safe}_log_ratio_sigma0ref_over_oldCT{args.output_suffix}.png"
                ratio_pdf = args.figures / f"{safe}_log_ratio_sigma0ref_over_oldCT{args.output_suffix}.pdf"
                plot_ratio(group, carrier, comp, ratio_png, ratio_pdf)
                figure_rows.append(
                    {
                        "figure_id": figure_id,
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "figure_type": "log_ratio",
                        "figure_path_png": str(ratio_png),
                        "figure_path_pdf": str(ratio_pdf),
                        "title": f"{group}, {carrier}: log10 sigma0_ref over old C(T)",
                        "n_old_ct_points": len(old),
                        "n_sigma0_ref_points": len(ref),
                        "description": "Nearest-temperature log-ratio comparison.",
                    }
                )
                figure_id += 1

    figure_index = pd.DataFrame(figure_rows)

    log("writing CSV outputs...")
    script_summary_path = output_path(args.output, "focus_ct_vs_sigma0ref_fit_tau_script_parse_summary", args.output_suffix)
    parse_summary.to_csv(script_summary_path, index=False)
    old_ct.to_csv(output_path(args.output, "focus_ct_vs_sigma0ref_old_ct_curves_normalized", args.output_suffix), index=False)
    sigma0_ref.to_csv(output_path(args.output, "focus_ct_vs_sigma0ref_sigma0_ref_curves", args.output_suffix), index=False)
    comparison_path = output_path(args.output, "focus_ct_vs_sigma0ref_curve_comparison_table", args.output_suffix)
    comparison.to_csv(comparison_path, index=False)
    summary_path = output_path(args.output, "focus_ct_vs_sigma0ref_summary_by_group_carrier", args.output_suffix)
    summary.to_csv(summary_path, index=False)
    figure_index_path = output_path(args.output, "focus_ct_vs_sigma0ref_figure_index", args.output_suffix)
    figure_index.to_csv(figure_index_path, index=False)
    mapping.to_csv(output_path(args.output, "focus_ct_vs_sigma0ref_material_mapping", args.output_suffix), index=False)
    unmatched.to_csv(output_path(args.output, "focus_ct_vs_sigma0ref_unmatched_old_material_labels", args.output_suffix), index=False)

    log("writing report...")
    write_report(args.report, args.old_ct_script, selected, sigma0_ref_path, target_groups, summary, figure_index, unmatched, comparison)

    run_sanity_checks(
        args.old_ct_script,
        selected,
        old_ct,
        sigma0_ref,
        target_groups,
        comparison_path,
        summary_path,
        figure_index,
        args.report,
        sigma0_ref_path,
    )
    elapsed = time.time() - start
    log(f"done. elapsed_seconds={elapsed:.2f}")
    print(f"old_ct_script: {args.old_ct_script}")
    print(f"old_ct_csv: {selected['path']}")
    print(f"old_ct_column: {selected['ct_col']}")
    print(f"target_groups: {', '.join(target_groups)}")
    print(f"group_carrier_rows: {len(summary)}")
    print(f"overlay_figures: {int(figure_index['figure_type'].eq('overlay').sum()) if not figure_index.empty else 0}")
    print(f"log_ratio_figures: {int(figure_index['figure_type'].eq('log_ratio').sum()) if not figure_index.empty else 0}")
    if not comparison.empty:
        print(f"median_log10_sigma0ref_over_oldCT: {comparison['log10_sigma0ref_over_oldCT'].median()}")
    print(f"output_dir: {args.output}")
    print(f"figure_dir: {args.figures}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
