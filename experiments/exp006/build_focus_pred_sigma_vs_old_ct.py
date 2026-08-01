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
DEFAULT_PREDICTIONS = [
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_test_predictions_valid.parquet",
    EXP_DIR / "data" / "processed" / "step6b_broad_family" / "step5b_test_predictions_valid.csv",
]
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "focus_pred_sigma_vs_old_ct"
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_pred_sigma_vs_old_ct"
DEFAULT_REPORT = EXP_DIR / "reports" / "focus_pred_sigma_vs_old_ct" / "focus_pred_sigma_vs_old_ct_report.md"
DEFAULT_CONFIG_ID = "sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median"
DEFAULT_TARGET_GROUPS = [
    "broad::SnTe_like",
    "broad::PbTe_like",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::SiGe_like",
    "broad::oxide",
    "broad::sulfide",
]

OLD_CT_COLUMNS = ["prefactor_C_S_per_m_step12", "C_T", "CT", "C(T)", "C_value", "C_S_per_m", "C_ref", "sigma_ref", "sigma_median", "sigma_median_S_per_m", "conductivity_ref"]
OLD_TEMPERATURE_COLUMNS = ["temperature_bin_K_step12", "T_K", "T", "temperature", "temperature_K", "T_bin_center_K"]
OLD_MATERIAL_COLUMNS = ["material_system", "material_family", "material_family_raw", "material_group_key", "material_group", "family", "system", "composition_group", "composition", "prefactor_group_key_step12"]
OLD_CARRIER_COLUMNS = ["n_or_p", "carrier_type", "pn_type", "type", "conduction_type"]


def log(message: str) -> None:
    print(f"[pred_vs_ct] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare broad-family sigma_pred points with old SS2026 C(T) curves.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--old-ct-script", type=Path, required=True)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--target-groups", nargs="+", default=DEFAULT_TARGET_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-rows-per-group", type=int, default=None)
    parser.add_argument("--include-exp-sigma", action="store_true")
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
    raise ValueError(f"Unsupported table: {path}")


def output_path(directory: Path, stem: str, suffix: str, extension: str = ".csv") -> Path:
    return directory / f"{stem}{suffix}{extension}"


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


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


def header_columns(path: Path) -> list[str]:
    if path.suffix.casefold() == ".parquet":
        return list(pd.read_parquet(path, columns=[]).columns)
    return pd.read_csv(path, nrows=0).columns.tolist()


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


def detect_old_ct(script_path: Path) -> dict[str, Any]:
    log("reading old C(T) script...")
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    output_dir = infer_default_output_dir(script_path)
    if output_dir is None:
        raise RuntimeError("Could not infer Step12 output directory from old C(T) script")
    names = script_csv_names(script_path)
    candidate_paths = [output_dir / name for name in names if (output_dir / name).exists()]
    preferred = ["sigma_predictions_step12.csv", "initial_tau_fit_predictions_step12.csv", "prefactor_baseline_audit_step12.csv"]
    candidate_paths = sorted(candidate_paths, key=lambda p: preferred.index(p.name) if p.name in preferred else len(preferred))
    for path in candidate_paths:
        header = header_columns(path)
        ct_col = next((c for c in OLD_CT_COLUMNS if c in header), "")
        temp_col = next((c for c in OLD_TEMPERATURE_COLUMNS if c in header), "")
        material_col = next((c for c in OLD_MATERIAL_COLUMNS if c in header), "")
        carrier_col = next((c for c in OLD_CARRIER_COLUMNS if c in header), "")
        if path.name == "sigma_predictions_step12.csv" and ct_col == "prefactor_C_S_per_m_step12" and temp_col and material_col:
            return {
                "path": path,
                "ct_col": ct_col,
                "temp_col": temp_col,
                "material_col": material_col,
                "carrier_col": carrier_col,
                "notes": "selected primary Step12 output containing prefactor_C_S_per_m_step12",
            }
    raise RuntimeError("No valid old C(T) output CSV was detected from fit_tau_eff_step12.py")


def old_material_label(df: pd.DataFrame, material_col: str) -> pd.Series:
    label = pd.Series("", index=df.index, dtype="object")
    for column in [material_col, "material_system", "composition", "prefactor_group_key_step12"]:
        if column and column in df.columns:
            values = df[column].fillna("").astype(str).str.strip()
            usable = label.str.strip().eq("") | label.str.casefold().isin({"unknown", "nan", "none", "all_data"})
            label = label.where(~usable, values)
    return label.replace("", "unknown")


def load_old_ct(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("loading old C(T) curves...")
    path = selected["path"]
    header = header_columns(path)
    keep = list(
        dict.fromkeys(
            [
                c
                for c in [
                    selected["ct_col"],
                    selected["temp_col"],
                    selected["material_col"],
                    selected["carrier_col"],
                    "material_system",
                    "composition",
                    "prefactor_group_key_step12",
                ]
                if c and c in header
            ]
        )
    )
    raw = pd.read_csv(path, usecols=keep, low_memory=False)
    raw["source_file"] = str(path)
    raw["old_material_label"] = old_material_label(raw, selected["material_col"])
    mapped = raw["old_material_label"].map(map_old_material_label)
    raw["material_group_key_mapped"] = mapped.map(lambda x: x[0])
    raw["mapping_status"] = mapped.map(lambda x: x[1])
    raw["mapping_rule"] = mapped.map(lambda x: x[2])
    carrier_col = selected.get("carrier_col", "")
    raw["carrier_type"] = raw[carrier_col].fillna("").astype(str).str.strip().str.lower() if carrier_col in raw.columns else "all"
    raw["T_K"] = pd.to_numeric(raw[selected["temp_col"]], errors="coerce")
    raw["old_C_T_S_per_m"] = pd.to_numeric(raw[selected["ct_col"]], errors="coerce")
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
    if ok["carrier_type"].eq("all").any():
        common = ok[ok["carrier_type"].eq("all")].copy()
        ok = ok[~ok["carrier_type"].eq("all")].copy()
        ok = pd.concat([ok, common.assign(carrier_type="p"), common.assign(carrier_type="n")], ignore_index=True)
    ok = ok[ok["carrier_type"].isin(["p", "n"])].copy()
    normalized = (
        ok.groupby(["source_file", "old_material_label", "material_group_key_mapped", "carrier_type", "T_K"], dropna=False, sort=True)
        .agg(old_C_T_S_per_m=("old_C_T_S_per_m", "median"))
        .reset_index()
    )
    normalized["log10_old_C_T_S_per_m"] = np.log10(normalized["old_C_T_S_per_m"])
    normalized["old_ct_parse_status"] = "ok"
    line = (
        normalized.groupby(["material_group_key_mapped", "carrier_type", "T_K"], dropna=False, sort=True)
        .agg(old_C_T_S_per_m=("old_C_T_S_per_m", "median"))
        .reset_index()
    )
    line["log10_old_C_T_S_per_m"] = np.log10(line["old_C_T_S_per_m"])
    return normalized, line, mapping, unmatched


def load_predictions(path: Path, config_id: str) -> pd.DataFrame:
    log("loading prediction rows...")
    columns = [
        "config_id",
        "prediction_status",
        "material_group_key",
        "material_group_key_for_prediction",
        "carrier_type",
        "T_K",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "sigma_S_per_m",
        "log10_sigma_S_per_m",
        "row_id",
        "paper_id",
        "sample_id",
        "sample_key",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "formula_raw",
        "material_name_raw",
    ]
    df = read_table(path, columns)
    df = df[df["config_id"].astype(str).eq(config_id) & df["prediction_status"].astype(str).eq("ok")].copy()
    df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
    df["sigma_pred_S_per_m"] = pd.to_numeric(df["sigma_pred_S_per_m"], errors="coerce")
    df["log10_sigma_pred_S_per_m"] = pd.to_numeric(df["log10_sigma_pred_S_per_m"], errors="coerce")
    df["log10_sigma_pred_S_per_m"] = df["log10_sigma_pred_S_per_m"].where(
        np.isfinite(df["log10_sigma_pred_S_per_m"]), np.log10(df["sigma_pred_S_per_m"].where(df["sigma_pred_S_per_m"] > 0))
    )
    df = df[np.isfinite(df["T_K"]) & finite_positive(df["sigma_pred_S_per_m"]) & df["carrier_type"].isin(["p", "n"])].copy()
    return df[[c for c in columns if c in df.columns]]


def limited_for_plot(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df
    return df.sort_values("T_K").iloc[np.linspace(0, len(df) - 1, max_rows).round().astype(int)].copy()


def nearest_comparison(pred: pd.DataFrame, old_line: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            pred_sub = pred[(pred["material_group_key"].eq(group)) & (pred["carrier_type"].eq(carrier))]
            old_sub = old_line[(old_line["material_group_key_mapped"].eq(group)) & (old_line["carrier_type"].eq(carrier))].sort_values("T_K")
            if pred_sub.empty or old_sub.empty:
                continue
            old_t = old_sub["T_K"].to_numpy(dtype=float)
            for _, row in pred_sub.iterrows():
                t_pred = float(row["T_K"])
                pos = int(np.nanargmin(np.abs(old_t - t_pred)))
                old = old_sub.iloc[pos]
                pred_sigma = float(row["sigma_pred_S_per_m"])
                old_ct = float(old["old_C_T_S_per_m"])
                rows.append(
                    {
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "row_id": row.get("row_id", ""),
                        "T_K_pred": t_pred,
                        "sigma_pred_S_per_m": pred_sigma,
                        "log10_sigma_pred_S_per_m": math.log10(pred_sigma),
                        "T_K_old_ct": float(old["T_K"]),
                        "old_C_T_S_per_m": old_ct,
                        "log10_old_C_T_S_per_m": math.log10(old_ct),
                        "T_delta_K": float(old["T_K"] - t_pred),
                        "log10_pred_over_oldCT": math.log10(pred_sigma / old_ct),
                        "match_method": "nearest_old_CT_temperature",
                    }
                )
    return pd.DataFrame(rows)


def plot_main(group: str, carrier: str, pred: pd.DataFrame, old: pd.DataFrame, png: Path, pdf: Path) -> None:
    log("creating sigma_pred vs old C(T) figure...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred["T_K"], pred["sigma_pred_S_per_m"], s=12, alpha=0.42, color="#1f77b4", edgecolors="none", label="Predicted sigma")
    ax.plot(old["T_K"], old["old_C_T_S_per_m"], linewidth=2.4, color="#d62728", label="Old C(T) from SS2026")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Electrical conductivity sigma [S/m]")
    ax.set_title(f"{group}, {carrier}: predicted sigma vs old C(T)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_with_exp(group: str, carrier: str, pred: pd.DataFrame, old: pd.DataFrame, png: Path, pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(pred["T_K"], pred["sigma_S_per_m"], s=9, alpha=0.18, color="#7f7f7f", edgecolors="none", label="Experimental sigma")
    ax.scatter(pred["T_K"], pred["sigma_pred_S_per_m"], s=12, alpha=0.42, color="#1f77b4", edgecolors="none", label="Predicted sigma")
    ax.plot(old["T_K"], old["old_C_T_S_per_m"], linewidth=2.4, color="#d62728", label="Old C(T) from SS2026")
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Electrical conductivity sigma [S/m]")
    ax.set_title(f"{group}, {carrier}: predicted/experimental sigma vs old C(T)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def build_summary(pred: pd.DataFrame, old_line: pd.DataFrame, comparison: pd.DataFrame, target_groups: list[str]) -> pd.DataFrame:
    rows = []
    for group in target_groups:
        for carrier in ["p", "n"]:
            p = pred[(pred["material_group_key"].eq(group)) & (pred["carrier_type"].eq(carrier))]
            o = old_line[(old_line["material_group_key_mapped"].eq(group)) & (old_line["carrier_type"].eq(carrier))]
            c = comparison[(comparison["material_group_key"].eq(group)) & (comparison["carrier_type"].eq(carrier))]
            warnings = []
            if p.empty:
                warnings.append("no_prediction_points")
            if o.empty:
                warnings.append("no_old_ct")
            if c.empty:
                warnings.append("no_nearest_comparison")
            rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "prediction_points": len(p),
                    "old_ct_points": len(o),
                    "T_pred_min_K": p["T_K"].min() if not p.empty else np.nan,
                    "T_pred_max_K": p["T_K"].max() if not p.empty else np.nan,
                    "T_old_ct_min_K": o["T_K"].min() if not o.empty else np.nan,
                    "T_old_ct_max_K": o["T_K"].max() if not o.empty else np.nan,
                    "sigma_pred_median_S_per_m": p["sigma_pred_S_per_m"].median() if not p.empty else np.nan,
                    "old_C_T_median_S_per_m": o["old_C_T_S_per_m"].median() if not o.empty else np.nan,
                    "median_log10_pred_over_oldCT_nearest": c["log10_pred_over_oldCT"].median() if not c.empty else np.nan,
                    "warning": ";".join(warnings),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    pred_path: Path,
    old_script: Path,
    selected: dict[str, Any],
    target_groups: list[str],
    summary: pd.DataFrame,
    figure_index: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    ratio = pd.to_numeric(comparison.get("log10_pred_over_oldCT", pd.Series(dtype=float)), errors="coerce").dropna()
    lines = [
        "# Predicted Sigma vs Old C(T)",
        "",
        "## Inputs",
        f"- prediction file: `{pred_path}`",
        f"- old C(T) source script: `{old_script}`",
        f"- detected old C(T) CSV: `{selected.get('path', '')}`",
        f"- adopted old C(T) column: `{selected.get('ct_col', '')}`",
        f"- temperature column: `{selected.get('temp_col', '')}`",
        f"- material column: `{selected.get('material_col', '')}`",
        f"- carrier column: `{selected.get('carrier_col', '')}`",
        "",
        "## Targets",
        f"- material groups: {', '.join(target_groups)}",
        "- carrier types: p, n",
        "",
        "## How To Read The Figures",
        "- Points are current broad_family predicted electrical conductivity, sigma_pred.",
        "- Lines are SS2026 old C(T).",
        "- If points lie near the old C(T) line, the current prediction has a similar scale and temperature dependence to the old C(T) baseline.",
        "- If points are far from the line, the S-input prediction gives values different from the old C(T) baseline.",
        "- Exact agreement is not required: old C(T) is based on measured sigma, while sigma_pred is predicted by the S-input method.",
        "",
        "## Summary",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['material_group_key']} / {row['carrier_type']}: predictions={row['prediction_points']}, old_ct={row['old_ct_points']}, "
            f"median_log10_pred_over_oldCT={row['median_log10_pred_over_oldCT_nearest']}, warning={row['warning']}"
        )
    lines.extend(["", "## Overall Ratio Summary"])
    if ratio.empty:
        lines.append("- no matched prediction and old C(T) points")
    else:
        lines.append(f"- count: {int(ratio.count())}")
        lines.append(f"- median log10(sigma_pred / old C(T)): {float(ratio.median()):.6g}")
        lines.append(f"- min/max: {float(ratio.min()):.6g} / {float(ratio.max()):.6g}")
    lines.extend(["", "## Figures"])
    if figure_index.empty:
        lines.append("- none")
    else:
        for _, row in figure_index.iterrows():
            lines.append(f"- {row['material_group_key']} / {row['carrier_type']} / {row['figure_type']}: `{row['figure_path_png']}`")
    lines.extend(["", "## Missing Combinations"])
    for _, row in summary[summary["warning"].fillna("").astype(str).str.len() > 0].iterrows():
        lines.append(f"- {row['material_group_key']} / {row['carrier_type']}: {row['warning']}")
    lines.extend(
        [
            "",
            "## Notes",
            "- Points are current predicted sigma.",
            "- Lines are SS2026 old C(T).",
            "- Experimental sigma points are not included in the main figures.",
            "- sigma0_ref is not included in these figures.",
            "- No new sigma_pred is calculated.",
            "- Step4 full-data reference curves are not used.",
            "- Starrydata2 raw data are not read.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    pred_path: Path,
    pred: pd.DataFrame,
    config_id: str,
    old_ct: pd.DataFrame,
    target_groups: list[str],
    figure_index: pd.DataFrame,
    summary_path: Path,
    report_path: Path,
    selected: dict[str, Any],
) -> None:
    log("running sanity checks...")
    if not pred_path.exists():
        raise RuntimeError("prediction file is missing")
    if pred.empty:
        raise RuntimeError("prediction rows are empty after filtering")
    if not pred["config_id"].astype(str).eq(config_id).all():
        raise RuntimeError("prediction rows include other config_id values")
    if "prediction_status" in pred and not pred["prediction_status"].astype(str).eq("ok").all():
        raise RuntimeError("prediction rows include non-ok status")
    if not finite_positive(pred["sigma_pred_S_per_m"]).all():
        raise RuntimeError("sigma_pred_S_per_m contains invalid values")
    if not Path(selected["path"]).exists():
        raise RuntimeError("old C(T) file is missing")
    if old_ct.empty or not finite_positive(old_ct["old_C_T_S_per_m"]).all():
        raise RuntimeError("old_C_T_S_per_m is empty or invalid")
    if not target_groups:
        raise RuntimeError("no target groups")
    if figure_index.empty:
        raise RuntimeError("no figures were created")
    if not summary_path.exists():
        raise RuntimeError("summary CSV was not created")
    if not report_path.exists():
        raise RuntimeError("report was not created")
    for path in [pred_path, Path(selected["path"])]:
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

    pred_path = args.predictions or resolve_first(DEFAULT_PREDICTIONS, "prediction")
    target_groups = list(args.target_groups)
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]

    pred = load_predictions(pred_path, args.config_id)
    selected = detect_old_ct(args.old_ct_script)
    old_normalized, old_line, mapping, unmatched = load_old_ct(selected)

    log("normalizing old C(T) columns...")
    log("mapping old material labels...")
    log("filtering target groups...")
    pred = pred[pred["material_group_key"].isin(target_groups)].copy()
    old_normalized = old_normalized[old_normalized["material_group_key_mapped"].isin(target_groups)].copy()
    old_line = old_line[old_line["material_group_key_mapped"].isin(target_groups)].copy()

    if args.max_rows_per_group is not None:
        limited_parts = []
        for _, group_df in pred.groupby(["material_group_key", "carrier_type"], sort=False):
            limited_parts.append(limited_for_plot(group_df.sort_values("T_K"), args.max_rows_per_group))
        pred = pd.concat(limited_parts, ignore_index=True) if limited_parts else pred.iloc[0:0].copy()

    comparison = nearest_comparison(pred, old_line, target_groups)
    summary = build_summary(pred, old_line, comparison, target_groups)

    figure_rows: list[dict[str, Any]] = []
    figure_id = 1
    for group in target_groups:
        for carrier in ["p", "n"]:
            log(f"processing group/carrier {group} / {carrier}")
            p = pred[(pred["material_group_key"].eq(group)) & (pred["carrier_type"].eq(carrier))].sort_values("T_K")
            o = old_line[(old_line["material_group_key_mapped"].eq(group)) & (old_line["carrier_type"].eq(carrier))].sort_values("T_K")
            if p.empty or o.empty:
                log(f"warning: missing prediction or old C(T) for {group} / {carrier}; skipping figure")
                continue
            safe = f"{safe_name(group)}_{carrier}"
            png = args.figures / f"{safe}_sigma_pred_points_vs_oldCT_line{args.output_suffix}.png"
            pdf = args.figures / f"{safe}_sigma_pred_points_vs_oldCT_line{args.output_suffix}.pdf"
            plot_main(group, carrier, p, o, png, pdf)
            figure_rows.append(
                {
                    "figure_id": figure_id,
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": "sigma_pred_vs_old_ct",
                    "figure_path_png": str(png),
                    "figure_path_pdf": str(pdf),
                    "title": f"{group}, {carrier}: predicted sigma vs old C(T)",
                    "n_prediction_points": len(p),
                    "n_old_ct_points": len(o),
                    "description": "Main figure: predicted sigma points and SS2026 old C(T) line only.",
                }
            )
            figure_id += 1
            if args.include_exp_sigma:
                png_exp = args.figures / f"{safe}_sigma_pred_exp_points_vs_oldCT_line{args.output_suffix}.png"
                pdf_exp = args.figures / f"{safe}_sigma_pred_exp_points_vs_oldCT_line{args.output_suffix}.pdf"
                plot_with_exp(group, carrier, p, o, png_exp, pdf_exp)
                figure_rows.append(
                    {
                        "figure_id": figure_id,
                        "material_group_key": group,
                        "carrier_type": carrier,
                        "figure_type": "sigma_pred_exp_vs_old_ct",
                        "figure_path_png": str(png_exp),
                        "figure_path_pdf": str(pdf_exp),
                        "title": f"{group}, {carrier}: predicted/experimental sigma vs old C(T)",
                        "n_prediction_points": len(p),
                        "n_old_ct_points": len(o),
                        "description": "Optional check figure including experimental sigma points.",
                    }
                )
                figure_id += 1

    figure_index = pd.DataFrame(figure_rows)

    log("writing CSV outputs...")
    pred.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_prediction_rows", args.output_suffix), index=False)
    old_normalized.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_old_ct_curves_normalized", args.output_suffix), index=False)
    summary_path = output_path(args.output, "focus_pred_sigma_vs_old_ct_summary_by_group_carrier", args.output_suffix)
    summary.to_csv(summary_path, index=False)
    comparison.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_nearest_comparison_table", args.output_suffix), index=False)
    figure_index.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_figure_index", args.output_suffix), index=False)
    mapping.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_material_mapping", args.output_suffix), index=False)
    unmatched.to_csv(output_path(args.output, "focus_pred_sigma_vs_old_ct_unmatched_old_material_labels", args.output_suffix), index=False)

    log("writing report...")
    write_report(args.report, pred_path, args.old_ct_script, selected, target_groups, summary, figure_index, comparison)
    run_sanity_checks(pred_path, pred, args.config_id, old_line, target_groups, figure_index, summary_path, args.report, selected)

    elapsed = time.time() - start
    log(f"done. elapsed_seconds={elapsed:.2f}")
    print(f"prediction_file: {pred_path}")
    print(f"old_ct_script: {args.old_ct_script}")
    print(f"old_ct_csv: {selected['path']}")
    print(f"old_ct_column: {selected['ct_col']}")
    print(f"target_groups: {', '.join(target_groups)}")
    print(f"group_carrier_rows: {len(summary)}")
    print(f"figures: {len(figure_index)}")
    if not comparison.empty:
        print(f"median_log10_pred_over_oldCT: {comparison['log10_pred_over_oldCT'].median()}")
    print(f"output_dir: {args.output}")
    print(f"figure_dir: {args.figures}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
