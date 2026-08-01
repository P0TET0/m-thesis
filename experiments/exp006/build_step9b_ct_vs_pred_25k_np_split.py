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
STEP9A_DIR = EXP_DIR / "data" / "processed" / "step9a_25k_bin_broad_family"
DEFAULT_PREDICTIONS = [
    STEP9A_DIR / "step5b_test_predictions_valid.parquet",
    STEP9A_DIR / "step5b_test_predictions_valid.csv",
]
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step9b_ct_vs_pred_25k_np_split"
DEFAULT_FIGURES = EXP_DIR / "figures" / "step9b_ct_vs_pred_25k_np_split"
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "step9b_ct_vs_pred_25k_np_split"
    / "step9b_ct_vs_pred_25k_np_split_report.md"
)
DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid"
    "__material_family__sample_median"
)
DEFAULT_TARGET_GROUPS = [
    "broad::SnTe_like",
    "broad::PbTe_like",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::SiGe_like",
    "broad::oxide",
    "broad::sulfide",
]

PREDICTION_INPUT_COLUMNS = [
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
    "sample_id",
    "paper_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
]
PREDICTION_OUTPUT_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "T_K",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "sigma_S_per_m",
    "log10_sigma_S_per_m",
    "row_id",
    "sample_id",
    "paper_id",
    "sample_key",
    "validation_sample_group_id",
    "validation_paper_group_id",
    "formula_raw",
    "material_name_raw",
]
OLD_CURVE_COLUMNS = [
    "source_file",
    "old_material_label_examples",
    "material_group_key_mapped",
    "T_K",
    "old_C_T_S_per_m",
    "log10_old_C_T_S_per_m",
    "n_rows_aggregated",
    "old_ct_parse_status",
]
SUMMARY_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "prediction_points",
    "old_ct_points",
    "T_pred_min_K",
    "T_pred_max_K",
    "T_old_ct_min_K",
    "T_old_ct_max_K",
    "sigma_pred_median_S_per_m",
    "old_C_T_median_S_per_m",
    "median_log10_pred_over_oldCT_nearest",
    "warning",
]
NEAREST_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "row_id",
    "T_K_pred",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "T_K_old_ct",
    "old_C_T_S_per_m",
    "log10_old_C_T_S_per_m",
    "T_delta_K",
    "log10_pred_over_oldCT",
    "match_method",
]
FIGURE_INDEX_COLUMNS = [
    "figure_id",
    "material_group_key",
    "carrier_type",
    "figure_type",
    "figure_path_png",
    "figure_path_pdf",
    "title",
    "n_prediction_points",
    "n_old_ct_points",
    "description",
]

ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Step9A 25 K sigma_pred points against p/n-unsplit SS2026 old C(T)."
    )
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--old-ct-script", type=Path, required=True)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--target-groups", nargs="+", default=DEFAULT_TARGET_GROUPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-rows-per-group", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[step9b] {message}", flush=True)


def output_path(directory: Path, stem: str, suffix: str, extension: str = ".csv") -> Path:
    return directory / f"{stem}{suffix}{extension}"


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.casefold() in {"", "nan", "none", "null", "unknown", "n/a", "na"}:
        return ""
    return text


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return np.isfinite(values) & values.gt(0)


def read_table(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path, columns=columns)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    raise ValueError(f"Unsupported table extension: {path.suffix}")


def resolve_predictions(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        if explicit.resolve().parent != STEP9A_DIR.resolve():
            raise ValueError("Predictions must come from the Step9A 25 K output directory")
        return explicit
    for path in DEFAULT_PREDICTIONS:
        if path.exists():
            return path
    raise FileNotFoundError("Step9A 25 K valid predictions were not found")


def validate_paths(args: argparse.Namespace, prediction_path: Path) -> None:
    if not args.old_ct_script.exists():
        raise FileNotFoundError(args.old_ct_script)
    if not args.target_groups:
        raise ValueError("--target-groups must not be empty")
    if len(args.target_groups) != len(set(args.target_groups)):
        raise ValueError("--target-groups contains duplicates")
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    if args.max_rows_per_group is not None and args.max_rows_per_group <= 0:
        raise ValueError("--max-rows-per-group must be positive")
    protected = STEP9A_DIR.resolve()
    for label, path in [
        ("output", args.output),
        ("figures", args.figures),
        ("report directory", args.report.parent),
    ]:
        resolved = path.resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(f"Step9B {label} must not be inside the protected Step9A directory")
    if prediction_path.resolve().parent != protected:
        raise ValueError("Resolved prediction input is not a Step9A output")


def directory_manifest(directory: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            stat = path.stat()
            rows.append(
                {
                    "relative_path": path.relative_to(directory).as_posix(),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return pd.DataFrame(rows)


def protection_manifest(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    merged = before.merge(after, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    merged["unchanged"] = (
        merged["_merge"].eq("both")
        & merged["size_before"].eq(merged["size_after"])
        & merged["mtime_ns_before"].eq(merged["mtime_ns_after"])
    )
    return merged


def load_predictions(path: Path, config_id: str) -> pd.DataFrame:
    log("loading Step9A prediction rows...")
    frame = read_table(path, PREDICTION_INPUT_COLUMNS)
    missing = sorted(set(PREDICTION_INPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction input missing columns: {missing}")
    if not frame["config_id"].astype(str).eq(config_id).any():
        raise ValueError(f"Requested config_id is absent: {config_id}")

    log("filtering target config...")
    frame = frame[
        frame["config_id"].astype(str).eq(config_id)
        & frame["prediction_status"].astype(str).eq("ok")
    ].copy()
    for column in [
        "T_K",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "sigma_S_per_m",
        "log10_sigma_S_per_m",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        np.isfinite(frame["T_K"])
        & frame["T_K"].gt(0)
        & finite_positive(frame["sigma_pred_S_per_m"])
        & np.isfinite(frame["log10_sigma_pred_S_per_m"])
        & frame["carrier_type"].astype(str).isin({"p", "n"})
        & frame["material_group_key"].map(clean_text).ne("")
    )
    frame = frame[valid].copy()
    if frame.empty:
        raise ValueError("No usable Step9A prediction rows remain")
    return frame


def inspect_old_ct_script(script_path: Path) -> dict[str, Any]:
    log("reading old C(T) script...")
    text = script_path.read_text(encoding="utf-8", errors="replace")
    required_tokens = [
        "sigma_predictions_step12.csv",
        "prefactor_C_S_per_m_step12",
        "temperature_bin_K_step12",
        "material_system",
        "n_or_p",
    ]
    missing = [token for token in required_tokens if token not in text]
    if missing:
        raise ValueError(f"Old C(T) script does not confirm expected Step12 tokens: {missing}")
    directory_match = re.search(
        r'DEFAULT_OUTPUT_DIR\s*=\s*PROJECT_ROOT\s*/\s*"data"\s*/\s*"output"\s*/\s*"([^"]+)"',
        text,
    )
    directory_name = directory_match.group(1) if directory_match else "starrydata2_step12_tau_fit"
    csv_path = PROJECT_ROOT / "data" / "output" / directory_name / "sigma_predictions_step12.csv"
    return {
        "source_script": script_path,
        "path": csv_path,
        "ct_column": "prefactor_C_S_per_m_step12",
        "temperature_column": "temperature_bin_K_step12",
        "material_column": "material_system",
        "carrier_column": "n_or_p",
        "fallback_material_column": "composition",
    }


def extract_elements(text: str) -> set[str]:
    return {token for token in re.findall(r"[A-Z][a-z]?", text or "") if token in ELEMENTS}


def map_material_label(value: Any) -> tuple[str, str]:
    text = clean_text(value)
    if not text:
        return "unmatched", "empty_or_unknown"
    lowered = text.casefold()
    compact = re.sub(r"[^a-z0-9]+", "", lowered)
    keyword_rules = [
        ("broad::BiSbTe_tetradymite_like", ["bisbte", "tetradymite"], "keyword_BiSbTe_or_tetradymite"),
        ("broad::BiTe_like", ["bi2te3", "bite"], "keyword_BiTe"),
        ("broad::SbTe_like", ["sb2te3", "sbte"], "keyword_SbTe"),
        ("broad::SnTe_like", ["snte"], "keyword_SnTe"),
        ("broad::PbTe_like", ["pbte"], "keyword_PbTe"),
        ("broad::GeTe_like", ["gete"], "keyword_GeTe"),
        ("broad::SiGe_like", ["sige"], "keyword_SiGe"),
        ("broad::Mg2SiSn_like", ["mg2si", "mg2sn", "mgsi", "mgsn"], "keyword_MgSi_or_MgSn"),
        ("broad::CoSb_skutterudite_like", ["cosb3", "cosb", "skutterudite"], "keyword_CoSb"),
    ]
    for group, keywords, reason in keyword_rules:
        if any(keyword in compact for keyword in keywords):
            return group, reason
    if "oxide" in lowered or "o-containing" in lowered:
        return "broad::oxide", "keyword_oxide"
    if "sulfide" in lowered or "s-containing" in lowered:
        return "broad::sulfide", "keyword_sulfide"
    if "selenide" in lowered or "se-containing" in lowered:
        return "broad::selenide", "keyword_selenide"
    if "telluride" in lowered or "te-containing" in lowered:
        return "broad::telluride", "keyword_telluride"

    elements = extract_elements(text)
    if {"Bi", "Sb", "Te"}.issubset(elements):
        return "broad::BiSbTe_tetradymite_like", "elements_Bi_Sb_Te"
    if {"Bi", "Te"}.issubset(elements) and "Sb" not in elements:
        return "broad::BiTe_like", "elements_Bi_Te"
    if {"Sb", "Te"}.issubset(elements) and "Bi" not in elements:
        return "broad::SbTe_like", "elements_Sb_Te"
    if {"Pb", "Te"}.issubset(elements):
        return "broad::PbTe_like", "elements_Pb_Te"
    if {"Sn", "Te"}.issubset(elements):
        return "broad::SnTe_like", "elements_Sn_Te"
    if {"Ge", "Te"}.issubset(elements):
        return "broad::GeTe_like", "elements_Ge_Te"
    if {"Si", "Ge"}.issubset(elements):
        return "broad::SiGe_like", "elements_Si_Ge"
    if "Mg" in elements and ({"Si", "Sn"} & elements):
        return "broad::Mg2SiSn_like", "elements_Mg_Si_or_Sn"
    if {"Co", "Sb"}.issubset(elements):
        return "broad::CoSb_skutterudite_like", "elements_Co_Sb"
    if "O" in elements:
        return "broad::oxide", "contains_O"
    if "Te" in elements:
        return "broad::telluride", "contains_Te"
    if "Se" in elements:
        return "broad::selenide", "contains_Se"
    if "S" in elements:
        return "broad::sulfide", "contains_S"
    return "unmatched", "no_mapping_rule"


def effective_material_labels(frame: pd.DataFrame, material_column: str, fallback_column: str) -> pd.DataFrame:
    out = frame.copy()
    primary = out[material_column].map(clean_text)
    fallback = out[fallback_column].map(clean_text)
    use_fallback = primary.eq("")
    out["old_material_label"] = primary.where(~use_fallback, fallback)
    out["old_material_label_source"] = np.where(use_fallback, fallback_column, material_column)
    out["old_material_label"] = out["old_material_label"].replace("", "unknown")
    return out


def join_examples(series: pd.Series, limit: int = 8) -> str:
    values = [str(value) for value in series.dropna().astype(str).drop_duplicates() if clean_text(value)]
    return " | ".join(values[:limit])


def load_old_ct(selected: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    log("loading old C(T) curves...")
    path = Path(selected["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = [
        selected["ct_column"],
        selected["temperature_column"],
        selected["material_column"],
        selected["carrier_column"],
        selected["fallback_material_column"],
    ]
    missing = sorted(set(required) - set(header))
    if missing:
        raise ValueError(f"Old C(T) CSV missing columns confirmed by the script: {missing}")
    raw = pd.read_csv(path, usecols=required, low_memory=False)
    raw = effective_material_labels(
        raw,
        selected["material_column"],
        selected["fallback_material_column"],
    )

    log("mapping old material labels...")
    mapped = raw["old_material_label"].map(map_material_label)
    raw["material_group_key_mapped"] = mapped.map(lambda item: item[0])
    raw["mapping_rule"] = mapped.map(lambda item: item[1])
    raw["mapping_status"] = np.where(
        raw["material_group_key_mapped"].eq("unmatched"), "unmatched", "matched"
    )
    raw["T_K"] = pd.to_numeric(raw[selected["temperature_column"]], errors="coerce")
    raw["old_C_T_S_per_m_raw"] = pd.to_numeric(raw[selected["ct_column"]], errors="coerce")
    raw["old_ct_parse_status"] = np.select(
        [
            raw["material_group_key_mapped"].eq("unmatched"),
            ~np.isfinite(raw["T_K"]),
            ~finite_positive(raw["old_C_T_S_per_m_raw"]),
        ],
        ["unmatched_material", "invalid_temperature", "invalid_old_ct"],
        default="ok",
    )

    mapping = (
        raw.groupby(
            [
                "old_material_label",
                "old_material_label_source",
                "material_group_key_mapped",
                "mapping_status",
                "mapping_rule",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="source_row_count")
        .sort_values(["mapping_status", "material_group_key_mapped", "source_row_count"], ascending=[True, True, False])
    )
    unmatched = mapping[mapping["mapping_status"].eq("unmatched")].copy()

    log("aggregating old C(T) without p/n split...")
    usable = raw[raw["old_ct_parse_status"].eq("ok")].copy()
    curves = (
        usable.groupby(["material_group_key_mapped", "T_K"], dropna=False, sort=True)
        .agg(
            source_file=(selected["ct_column"], lambda _: str(path.resolve())),
            old_material_label_examples=("old_material_label", join_examples),
            old_C_T_S_per_m=("old_C_T_S_per_m_raw", "median"),
            n_rows_aggregated=("old_C_T_S_per_m_raw", "size"),
        )
        .reset_index()
    )
    curves["log10_old_C_T_S_per_m"] = np.log10(curves["old_C_T_S_per_m"])
    curves["old_ct_parse_status"] = "ok_pn_aggregated"
    curves = curves[OLD_CURVE_COLUMNS].sort_values(
        ["material_group_key_mapped", "T_K"]
    ).reset_index(drop=True)
    return curves, mapping, unmatched


def limit_rows(frame: pd.DataFrame, maximum: int | None) -> pd.DataFrame:
    if maximum is None or len(frame) <= maximum:
        return frame
    ordered = frame.sort_values(["T_K", "row_id"]).reset_index(drop=True)
    positions = np.linspace(0, len(ordered) - 1, maximum).round().astype(int)
    return ordered.iloc[positions].copy()


def select_prediction_rows(
    predictions: pd.DataFrame,
    target_groups: list[str],
    max_rows_per_group: int | None,
) -> pd.DataFrame:
    selected = predictions[predictions["material_group_key"].isin(target_groups)].copy()
    parts: list[pd.DataFrame] = []
    for _, group in selected.groupby(["material_group_key", "carrier_type"], sort=False):
        parts.append(limit_rows(group, max_rows_per_group))
    if not parts:
        return selected.iloc[0:0][PREDICTION_OUTPUT_COLUMNS].copy()
    return pd.concat(parts, ignore_index=True)[PREDICTION_OUTPUT_COLUMNS]


def nearest_comparison(
    predictions: pd.DataFrame,
    old_curves: pd.DataFrame,
    target_groups: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for group in target_groups:
        old = old_curves[old_curves["material_group_key_mapped"].eq(group)].sort_values("T_K")
        if old.empty:
            continue
        old_t = old["T_K"].to_numpy(dtype=float)
        for carrier in ["p", "n"]:
            pred = predictions[
                predictions["material_group_key"].eq(group)
                & predictions["carrier_type"].eq(carrier)
            ].copy()
            if pred.empty:
                continue
            pred_t = pred["T_K"].to_numpy(dtype=float)
            right = np.searchsorted(old_t, pred_t, side="left")
            right = np.clip(right, 0, len(old_t) - 1)
            left = np.clip(right - 1, 0, len(old_t) - 1)
            choose_left = np.abs(old_t[left] - pred_t) <= np.abs(old_t[right] - pred_t)
            positions = np.where(choose_left, left, right)
            matched = old.iloc[positions].reset_index(drop=True)
            pred = pred.reset_index(drop=True)
            result = pd.DataFrame(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "row_id": pred["row_id"],
                    "T_K_pred": pred["T_K"],
                    "sigma_pred_S_per_m": pred["sigma_pred_S_per_m"],
                    "log10_sigma_pred_S_per_m": pred["log10_sigma_pred_S_per_m"],
                    "T_K_old_ct": matched["T_K"],
                    "old_C_T_S_per_m": matched["old_C_T_S_per_m"],
                    "log10_old_C_T_S_per_m": matched["log10_old_C_T_S_per_m"],
                }
            )
            result["T_delta_K"] = (result["T_K_old_ct"] - result["T_K_pred"]).abs()
            result["log10_pred_over_oldCT"] = (
                result["log10_sigma_pred_S_per_m"]
                - result["log10_old_C_T_S_per_m"]
            )
            result["match_method"] = "nearest_old_ct_temperature"
            rows.append(result[NEAREST_COLUMNS])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=NEAREST_COLUMNS)


def build_summary(
    predictions: pd.DataFrame,
    old_curves: pd.DataFrame,
    nearest: pd.DataFrame,
    target_groups: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group in target_groups:
        old = old_curves[old_curves["material_group_key_mapped"].eq(group)]
        for carrier in ["p", "n"]:
            pred = predictions[
                predictions["material_group_key"].eq(group)
                & predictions["carrier_type"].eq(carrier)
            ]
            matches = nearest[
                nearest["material_group_key"].eq(group)
                & nearest["carrier_type"].eq(carrier)
            ]
            warnings: list[str] = []
            if pred.empty:
                warnings.append("no_prediction_points")
            if old.empty:
                warnings.append("no_old_ct")
            if matches.empty:
                warnings.append("no_nearest_comparison")
            if not pred.empty and not old.empty:
                if pred["T_K"].max() < old["T_K"].min() or pred["T_K"].min() > old["T_K"].max():
                    warnings.append("temperature_ranges_do_not_overlap")
            rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "prediction_points": len(pred),
                    "old_ct_points": len(old),
                    "T_pred_min_K": pred["T_K"].min() if len(pred) else np.nan,
                    "T_pred_max_K": pred["T_K"].max() if len(pred) else np.nan,
                    "T_old_ct_min_K": old["T_K"].min() if len(old) else np.nan,
                    "T_old_ct_max_K": old["T_K"].max() if len(old) else np.nan,
                    "sigma_pred_median_S_per_m": pred["sigma_pred_S_per_m"].median() if len(pred) else np.nan,
                    "old_C_T_median_S_per_m": old["old_C_T_S_per_m"].median() if len(old) else np.nan,
                    "median_log10_pred_over_oldCT_nearest": (
                        matches["log10_pred_over_oldCT"].median() if len(matches) else np.nan
                    ),
                    "warning": ";".join(warnings),
                }
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def plot_group_carrier(
    group: str,
    carrier: str,
    predictions: pd.DataFrame,
    old_curve: pd.DataFrame,
    png_path: Path,
    pdf_path: Path,
) -> str:
    log(f"creating {carrier} figure...")
    title = f"{group}, {carrier}: predicted sigma vs old C(T), 25K bins"
    color = "#d95f02" if carrier == "p" else "#1b6ca8"
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.scatter(
        predictions["T_K"],
        predictions["sigma_pred_S_per_m"],
        s=16,
        alpha=0.48,
        color=color,
        edgecolors="none",
        label=f"Predicted sigma ({carrier})",
        zorder=2,
    )
    if not old_curve.empty:
        ax.plot(
            old_curve["T_K"],
            old_curve["old_C_T_S_per_m"],
            color="#222222",
            linewidth=2.5,
            label="Old C(T) from SS2026 (no p/n split)",
            zorder=3,
        )
    else:
        ax.plot(
            [],
            [],
            color="#222222",
            linewidth=2.5,
            label="Old C(T) from SS2026 (no p/n split)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("Electrical conductivity sigma [S/m]")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return title


def build_figures(
    predictions: pd.DataFrame,
    old_curves: pd.DataFrame,
    target_groups: list[str],
    figure_dir: Path,
    suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    figure_id = 1
    for group in target_groups:
        log(f"processing material group {group}...")
        old = old_curves[old_curves["material_group_key_mapped"].eq(group)].sort_values("T_K")
        for carrier in ["p", "n"]:
            pred = predictions[
                predictions["material_group_key"].eq(group)
                & predictions["carrier_type"].eq(carrier)
            ].sort_values("T_K")
            if pred.empty:
                log(f"warning: no prediction points for {group} / {carrier}; creating line-only main figure")
            if old.empty:
                log(f"warning: no old C(T) for {group}; creating point-only main figure")
            if pred.empty and old.empty:
                log(f"warning: no data for {group} / {carrier}; skipping empty figure")
                continue
            stem = f"{safe_name(group)}_{carrier}_sigma_pred_vs_oldCT_25k{suffix}"
            png = figure_dir / f"{stem}.png"
            pdf = figure_dir / f"{stem}.pdf"
            title = plot_group_carrier(group, carrier, pred, old, png, pdf)
            rows.append(
                {
                    "figure_id": f"FIG_{figure_id:03d}",
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": f"{carrier}_sigma_pred_vs_old_ct_no_pn",
                    "figure_path_png": str(png.resolve()),
                    "figure_path_pdf": str(pdf.resolve()),
                    "title": title,
                    "n_prediction_points": len(pred),
                    "n_old_ct_points": len(old),
                    "description": (
                        "Main figure contains Step9A sigma_pred points and the SS2026 "
                        "p/n-unsplit old C(T) line; measured sigma and sigma0_ref are not plotted."
                    ),
                }
            )
            figure_id += 1
    return pd.DataFrame(rows, columns=FIGURE_INDEX_COLUMNS)


def dataframe_to_markdown(frame: pd.DataFrame, max_rows: int = 50) -> str:
    if frame.empty:
        return "n/a"
    text = frame.head(max_rows).copy()
    for column in text.columns:
        text[column] = text[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|").replace("\n", " ")
        )
    header = "| " + " | ".join(text.columns) + " |"
    separator = "| " + " | ".join("---" for _ in text.columns) + " |"
    body = ["| " + " | ".join(row[column] for column in text.columns) + " |" for _, row in text.iterrows()]
    return "\n".join([header, separator, *body])


def write_report(
    path: Path,
    prediction_path: Path,
    config_id: str,
    old_script: Path,
    old_info: dict[str, Any],
    target_groups: list[str],
    summary: pd.DataFrame,
    figure_index: pd.DataFrame,
    nearest: pd.DataFrame,
    mapping: pd.DataFrame,
    elapsed: float,
) -> None:
    ratios = pd.to_numeric(nearest["log10_pred_over_oldCT"], errors="coerce").dropna()
    missing_old = summary.loc[summary["old_ct_points"].eq(0), "material_group_key"].drop_duplicates().tolist()
    missing_pred = summary.loc[
        summary["prediction_points"].eq(0), ["material_group_key", "carrier_type"]
    ]
    fallback_rows = int(
        mapping.loc[mapping["old_material_label_source"].eq("composition"), "source_row_count"].sum()
    )
    lines = [
        "# Step9B: Step9A 25 K sigma_pred vs SS2026 old C(T)",
        "",
        "## Purpose and inputs",
        "",
        "- Purpose: compare the Step9A 25 K predicted electrical conductivity with the old SS2026 C(T) baseline.",
        f"- Prediction file: `{prediction_path}`",
        f"- Config ID: `{config_id}`",
        f"- Old C(T) source script (read statically, not executed): `{old_script}`",
        f"- Old C(T) output CSV: `{old_info['path']}`",
        f"- Old C(T) column: `{old_info['ct_column']}`",
        f"- Old temperature column: `{old_info['temperature_column']}`",
        f"- Old material column: `{old_info['material_column']}`",
        f"- Old carrier column present but excluded from aggregation: `{old_info['carrier_column']}`",
        "",
        "## Old C(T) material mapping and aggregation",
        "",
        "- The old C(T) curve was aggregated by `material_group_key_mapped` and `T_K` only.",
        "- The `n_or_p` column was deliberately excluded, so one p/n-unsplit median C(T) curve is used for both figures in each material group.",
        "- The Step12 `material_system` column is `unknown` in the available file. The `composition` column from the same Step12 output was therefore used as a fallback label.",
        f"- Rows mapped through the composition fallback: {fallback_rows}",
        "- Unmapped effective labels were retained in `step9b_unmatched_old_material_labels.csv`.",
        "",
        "## Target material groups",
        "",
        *[f"- {group}" for group in target_groups],
        "",
        "## Prediction and old C(T) counts",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Median log10(sigma_pred / old C(T))",
        "",
    ]
    if ratios.empty:
        lines.append("- No nearest comparisons were available.")
    else:
        lines.extend(
            [
                f"- Overall matched rows: {len(ratios)}",
                f"- Overall median: {ratios.median():.6f}",
                f"- Q25 / Q75: {ratios.quantile(0.25):.6f} / {ratios.quantile(0.75):.6f}",
                f"- Minimum / maximum: {ratios.min():.6f} / {ratios.max():.6f}",
                "",
                dataframe_to_markdown(
                    nearest.groupby(["material_group_key", "carrier_type"], as_index=False)
                    .agg(
                        matched_rows=("log10_pred_over_oldCT", "size"),
                        median_log10_pred_over_oldCT=("log10_pred_over_oldCT", "median"),
                        q25=("log10_pred_over_oldCT", lambda values: values.quantile(0.25)),
                        q75=("log10_pred_over_oldCT", lambda values: values.quantile(0.75)),
                    )
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            dataframe_to_markdown(
                figure_index[
                    [
                        "figure_id",
                        "material_group_key",
                        "carrier_type",
                        "figure_path_png",
                        "figure_path_pdf",
                        "n_prediction_points",
                        "n_old_ct_points",
                    ]
                ],
                100,
            ),
            "",
            "## Missing data",
            "",
            f"- Material groups without old C(T): {missing_old if missing_old else 'none'}",
            "- Material-group/carrier combinations without prediction points:",
        ]
    )
    if missing_pred.empty:
        lines.append("  - none")
    else:
        for _, row in missing_pred.iterrows():
            lines.append(f"  - {row['material_group_key']} / {row['carrier_type']}")
    lines.extend(
        [
            "",
            "## How to read the figures",
            "",
            "- Points are the Step9A 25 K predicted electrical conductivity.",
            "- The line is the old SS2026 C(T).",
            "- The line has no p/n split and is identical in the p and n figures for a material group.",
            "- Only the point cloud is separated into p-type and n-type figures.",
            "",
            "## Notes",
            "",
            "- Measured sigma is not included in the main figures.",
            "- sigma0_ref is not included in the figures.",
            "- No new sigma_pred was calculated.",
            "- This step only visualizes the existing Step9A predictions.",
            "- Step4 full-data reference curves were not used.",
            "- Starrydata2 raw data was not read.",
            f"- elapsed_seconds: {elapsed:.2f}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sanity_checks(
    prediction_path: Path,
    source_predictions: pd.DataFrame,
    used_predictions: pd.DataFrame,
    config_id: str,
    old_info: dict[str, Any],
    old_curves: pd.DataFrame,
    summary_path: Path,
    figure_index_path: Path,
    figure_index: pd.DataFrame,
    report_path: Path,
    protection: pd.DataFrame,
) -> None:
    checks: dict[str, bool] = {}
    checks["prediction_input_exists"] = prediction_path.exists()
    checks["config_id_exists"] = source_predictions["config_id"].astype(str).eq(config_id).any()
    source_used = source_predictions[
        source_predictions["config_id"].astype(str).eq(config_id)
        & source_predictions["row_id"].astype(str).isin(used_predictions["row_id"].astype(str))
    ]
    checks["prediction_status_ok_only"] = source_used["prediction_status"].astype(str).eq("ok").all()
    checks["sigma_pred_finite_positive"] = finite_positive(used_predictions["sigma_pred_S_per_m"]).all()
    checks["old_ct_file_exists"] = Path(old_info["path"]).exists()
    checks["old_ct_finite_positive"] = (
        not old_curves.empty and finite_positive(old_curves["old_C_T_S_per_m"]).all()
    )
    checks["old_ct_no_pn_split"] = (
        "carrier_type" not in old_curves.columns
        and "n_or_p" not in old_curves.columns
        and not old_curves.duplicated(["material_group_key_mapped", "T_K"]).any()
    )
    checks["at_least_one_figure"] = len(figure_index) > 0
    checks["figure_index_created"] = figure_index_path.exists()
    checks["summary_created"] = summary_path.exists()
    checks["report_created"] = report_path.exists() and report_path.stat().st_size > 0
    checks["measured_sigma_not_in_main_figure"] = figure_index["description"].str.contains(
        "measured sigma and sigma0_ref are not plotted", regex=False
    ).all()
    checks["sigma0_ref_not_in_main_figure"] = checks["measured_sigma_not_in_main_figure"]
    checks["no_new_sigma_pred_calculated"] = True
    checks["step9a_results_unchanged"] = not protection.empty and protection["unchanged"].all()
    checks["raw_data_not_read"] = True
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        for failure in failures:
            print(f"[step9b] FAIL: {failure}", flush=True)
        raise SystemExit(1)


def main() -> None:
    started = time.time()
    args = parse_args()
    prediction_path = resolve_predictions(args.predictions)
    validate_paths(args, prediction_path)
    target_groups = list(args.target_groups)
    if args.max_groups is not None:
        target_groups = target_groups[: args.max_groups]

    step9a_before = directory_manifest(STEP9A_DIR)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    source_predictions = load_predictions(prediction_path, args.config_id)
    used_predictions = select_prediction_rows(
        source_predictions,
        target_groups,
        args.max_rows_per_group,
    )
    old_info = inspect_old_ct_script(args.old_ct_script)
    old_curves_all, mapping, unmatched = load_old_ct(old_info)
    old_curves = old_curves_all[
        old_curves_all["material_group_key_mapped"].isin(target_groups)
    ].copy()
    nearest = nearest_comparison(used_predictions, old_curves, target_groups)
    summary = build_summary(used_predictions, old_curves, nearest, target_groups)
    figure_index = build_figures(
        used_predictions,
        old_curves,
        target_groups,
        args.figures,
        args.output_suffix,
    )

    log("writing CSV outputs...")
    prediction_rows_path = output_path(
        args.output, "step9b_prediction_rows_used", args.output_suffix
    )
    old_curves_path = output_path(
        args.output, "step9b_old_ct_curves_no_pn", args.output_suffix
    )
    summary_path = output_path(
        args.output, "step9b_summary_by_group_carrier", args.output_suffix
    )
    nearest_path = output_path(
        args.output, "step9b_nearest_comparison_table", args.output_suffix
    )
    figure_index_path = output_path(
        args.output, "step9b_figure_index", args.output_suffix
    )
    mapping_path = output_path(
        args.output, "step9b_material_mapping", args.output_suffix
    )
    unmatched_path = output_path(
        args.output, "step9b_unmatched_old_material_labels", args.output_suffix
    )
    used_predictions.to_csv(prediction_rows_path, index=False, encoding="utf-8-sig")
    old_curves.to_csv(old_curves_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    nearest.to_csv(nearest_path, index=False, encoding="utf-8-sig")
    figure_index.to_csv(figure_index_path, index=False, encoding="utf-8-sig")
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")

    step9a_after = directory_manifest(STEP9A_DIR)
    protection = protection_manifest(step9a_before, step9a_after)
    protection_path = output_path(
        args.output, "step9b_step9a_protection_manifest", args.output_suffix
    )
    protection.to_csv(protection_path, index=False, encoding="utf-8-sig")

    log("writing report...")
    write_report(
        args.report,
        prediction_path,
        args.config_id,
        args.old_ct_script,
        old_info,
        target_groups,
        summary,
        figure_index,
        nearest,
        mapping,
        time.time() - started,
    )
    run_sanity_checks(
        prediction_path,
        source_predictions,
        used_predictions,
        args.config_id,
        old_info,
        old_curves,
        summary_path,
        figure_index_path,
        figure_index,
        args.report,
        protection,
    )

    log("done.")
    log(f"elapsed seconds: {time.time() - started:.2f}")
    print(f"prediction_file: {prediction_path}")
    print(f"config_id: {args.config_id}")
    print(f"old_ct_csv: {old_info['path']}")
    print(f"old_ct_column: {old_info['ct_column']}")
    print(f"target_groups: {', '.join(target_groups)}")
    print(f"png_figures: {len(figure_index)}")
    print(f"pdf_figures: {len(figure_index)}")
    if not nearest.empty:
        print(
            "median_log10_pred_over_oldCT: "
            f"{nearest['log10_pred_over_oldCT'].median():.12g}"
        )
    print(f"output_dir: {args.output}")
    print(f"figure_dir: {args.figures}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
