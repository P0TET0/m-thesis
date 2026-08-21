"""Build Step9B overlays of Step9A 25 K sigma0_ref(T) and Step31 SS2026 C(T)."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
STEP9A_DIR = EXP_DIR / "data" / "processed" / "step9a_25k_bin_broad_family"
DEFAULT_SIGMA_INPUTS = [
    STEP9A_DIR / "step5b_train_reference_curve_bins.parquet",
    STEP9A_DIR / "step5b_train_reference_curve_bins.csv",
]
DEFAULT_STEP31_SCRIPT = (
    PROJECT_ROOT / "experiments" / "exp005" / "plot_reclassified_material_system_ct_curves_step31.py"
)
DEFAULT_STEP31_OUTPUT = PROJECT_ROOT / "data" / "output" / "starrydata2_step31_reclassified_ct_curves"
DEFAULT_STEP31_REPORT = DEFAULT_STEP31_OUTPUT / "step31_ct_curve_report.txt"
DEFAULT_OUTPUT = EXP_DIR / "data" / "processed" / "step9b_25k_vs_step31_ct_overlay"
DEFAULT_FIGURES = EXP_DIR / "figures" / "step9b_25k_vs_step31_ct_overlay"
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "step9b_25k_vs_step31_ct_overlay"
    / "step9b_25k_vs_step31_ct_overlay_report.md"
)
DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid"
    "__material_family__sample_median"
)

CT_CANDIDATES = [
    "C_T_sigma_median_S_per_m_step31",
    "C_T_S_per_m",
    "C_T",
    "ct_S_per_m",
    "ct_value",
    "sigma_ref_S_per_m",
    "sigma_reference_S_per_m",
    "prefactor_C_S_per_m",
    "median_sigma_S_per_m",
    "ref_sigma_S_per_m",
    "log10_C_T_S_per_m",
    "log10_C_T",
]
TEMPERATURE_CANDIDATES = [
    "temp_bin_center_K",
    "T_K",
    "temperature_K",
    "temperature_bin_K",
    "T_bin_center_K",
    "temperature_center_K",
    "temp_K",
]
MATERIAL_CANDIDATES = [
    "material_system_reclass_step31",
    "material_group_key",
    "reclassified_material_system",
    "material_system",
    "material_family",
    "material_family_reclassified",
    "group_key",
]
CARRIER_CANDIDATES = [
    "n_or_p_reclass_step31",
    "carrier_type",
    "n_or_p",
    "pn_type",
    "conduction_type",
]

CANONICAL_GROUPS = [
    "broad::BiSbTe_tetradymite_like",
    "broad::PbTe_like",
    "broad::SnTe_like",
    "broad::BiTe_like",
    "broad::SbTe_like",
    "broad::SiGe_like",
    "broad::Mg2SiSn_like",
    "broad::CoSb_skutterudite_like",
    "broad::GeTe_like",
    "broad::oxide",
    "broad::sulfide",
    "broad::selenide",
    "broad::telluride",
    "broad::other_formula_system",
    "unknown_material_group",
]

HOST_TO_BROAD = {
    "BiSbTe": "broad::BiSbTe_tetradymite_like",
    "PbTe": "broad::PbTe_like",
    "SnTe": "broad::SnTe_like",
    "BiTe": "broad::BiTe_like",
    "SbTe": "broad::SbTe_like",
    "SiGe": "broad::SiGe_like",
    "Mg2SiSn": "broad::Mg2SiSn_like",
    "CoSb_skutterudite": "broad::CoSb_skutterudite_like",
    "GeTe": "broad::GeTe_like",
    "oxide": "broad::oxide",
    "sulfide": "broad::sulfide",
    "selenide": "broad::selenide",
    "telluride": "broad::telluride",
    "other_formula_system": "broad::other_formula_system",
}

SIGMA_OUTPUT_COLUMNS = [
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
NORMALIZED_CT_COLUMNS = [
    "source_file",
    "material_group_key",
    "carrier_type",
    "T_K",
    "step31_C_T_S_per_m",
    "log10_step31_C_T_S_per_m",
    "parse_status",
]
COMPARISON_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "T_K_25k",
    "sigma0_ref_S_per_m",
    "log10_sigma0_ref_S_per_m",
    "T_K_step31",
    "step31_C_T_S_per_m",
    "log10_step31_C_T_S_per_m",
    "T_delta_K",
    "log10_sigma0ref25k_over_step31ct",
    "match_method",
]
SUMMARY_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "sigma0ref25k_points",
    "step31_ct_points",
    "comparison_points",
    "T_min_K",
    "T_max_K",
    "sigma0ref25k_median_S_per_m",
    "step31_ct_median_S_per_m",
    "median_log10_sigma0ref25k_over_step31ct",
    "warning",
]
FIGURE_INDEX_COLUMNS = [
    "figure_id",
    "material_group_key",
    "carrier_type",
    "figure_type",
    "figure_path_png",
    "figure_path_pdf",
    "title",
    "n_sigma0ref25k_points",
    "n_step31_ct_points",
    "description",
]
PARSE_SUMMARY_COLUMNS = [
    "step31_script_path",
    "step31_report_path",
    "detected_output_file",
    "detected_ct_column",
    "detected_temperature_column",
    "detected_material_column",
    "detected_carrier_column",
    "detection_status",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma0ref25k", type=Path, default=None)
    parser.add_argument("--step31-script", type=Path, default=DEFAULT_STEP31_SCRIPT)
    parser.add_argument("--step31-report", type=Path, default=DEFAULT_STEP31_REPORT)
    parser.add_argument("--step31-output-dir", type=Path, default=DEFAULT_STEP31_OUTPUT)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--target-groups", nargs="+", default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def log(message: str, started: float) -> None:
    print(f"[step9b] {message} (elapsed {time.perf_counter() - started:.1f}s)", flush=True)


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def safe_group(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def with_suffix(directory: Path, stem: str, suffix: str) -> Path:
    return directory / f"{stem}{suffix}.csv"


def first_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def resolve_sigma_input(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit.resolve()
        if explicit.suffix.casefold() == ".parquet":
            fallback = explicit.with_suffix(".csv")
            if fallback.exists():
                return fallback.resolve()
        raise FileNotFoundError(explicit)
    for candidate in DEFAULT_SIGMA_INPUTS:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Step9A 25 K sigma0_ref curve input was not found")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.casefold() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported input table: {path}")


def validate_args(args: argparse.Namespace, sigma_path: Path) -> None:
    for label, path in [
        ("Step31 script", args.step31_script),
        ("Step31 report", args.step31_report),
        ("Step31 output directory", args.step31_output_dir),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if "25k" not in sigma_path.parent.name.casefold():
        raise ValueError(f"sigma0_ref input is not in a Step9A 25 K directory: {sigma_path}")
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    if re.search(r"[\\/]", args.output_suffix):
        raise ValueError("--output-suffix must not contain a path separator")
    protected = [STEP9A_DIR.resolve(), args.step31_output_dir.resolve()]
    for label, destination in [
        ("output", args.output.resolve()),
        ("figures", args.figures.resolve()),
        ("report", args.report.resolve()),
    ]:
        if any(destination == source or source in destination.parents for source in protected):
            raise ValueError(f"{label} must not be inside a protected input directory: {destination}")


def load_sigma_curves(path: Path, config_id: str) -> pd.DataFrame:
    frame = read_table(path)
    required = set(SIGMA_OUTPUT_COLUMNS + [
        "config_id",
        "split_scheme",
        "reference_source_subset",
        "group_scheme",
        "curve_method",
    ])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Step9A sigma0_ref input is missing columns: {sorted(missing)}")
    selected = frame.loc[
        frame["config_id"].astype(str).eq(config_id)
        & frame["split_scheme"].astype(str).eq("sample_holdout")
        & frame["reference_source_subset"].astype(str).eq("conservative_valid")
        & frame["group_scheme"].astype(str).eq("material_family")
        & frame["curve_method"].astype(str).eq("sample_median"),
        SIGMA_OUTPUT_COLUMNS,
    ].copy()
    if selected.empty:
        raise ValueError(f"No Step9A rows match config_id={config_id}")
    selected["carrier_type"] = selected["carrier_type"].astype(str).str.strip().str.lower()
    selected["T_bin_center_K"] = pd.to_numeric(selected["T_bin_center_K"], errors="coerce")
    selected["sigma0_ref_S_per_m"] = pd.to_numeric(selected["sigma0_ref_S_per_m"], errors="coerce")
    valid = (
        np.isfinite(selected["T_bin_center_K"])
        & np.isfinite(selected["sigma0_ref_S_per_m"])
        & selected["sigma0_ref_S_per_m"].gt(0)
        & selected["carrier_type"].isin(["p", "n"])
    )
    if not valid.all():
        raise ValueError(f"Step9A selected rows contain {(~valid).sum()} invalid rows")
    bin_remainders = np.mod(selected["T_bin_center_K"].to_numpy(dtype=float), 25.0)
    if not np.all(np.isclose(bin_remainders, 0.0, atol=1e-8)):
        raise ValueError("Step9A T_bin_center_K is not on a 25 K grid")
    expected_log = np.log10(selected["sigma0_ref_S_per_m"].to_numpy(dtype=float))
    supplied_log = pd.to_numeric(selected["log10_sigma0_ref_S_per_m"], errors="coerce").to_numpy()
    if not np.allclose(expected_log, supplied_log, rtol=1e-10, atol=1e-10):
        raise ValueError("Step9A sigma0_ref and log10_sigma0_ref columns are inconsistent")
    return selected.sort_values(["material_group_key", "carrier_type", "T_bin_center_K"]).reset_index(drop=True)


def detect_step31_output(
    output_dir: Path, script_path: Path, report_path: Path
) -> tuple[Path, str, str, str, str, str]:
    context = script_path.read_text(encoding="utf-8", errors="replace") + "\n" + report_path.read_text(
        encoding="utf-8", errors="replace"
    )
    candidates: list[tuple[int, Path, str, str, str, str, str]] = []
    rejected: list[str] = []
    for path in sorted(output_dir.glob("*.csv")):
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except Exception as exc:  # pragma: no cover - defensive audit path
            rejected.append(f"{path.name}: unreadable ({exc})")
            continue
        ct_col = first_column(columns, CT_CANDIDATES)
        t_col = first_column(columns, TEMPERATURE_CANDIDATES)
        material_col = first_column(columns, MATERIAL_CANDIDATES)
        carrier_col = first_column(columns, CARRIER_CANDIDATES)
        if not all([ct_col, t_col, material_col, carrier_col]):
            rejected.append(f"{path.name}: required C(T)/T/material/carrier columns not all present")
            continue
        score = 100
        score += 30 if "np_split" in path.name.casefold() else 0
        score += 10 if path.name in context else 0
        score += 5 if "step31" in path.name.casefold() else 0
        note = f"score={score}; selected from {len(list(output_dir.glob('*.csv')))} CSV files"
        candidates.append((score, path, ct_col, t_col, material_col, carrier_col, note))
    if not candidates:
        raise ValueError("No Step31 CSV contains detectable C(T), temperature, material, and carrier columns")
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    _, path, ct_col, t_col, material_col, carrier_col, note = candidates[0]
    if rejected:
        note += f"; rejected={len(rejected)}"
    return path.resolve(), ct_col, t_col, material_col, carrier_col, note


def normalize_material_label(value: Any) -> tuple[str, str, str]:
    source = clean_text(value)
    if not source:
        return "", "unmatched", "blank_label"
    if source == "unknown_material_group":
        return source, "matched", "exact_unknown_material_group"
    if source in CANONICAL_GROUPS:
        return source, "matched", "exact_broad_label"
    bare = source
    if source.startswith("host::"):
        bare = source.split("::", 1)[1]
    elif source.startswith("broad::"):
        bare = source.split("::", 1)[1]
    if bare in HOST_TO_BROAD:
        rule = "host_to_broad" if source.startswith("host::") else "bare_to_broad"
        return HOST_TO_BROAD[bare], "matched", rule
    if bare.endswith("_like") and f"broad::{bare}" in CANONICAL_GROUPS:
        return f"broad::{bare}", "matched", "add_broad_prefix"
    return source, "unmatched", "no_known_mapping_rule"


def load_step31_curves(
    path: Path, ct_col: str, t_col: str, material_col: str, carrier_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path, low_memory=False)
    records: list[dict[str, Any]] = []
    mappings: dict[str, tuple[str, str, str]] = {}
    is_log_ct = ct_col.casefold().startswith("log10")
    for row in raw[[ct_col, t_col, material_col, carrier_col]].itertuples(index=False, name=None):
        ct_raw, t_raw, material_raw, carrier_raw = row
        source_label = clean_text(material_raw)
        mapped, status, rule = normalize_material_label(source_label)
        mappings[source_label] = (mapped, status, rule)
        carrier = clean_text(carrier_raw).lower()
        t_value = pd.to_numeric(pd.Series([t_raw]), errors="coerce").iloc[0]
        ct_value_raw = pd.to_numeric(pd.Series([ct_raw]), errors="coerce").iloc[0]
        ct_value = 10.0 ** float(ct_value_raw) if is_log_ct and pd.notna(ct_value_raw) else ct_value_raw
        valid = (
            status == "matched"
            and carrier in {"p", "n"}
            and pd.notna(t_value)
            and np.isfinite(float(t_value))
            and pd.notna(ct_value)
            and np.isfinite(float(ct_value))
            and float(ct_value) > 0
        )
        parse_status = "ok" if valid else (
            "unmatched_material" if status != "matched" else "invalid_carrier_or_numeric_value"
        )
        records.append(
            {
                "source_file": str(path),
                "material_group_key": mapped,
                "carrier_type": carrier,
                "T_K": float(t_value) if pd.notna(t_value) else np.nan,
                "step31_C_T_S_per_m": float(ct_value) if pd.notna(ct_value) else np.nan,
                "log10_step31_C_T_S_per_m": np.log10(float(ct_value)) if valid else np.nan,
                "parse_status": parse_status,
            }
        )
    normalized = pd.DataFrame(records, columns=NORMALIZED_CT_COLUMNS)
    valid = normalized[normalized["parse_status"].eq("ok")].copy()
    if valid.empty:
        raise ValueError("The detected Step31 C(T) file has no valid n/p-split positive curves")
    if valid.duplicated(["material_group_key", "carrier_type", "T_K"]).any():
        valid = (
            valid.groupby(["source_file", "material_group_key", "carrier_type", "T_K"], as_index=False)
            .agg(step31_C_T_S_per_m=("step31_C_T_S_per_m", "median"))
        )
        valid["log10_step31_C_T_S_per_m"] = np.log10(valid["step31_C_T_S_per_m"])
        valid["parse_status"] = "ok_duplicate_temperature_median"
        normalized = pd.concat(
            [valid[NORMALIZED_CT_COLUMNS], normalized[~normalized["parse_status"].eq("ok")]],
            ignore_index=True,
        )
    mapping_rows = [
        {
            "source_label": source,
            "normalized_material_group_key": mapped,
            "mapping_status": status,
            "mapping_rule": rule,
        }
        for source, (mapped, status, rule) in sorted(mappings.items())
    ]
    mapping = pd.DataFrame(
        mapping_rows,
        columns=["source_label", "normalized_material_group_key", "mapping_status", "mapping_rule"],
    )
    unmatched = mapping[mapping["mapping_status"].eq("unmatched")].copy()
    return (
        normalized.sort_values(["material_group_key", "carrier_type", "T_K"]).reset_index(drop=True),
        mapping,
        unmatched,
    )


def target_groups_from_args(
    raw_targets: list[str] | None, sigma: pd.DataFrame, step31: pd.DataFrame, max_groups: int | None
) -> list[str]:
    sigma_groups = set(sigma["material_group_key"].astype(str))
    step31_groups = set(step31.loc[step31["parse_status"].str.startswith("ok"), "material_group_key"].astype(str))
    common = sigma_groups & step31_groups
    if raw_targets:
        requested = []
        for token in raw_targets:
            requested.extend(piece.strip() for piece in token.split(",") if piece.strip())
        if len(requested) != len(set(requested)):
            raise ValueError("--target-groups contains duplicates")
        missing = [group for group in requested if group not in common]
        if missing:
            raise ValueError(f"Target groups are not common to both inputs: {missing}")
        groups = requested
    else:
        groups = [group for group in CANONICAL_GROUPS if group in common]
        groups.extend(sorted(common - set(groups)))
    if max_groups is not None:
        groups = groups[:max_groups]
    if not groups:
        raise ValueError("No common material groups are available")
    return groups


def nearest_comparison(sigma: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    if sigma.empty or ct.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    ct_sorted = ct.sort_values("T_K").reset_index(drop=True)
    ct_t = ct_sorted["T_K"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for item in sigma.sort_values("T_bin_center_K").itertuples(index=False):
        t_25k = float(item.T_bin_center_K)
        index = int(np.argmin(np.abs(ct_t - t_25k)))
        matched = ct_sorted.iloc[index]
        rows.append(
            {
                "material_group_key": item.material_group_key,
                "carrier_type": item.carrier_type,
                "T_K_25k": t_25k,
                "sigma0_ref_S_per_m": float(item.sigma0_ref_S_per_m),
                "log10_sigma0_ref_S_per_m": float(item.log10_sigma0_ref_S_per_m),
                "T_K_step31": float(matched["T_K"]),
                "step31_C_T_S_per_m": float(matched["step31_C_T_S_per_m"]),
                "log10_step31_C_T_S_per_m": float(matched["log10_step31_C_T_S_per_m"]),
                "T_delta_K": abs(t_25k - float(matched["T_K"])),
                "log10_sigma0ref25k_over_step31ct": (
                    float(item.log10_sigma0_ref_S_per_m)
                    - float(matched["log10_step31_C_T_S_per_m"])
                ),
                "match_method": "nearest_temperature",
            }
        )
    return pd.DataFrame(rows, columns=COMPARISON_COLUMNS)


def planned_csv_paths(args: argparse.Namespace) -> dict[str, Path]:
    suffix = args.output_suffix
    return {
        "parse": with_suffix(args.output, "step9b_step31_ct_parse_summary", suffix),
        "sigma": with_suffix(args.output, "step9b_25k_sigma0ref_curves", suffix),
        "ct": with_suffix(args.output, "step9b_step31_ct_curves_normalized", suffix),
        "comparison": with_suffix(args.output, "step9b_curve_comparison_table", suffix),
        "summary": with_suffix(args.output, "step9b_summary_by_group_carrier", suffix),
        "figures": with_suffix(args.output, "step9b_figure_index", suffix),
        "mapping": with_suffix(args.output, "step9b_material_mapping", suffix),
        "unmatched": with_suffix(args.output, "step9b_unmatched_material_labels", suffix),
    }


def ensure_outputs_absent(paths: Iterable[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing Step9B outputs:\n" + "\n".join(existing))


def create_figures(
    group: str,
    carrier: str,
    sigma: pd.DataFrame,
    ct: pd.DataFrame,
    comparison: pd.DataFrame,
    figures_dir: Path,
    suffix: str,
    figure_counter: int,
) -> list[dict[str, Any]]:
    safe = safe_group(group)
    title = f"{group}, {carrier}: 25 K sigma0_ref(T) vs SS2026 C(T)"
    overlay_stem = f"{safe}_{carrier}_25k_sigma0ref_vs_step31_ct_overlay{suffix}"
    overlay_png = figures_dir / f"{overlay_stem}.png"
    overlay_pdf = figures_dir / f"{overlay_stem}.pdf"
    ratio_stem = f"{safe}_{carrier}_log_ratio_sigma0ref25k_over_step31ct{suffix}"
    ratio_png = figures_dir / f"{ratio_stem}.png"
    # Keep the requested duplicated carrier token in the PDF filename.
    ratio_pdf = figures_dir / f"{safe}_{carrier}_{carrier}_log_ratio_sigma0ref25k_over_step31ct{suffix}.pdf"
    ensure_outputs_absent([overlay_png, overlay_pdf, ratio_png, ratio_pdf])

    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.plot(
        sigma["T_bin_center_K"],
        sigma["sigma0_ref_S_per_m"],
        color="#1f77b4",
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="25 K sigma0_ref(T) from exp006 Step9A",
    )
    ax.plot(
        ct["T_K"],
        ct["step31_C_T_S_per_m"],
        color="#d62728",
        marker="s",
        markersize=3.2,
        linewidth=1.5,
        label="SS2026 C(T) from exp005 Step31",
    )
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("S/m")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(overlay_png, dpi=220)
    fig.savefig(overlay_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(
        comparison["T_K_25k"],
        comparison["log10_sigma0ref25k_over_step31ct"],
        color="#6a3d9a",
        marker="o",
        markersize=3.5,
        linewidth=1.4,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Temperature T [K]")
    ax.set_ylabel("log10(sigma0_ref_25k / C_T_step31)")
    ax.set_title(f"{group}, {carrier}: log ratio (nearest-temperature match)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ratio_png, dpi=220)
    fig.savefig(ratio_pdf)
    plt.close(fig)

    common = {
        "material_group_key": group,
        "carrier_type": carrier,
        "title": title,
        "n_sigma0ref25k_points": len(sigma),
        "n_step31_ct_points": len(ct),
    }
    return [
        {
            "figure_id": f"step9b_{figure_counter:03d}",
            **common,
            "figure_type": "overlay",
            "figure_path_png": str(overlay_png.resolve()),
            "figure_path_pdf": str(overlay_pdf.resolve()),
            "description": "Step9A 25 K sigma0_ref(T) and Step31 SS2026 C(T) overlay; no new sigma_pred is calculated.",
        },
        {
            "figure_id": f"step9b_{figure_counter + 1:03d}",
            **common,
            "figure_type": "log_ratio_nearest",
            "figure_path_png": str(ratio_png.resolve()),
            "figure_path_pdf": str(ratio_pdf.resolve()),
            "description": "Nearest-temperature log10 ratio with T_delta_K recorded in the comparison CSV.",
        },
    ]


def finite_median(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.median()) if len(numeric) else np.nan


def write_report(
    path: Path,
    sigma_path: Path,
    step31_path: Path,
    ct_col: str,
    t_col: str,
    material_col: str,
    carrier_col: str,
    groups: list[str],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    figures: pd.DataFrame,
    unmatched: pd.DataFrame,
    csv_paths: dict[str, Path],
    source_unchanged: bool,
) -> None:
    created_combinations = int(summary["comparison_points"].gt(0).sum())
    missing = summary.loc[summary["warning"].ne(""), ["material_group_key", "carrier_type", "warning"]]
    overall_ratio = finite_median(comparison["log10_sigma0ref25k_over_step31ct"])
    lines = [
        "# Step9B: 25 K sigma0_ref(T) vs Step31 SS2026 C(T) overlay",
        "",
        "## Inputs and detected Step31 schema",
        "",
        f"- exp006 input: `{sigma_path}`",
        f"- exp005 Step31 C(T) CSV: `{step31_path}`",
        f"- adopted C(T) column: `{ct_col}`",
        f"- adopted temperature column: `{t_col}`",
        f"- adopted material column: `{material_col}`",
        f"- adopted carrier column: `{carrier_col}`",
        "",
        "## Scope and results",
        "",
        f"- target material groups ({len(groups)}): {', '.join(groups)}",
        "- target carrier_type: p, n",
        f"- material-group x carrier combinations with both curves: {created_combinations}",
        f"- overlay figures: {int(figures['figure_type'].eq('overlay').sum())}",
        f"- log-ratio figures: {int(figures['figure_type'].eq('log_ratio_nearest').sum())}",
        f"- combinations skipped because one side was missing: {len(missing)}",
        f"- overall median log10(sigma0_ref_25k / C_T_step31): {overall_ratio:.6g}" if np.isfinite(overall_ratio) else "- overall median log10 ratio: unavailable",
        "",
        "### Median ratio by material group",
        "",
        "| material_group_key | median log10 ratio |",
        "|---|---:|",
    ]
    for group in groups:
        subset = comparison[comparison["material_group_key"].eq(group)]
        value = finite_median(subset["log10_sigma0ref25k_over_step31ct"])
        lines.append(f"| {group} | {value:.6g} |" if np.isfinite(value) else f"| {group} | unavailable |")
    lines.extend(["", "### Missing-side warnings", ""])
    if missing.empty:
        lines.append("None.")
    else:
        for row in missing.itertuples(index=False):
            lines.append(f"- {row.material_group_key}, {row.carrier_type}: {row.warning}")
    lines.extend(["", "## Figure index", ""])
    if figures.empty:
        lines.append("No figures were created.")
    else:
        lines.extend(["| figure_id | group | carrier | type | PNG | PDF |", "|---|---|---|---|---|---|"])
        for row in figures.itertuples(index=False):
            lines.append(
                f"| {row.figure_id} | {row.material_group_key} | {row.carrier_type} | {row.figure_type} | "
                f"`{row.figure_path_png}` | `{row.figure_path_pdf}` |"
            )
    lines.extend(["", "## Unmatched material labels", ""])
    if unmatched.empty:
        lines.append("None.")
    else:
        for row in unmatched.itertuples(index=False):
            lines.append(f"- `{row.source_label}` ({row.mapping_rule})")
    lines.extend(["", "## CSV outputs", ""])
    for label, output_path in csv_paths.items():
        lines.append(f"- {label}: `{output_path.resolve()}`")
    lines.extend(
        [
            "",
            "## Interpretation and safeguards",
            "",
            "- The exp006 curve is the 25 K temperature-bin broad_family sigma0_ref(T) from Step9A.",
            "- The exp005 curve is SS2026 C(T) from Step31, not the old Step12 prefactor C(T).",
            "- No new sigma_pred was calculated.",
            "- Starrydata2 raw data was not read.",
            "- Existing Step0-Step9A and Step31 outputs were not overwritten.",
            f"- Input file metadata remained unchanged during this run: {source_unchanged}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    sigma_path = resolve_sigma_input(args.sigma0ref25k)
    validate_args(args, sigma_path)
    input_metadata_before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in [sigma_path, args.step31_script.resolve(), args.step31_report.resolve()]
    }

    log("loading exp006 25K sigma0_ref curves...", started)
    sigma = load_sigma_curves(sigma_path, args.config_id)
    log("reading Step31 script/report...", started)
    log("detecting Step31 C(T) output file...", started)
    step31_path, ct_col, t_col, material_col, carrier_col, detection_note = detect_step31_output(
        args.step31_output_dir, args.step31_script, args.step31_report
    )
    log("loading Step31 C(T) curves...", started)
    normalized_ct, mapping, unmatched = load_step31_curves(
        step31_path, ct_col, t_col, material_col, carrier_col
    )
    log("normalizing material labels...", started)
    log("filtering common material groups...", started)
    groups = target_groups_from_args(args.target_groups, sigma, normalized_ct, args.max_groups)

    csv_paths = planned_csv_paths(args)
    ensure_outputs_absent([*csv_paths.values(), args.report])
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    parse_summary = pd.DataFrame(
        [
            {
                "step31_script_path": str(args.step31_script.resolve()),
                "step31_report_path": str(args.step31_report.resolve()),
                "detected_output_file": str(step31_path),
                "detected_ct_column": ct_col,
                "detected_temperature_column": t_col,
                "detected_material_column": material_col,
                "detected_carrier_column": carrier_col,
                "detection_status": "ok",
                "notes": detection_note,
            }
        ],
        columns=PARSE_SUMMARY_COLUMNS,
    )

    comparison_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    figure_counter = 1
    valid_ct = normalized_ct[normalized_ct["parse_status"].str.startswith("ok")]
    for group in groups:
        for carrier in ["p", "n"]:
            log(f"processing group/carrier {group}, {carrier}...", started)
            sigma_part = sigma[
                sigma["material_group_key"].eq(group) & sigma["carrier_type"].eq(carrier)
            ].copy()
            ct_part = valid_ct[
                valid_ct["material_group_key"].eq(group) & valid_ct["carrier_type"].eq(carrier)
            ].copy()
            warning_parts = []
            if sigma_part.empty:
                warning_parts.append("missing Step9A 25 K sigma0_ref(T)")
            if ct_part.empty:
                warning_parts.append("missing Step31 C(T)")
            comparison = nearest_comparison(sigma_part, ct_part)
            if not comparison.empty:
                comparison_parts.append(comparison)
                log("creating overlay figure...", started)
                log("creating log-ratio figure...", started)
                new_rows = create_figures(
                    group,
                    carrier,
                    sigma_part,
                    ct_part,
                    comparison,
                    args.figures,
                    args.output_suffix,
                    figure_counter,
                )
                figure_rows.extend(new_rows)
                figure_counter += 2
            temperatures = pd.concat(
                [sigma_part["T_bin_center_K"], ct_part["T_K"]], ignore_index=True
            )
            summary_rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "sigma0ref25k_points": len(sigma_part),
                    "step31_ct_points": len(ct_part),
                    "comparison_points": len(comparison),
                    "T_min_K": float(temperatures.min()) if len(temperatures) else np.nan,
                    "T_max_K": float(temperatures.max()) if len(temperatures) else np.nan,
                    "sigma0ref25k_median_S_per_m": finite_median(sigma_part["sigma0_ref_S_per_m"]),
                    "step31_ct_median_S_per_m": finite_median(ct_part["step31_C_T_S_per_m"]),
                    "median_log10_sigma0ref25k_over_step31ct": finite_median(
                        comparison["log10_sigma0ref25k_over_step31ct"]
                    ),
                    "warning": "; ".join(warning_parts),
                }
            )

    comparison_table = (
        pd.concat(comparison_parts, ignore_index=True)
        if comparison_parts
        else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    figure_index = pd.DataFrame(figure_rows, columns=FIGURE_INDEX_COLUMNS)
    input_metadata_after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in input_metadata_before
    }
    source_unchanged = input_metadata_before == input_metadata_after
    if not source_unchanged:
        raise AssertionError("An input file changed during the Step9B run")

    log("writing CSV outputs...", started)
    parse_summary.to_csv(csv_paths["parse"], index=False)
    sigma.to_csv(csv_paths["sigma"], index=False)
    normalized_ct.to_csv(csv_paths["ct"], index=False)
    comparison_table.to_csv(csv_paths["comparison"], index=False)
    summary.to_csv(csv_paths["summary"], index=False)
    figure_index.to_csv(csv_paths["figures"], index=False)
    mapping.to_csv(csv_paths["mapping"], index=False)
    unmatched.to_csv(csv_paths["unmatched"], index=False)
    log("writing report...", started)
    write_report(
        args.report,
        sigma_path,
        step31_path,
        ct_col,
        t_col,
        material_col,
        carrier_col,
        groups,
        summary,
        comparison_table,
        figure_index,
        unmatched,
        csv_paths,
        source_unchanged,
    )
    log("running sanity checks...", started)
    if comparison_table.empty or figure_index.empty:
        raise AssertionError("No comparison rows or figures were created")
    if not set(figure_index["figure_type"]) >= {"overlay", "log_ratio_nearest"}:
        raise AssertionError("Both required figure types were not created")
    log(f"done. groups={len(groups)}, combinations={summary['comparison_points'].gt(0).sum()}", started)


if __name__ == "__main__":
    main()
