"""Plot saved Step9A sigma_pred rows against carrier-specific Step31 C(T)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
STEP9A_DIR = EXP_DIR / "data" / "processed" / "step9a_25k_bin_broad_family"
DATA_DIR = EXP_DIR / "data" / "processed" / "step9b_25k_vs_step31_ct_overlay"
DEFAULT_PREDICTIONS = STEP9A_DIR / "step5b_test_predictions_valid.parquet"
DEFAULT_CT = DATA_DIR / "step9b_step31_ct_curves_normalized.csv"
DEFAULT_BASE_SUMMARY = DATA_DIR / "step9b_summary_by_group_carrier.csv"
DEFAULT_FIGURES = EXP_DIR / "figures" / "step9b_25k_vs_step31_ct_overlay" / "ct_vs_pred_style"
DEFAULT_INDEX = DATA_DIR / "step9b_figure_index_ct_vs_pred_style.csv"
DEFAULT_COMPARISON = DATA_DIR / "step9b_pred_vs_step31_ct_comparison.csv"
DEFAULT_PRED_SUMMARY = DATA_DIR / "step9b_pred_vs_step31_ct_summary.csv"
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "step9b_25k_vs_step31_ct_overlay"
    / "step9b_25k_vs_step31_ct_style_matched_report.md"
)
DEFAULT_CONFIG_ID = (
    "sample_holdout__ref_conservative_valid__eval_all_valid"
    "__material_family__sample_median"
)

PREDICTION_COLUMNS = [
    "config_id",
    "prediction_status",
    "material_group_key",
    "carrier_type",
    "T_K",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "row_id",
    "sample_id",
    "paper_id",
    "F0_eta",
    "sigma0_ref_S_per_m",
]
INDEX_COLUMNS = [
    "figure_id",
    "material_group_key",
    "carrier_type",
    "figure_type",
    "figure_path_png",
    "figure_path_pdf",
    "title",
    "n_sigma_pred_points",
    "n_step31_ct_points",
    "description",
]
COMPARISON_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "row_id",
    "sample_id",
    "paper_id",
    "T_K_pred",
    "sigma_pred_S_per_m",
    "log10_sigma_pred_S_per_m",
    "F0_eta",
    "sigma0_ref_S_per_m",
    "T_K_step31",
    "step31_C_T_S_per_m",
    "log10_step31_C_T_S_per_m",
    "T_delta_K",
    "log10_sigma_pred_over_step31ct",
    "match_method",
]
SUMMARY_COLUMNS = [
    "material_group_key",
    "carrier_type",
    "sigma_pred_points",
    "step31_ct_points",
    "comparison_points",
    "T_pred_min_K",
    "T_pred_max_K",
    "sigma_pred_median_S_per_m",
    "step31_ct_median_S_per_m",
    "median_log10_sigma_pred_over_step31ct",
    "warning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--step31-ct", type=Path, default=DEFAULT_CT)
    parser.add_argument("--base-summary", type=Path, default=DEFAULT_BASE_SUMMARY)
    parser.add_argument("--config-id", default=DEFAULT_CONFIG_ID)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--figure-index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--comparison-summary", type=Path, default=DEFAULT_PRED_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--target-groups", nargs="+", default=None)
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite this script's existing sigma_pred-vs-Step31 outputs.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def suffixed(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def ensure_absent(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs:\n" + "\n".join(existing))


def validate_frame(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_predictions(path: Path, config_id: str) -> pd.DataFrame:
    if path.suffix.casefold() == ".parquet":
        frame = pd.read_parquet(path, columns=PREDICTION_COLUMNS)
    else:
        frame = pd.read_csv(path, usecols=PREDICTION_COLUMNS, low_memory=False)
    if not frame["config_id"].astype(str).eq(config_id).any():
        raise ValueError(f"Requested config_id is absent: {config_id}")
    frame = frame[
        frame["config_id"].astype(str).eq(config_id)
        & frame["prediction_status"].astype(str).eq("ok")
    ].copy()
    for column in [
        "T_K",
        "sigma_pred_S_per_m",
        "log10_sigma_pred_S_per_m",
        "F0_eta",
        "sigma0_ref_S_per_m",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    valid = (
        np.isfinite(frame["T_K"])
        & frame["T_K"].gt(0)
        & np.isfinite(frame["sigma_pred_S_per_m"])
        & frame["sigma_pred_S_per_m"].gt(0)
        & np.isfinite(frame["log10_sigma_pred_S_per_m"])
        & np.isfinite(frame["F0_eta"])
        & frame["F0_eta"].gt(0)
        & np.isfinite(frame["sigma0_ref_S_per_m"])
        & frame["sigma0_ref_S_per_m"].gt(0)
        & frame["carrier_type"].astype(str).isin({"p", "n"})
        & frame["material_group_key"].notna()
    )
    frame = frame[valid].copy()
    if frame.empty:
        raise ValueError("No usable saved sigma_pred rows remain")
    expected = frame["sigma0_ref_S_per_m"] * frame["F0_eta"]
    if not np.allclose(frame["sigma_pred_S_per_m"], expected, rtol=1e-10, atol=0.0):
        raise ValueError("Saved sigma_pred is inconsistent with sigma0_ref * F0_eta")
    return frame


def nearest_comparison(pred: pd.DataFrame, ct: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or ct.empty:
        return pd.DataFrame(columns=COMPARISON_COLUMNS)
    ordered_ct = ct.sort_values("T_K").reset_index(drop=True)
    ct_t = ordered_ct["T_K"].to_numpy(dtype=float)
    pred_t = pred["T_K"].to_numpy(dtype=float)
    right = np.searchsorted(ct_t, pred_t, side="left")
    right = np.clip(right, 0, len(ct_t) - 1)
    left = np.clip(right - 1, 0, len(ct_t) - 1)
    choose_left = np.abs(ct_t[left] - pred_t) <= np.abs(ct_t[right] - pred_t)
    positions = np.where(choose_left, left, right)
    matched = ordered_ct.iloc[positions].reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    result = pd.DataFrame(
        {
            "material_group_key": pred["material_group_key"],
            "carrier_type": pred["carrier_type"],
            "row_id": pred["row_id"],
            "sample_id": pred["sample_id"],
            "paper_id": pred["paper_id"],
            "T_K_pred": pred["T_K"],
            "sigma_pred_S_per_m": pred["sigma_pred_S_per_m"],
            "log10_sigma_pred_S_per_m": pred["log10_sigma_pred_S_per_m"],
            "F0_eta": pred["F0_eta"],
            "sigma0_ref_S_per_m": pred["sigma0_ref_S_per_m"],
            "T_K_step31": matched["T_K"],
            "step31_C_T_S_per_m": matched["step31_C_T_S_per_m"],
            "log10_step31_C_T_S_per_m": matched["log10_step31_C_T_S_per_m"],
        }
    )
    result["T_delta_K"] = (result["T_K_step31"] - result["T_K_pred"]).abs()
    result["log10_sigma_pred_over_step31ct"] = (
        result["log10_sigma_pred_S_per_m"] - result["log10_step31_C_T_S_per_m"]
    )
    result["match_method"] = "nearest_step31_temperature"
    return result[COMPARISON_COLUMNS]


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    return float(values.median()) if len(values) else np.nan


def main() -> None:
    args = parse_args()
    for path in [args.predictions, args.step31_ct, args.base_summary]:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.max_groups is not None and args.max_groups <= 0:
        raise ValueError("--max-groups must be positive")
    if re.search(r"[\\/]", args.output_suffix):
        raise ValueError("--output-suffix must not contain a path separator")

    predictions = load_predictions(args.predictions, args.config_id)
    ct = pd.read_csv(args.step31_ct, low_memory=False)
    base_summary = pd.read_csv(args.base_summary, low_memory=False)
    validate_frame(
        ct,
        {
            "material_group_key",
            "carrier_type",
            "T_K",
            "step31_C_T_S_per_m",
            "log10_step31_C_T_S_per_m",
            "parse_status",
        },
        "Step31 C(T) input",
    )
    validate_frame(base_summary, {"material_group_key", "carrier_type"}, "base summary")
    valid_ct = ct[ct["parse_status"].astype(str).str.startswith("ok")].copy()

    groups = base_summary["material_group_key"].drop_duplicates().astype(str).tolist()
    if args.target_groups:
        requested: list[str] = []
        for token in args.target_groups:
            requested.extend(value.strip() for value in token.split(",") if value.strip())
        missing = [group for group in requested if group not in groups]
        if missing:
            raise ValueError(f"Requested groups are absent from the Step9B groups: {missing}")
        groups = requested
    if args.max_groups is not None:
        groups = groups[: args.max_groups]

    index_path = suffixed(args.figure_index, args.output_suffix)
    comparison_path = suffixed(args.comparison, args.output_suffix)
    summary_path = suffixed(args.comparison_summary, args.output_suffix)
    report_path = suffixed(args.report, args.output_suffix)
    output_paths = [index_path, comparison_path, summary_path, report_path]
    if not args.overwrite:
        ensure_absent(output_paths)
    args.figures.mkdir(parents=True, exist_ok=True)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, object]] = []
    comparison_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    line_only: list[str] = []
    figure_id = 1
    for group in groups:
        for carrier in ["p", "n"]:
            pred = predictions[
                predictions["material_group_key"].eq(group)
                & predictions["carrier_type"].eq(carrier)
            ].sort_values(["T_K", "row_id"])
            ct_part = valid_ct[
                valid_ct["material_group_key"].eq(group)
                & valid_ct["carrier_type"].eq(carrier)
            ].sort_values("T_K")
            if pred.empty:
                line_only.append(f"{group}, {carrier}: no saved sigma_pred rows")
            if pred.empty and ct_part.empty:
                continue

            comparison = nearest_comparison(pred, ct_part)
            if not comparison.empty:
                comparison_parts.append(comparison)
            stem = f"{safe_name(group)}_{carrier}_sigma_pred_vs_step31CT_25k{args.output_suffix}"
            png_path = args.figures / f"{stem}.png"
            pdf_path = args.figures / f"{stem}.pdf"
            if not args.overwrite:
                ensure_absent([png_path, pdf_path])

            title = f"{group}, {carrier}: predicted sigma vs Step31 C(T), 25K bins"
            color = "#d95f02" if carrier == "p" else "#1b6ca8"
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            ax.scatter(
                pred["T_K"],
                pred["sigma_pred_S_per_m"],
                s=16,
                alpha=0.48,
                color=color,
                edgecolors="none",
                label=f"Predicted sigma ({carrier})",
                zorder=2,
            )
            ax.plot(
                ct_part["T_K"],
                ct_part["step31_C_T_S_per_m"],
                color="#222222",
                linewidth=2.5,
                label=f"SS2026 Step31 C(T) ({carrier})",
                zorder=3,
            )
            ax.set_yscale("log")
            ax.set_xlabel("Temperature T [K]")
            ax.set_ylabel("Electrical conductivity sigma [S/m]")
            ax.set_title(title)
            maxima = [1300.0]
            if not pred.empty:
                maxima.append(float(pred["T_K"].max()))
            if not ct_part.empty:
                maxima.append(float(ct_part["T_K"].max()))
            ax.set_xlim(0.0, max(maxima))
            ax.grid(True, which="both", alpha=0.22)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(png_path, dpi=240, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)

            warning = []
            if pred.empty:
                warning.append("missing saved sigma_pred rows; line-only figure")
            if ct_part.empty:
                warning.append("missing Step31 C(T)")
            summary_rows.append(
                {
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "sigma_pred_points": len(pred),
                    "step31_ct_points": len(ct_part),
                    "comparison_points": len(comparison),
                    "T_pred_min_K": float(pred["T_K"].min()) if len(pred) else np.nan,
                    "T_pred_max_K": float(pred["T_K"].max()) if len(pred) else np.nan,
                    "sigma_pred_median_S_per_m": finite_median(pred["sigma_pred_S_per_m"]),
                    "step31_ct_median_S_per_m": finite_median(ct_part["step31_C_T_S_per_m"]),
                    "median_log10_sigma_pred_over_step31ct": finite_median(
                        comparison["log10_sigma_pred_over_step31ct"]
                    ),
                    "warning": "; ".join(warning),
                }
            )
            index_rows.append(
                {
                    "figure_id": f"STYLE_{figure_id:03d}",
                    "material_group_key": group,
                    "carrier_type": carrier,
                    "figure_type": "sigma_pred_points_vs_step31_ct_line",
                    "figure_path_png": str(png_path.resolve()),
                    "figure_path_pdf": str(pdf_path.resolve()),
                    "title": title,
                    "n_sigma_pred_points": len(pred),
                    "n_step31_ct_points": len(ct_part),
                    "description": (
                        "Saved Step9A holdout/test sigma_pred points and black carrier-specific "
                        "Step31 SS2026 C(T) line; no prediction was recalculated."
                    ),
                }
            )
            figure_id += 1
            print(f"[step9b-style] created {group}, {carrier}: sigma_pred={len(pred)}", flush=True)

    index = pd.DataFrame(index_rows, columns=INDEX_COLUMNS)
    comparison_table = (
        pd.concat(comparison_parts, ignore_index=True)
        if comparison_parts
        else pd.DataFrame(columns=COMPARISON_COLUMNS)
    )
    pred_summary = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    if index.empty or comparison_table.empty:
        raise AssertionError("No sigma_pred-vs-Step31 outputs were created")
    index.to_csv(index_path, index=False)
    comparison_table.to_csv(comparison_path, index=False)
    pred_summary.to_csv(summary_path, index=False)

    ratio = finite_median(comparison_table["log10_sigma_pred_over_step31ct"])
    report_lines = [
        "# Step9B saved sigma_pred vs Step31 C(T)",
        "",
        "This replaces the former sigma0_ref points in the ct-vs-pred-style supplementary figures "
        "with already-saved Step9A holdout/test sigma_pred rows.",
        "Before replacement, the colored points were the train-derived 25 K sigma0_ref reference "
        "coefficient (before multiplying by row-specific F0_eta), compared with Step31's "
        "carrier-specific median measured conductivity C(T).",
        "",
        f"- prediction input: `{args.predictions.resolve()}`",
        f"- config_id: `{args.config_id}`",
        f"- Step31 C(T) input: `{args.step31_ct.resolve()}`",
        f"- target material groups: {len(groups)}",
        f"- figures created: {len(index)} PNG + {len(index)} PDF",
        f"- saved sigma_pred rows plotted/matched: {len(comparison_table)}",
        f"- line-only combinations: {len(line_only)}",
        f"- median log10(sigma_pred / Step31 C(T)): {ratio:.6g}",
        "- x-axis policy: every figure includes 0-1300 K; data above 1300 K remain visible.",
        f"- figure index: `{index_path.resolve()}`",
        f"- nearest comparison: `{comparison_path.resolve()}`",
        f"- summary: `{summary_path.resolve()}`",
        f"- figure directory: `{args.figures.resolve()}`",
        "",
        "No sigma_pred was recalculated, and no Step9A or Step31 input was modified.",
    ]
    if line_only:
        report_lines.extend(["", "## Line-only combinations", ""] + [f"- {value}" for value in line_only])
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    expected_ratio = (
        comparison_table["log10_sigma_pred_S_per_m"]
        - comparison_table["log10_step31_C_T_S_per_m"]
    )
    if not np.allclose(expected_ratio, comparison_table["log10_sigma_pred_over_step31ct"]):
        raise AssertionError("Comparison log ratio is inconsistent")
    for row in index.itertuples(index=False):
        for path in [Path(row.figure_path_png), Path(row.figure_path_pdf)]:
            if not path.exists() or path.stat().st_size == 0:
                raise AssertionError(f"Missing/empty figure: {path}")

    if args.overwrite:
        legacy = list(args.figures.glob("*_sigma0ref_vs_step31CT_25k_ct_vs_pred_style*.png"))
        legacy += list(args.figures.glob("*_sigma0ref_vs_step31CT_25k_ct_vs_pred_style*.pdf"))
        for path in legacy:
            path.unlink()
        if legacy:
            print(f"[step9b-style] removed {len(legacy)} replaced sigma0_ref figure files", flush=True)
    print(
        f"[step9b-style] done: figures={len(index)}, comparisons={len(comparison_table)}, "
        f"index={index_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
