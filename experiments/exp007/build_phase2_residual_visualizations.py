"""Create exp007 Phase 2 visual diagnostics for sigma prediction residuals."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import PercentFormatter
except ImportError as exc:  # pragma: no cover - gives a useful runtime message
    raise SystemExit(
        "Phase 2 requires matplotlib. Install it with `python -m pip install matplotlib`."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
EXP007 = Path(__file__).resolve().parent
SOURCE = EXP007 / "data" / "processed" / "phase1_sigma_residual_rows.csv"
PHASE1_CHECKS = EXP007 / "data" / "processed" / "phase1_validation_checks.csv"
DEFAULT_FIGURE_DIR = EXP007 / "figures" / "phase2"
DEFAULT_DATA_DIR = EXP007 / "data" / "processed"
DEFAULT_REPORT = EXP007 / "reports" / "phase2_residual_visualization.md"
DEFAULT_VALIDATION = EXP007 / "reports" / "phase2_validation_log.txt"

LOG2 = math.log10(2.0)
LOG5 = math.log10(5.0)
PLOT_RESIDUAL_LIMIT = 4.0

COLORS = {
    "blue": "#4472C4",
    "orange": "#ED7D31",
    "green": "#70AD47",
    "red": "#C00000",
    "purple": "#8064A2",
    "gray": "#7F7F7F",
    "light_gray": "#D9E1F2",
    "dark": "#262626",
}

FIGURE_STEMS = [
    "phase2_figure1_global_residual_overview",
    "phase2_figure2_material_group_profiles",
    "phase2_figure3_temperature_profiles",
    "phase2_figure4_oxide_focus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-log", type=Path, default=DEFAULT_VALIDATION)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_group_label(value: str) -> str:
    text = str(value).replace("broad::", "").replace("_", " ")
    replacements = {
        "unknown material group": "unknown",
        "other formula system": "other formula system",
        "BiSbTe tetradymite like": "BiSbTe tetradymite-like",
        "CoSb skutterudite like": "CoSb skutterudite-like",
    }
    return replacements.get(text, text)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def numeric_metrics(errors: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(errors, errors="coerce").dropna().to_numpy(dtype=float)
    absolute = np.abs(values)
    return {
        "n_rows": int(values.size),
        "mean_residual_log10": float(np.mean(values)),
        "median_residual_log10": float(np.median(values)),
        "q25_residual_log10": float(np.quantile(values, 0.25)),
        "q75_residual_log10": float(np.quantile(values, 0.75)),
        "q10_residual_log10": float(np.quantile(values, 0.10)),
        "q90_residual_log10": float(np.quantile(values, 0.90)),
        "mae_log10": float(np.mean(absolute)),
        "rmse_log10": float(np.sqrt(np.mean(values**2))),
        "within_factor_2_fraction": float(np.mean(absolute <= LOG2)),
        "within_factor_5_fraction": float(np.mean(absolute <= LOG5)),
        "within_factor_10_fraction": float(np.mean(absolute <= 1.0)),
        "overprediction_fraction": float(np.mean(values > 0)),
        "underprediction_fraction": float(np.mean(values < 0)),
    }


def material_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_col = "material_group_key_for_prediction"
    for key, group in rows.groupby(group_col, sort=False):
        record = {group_col: key, "material_group_label": clean_group_label(key)}
        record.update(numeric_metrics(group["residual_log10_pred_over_observed"]))
        record["n_samples"] = int(group["validation_sample_group_id"].nunique())
        sample_medians = group.groupby("validation_sample_group_id")[
            "residual_log10_pred_over_observed"
        ].median()
        sample_metrics = numeric_metrics(sample_medians)
        for metric, value in sample_metrics.items():
            record[f"sample_equal_{metric}"] = value
        records.append(record)
    return pd.DataFrame(records).sort_values("mae_log10", ascending=False).reset_index(drop=True)


def temperature_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for label, group in rows.groupby("T_bin_label", sort=False):
        record = {
            "T_bin_label": label,
            "T_bin_center_K": float(pd.to_numeric(group["T_bin_center_K"]).median()),
            "T_bin_left_K": float(pd.to_numeric(group["T_bin_left_K"]).median()),
            "T_bin_right_K": float(pd.to_numeric(group["T_bin_right_K"]).median()),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
        }
        record.update(numeric_metrics(group["residual_log10_pred_over_observed"]))
        records.append(record)
    return pd.DataFrame(records).sort_values("T_bin_center_K").reset_index(drop=True)


def oxide_summary(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for is_oxide, group in rows.groupby("is_oxide_broad", sort=False):
        record = {
            "scope": "broad oxide" if bool(is_oxide) else "non-oxide",
            "is_oxide_broad": bool(is_oxide),
            "n_samples": int(group["validation_sample_group_id"].nunique()),
            "n_papers": int(group["validation_paper_group_id"].nunique()),
        }
        record.update(numeric_metrics(group["residual_log10_pred_over_observed"]))
        sample_medians = group.groupby("validation_sample_group_id")[
            "residual_log10_pred_over_observed"
        ].median()
        sample_metrics = numeric_metrics(sample_medians)
        for metric, value in sample_metrics.items():
            record[f"sample_equal_{metric}"] = value
        records.append(record)
    return pd.DataFrame(records).sort_values("is_oxide_broad", ascending=False).reset_index(drop=True)


def add_residual_reference_lines(ax: plt.Axes, horizontal: bool = False) -> None:
    method = ax.axhline if horizontal else ax.axvline
    method(0.0, color=COLORS["dark"], linewidth=1.2, zorder=4)
    for value in (-1.0, 1.0):
        method(value, color=COLORS["red"], linewidth=1.0, linestyle="--", alpha=0.85)
    for value in (-LOG2, LOG2):
        method(value, color=COLORS["gray"], linewidth=0.9, linestyle=":", alpha=0.9)


def robust_log_sigma_limits(rows: pd.DataFrame) -> tuple[float, float]:
    values = pd.concat(
        [
            rows["log10_sigma_observed_S_per_m"],
            rows["log10_sigma_predicted_S_per_m"],
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    low = max(-3.0, math.floor(float(values.quantile(0.005)) * 2.0) / 2.0)
    high = min(10.0, math.ceil(float(values.quantile(0.995)) * 2.0) / 2.0)
    return low, high


def binned_trend(
    x: pd.Series, y: pd.Series, bins: int = 14
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"x": pd.to_numeric(x), "y": pd.to_numeric(y)}).dropna()
    frame = frame.loc[np.isfinite(frame["x"]) & np.isfinite(frame["y"])]
    frame["bin"] = pd.qcut(frame["x"], q=bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True)
    return (
        grouped["x"].median().to_numpy(dtype=float),
        grouped["y"].median().to_numpy(dtype=float),
        grouped["y"].quantile(0.25).to_numpy(dtype=float),
        grouped["y"].quantile(0.75).to_numpy(dtype=float),
    )


def figure_global_overview(rows: pd.DataFrame, figure_dir: Path) -> None:
    residual = rows["residual_log10_pred_over_observed"]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    fig.suptitle("Electrical-conductivity residual overview: broad-family 100 K baseline")

    ax = axes[0, 0]
    central = residual.between(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT)
    bins = np.linspace(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT, 81)
    ax.hist(residual.loc[central], bins=bins, color=COLORS["blue"], alpha=0.82)
    add_residual_reference_lines(ax)
    left = int((residual < -PLOT_RESIDUAL_LIMIT).sum())
    right = int((residual > PLOT_RESIDUAL_LIMIT).sum())
    ax.text(
        0.02,
        0.97,
        f"Outside view: {left:,} below −{PLOT_RESIDUAL_LIMIT:g}, {right:,} above +{PLOT_RESIDUAL_LIMIT:g}",
        transform=ax.transAxes,
        va="top",
    )
    ax.set(xlabel="Signed residual log10(predicted / observed)", ylabel="Rows", title="A  Residual distribution")

    low, high = robust_log_sigma_limits(rows)
    ax = axes[0, 1]
    in_view = (
        rows["log10_sigma_observed_S_per_m"].between(low, high)
        & rows["log10_sigma_predicted_S_per_m"].between(low, high)
    )
    hb = ax.hexbin(
        rows.loc[in_view, "log10_sigma_observed_S_per_m"],
        rows.loc[in_view, "log10_sigma_predicted_S_per_m"],
        gridsize=55,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(),
    )
    line = np.array([low, high])
    ax.plot(line, line, color=COLORS["dark"], linewidth=1.2, label="1:1")
    ax.plot(line, line + 1, color=COLORS["red"], linestyle="--", linewidth=1.0, label="factor 10")
    ax.plot(line, line - 1, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
    ax.set(
        xlabel="Observed log10 conductivity (S/m)",
        ylabel="Predicted log10 conductivity (S/m)",
        title="B  Observed versus predicted",
    )
    ax.text(0.02, 0.97, f"Rows outside view: {(~in_view).sum():,}", transform=ax.transAxes, va="top")
    ax.legend(loc="lower right")
    fig.colorbar(hb, ax=ax, label="Rows per hexagon", shrink=0.83)

    ax = axes[1, 0]
    y_view = residual.between(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT)
    view = in_view & y_view
    hb = ax.hexbin(
        rows.loc[view, "log10_sigma_observed_S_per_m"],
        rows.loc[view, "residual_log10_pred_over_observed"],
        gridsize=55,
        mincnt=1,
        cmap="cividis",
        norm=LogNorm(),
    )
    tx, median, q25, q75 = binned_trend(
        rows.loc[in_view, "log10_sigma_observed_S_per_m"],
        rows.loc[in_view, "residual_log10_pred_over_observed"],
    )
    ax.fill_between(tx, q25, q75, color=COLORS["orange"], alpha=0.22, label="binned IQR")
    ax.plot(tx, median, color=COLORS["orange"], linewidth=2.0, marker="o", markersize=3.5, label="binned median")
    add_residual_reference_lines(ax, horizontal=True)
    ax.set(xlim=(low, high), ylim=(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT))
    ax.set(
        xlabel="Observed log10 conductivity (S/m)",
        ylabel="Signed residual log10(predicted / observed)",
        title="C  Residual versus observed conductivity",
    )
    ax.legend(loc="upper right")
    fig.colorbar(hb, ax=ax, label="Rows per hexagon", shrink=0.83)

    ax = axes[1, 1]
    bands = ["within_factor_2", "factor_2_to_5", "factor_5_to_10", "beyond_factor_10"]
    labels = ["≤2×", "2–5×", "5–10×", ">10×"]
    under = []
    over = []
    total = len(rows)
    for band in bands:
        subset = rows["residual_band"].eq(band)
        under.append(int((subset & rows["prediction_direction"].eq("underprediction")).sum()))
        over.append(int((subset & rows["prediction_direction"].eq("overprediction")).sum()))
    x = np.arange(len(bands))
    under_pct = np.array(under) / total * 100.0
    over_pct = np.array(over) / total * 100.0
    ax.bar(x, under_pct, color=COLORS["blue"], label="underprediction")
    ax.bar(x, over_pct, bottom=under_pct, color=COLORS["orange"], label="overprediction")
    for idx, (u, o) in enumerate(zip(under, over)):
        ax.text(idx, (u + o) / total * 100.0 + 0.7, f"{u + o:,}\n({(u + o) / total:.1%})", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of evaluable rows (%)")
    ax.set_title("D  Error-magnitude bands and direction")
    ax.set_ylim(0, float(np.max(under_pct + over_pct)) * 1.28)
    ax.legend(loc="upper right")

    save_figure(fig, figure_dir, FIGURE_STEMS[0])


def figure_material_profiles(summary: pd.DataFrame, figure_dir: Path) -> None:
    plot = summary.sort_values("mae_log10", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot))
    labels = [f"{label}  (n={n:,})" for label, n in zip(plot["material_group_label"], plot["n_rows"])]
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 9.2), sharey=True, constrained_layout=True)
    fig.suptitle("Residual profiles by broad material group")

    ax = axes[0]
    ax.hlines(y, plot["mae_log10"], plot["sample_equal_mae_log10"], color=COLORS["light_gray"], linewidth=2)
    ax.scatter(plot["mae_log10"], y, color=COLORS["blue"], s=42, label="row-equal MAE", zorder=3)
    ax.scatter(plot["sample_equal_mae_log10"], y, color=COLORS["orange"], marker="s", s=38, label="sample-equal MAE", zorder=3)
    ax.set_yticks(y, labels)
    ax.set_xlabel("MAE of log10 ratio")
    ax.set_title("A  Absolute error")
    ax.legend(loc="lower right")

    ax = axes[1]
    lower = plot["median_residual_log10"] - plot["q25_residual_log10"]
    upper = plot["q75_residual_log10"] - plot["median_residual_log10"]
    ax.errorbar(
        plot["median_residual_log10"],
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color=COLORS["green"],
        ecolor=COLORS["gray"],
        elinewidth=2,
        capsize=3,
    )
    add_residual_reference_lines(ax)
    ax.set_xlabel("Signed residual log10(predicted / observed)")
    ax.set_title("B  Median and interquartile range")

    ax = axes[2]
    ax.hlines(
        y,
        plot["within_factor_2_fraction"] * 100,
        plot["within_factor_10_fraction"] * 100,
        color=COLORS["light_gray"],
        linewidth=2,
    )
    ax.scatter(plot["within_factor_2_fraction"] * 100, y, color=COLORS["purple"], s=42, label="within factor 2")
    ax.scatter(plot["within_factor_10_fraction"] * 100, y, color=COLORS["green"], marker="s", s=38, label="within factor 10")
    ax.set(xlim=(0, 102), xlabel="Rows within error threshold (%)", title="C  Accuracy thresholds")
    ax.legend(loc="lower right")

    save_figure(fig, figure_dir, FIGURE_STEMS[1])


def figure_temperature_profiles(summary: pd.DataFrame, figure_dir: Path) -> None:
    plot = summary.sort_values("T_bin_center_K")
    x = plot["T_bin_center_K"].to_numpy(dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), constrained_layout=True)
    fig.suptitle("Residual profiles across 100 K reference-temperature bins")

    ax = axes[0, 0]
    ax.fill_between(x, plot["q25_residual_log10"], plot["q75_residual_log10"], color=COLORS["blue"], alpha=0.18, label="IQR")
    ax.plot(x, plot["median_residual_log10"], color=COLORS["blue"], marker="o", label="median")
    add_residual_reference_lines(ax, horizontal=True)
    ax.set(xlabel="Temperature-bin center (K)", ylabel="Signed residual log10(predicted / observed)", title="A  Bias by temperature bin")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    ax.plot(x, plot["mae_log10"], color=COLORS["orange"], marker="o", label="MAE")
    ax.plot(x, plot["rmse_log10"], color=COLORS["red"], marker="s", label="RMSE")
    ax.set(xlabel="Temperature-bin center (K)", ylabel="log10 error", title="B  Error magnitude")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, plot["within_factor_2_fraction"] * 100, color=COLORS["purple"], marker="o", label="within factor 2")
    ax.plot(x, plot["within_factor_10_fraction"] * 100, color=COLORS["green"], marker="s", label="within factor 10")
    ax.set(xlabel="Temperature-bin center (K)", ylabel="Rows within threshold (%)", ylim=(0, 102), title="C  Accuracy thresholds")
    ax.legend()

    ax = axes[1, 1]
    widths = np.maximum(30.0, plot["T_bin_right_K"] - plot["T_bin_left_K"]) * 0.75
    ax.bar(x, plot["n_rows"], width=widths, color=COLORS["gray"], alpha=0.8)
    ax.set(xlabel="Temperature-bin center (K)", ylabel="Evaluable rows", title="D  Evaluation support")

    save_figure(fig, figure_dir, FIGURE_STEMS[2])


def figure_oxide_focus(rows: pd.DataFrame, figure_dir: Path) -> None:
    oxide = rows.loc[rows["is_oxide_broad"]].copy()
    nonoxide = rows.loc[~rows["is_oxide_broad"]].copy()
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    fig.suptitle("Broad-oxide residual diagnostics")

    ax = axes[0, 0]
    bins = np.linspace(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT, 81)
    for frame, label, color in (
        (oxide, f"broad oxide (n={len(oxide):,})", COLORS["orange"]),
        (nonoxide, f"non-oxide (n={len(nonoxide):,})", COLORS["blue"]),
    ):
        values = frame["residual_log10_pred_over_observed"]
        central = values.between(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT)
        ax.hist(values.loc[central], bins=bins, density=True, histtype="step", linewidth=2.0, label=label, color=color)
    add_residual_reference_lines(ax)
    ax.set(xlabel="Signed residual log10(predicted / observed)", ylabel="Density within displayed range", title="A  Oxide versus non-oxide")
    ax.legend(loc="upper right")

    low, high = robust_log_sigma_limits(rows)
    ax = axes[0, 1]
    in_view = (
        oxide["log10_sigma_observed_S_per_m"].between(low, high)
        & oxide["log10_sigma_predicted_S_per_m"].between(low, high)
    )
    hb = ax.hexbin(
        oxide.loc[in_view, "log10_sigma_observed_S_per_m"],
        oxide.loc[in_view, "log10_sigma_predicted_S_per_m"],
        gridsize=45,
        mincnt=1,
        cmap="inferno",
        norm=LogNorm(),
    )
    line = np.array([low, high])
    ax.plot(line, line, color=COLORS["dark"], linewidth=1.2)
    ax.plot(line, line + 1, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.plot(line, line - 1, color=COLORS["red"], linestyle="--", linewidth=1.0)
    ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
    ax.set(xlabel="Observed log10 conductivity (S/m)", ylabel="Predicted log10 conductivity (S/m)", title="B  Broad-oxide parity")
    fig.colorbar(hb, ax=ax, label="Rows per hexagon", shrink=0.83)

    ax = axes[1, 0]
    temp_view = oxide["residual_log10_pred_over_observed"].between(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT)
    hb = ax.hexbin(
        oxide.loc[temp_view, "T_K"],
        oxide.loc[temp_view, "residual_log10_pred_over_observed"],
        gridsize=45,
        mincnt=1,
        cmap="cividis",
        norm=LogNorm(),
    )
    tx, median, q25, q75 = binned_trend(oxide["T_K"], oxide["residual_log10_pred_over_observed"], bins=12)
    ax.fill_between(tx, q25, q75, color=COLORS["orange"], alpha=0.22)
    ax.plot(tx, median, color=COLORS["orange"], marker="o", linewidth=2.0, markersize=3.5)
    add_residual_reference_lines(ax, horizontal=True)
    ax.set(ylim=(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT), xlabel="Measurement temperature (K)", ylabel="Signed residual log10(predicted / observed)", title="C  Broad-oxide residual versus temperature")
    fig.colorbar(hb, ax=ax, label="Rows per hexagon", shrink=0.83)

    ax = axes[1, 1]
    s_view = temp_view & oxide["S_abs_uV_per_K"].between(0, 155)
    hb = ax.hexbin(
        oxide.loc[s_view, "S_abs_uV_per_K"],
        oxide.loc[s_view, "residual_log10_pred_over_observed"],
        gridsize=45,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm(),
    )
    tx, median, q25, q75 = binned_trend(oxide["S_abs_uV_per_K"], oxide["residual_log10_pred_over_observed"], bins=12)
    ax.fill_between(tx, q25, q75, color=COLORS["green"], alpha=0.20)
    ax.plot(tx, median, color=COLORS["green"], marker="o", linewidth=2.0, markersize=3.5)
    add_residual_reference_lines(ax, horizontal=True)
    ax.set(xlim=(0, 155), ylim=(-PLOT_RESIDUAL_LIMIT, PLOT_RESIDUAL_LIMIT), xlabel="Absolute Seebeck coefficient (µV/K)", ylabel="Signed residual log10(predicted / observed)", title="D  Broad-oxide residual versus |S|")
    fig.colorbar(hb, ax=ax, label="Rows per hexagon", shrink=0.83)

    save_figure(fig, figure_dir, FIGURE_STEMS[3])


def append_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> None:
    args = parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing Phase 1 residual source: {SOURCE}")
    if not PHASE1_CHECKS.is_file():
        raise FileNotFoundError(f"Missing Phase 1 validation file: {PHASE1_CHECKS}")

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.validation_log.parent.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    source_hash_before = sha256(SOURCE)
    source = pd.read_csv(SOURCE, low_memory=False)
    phase1_checks = pd.read_csv(PHASE1_CHECKS)
    required = {
        "row_id",
        "is_evaluable",
        "is_oxide_broad",
        "validation_sample_group_id",
        "validation_paper_group_id",
        "material_group_key_for_prediction",
        "T_K",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "T_bin_label",
        "S_abs_uV_per_K",
        "log10_sigma_observed_S_per_m",
        "log10_sigma_predicted_S_per_m",
        "residual_log10_pred_over_observed",
        "abs_residual_log10",
        "residual_band",
        "prediction_direction",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise KeyError(f"Phase 1 residual source missing columns: {missing}")
    if not phase1_checks["status"].eq("PASS").all():
        raise RuntimeError("Phase 1 validation is not fully PASS")

    rows = source.loc[source["is_evaluable"].eq(True)].copy()
    numeric_columns = [
        "T_K",
        "T_bin_left_K",
        "T_bin_right_K",
        "T_bin_center_K",
        "S_abs_uV_per_K",
        "log10_sigma_observed_S_per_m",
        "log10_sigma_predicted_S_per_m",
        "residual_log10_pred_over_observed",
        "abs_residual_log10",
    ]
    for column in numeric_columns:
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    valid_numeric = np.isfinite(rows[numeric_columns]).all(axis=1)
    if not valid_numeric.all():
        raise RuntimeError(f"Non-finite plot values in {int((~valid_numeric).sum())} evaluable rows")

    materials = material_summary(rows)
    temperatures = temperature_summary(rows)
    oxides = oxide_summary(rows)
    global_metrics = numeric_metrics(rows["residual_log10_pred_over_observed"])
    global_metrics.update(
        {
            "n_test_rows": int(len(source)),
            "n_evaluable_rows": int(len(rows)),
            "n_samples": int(rows["validation_sample_group_id"].nunique()),
            "n_papers": int(rows["validation_paper_group_id"].nunique()),
            "n_oxide_rows": int(rows["is_oxide_broad"].sum()),
            "n_factor_10_outliers": int((rows["abs_residual_log10"] > 1.0).sum()),
            "display_residual_limit": PLOT_RESIDUAL_LIMIT,
            "n_below_display_limit": int((rows["residual_log10_pred_over_observed"] < -PLOT_RESIDUAL_LIMIT).sum()),
            "n_above_display_limit": int((rows["residual_log10_pred_over_observed"] > PLOT_RESIDUAL_LIMIT).sum()),
        }
    )
    scope = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in global_metrics.items()]
    )

    materials.to_csv(args.data_dir / "phase2_material_visual_summary.csv", index=False)
    temperatures.to_csv(args.data_dir / "phase2_temperature_visual_summary.csv", index=False)
    oxides.to_csv(args.data_dir / "phase2_oxide_visual_summary.csv", index=False)
    scope.to_csv(args.data_dir / "phase2_visual_scope_summary.csv", index=False)

    figure_global_overview(rows, args.figure_dir)
    figure_material_profiles(materials, args.figure_dir)
    figure_temperature_profiles(temperatures, args.figure_dir)
    figure_oxide_focus(rows, args.figure_dir)

    source_hash_after = sha256(SOURCE)
    figure_records = []
    for stem in FIGURE_STEMS:
        for suffix in (".png", ".pdf"):
            path = args.figure_dir / f"{stem}{suffix}"
            figure_records.append(
                {
                    "figure_file": path.relative_to(ROOT).as_posix(),
                    "format": suffix.lstrip("."),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = pd.DataFrame(figure_records)
    manifest.to_csv(args.data_dir / "phase2_figure_manifest.csv", index=False)

    checks: list[dict[str, Any]] = []
    append_check(checks, "phase1_validation_all_pass", phase1_checks["status"].eq("PASS").all(), f"checks={len(phase1_checks)}")
    append_check(checks, "evaluable_row_count_preserved", len(rows) == 18968, f"rows={len(rows)}")
    append_check(checks, "material_summary_reconciles", int(materials["n_rows"].sum()) == len(rows), f"sum={int(materials['n_rows'].sum())}")
    append_check(checks, "temperature_summary_reconciles", int(temperatures["n_rows"].sum()) == len(rows), f"sum={int(temperatures['n_rows'].sum())}")
    append_check(checks, "oxide_summary_reconciles", int(oxides["n_rows"].sum()) == len(rows), f"sum={int(oxides['n_rows'].sum())}")
    append_check(checks, "all_figure_files_created", len(manifest) == 8 and manifest["size_bytes"].gt(10_000).all(), f"files={len(manifest)}")
    append_check(checks, "phase1_source_unchanged", source_hash_before == source_hash_after, f"before={source_hash_before}; after={source_hash_after}")
    validation = pd.DataFrame(checks)
    validation.to_csv(args.data_dir / "phase2_validation_checks.csv", index=False)
    input_manifest = pd.DataFrame(
        [
            {
                "source_file": SOURCE.relative_to(ROOT).as_posix(),
                "source_role": "phase1_primary_residual_rows",
                "size_bytes": SOURCE.stat().st_size,
                "sha256_before": source_hash_before,
                "sha256_after": source_hash_after,
                "source_unchanged_during_run": source_hash_before == source_hash_after,
            }
        ]
    )
    input_manifest.to_csv(args.data_dir / "phase2_input_manifest.csv", index=False)
    if not validation["status"].eq("PASS").all():
        raise RuntimeError(validation.loc[validation["status"].ne("PASS")].to_string(index=False))

    oxide_row = oxides.loc[oxides["is_oxide_broad"]].iloc[0]
    nonoxide_row = oxides.loc[~oxides["is_oxide_broad"]].iloc[0]
    worst_material = materials.iloc[0]
    sufficiently_supported = temperatures.loc[temperatures["n_rows"] >= 50]
    worst_temperature = sufficiently_supported.sort_values("mae_log10", ascending=False).iloc[0]
    report_lines = [
        "# exp007 Phase 2 residual visualization",
        "",
        "## Scope",
        "",
        "Phase 2 visualizes how the frozen broad-family 100 K conductivity prediction misses the observed values. It does not change the model or the material taxonomy.",
        "",
        f"- Evaluable rows: {len(rows):,}",
        f"- Evaluable samples: {rows['validation_sample_group_id'].nunique():,}",
        f"- Broad-oxide rows: {int(rows['is_oxide_broad'].sum()):,}",
        f"- Factor-10 outlier rows: {int((rows['abs_residual_log10'] > 1.0).sum()):,}",
        "",
        "## Figure set",
        "",
        "1. Global residual distribution, parity, residual-versus-observed relationship, and error bands.",
        "2. Broad material-group error, bias, and threshold accuracy.",
        "3. Error, bias, support, and threshold accuracy across 100 K temperature bins.",
        "4. Broad-oxide comparison and oxide residual relationships with conductivity, temperature, and |S|.",
        "",
        "## Descriptive observations",
        "",
        f"- The median signed residual is {global_metrics['median_residual_log10']:.3f}, while the mean is {global_metrics['mean_residual_log10']:.3f}; the large-error tail therefore affects the mean more strongly than the median.",
        f"- Overprediction and underprediction account for {global_metrics['overprediction_fraction']:.1%} and {global_metrics['underprediction_fraction']:.1%} of evaluable rows, respectively.",
        f"- Broad oxides have row-equal MAE {oxide_row['mae_log10']:.3f} and factor-10 accuracy {oxide_row['within_factor_10_fraction']:.1%}, compared with {nonoxide_row['mae_log10']:.3f} and {nonoxide_row['within_factor_10_fraction']:.1%} for non-oxides.",
        f"- The largest broad-group row-equal MAE is `{worst_material['material_group_key_for_prediction']}` at {worst_material['mae_log10']:.3f} across {int(worst_material['n_rows']):,} rows.",
        f"- Among temperature bins with at least 50 rows, `{worst_temperature['T_bin_label']}` has the largest row-equal MAE ({worst_temperature['mae_log10']:.3f}).",
        "",
        "These are descriptive patterns, not causal conclusions. The next phase should inspect the underlying material labels and data quality before changing the oxide taxonomy.",
        "",
        "## Plot-range policy",
        "",
        f"Residual-density panels show −{PLOT_RESIDUAL_LIMIT:g} to +{PLOT_RESIDUAL_LIMIT:g} log10 units so the main distribution remains readable. Counts outside that display range are retained in all metrics and are annotated in the global overview.",
        "",
        "## Validation",
        "",
        f"- Figure files created: {len(manifest)}",
        f"- Phase 1 input unchanged: {source_hash_before == source_hash_after}",
        "- All Phase 2 reconciliation checks: PASS",
    ]
    args.report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    log_lines = [
        "exp007 Phase 2 visualization validation",
        "========================================",
        *[
            f"{row.status} {row.check}: {row.detail}"
            for row in validation.itertuples(index=False)
        ],
        "",
        f"figure_files={len(manifest)}",
        f"evaluable_rows={len(rows)}",
        f"broad_oxide_rows={int(rows['is_oxide_broad'].sum())}",
        "Final status: PASS",
    ]
    args.validation_log.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Figures: {args.figure_dir}")
    print(f"Report: {args.report}")
    print(f"Validation: {args.validation_log}")
    print(f"Created {len(manifest)} figure files from {len(rows):,} evaluable rows")


if __name__ == "__main__":
    main()
