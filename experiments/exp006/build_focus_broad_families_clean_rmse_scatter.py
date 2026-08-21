import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_focus_broad_families_uniform_axis_scatter import (
    DEFAULT_BROAD_DATA,
    DEFAULT_OUTPUT,
    DEFAULT_SIGE_DATA,
    FIXED_LOWER,
    FIXED_UPPER,
    GROUP_SPECS,
    clean_rows,
    source_path,
)


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_FIGURES = (
    EXP_DIR
    / "figures"
    / "focus_broad_families_uniform_axes"
    / "material_panels_transparent"
)
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "focus_broad_families_uniform_axes"
    / "focus_broad_families_rmse_fixed_axes_1e2_1e8.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one clean, transparent-outside scatter plot per material "
            "classification and write RMSE values to a separate table."
        )
    )
    parser.add_argument("--broad-data", type=Path, default=DEFAULT_BROAD_DATA)
    parser.add_argument("--sige-data", type=Path, default=DEFAULT_SIGE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--axis-lower", type=float, default=FIXED_LOWER)
    parser.add_argument("--axis-upper", type=float, default=FIXED_UPPER)
    return parser.parse_args()


def rows_visible_in_axes(
    rows: pd.DataFrame, lower: float, upper: float
) -> pd.DataFrame:
    visible = rows["sigma_S_per_m"].between(lower, upper) & rows[
        "sigma_pred_S_per_m"
    ].between(lower, upper)
    return rows.loc[visible].copy()


def rmse_log10(rows: pd.DataFrame) -> float:
    residual = np.log10(
        rows["sigma_pred_S_per_m"].to_numpy(dtype=float)
        / rows["sigma_S_per_m"].to_numpy(dtype=float)
    )
    return float(np.sqrt(np.mean(residual**2)))


def save_clean_scatter(
    rows: pd.DataFrame,
    classification_ja: str,
    png_path: Path,
    pdf_path: Path,
    lower: float,
    upper: float,
) -> None:
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # Only the plotting field is opaque white. The surrounding canvas is RGBA
    # transparent so the figure can be placed on any document background.
    fig.patch.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor("white")

    markers = {"p": "o", "n": "^"}
    colors = {"p": "#1f77b4", "n": "#d62728"}
    for carrier in ["p", "n"]:
        carrier_rows = rows[rows["carrier_type"].astype("string").eq(carrier)]
        if carrier_rows.empty:
            continue
        ax.scatter(
            carrier_rows["sigma_S_per_m"],
            carrier_rows["sigma_pred_S_per_m"],
            s=20,
            alpha=0.68,
            marker=markers[carrier],
            color=colors[carrier],
            edgecolors="none",
            label=f"{carrier}-type",
            rasterized=True,
        )

    line = np.geomspace(lower, upper, 400)
    ax.plot(line, line, color="black", linewidth=1.25, label="y=x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, which="both", linewidth=0.35, alpha=0.27)
    ax.tick_params(axis="both", which="major", labelsize=10, length=4)
    ax.tick_params(axis="both", which="minor", length=2.5)

    # Keep material identification within the white plotting field.
    ax.text(
        0.035,
        0.965,
        classification_ja,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="black",
    )
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
        handlelength=2.2,
    )
    fig.subplots_adjust(left=0.145, right=0.975, bottom=0.115, top=0.975)
    fig.savefig(
        png_path,
        dpi=240,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor="none",
        transparent=False,
    )
    fig.savefig(
        pdf_path,
        dpi=300,
        facecolor=(0.0, 0.0, 0.0, 0.0),
        edgecolor="none",
        transparent=False,
    )
    plt.close(fig)


def write_report(report_path: Path, rmse_table: pd.DataFrame, csv_path: Path) -> None:
    lines = [
        "# 材料系別散布図のRMSE（固定軸範囲）",
        "",
        "実測値と予測値の両方が10^2～10^8 S/mに入る点へ先に絞り、",
        "その点について`log10(予測値/実測値)`の二乗平均平方根を計算した。0に近いほどy=xに近い。",
        "RMSE値は散布図内には記載せず、この表とCSVに分離した。",
        "",
        f"CSV: `{csv_path}`",
        "",
        "| 材料系 | 使用点数 | RMSE (log10) |",
        "|---|---:|---:|",
    ]
    for _, row in rmse_table.iterrows():
        lines.append(
            f"| {row['material_classification_ja']} | {int(row['n_points_used'])} | "
            f"{row['rmse_log10_pred_over_exp']:.3f} |"
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.axis_lower < args.axis_upper):
        raise ValueError("axis limits must satisfy 0 < lower < upper")
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    rmse_rows = []
    figure_rows = []
    for material_order, (
        classification_ja,
        group_key,
        safe_name,
        source_kind,
    ) in enumerate(GROUP_SPECS, start=1):
        path = source_path(args.broad_data, args.sige_data, safe_name, source_kind)
        if not path.exists():
            raise FileNotFoundError(f"missing source for {classification_ja}: {path}")
        all_rows = clean_rows(pd.read_csv(path, low_memory=False))
        plotted_rows = rows_visible_in_axes(
            all_rows, args.axis_lower, args.axis_upper
        )
        if plotted_rows.empty:
            raise ValueError(f"no points inside fixed axes for {classification_ja}")

        base = f"{safe_name}_scatter_pred_vs_exp_all_clean_transparent"
        png_path = args.figures / f"{base}.png"
        pdf_path = args.figures / f"{base}.pdf"
        save_clean_scatter(
            plotted_rows,
            classification_ja,
            png_path,
            pdf_path,
            args.axis_lower,
            args.axis_upper,
        )
        rmse_rows.append(
            {
                "material_order": material_order,
                "material_classification_ja": classification_ja,
                "n_points_used": len(plotted_rows),
                "rmse_log10_pred_over_exp": rmse_log10(plotted_rows),
            }
        )
        figure_rows.append(
            {
                "material_order": material_order,
                "material_classification_ja": classification_ja,
                "material_group_key": group_key,
                "safe_group_name": safe_name,
                "n_rows_source": len(all_rows),
                "n_rows_plotted": len(plotted_rows),
                "axis_lower_S_per_m": args.axis_lower,
                "axis_upper_S_per_m": args.axis_upper,
                "source_csv": str(path),
                "figure_path_png": str(png_path),
                "figure_path_pdf": str(pdf_path),
            }
        )
        print(f"[clean_rmse_scatter] created: {classification_ja}", flush=True)

    rmse_table = pd.DataFrame(rmse_rows)
    rmse_path = (
        args.output / "focus_broad_families_rmse_fixed_axes_1e2_1e8.csv"
    )
    rmse_table.to_csv(rmse_path, index=False, encoding="utf-8-sig")
    figure_index = pd.DataFrame(figure_rows)
    figure_index_path = (
        args.output / "focus_broad_families_clean_transparent_figure_index.csv"
    )
    figure_index.to_csv(figure_index_path, index=False, encoding="utf-8-sig")
    write_report(args.report, rmse_table, rmse_path)

    if len(rmse_table) != 15 or len(figure_index) != 15:
        raise ValueError("expected exactly 15 material-classification outputs")
    if not np.isfinite(rmse_table["rmse_log10_pred_over_exp"]).all():
        raise ValueError("RMSE table contains non-finite values")
    for column in ["figure_path_png", "figure_path_pdf"]:
        for raw_path in figure_index[column]:
            output_path = Path(raw_path)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ValueError(f"missing or empty figure: {output_path}")
    for path in [rmse_path, figure_index_path, args.report]:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty output: {path}")

    print(f"[clean_rmse_scatter] figures: {len(figure_index)} PNG + PDF", flush=True)
    print(f"[clean_rmse_scatter] RMSE: {rmse_path}", flush=True)
    print(f"[clean_rmse_scatter] report: {args.report}", flush=True)


if __name__ == "__main__":
    main()
