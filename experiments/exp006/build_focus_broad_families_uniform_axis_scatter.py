import argparse
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent
DEFAULT_BROAD_DATA = EXP_DIR / "data" / "processed" / "focus_broad_families"
DEFAULT_SIGE_DATA = EXP_DIR / "data" / "processed" / "focus_sige_like"
DEFAULT_OUTPUT = (
    EXP_DIR / "data" / "processed" / "focus_broad_families_uniform_axes"
)
DEFAULT_FIGURES = EXP_DIR / "figures" / "focus_broad_families_uniform_axes"
DEFAULT_REPORT = (
    EXP_DIR
    / "reports"
    / "focus_broad_families_uniform_axes"
    / "focus_broad_families_uniform_axes_report.md"
)

FIXED_LOWER = 1.0e2
FIXED_UPPER = 1.0e8
OUTSIDE_MAIN_THRESHOLD = 0.5

# The order and labels follow the 15 material classifications requested for exp006.
GROUP_SPECS = [
    ("Si-Ge系", "broad::SiGe_like", "focus_sige_like", "sige"),
    ("Sn-Te系", "broad::SnTe_like", "broad_SnTe_like", "broad"),
    ("Pb-Te系", "broad::PbTe_like", "broad_PbTe_like", "broad"),
    ("Bi-Te系", "broad::BiTe_like", "broad_BiTe_like", "broad"),
    ("Sb-Te系", "broad::SbTe_like", "broad_SbTe_like", "broad"),
    (
        "Bi-Sb-Te系",
        "broad::BiSbTe_tetradymite_like",
        "broad_BiSbTe_tetradymite_like",
        "broad",
    ),
    ("Ge-Te系", "broad::GeTe_like", "broad_GeTe_like", "broad"),
    ("Mg-Si/Sn系", "broad::Mg2SiSn_like", "broad_Mg2SiSn_like", "broad"),
    (
        "Co-Sb系・スクッテルダイト系",
        "broad::CoSb_skutterudite_like",
        "broad_CoSb_skutterudite_like",
        "broad",
    ),
    ("セレン化物系", "broad::selenide", "broad_selenide", "broad"),
    ("テルル化物系", "broad::telluride", "broad_telluride", "broad"),
    ("硫化物系", "broad::sulfide", "broad_sulfide", "broad"),
    ("酸化物系", "broad::oxide", "broad_oxide", "broad"),
    (
        "その他の化学式系",
        "broad::other_formula_system",
        "broad_other_formula_system",
        "broad",
    ),
    ("分類不能", "unknown_material_group", "unknown_material_group", "broad"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fixed-axis measured-versus-predicted conductivity scatter "
            "plots for all 15 focus material classifications."
        )
    )
    parser.add_argument("--broad-data", type=Path, default=DEFAULT_BROAD_DATA)
    parser.add_argument("--sige-data", type=Path, default=DEFAULT_SIGE_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--axis-lower", type=float, default=FIXED_LOWER)
    parser.add_argument("--axis-upper", type=float, default=FIXED_UPPER)
    parser.add_argument(
        "--outside-main-threshold",
        type=float,
        default=OUTSIDE_MAIN_THRESHOLD,
        help="Create an additional full-data-range figure when outside fraction exceeds this value.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[uniform_axes] {message}", flush=True)


def source_path(
    broad_data: Path, sige_data: Path, safe_name: str, source_kind: str
) -> Path:
    if source_kind == "sige":
        return sige_data / "focus_sige_like_prediction_rows.csv"
    return broad_data / f"focus_broad_families_prediction_rows_{safe_name}.csv"


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = ["sigma_S_per_m", "sigma_pred_S_per_m", "carrier_type"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"input is missing required columns: {missing}")
    clean = df.copy()
    clean["sigma_S_per_m"] = pd.to_numeric(
        clean["sigma_S_per_m"], errors="coerce"
    )
    clean["sigma_pred_S_per_m"] = pd.to_numeric(
        clean["sigma_pred_S_per_m"], errors="coerce"
    )
    valid = (
        np.isfinite(clean["sigma_S_per_m"])
        & np.isfinite(clean["sigma_pred_S_per_m"])
        & clean["sigma_S_per_m"].gt(0)
        & clean["sigma_pred_S_per_m"].gt(0)
        & clean["carrier_type"].astype("string").isin(["p", "n"])
    )
    return clean.loc[valid].copy()


def subset_rows(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "all":
        return df.copy()
    return df[df["carrier_type"].astype("string").eq(subset)].copy()


def fixed_range_counts(
    df: pd.DataFrame, lower: float, upper: float
) -> tuple[int, int, float]:
    inside = df["sigma_S_per_m"].between(lower, upper) & df[
        "sigma_pred_S_per_m"
    ].between(lower, upper)
    n_inside = int(inside.sum())
    n_outside = int(len(df) - n_inside)
    outside_fraction = float(n_outside / len(df)) if len(df) else float("nan")
    return n_inside, n_outside, outside_fraction


def error_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    errors = np.log10(
        df["sigma_pred_S_per_m"].to_numpy(dtype=float)
        / df["sigma_S_per_m"].to_numpy(dtype=float)
    )
    mae = float(np.mean(np.abs(errors)))
    factor2 = float(np.mean(np.abs(errors) <= math.log10(2.0)))
    factor10 = float(np.mean(np.abs(errors) <= 1.0))
    return mae, factor2, factor10


def full_data_limits(df: pd.DataFrame) -> tuple[float, float]:
    values = np.concatenate(
        [
            df["sigma_S_per_m"].to_numpy(dtype=float),
            df["sigma_pred_S_per_m"].to_numpy(dtype=float),
        ]
    )
    lower = 10.0 ** math.floor(math.log10(float(values.min())))
    upper = 10.0 ** math.ceil(math.log10(float(values.max())))
    if lower == upper:
        upper *= 10.0
    return lower, upper


def display_number(value: float) -> str:
    exponent = int(round(math.log10(value)))
    if math.isclose(value, 10.0**exponent):
        return f"10^{exponent}"
    return f"{value:.3g}"


def save_scatter(
    df: pd.DataFrame,
    title: str,
    png_path: Path,
    pdf_path: Path,
    lower: float,
    upper: float,
    n_inside_fixed: int,
    fixed_lower: float,
    fixed_upper: float,
    range_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    markers = {"p": "o", "n": "^"}
    colors = {"p": "#1f77b4", "n": "#d62728"}
    for carrier in ["p", "n"]:
        carrier_rows = df[df["carrier_type"].astype("string").eq(carrier)]
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
        )

    line = np.geomspace(lower, upper, 400)
    ax.plot(line, line, color="black", linewidth=1.2, label="y=x")
    ax.plot(line, line * 2.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.plot(line, line / 2.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.plot(line, line * 10.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.45)
    ax.plot(line, line / 10.0, color="gray", linewidth=0.7, linestyle=":", alpha=0.45)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Measured conductivity, $\sigma_{\mathrm{exp}}$ (S/m)")
    ax.set_ylabel(r"Predicted conductivity, $\sigma_{\mathrm{pred}}$ (S/m)")

    mae, factor2, factor10 = error_metrics(df)
    ax.set_title(
        f"{title}\n"
        f"{range_label}: x/y {display_number(lower)} to {display_number(upper)} S/m | "
        f"inside fixed range={n_inside_fixed}/{len(df)}\n"
        f"MAE(log10)={mae:.3f} | factor2={100.0 * factor2:.1f}% | "
        f"factor10={100.0 * factor10:.1f}%",
        fontsize=10,
        pad=8,
    )
    ax.grid(True, which="both", linewidth=0.4, alpha=0.28)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def write_report(
    report_path: Path,
    index: pd.DataFrame,
    lower: float,
    upper: float,
    threshold: float,
) -> None:
    fixed = index[index["axis_mode"].eq("fixed")]
    extra = index[index["axis_mode"].eq("full_data_range")]
    lines = [
        "# Focus broad families: uniform-axis scatter plots",
        "",
        "## Settings",
        "",
        f"- Fixed x/y range: `{lower:g}` to `{upper:g}` S/m",
        f"- Additional full-data-range threshold: outside fraction > `{threshold:g}`",
        "- Both measured and predicted conductivity must be inside the fixed range for a point to count as inside.",
        "",
        "## Outputs",
        "",
        f"- Fixed-axis plots: {len(fixed)}",
        f"- Additional full-data-range plots: {len(extra)}",
        "- Each plot is saved as PNG and PDF.",
        "",
        "## Fixed-range coverage",
        "",
        "| Material classification | Subset | Total | Inside | Outside | Inside fraction | Additional plot |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fixed.iterrows():
        lines.append(
            f"| {row['material_classification_ja']} | {row['carrier_subset']} | "
            f"{int(row['n_rows'])} | {int(row['n_inside_fixed'])} | "
            f"{int(row['n_outside_fixed'])} | {100.0 * row['inside_fraction_fixed']:.1f}% | "
            f"{'yes' if row['outside_is_main'] else 'no'} |"
        )
    if extra.empty:
        lines.extend(
            [
                "",
                "No classification/subset currently has more than half of its points outside the fixed range, so no additional full-data-range plot was needed.",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if not (args.axis_lower > 0 and args.axis_upper > args.axis_lower):
        raise ValueError("axis limits must satisfy 0 < lower < upper")
    if not 0.0 <= args.outside_main_threshold <= 1.0:
        raise ValueError("outside-main-threshold must be between 0 and 1")


def main() -> None:
    start = time.perf_counter()
    args = parse_args()
    validate_args(args)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, Any]] = []
    for classification_ja, group_key, safe_name, source_kind in GROUP_SPECS:
        path = source_path(args.broad_data, args.sige_data, safe_name, source_kind)
        if not path.exists():
            raise FileNotFoundError(f"missing source for {classification_ja}: {path}")
        rows = clean_rows(pd.read_csv(path, low_memory=False))
        if rows.empty:
            raise ValueError(f"no valid rows for {classification_ja}: {path}")

        figure_dir = args.figures / safe_name
        figure_dir.mkdir(parents=True, exist_ok=True)
        for subset in ["all", "p", "n"]:
            current = subset_rows(rows, subset)
            if current.empty:
                log(f"skip {classification_ja} {subset}: no rows")
                continue
            n_inside, n_outside, outside_fraction = fixed_range_counts(
                current, args.axis_lower, args.axis_upper
            )
            outside_is_main = outside_fraction > args.outside_main_threshold
            base = f"{safe_name}_scatter_pred_vs_exp_{subset}_fixed_1e2_1e8"
            png_path = figure_dir / f"{base}.png"
            pdf_path = figure_dir / f"{base}.pdf"
            title = f"{group_key} {subset}"
            save_scatter(
                current,
                title,
                png_path,
                pdf_path,
                args.axis_lower,
                args.axis_upper,
                n_inside,
                args.axis_lower,
                args.axis_upper,
                "fixed axes",
            )
            common = {
                "material_classification_ja": classification_ja,
                "material_group_key": group_key,
                "safe_group_name": safe_name,
                "carrier_subset": subset,
                "n_rows": len(current),
                "n_inside_fixed": n_inside,
                "n_outside_fixed": n_outside,
                "inside_fraction_fixed": n_inside / len(current),
                "outside_fraction_fixed": outside_fraction,
                "outside_is_main": outside_is_main,
                "source_csv": str(path),
            }
            index_rows.append(
                {
                    **common,
                    "axis_mode": "fixed",
                    "axis_lower_S_per_m": args.axis_lower,
                    "axis_upper_S_per_m": args.axis_upper,
                    "figure_path_png": str(png_path),
                    "figure_path_pdf": str(pdf_path),
                }
            )
            log(
                f"created {classification_ja} {subset}: "
                f"inside={n_inside}/{len(current)}, outside={n_outside}/{len(current)}"
            )

            if outside_is_main:
                full_lower, full_upper = full_data_limits(current)
                full_base = f"{safe_name}_scatter_pred_vs_exp_{subset}_full_data_range"
                full_png_path = figure_dir / f"{full_base}.png"
                full_pdf_path = figure_dir / f"{full_base}.pdf"
                save_scatter(
                    current,
                    title,
                    full_png_path,
                    full_pdf_path,
                    full_lower,
                    full_upper,
                    n_inside,
                    args.axis_lower,
                    args.axis_upper,
                    "full data range",
                )
                index_rows.append(
                    {
                        **common,
                        "axis_mode": "full_data_range",
                        "axis_lower_S_per_m": full_lower,
                        "axis_upper_S_per_m": full_upper,
                        "figure_path_png": str(full_png_path),
                        "figure_path_pdf": str(full_pdf_path),
                    }
                )
                log(f"created additional full-data-range plot: {classification_ja} {subset}")

    index = pd.DataFrame(index_rows)
    index_path = args.output / "focus_broad_families_uniform_axes_figure_index.csv"
    index.to_csv(index_path, index=False, encoding="utf-8-sig")
    write_report(
        args.report,
        index,
        args.axis_lower,
        args.axis_upper,
        args.outside_main_threshold,
    )

    fixed = index[index["axis_mode"].eq("fixed")]
    expected_classifications = {spec[0] for spec in GROUP_SPECS}
    if set(fixed["material_classification_ja"]) != expected_classifications:
        raise ValueError("not all 15 material classifications have a fixed-axis plot")
    if not (fixed["axis_lower_S_per_m"].eq(args.axis_lower)).all():
        raise ValueError("fixed-axis lower limits are inconsistent")
    if not (fixed["axis_upper_S_per_m"].eq(args.axis_upper)).all():
        raise ValueError("fixed-axis upper limits are inconsistent")
    for column in ["figure_path_png", "figure_path_pdf"]:
        for raw_path in index[column]:
            output_path = Path(raw_path)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ValueError(f"missing or empty figure: {output_path}")
    if not args.report.exists() or args.report.stat().st_size == 0:
        raise ValueError(f"missing or empty report: {args.report}")

    log(f"fixed-axis plots: {len(fixed)}")
    log(f"additional full-data-range plots: {len(index) - len(fixed)}")
    log(f"figure index: {index_path}")
    log(f"report: {args.report}")
    log(f"elapsed seconds: {time.perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
