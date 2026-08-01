import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "starrydata2_step5_core_properties"
    / "property_core_curves_step5.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "output"
    / "starrydata2_step27_sigma_temperature_comparison"
)

SIGMA_PROPERTY = "Electrical conductivity"
RHO_PROPERTY = "Electrical resistivity"
TARGET_PROPERTIES = {SIGMA_PROPERTY, RHO_PROPERTY}
ELEMENT_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Starrydata2 temperature-dependent electrical conductivity curves "
            "for Si-Ge compositions and Pb-Te based comparison records."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--pbte-top-n", type=int, default=12)
    parser.add_argument(
        "--y-quantile-low",
        type=float,
        default=0.01,
        help="Lower quantile used for readable y-axis limits.",
    )
    parser.add_argument(
        "--y-quantile-high",
        type=float,
        default=0.995,
        help="Upper quantile used for readable y-axis limits.",
    )
    parser.add_argument(
        "--no-log-y",
        action="store_true",
        help="Use a linear conductivity axis instead of the default log scale.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "null"}:
        return ""
    return text


def parse_numeric_list(raw_value: Any) -> list[float]:
    if isinstance(raw_value, (list, tuple)):
        values = raw_value
    else:
        text = normalize_text(raw_value)
        if not text:
            return []
        try:
            values = json.loads(text)
        except json.JSONDecodeError:
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                return []

    if not isinstance(values, (list, tuple)):
        return []

    parsed: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            parsed.append(numeric)
    return parsed


def parse_elements(composition: Any) -> Counter[str]:
    text = normalize_text(composition).replace(" ", "")
    elements: Counter[str] = Counter()
    for element, amount_text in ELEMENT_RE.findall(text):
        amount = float(amount_text) if amount_text else 1.0
        elements[element] += amount
    return elements


def fraction_label(prefix: str, element_a: str, frac_a: float, element_b: str, frac_b: float) -> str:
    return f"{prefix} {element_a}{frac_a:.3g}{element_b}{frac_b:.3g}"


def classify_composition(composition: Any) -> dict[str, Any]:
    elements = parse_elements(composition)
    output: dict[str, Any] = {
        "composition_group": "other",
        "composition_ratio_label": "",
        "si_frac": np.nan,
        "ge_frac": np.nan,
        "pb_frac_in_pbte": np.nan,
        "te_frac_in_pbte": np.nan,
    }
    if not elements:
        return output

    if set(elements) == {"Si", "Ge"}:
        total = elements["Si"] + elements["Ge"]
        if total > 0:
            si_frac = elements["Si"] / total
            ge_frac = elements["Ge"] / total
            output.update(
                {
                    "composition_group": "SiGe",
                    "composition_ratio_label": fraction_label("SiGe", "Si", si_frac, "Ge", ge_frac),
                    "si_frac": si_frac,
                    "ge_frac": ge_frac,
                }
            )
        return output

    if {"Pb", "Te"}.issubset(elements):
        total = elements["Pb"] + elements["Te"]
        if total > 0:
            pb_frac = elements["Pb"] / total
            te_frac = elements["Te"] / total
            output.update(
                {
                    "composition_group": "PbTe-based",
                    "composition_ratio_label": fraction_label(
                        "PbTe-based", "Pb", pb_frac, "Te", te_frac
                    ),
                    "pb_frac_in_pbte": pb_frac,
                    "te_frac_in_pbte": te_frac,
                }
            )
        return output

    return output


def conductivity_factor(unit_y: Any) -> float:
    unit = normalize_text(unit_y).casefold().replace(" ", "")
    if unit in {"ohm^(-1)*m^(-1)", "s*m^(-1)", "s/m", "a**2*s**2/kg/m**2"}:
        return 1.0
    if unit in {"ohm^(-1)*cm^(-1)", "s*cm^(-1)", "s/cm"}:
        return 100.0
    return math.nan


def resistivity_factor(unit_y: Any) -> float:
    unit = normalize_text(unit_y).casefold().replace(" ", "")
    if unit in {"ohm*m", "kg*m**3/a**2/s**3"}:
        return 1.0
    if unit in {"ohm*cm"}:
        return 0.01
    return math.nan


def normalize_sigma(
    x_values: list[float], y_values: list[float], property_name: Any, unit_y: Any
) -> tuple[np.ndarray, np.ndarray, str]:
    n = min(len(x_values), len(y_values))
    if n < 2:
        return np.asarray([]), np.asarray([]), "too_few_points"

    x = np.asarray(x_values[:n], dtype=float)
    y = np.asarray(y_values[:n], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.asarray([]), np.asarray([]), "too_few_finite_points"

    property_text = normalize_text(property_name)
    if property_text == SIGMA_PROPERTY:
        factor = conductivity_factor(unit_y)
        if not math.isfinite(factor):
            return np.asarray([]), np.asarray([]), "unsupported_sigma_unit"
        sigma = y * factor
    elif property_text == RHO_PROPERTY:
        factor = resistivity_factor(unit_y)
        if not math.isfinite(factor):
            return np.asarray([]), np.asarray([]), "unsupported_rho_unit"
        rho = y * factor
        valid = np.isfinite(rho) & (rho > 0)
        x = x[valid]
        rho = rho[valid]
        sigma = 1.0 / rho
    else:
        return np.asarray([]), np.asarray([]), "not_sigma_or_rho"

    valid = np.isfinite(x) & np.isfinite(sigma) & (x > 0) & (sigma > 0)
    x = x[valid]
    sigma = sigma[valid]
    if len(x) < 2:
        return np.asarray([]), np.asarray([]), "too_few_positive_points"

    order = np.argsort(x)
    x = x[order]
    sigma = sigma[order]
    unique_t: list[float] = []
    unique_sigma: list[float] = []
    for t_value in np.unique(x):
        same_t = x == t_value
        unique_t.append(float(t_value))
        unique_sigma.append(float(np.mean(sigma[same_t])))

    if len(unique_t) < 2:
        return np.asarray([]), np.asarray([]), "too_few_unique_temperatures"
    return np.asarray(unique_t), np.asarray(unique_sigma), "ok"


def load_curves(input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    use_columns = [
        "curve_id",
        "curve_key",
        "sample_key",
        "SID",
        "DOI",
        "sample_id",
        "paper_title",
        "year",
        "composition",
        "material_system",
        "n_or_p",
        "figure_id",
        "prop_x",
        "property_step5",
        "unit_y",
        "x_values_json",
        "y_values_json",
    ]
    df = pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        usecols=lambda column: column in use_columns,
        low_memory=False,
    )
    required = {"composition", "property_step5", "prop_x", "x_values_json", "y_values_json"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"input missing required columns: {sorted(missing)}")

    df = df[
        df["property_step5"].isin(TARGET_PROPERTIES)
        & df["prop_x"].map(normalize_text).eq("Temperature")
    ].copy()

    group_info = df["composition"].apply(classify_composition).apply(pd.Series)
    df = pd.concat([df, group_info], axis=1)
    df = df[df["composition_group"].isin(["SiGe", "PbTe-based"])].copy()

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        x_values = parse_numeric_list(row["x_values_json"])
        y_values = parse_numeric_list(row["y_values_json"])
        t_values, sigma_values, status = normalize_sigma(
            x_values, y_values, row["property_step5"], row.get("unit_y", "")
        )
        base = row.drop(labels=["x_values_json", "y_values_json"]).to_dict()
        if status != "ok":
            skipped.append({**base, "skip_reason": status})
            continue
        records.append(
            {
                **base,
                "curve_index": len(records) + 1,
                "source_property": row["property_step5"],
                "temperature_K": t_values.tolist(),
                "sigma_S_per_m": sigma_values.tolist(),
                "T_min_K": float(t_values[0]),
                "T_max_K": float(t_values[-1]),
                "sigma_min_S_per_m": float(np.min(sigma_values)),
                "sigma_max_S_per_m": float(np.max(sigma_values)),
                "n_points_normalized": int(len(t_values)),
            }
        )
    return pd.DataFrame(records), pd.DataFrame(skipped)


def sorted_labels(df: pd.DataFrame, group_name: str) -> list[str]:
    subset = df[df["composition_group"].eq(group_name)].copy()
    if subset.empty:
        return []
    if group_name == "SiGe":
        summary = (
            subset.groupby("composition_ratio_label", dropna=False)["si_frac"]
            .median()
            .sort_values(kind="mergesort")
        )
        return summary.index.astype(str).tolist()
    counts = subset["composition_ratio_label"].value_counts()
    return counts.index.astype(str).tolist()


def plot_curves_by_label(
    curves: pd.DataFrame,
    output_path: Path,
    title: str,
    label_column: str,
    labels: list[str],
    *,
    log_y: bool,
    dpi: int,
    y_quantile_low: float,
    y_quantile_high: float,
    line_styles: dict[str, str] | None = None,
    line_colors: dict[str, Any] | None = None,
    median_colors: dict[str, Any] | None = None,
    line_alphas: dict[str, float] | None = None,
    line_widths: dict[str, float] | None = None,
    median_widths: dict[str, float] | None = None,
    median_line_styles: dict[str, str] | None = None,
    font_size: float = 10.0,
    title_font_size: float | None = None,
    label_font_size: float | None = None,
    tick_label_size: float | None = None,
    legend_font_size: float = 8.0,
    legend_line_width: float = 1.6,
    show_median_in_legend: bool = False,
    show_median: bool = True,
) -> None:
    if curves.empty:
        return

    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}
    if line_colors:
        colors.update(line_colors)

    fig, ax = plt.subplots(figsize=(13.5, 7))
    legend_handles: dict[str, Any] = {}
    median_handles: dict[str, Any] = {}
    for row in curves.itertuples(index=False):
        label = str(getattr(row, label_column))
        color = colors.get(label, "0.5")
        style = "-"
        if line_styles:
            style = line_styles.get(label, "-")
        alpha = line_alphas.get(label, 0.16) if line_alphas else 0.16
        linewidth = line_widths.get(label, 0.7) if line_widths else 0.7
        handle = ax.plot(
            getattr(row, "temperature_K"),
            getattr(row, "sigma_S_per_m"),
            color=color,
            linestyle=style,
            linewidth=linewidth,
            alpha=alpha,
        )[0]
        legend_handles.setdefault(
            label,
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=style,
                linewidth=legend_line_width,
                alpha=1.0,
            ),
        )

    if show_median:
        for label in labels:
            subset = curves[curves[label_column].astype(str).eq(label)]
            if subset.empty:
                continue
            grid_min = float(subset["T_min_K"].min())
            grid_max = float(subset["T_max_K"].max())
            if not math.isfinite(grid_min) or not math.isfinite(grid_max) or grid_max <= grid_min:
                continue
            grid = np.linspace(grid_min, grid_max, 180)
            interpolated: list[np.ndarray] = []
            for row in subset.itertuples(index=False):
                t_values = np.asarray(getattr(row, "temperature_K"), dtype=float)
                sigma_values = np.asarray(getattr(row, "sigma_S_per_m"), dtype=float)
                mask = (grid >= t_values[0]) & (grid <= t_values[-1])
                values = np.full_like(grid, np.nan, dtype=float)
                values[mask] = np.interp(grid[mask], t_values, sigma_values)
                interpolated.append(values)
            if not interpolated:
                continue
            stacked = np.vstack(interpolated)
            valid_count = np.isfinite(stacked).sum(axis=0)
            median_values = np.nanmedian(stacked, axis=0)
            median_values[valid_count < 2] = np.nan
            if np.isfinite(median_values).sum() < 2:
                continue
            median_color = (
                median_colors.get(label, colors.get(label, "0.5"))
                if median_colors
                else colors.get(label, "0.5")
            )
            median_style = median_line_styles.get(label, "-") if median_line_styles else "-"
            median_width = median_widths.get(label, 3.2) if median_widths else 3.2
            ax.plot(
                grid,
                median_values,
                color=median_color,
                linestyle=median_style,
                linewidth=median_width,
                alpha=1.0,
                zorder=10,
            )
            median_handles[label] = Line2D(
                [0],
                [0],
                color=median_color,
                linestyle=median_style,
                linewidth=max(median_width, legend_line_width),
                alpha=1.0,
            )

    ax.set_title(title, fontsize=title_font_size or font_size + 2)
    ax.set_xlabel("Temperature [K]", fontsize=label_font_size or font_size)
    ax.set_ylabel("Electrical conductivity [S/m]", fontsize=label_font_size or font_size)
    ax.tick_params(axis="both", labelsize=tick_label_size or font_size)
    if log_y:
        ax.set_yscale("log")
    set_readable_y_limits(curves, ax, y_quantile_low, y_quantile_high, log_y=log_y)
    ax.grid(True, alpha=0.25)

    handles = [legend_handles[label] for label in labels if label in legend_handles]
    shown_labels = [
        f"{label} (n={int(curves[curves[label_column].astype(str).eq(label)].shape[0])})"
        for label in labels
        if label in legend_handles
    ]
    if show_median and show_median_in_legend:
        for label in labels:
            if label not in median_handles:
                continue
            handles.append(median_handles[label])
            shown_labels.append(f"{label} trend")
    if len(shown_labels) > 6:
        ax.legend(
            handles,
            shown_labels,
            fontsize=legend_font_size,
            frameon=False,
            ncol=1,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.77, 1.0))
    else:
        ax.legend(
            handles,
            shown_labels,
            fontsize=legend_font_size,
            frameon=False,
            ncol=1,
            loc="best",
        )
        fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def set_readable_y_limits(
    curves: pd.DataFrame,
    ax: plt.Axes,
    y_quantile_low: float,
    y_quantile_high: float,
    *,
    log_y: bool,
) -> None:
    values: list[float] = []
    for row in curves.itertuples(index=False):
        values.extend(float(value) for value in getattr(row, "sigma_S_per_m"))
    if not values:
        return

    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) < 2:
        return

    low_q = min(max(y_quantile_low, 0.0), 0.49)
    high_q = min(max(y_quantile_high, 0.51), 1.0)
    y_low = float(np.quantile(y, low_q))
    y_high = float(np.quantile(y, high_q))
    if not math.isfinite(y_low) or not math.isfinite(y_high) or y_high <= y_low:
        return

    if log_y:
        y_low = max(y_low / 1.4, np.min(y) * 0.9)
        y_high = y_high * 1.4
    else:
        padding = (y_high - y_low) * 0.06
        y_low = max(0.0, y_low - padding)
        y_high = y_high + padding
    ax.set_ylim(y_low, y_high)


def write_curve_map(curves: pd.DataFrame, output_dir: Path) -> None:
    output = curves.copy()
    output["temperature_K"] = output["temperature_K"].apply(json.dumps)
    output["sigma_S_per_m"] = output["sigma_S_per_m"].apply(json.dumps)
    output.to_csv(output_dir / "step27_selected_curve_map.csv", index=False)


def write_report(curves: pd.DataFrame, skipped: pd.DataFrame, output_dir: Path) -> None:
    rows = [
        ("selected_curves", len(curves)),
        ("skipped_curves", len(skipped)),
        ("sige_curves", int(curves["composition_group"].eq("SiGe").sum()) if not curves.empty else 0),
        (
            "pbte_based_curves",
            int(curves["composition_group"].eq("PbTe-based").sum()) if not curves.empty else 0,
        ),
        (
            "sige_composition_ratio_groups",
            int(curves.loc[curves["composition_group"].eq("SiGe"), "composition_ratio_label"].nunique())
            if not curves.empty
            else 0,
        ),
        (
            "pbte_composition_ratio_groups",
            int(
                curves.loc[
                    curves["composition_group"].eq("PbTe-based"), "composition_ratio_label"
                ].nunique()
            )
            if not curves.empty
            else 0,
        ),
    ]
    if not skipped.empty:
        for reason, count in skipped["skip_reason"].value_counts().items():
            rows.append((f"skipped_{reason}", int(count)))

    report_df = pd.DataFrame(rows, columns=["metric", "value"])
    report_df.to_csv(output_dir / "step27_sigma_temperature_comparison_report.csv", index=False)
    report_text = "\n".join(f"{metric}: {value}" for metric, value in rows) + "\n"
    (output_dir / "step27_sigma_temperature_comparison_report.txt").write_text(
        report_text, encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_y = not args.no_log_y

    curves, skipped = load_curves(args.input)
    if curves.empty:
        raise SystemExit("no usable SiGe or PbTe-based conductivity curves were found")

    write_curve_map(curves, args.output_dir)
    if not skipped.empty:
        skipped.to_csv(args.output_dir / "step27_skipped_curve_map.csv", index=False)
    write_report(curves, skipped, args.output_dir)

    sige_labels = sorted_labels(curves, "SiGe")
    sige_curves = curves[curves["composition_group"].eq("SiGe")].copy()
    plot_curves_by_label(
        sige_curves,
        args.output_dir / "figure_01_sige_sigma_by_composition_ratio.png",
        "Starrydata2 Si-Ge electrical conductivity curves by composition ratio",
        "composition_ratio_label",
        sige_labels,
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
    )
    plot_curves_by_label(
        sige_curves,
        args.output_dir / "figure_01_sige_sigma_by_composition_ratio_without_trend_lines.png",
        "Starrydata2 Si-Ge electrical conductivity curves by composition ratio",
        "composition_ratio_label",
        sige_labels,
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
        show_median=False,
    )

    family_curves = curves.copy()
    family_curves["family_label"] = family_curves["composition_group"]
    plot_curves_by_label(
        family_curves,
        args.output_dir / "figure_02_sige_vs_pbte_based_sigma.png",
        "Starrydata2 Si-Ge vs Pb-Te based electrical conductivity curves",
        "family_label",
        ["SiGe", "PbTe-based"],
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
        line_styles={"SiGe": "-", "PbTe-based": "--"},
        line_colors={"SiGe": "#6baed6", "PbTe-based": "#fdae6b"},
        median_colors={"SiGe": "#08519c", "PbTe-based": "#b44a00"},
        line_alphas={"SiGe": 0.18, "PbTe-based": 0.12},
        line_widths={"SiGe": 0.75, "PbTe-based": 0.8},
        median_widths={"SiGe": 3.6, "PbTe-based": 4.0},
    )
    plot_curves_by_label(
        family_curves,
        args.output_dir / "figure_02_sige_vs_pbte_based_sigma_without_trend_lines.png",
        "Starrydata2 Si-Ge vs Pb-Te based electrical conductivity curves",
        "family_label",
        ["SiGe", "PbTe-based"],
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
        line_styles={"SiGe": "-", "PbTe-based": "--"},
        line_colors={"SiGe": "#6baed6", "PbTe-based": "#fdae6b"},
        line_alphas={"SiGe": 0.18, "PbTe-based": 0.12},
        line_widths={"SiGe": 0.75, "PbTe-based": 0.8},
        show_median=False,
    )

    pbte_curves = curves[curves["composition_group"].eq("PbTe-based")].copy()
    pbte_labels = (
        pbte_curves["composition_ratio_label"].value_counts().head(args.pbte_top_n).index.tolist()
    )
    plot_curves_by_label(
        pbte_curves[pbte_curves["composition_ratio_label"].isin(pbte_labels)].copy(),
        args.output_dir / "figure_03_pbte_based_top_composition_ratios.png",
        f"Starrydata2 Pb-Te based electrical conductivity curves: top {len(pbte_labels)} ratio groups",
        "composition_ratio_label",
        pbte_labels,
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
        line_styles={label: "--" for label in pbte_labels},
    )
    plot_curves_by_label(
        pbte_curves[pbte_curves["composition_ratio_label"].isin(pbte_labels)].copy(),
        args.output_dir / "figure_03_pbte_based_top_composition_ratios_without_trend_lines.png",
        f"Starrydata2 Pb-Te based electrical conductivity curves: top {len(pbte_labels)} ratio groups",
        "composition_ratio_label",
        pbte_labels,
        log_y=log_y,
        dpi=args.dpi,
        y_quantile_low=args.y_quantile_low,
        y_quantile_high=args.y_quantile_high,
        line_styles={label: "--" for label in pbte_labels},
        show_median=False,
    )

    print(f"output_dir: {args.output_dir}")
    print(f"selected_curves: {len(curves)}")
    print(f"SiGe curves: {int(curves['composition_group'].eq('SiGe').sum())}")
    print(f"PbTe-based curves: {int(curves['composition_group'].eq('PbTe-based').sum())}")
    print("figures:")
    print("- figure_01_sige_sigma_by_composition_ratio.png")
    print("- figure_01_sige_sigma_by_composition_ratio_without_trend_lines.png")
    print("- figure_02_sige_vs_pbte_based_sigma.png")
    print("- figure_02_sige_vs_pbte_based_sigma_without_trend_lines.png")
    print("- figure_03_pbte_based_top_composition_ratios.png")
    print("- figure_03_pbte_based_top_composition_ratios_without_trend_lines.png")


if __name__ == "__main__":
    main()
