# SiGe の電気伝導率曲線を組成ごとに出典別で色分け表示し、対応表 CSV や HTML も出力できるスクリプト。
import argparse
import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def parse_numeric_list(raw_value: Any) -> list[float]:
    if isinstance(raw_value, (list, tuple)):
        return [float(v) for v in raw_value]
    if raw_value is None or pd.isna(raw_value):
        return []

    text = str(raw_value).strip()
    if not text:
        return []

    parsed = None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            break
        except (TypeError, ValueError, SyntaxError, json.JSONDecodeError):
            continue

    if not isinstance(parsed, (list, tuple)):
        raise ValueError("x/y is not list-like")

    values: list[float] = []
    for value in parsed:
        if value is None:
            continue
        values.append(float(value))
    return values


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return safe.strip("._") or "composition"


def iter_xy_pairs(
    x_values: Iterable[float], y_values: Iterable[float]
) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(x_values, y_values):
        if x is None or y is None:
            continue
        try:
            xf = float(x)
            yf = float(y)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(xf) and math.isfinite(yf)):
            continue
        pairs.append((xf, yf))
    pairs.sort(key=lambda t: t[0])
    return pairs


def resolve_xy_columns(df: pd.DataFrame) -> tuple[str, str]:
    if {"x_list", "y_list"}.issubset(df.columns):
        return "x_list", "y_list"
    if {"x", "y"}.issubset(df.columns):
        return "x", "y"
    raise KeyError("missing x/y columns: expected x_list,y_list or x,y")


def source_style(prop_y: Any) -> tuple[str, str]:
    prop_text = str(prop_y).strip().lower()
    if prop_text == "electrical conductivity":
        return "#1f77b4", "Electrical conductivity (original)"
    if prop_text == "electrical resistivity":
        return "#ff7f0e", "1 / Electrical resistivity"
    return "#7f7f7f", f"Unknown source: {prop_y}"


def spread_label_positions(targets: list[float], min_gap: float = 0.03) -> list[float]:
    if not targets:
        return []

    count = len(targets)
    if count > 1:
        min_gap = min(min_gap, 0.95 / (count - 1))
    else:
        min_gap = 0.0

    adjusted: list[float] = []
    for target in targets:
        clipped = min(max(target, 0.0), 1.0)
        if not adjusted:
            adjusted.append(clipped)
            continue
        adjusted.append(max(clipped, adjusted[-1] + min_gap))

    overflow = adjusted[-1] - 1.0
    if overflow > 0:
        adjusted = [value - overflow for value in adjusted]

    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1] - min_gap)

    underflow = 0.0 - adjusted[0]
    if underflow > 0:
        adjusted = [value + underflow for value in adjusted]

    return [min(max(value, 0.0), 1.0) for value in adjusted]


def annotate_curve_ids_outside(
    ax: plt.Axes,
    annotation_points: list[dict[str, Any]],
    *,
    log_scale: bool,
) -> None:
    if not annotation_points:
        return

    y_low, y_high = sorted(ax.get_ylim())
    if y_high <= y_low:
        return

    if log_scale:
        y_low = max(y_low, 1e-12)

    points_with_target: list[dict[str, Any]] = []
    for point in annotation_points:
        y_value = float(point["y"])
        if log_scale and y_value <= 0:
            continue

        if log_scale:
            y_clamped = min(max(y_value, y_low), y_high)
            denom = math.log10(y_high) - math.log10(y_low)
            if denom <= 0:
                continue
            target = (math.log10(y_clamped) - math.log10(y_low)) / denom
        else:
            y_clamped = min(max(y_value, y_low), y_high)
            target = (y_clamped - y_low) / (y_high - y_low)

        point_copy = dict(point)
        point_copy["target_frac"] = target
        points_with_target.append(point_copy)

    if not points_with_target:
        return

    points_with_target.sort(key=lambda p: p["target_frac"])
    targets = [float(p["target_frac"]) for p in points_with_target]
    adjusted = spread_label_positions(targets)

    for point, y_fraction in zip(points_with_target, adjusted):
        ax.plot(
            [point["x"]],
            [point["y"]],
            marker="o",
            markersize=2.2,
            color=point["color"],
            alpha=0.85,
        )
        ax.annotate(
            str(point["curve_id"]),
            xy=(point["x"], point["y"]),
            xycoords="data",
            xytext=(1.02, y_fraction),
            textcoords="axes fraction",
            ha="left",
            va="center",
            fontsize=7,
            color=point["color"],
            bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8},
            arrowprops={"arrowstyle": "-", "color": point["color"], "alpha": 0.5, "lw": 0.7},
            annotation_clip=False,
            clip_on=False,
        )


def save_interactive_html(
    composition: Any,
    curves: list[dict[str, Any]],
    title: str,
    html_path: Path,
    *,
    log_y: bool,
) -> None:
    fig = go.Figure()
    shown_labels: set[str] = set()

    for curve in curves:
        label = str(curve["source_label"])
        show_legend = label not in shown_labels
        if show_legend:
            shown_labels.add(label)

        xs = curve["xs"]
        ys = curve["ys"]
        n_points = len(xs)
        customdata = [
            [
                curve["curve_id"],
                curve.get("DOI"),
                curve.get("SID"),
                curve.get("sample_id"),
                curve.get("figure_id"),
                curve.get("source_label"),
            ]
            for _ in range(n_points)
        ]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": curve["color"], "width": 1.4},
                opacity=0.7,
                name=label,
                legendgroup=label,
                showlegend=show_legend,
                customdata=customdata,
                hovertemplate=(
                    "curve_id: %{customdata[0]}<br>"
                    "source: %{customdata[5]}<br>"
                    "DOI: %{customdata[1]}<br>"
                    "SID: %{customdata[2]}<br>"
                    "sample_id: %{customdata[3]}<br>"
                    "figure_id: %{customdata[4]}<br>"
                    "T [K]: %{x:.4f}<br>"
                    "sigma [1/(ohm*m)]: %{y:.6g}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Temperature [K]",
        yaxis_title="Electrical conductivity [1/(ohm*m)]",
        template="plotly_white",
        hovermode="closest",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    if log_y:
        fig.update_yaxes(type="log")

    fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sige_output_dir = project_root / "data" / "output" / "sige"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=sige_output_dir / "sige_ElectricalConductivity_curves.csv",
        help="input csv from SiGe_ElectricalConductivity.py",
    )
    parser.add_argument(
        "--outdir",
        default=project_root / "data" / "output" / "by_composition_source_colored",
        help="directory for output png files",
    )
    parser.add_argument(
        "--split-csv",
        action="store_true",
        help="also save one csv file per composition",
    )
    parser.add_argument(
        "--save-curve-map",
        action="store_true",
        help="save one curve-id mapping csv per composition",
    )
    parser.add_argument(
        "--annotate-curve-id",
        action="store_true",
        help="draw curve ids (C1, C2, ...) near line end points",
    )
    parser.add_argument(
        "--curve-id-placement",
        choices=("outside", "end"),
        default="outside",
        help="placement for curve id labels when --annotate-curve-id is set",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="dpi for saved figures",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="plot conductivity axis on log scale",
    )
    parser.add_argument(
        "--save-interactive-html",
        action="store_true",
        help="also save one interactive html per composition with hover details",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    df = pd.read_csv(args.csv)
    required_cols = {"composition", "prop_y"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    x_col, y_col = resolve_xy_columns(df)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_outdir = outdir / "split_csv"
    if args.split_csv:
        csv_outdir.mkdir(parents=True, exist_ok=True)

    curve_map_outdir = outdir / "curve_map"
    if args.save_curve_map:
        curve_map_outdir.mkdir(parents=True, exist_ok=True)

    html_outdir = outdir / "interactive_html"
    if args.save_interactive_html:
        html_outdir.mkdir(parents=True, exist_ok=True)

    grouped = df[df["composition"].notna()].groupby("composition", sort=True)
    total_groups = grouped.ngroups

    saved_figures = 0
    skipped_groups = 0
    for composition, group in grouped:
        fig, ax = plt.subplots(figsize=(8, 5))
        curve_count = 0
        source_counts = {
            "Electrical conductivity (original)": 0,
            "1 / Electrical resistivity": 0,
            "Unknown": 0,
        }
        legend_labels: set[str] = set()
        curve_map_rows: list[dict[str, Any]] = []
        annotation_points: list[dict[str, Any]] = []
        interactive_curves: list[dict[str, Any]] = []

        for row in group.itertuples(index=False):
            try:
                x_values = parse_numeric_list(getattr(row, x_col))
                y_values = parse_numeric_list(getattr(row, y_col))
            except (ValueError, TypeError) as exc:
                logging.warning("skip row: composition=%s parse error: %s", composition, exc)
                continue

            pairs = iter_xy_pairs(x_values, y_values)
            if len(pairs) < 2:
                continue

            color, label = source_style(getattr(row, "prop_y"))
            if label in source_counts:
                source_counts[label] += 1
            else:
                source_counts["Unknown"] += 1

            xs, ys = zip(*pairs)
            curve_id = f"C{curve_count + 1}"
            draw_label = label if label not in legend_labels else None
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.45, label=draw_label)
            if args.annotate_curve_id and args.curve_id_placement == "end":
                ax.text(
                    xs[-1],
                    ys[-1],
                    curve_id,
                    color=color,
                    fontsize=7,
                    alpha=0.85,
                    va="center",
                    bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7},
                )
            annotation_points.append(
                {
                    "curve_id": curve_id,
                    "x": xs[-1],
                    "y": ys[-1],
                    "color": color,
                }
            )
            if draw_label is not None:
                legend_labels.add(label)
            curve_map_rows.append(
                {
                    "curve_id": curve_id,
                    "composition": composition,
                    "source_label": label,
                    "source_prop_y": getattr(row, "prop_y", None),
                    "DOI": getattr(row, "DOI", None),
                    "SID": getattr(row, "SID", None),
                    "sample_id": getattr(row, "sample_id", None),
                    "figure_id": getattr(row, "figure_id", None),
                    "T_min_curve": xs[0],
                    "T_max_curve": xs[-1],
                    "n_points": len(pairs),
                    "x_list": json.dumps(list(xs)),
                    "y_list": json.dumps(list(ys)),
                }
            )
            interactive_curves.append(
                {
                    "curve_id": curve_id,
                    "source_label": label,
                    "color": color,
                    "DOI": getattr(row, "DOI", None),
                    "SID": getattr(row, "SID", None),
                    "sample_id": getattr(row, "sample_id", None),
                    "figure_id": getattr(row, "figure_id", None),
                    "xs": list(xs),
                    "ys": list(ys),
                }
            )
            curve_count += 1

        if curve_count == 0:
            plt.close(fig)
            skipped_groups += 1
            continue

        title = (
            f"{composition} ({curve_count} curves, "
            f"cond={source_counts['Electrical conductivity (original)']}, "
            f"1/rho={source_counts['1 / Electrical resistivity']})"
        )
        ax.set_title(title)
        ax.set_xlabel("Temperature [K]")
        ax.set_ylabel("Electrical conductivity [1/(ohm*m)]")
        if args.log_y:
            ax.set_yscale("log")
        if legend_labels:
            ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.3)
        if args.annotate_curve_id and args.curve_id_placement == "outside":
            annotate_curve_ids_outside(ax, annotation_points, log_scale=args.log_y)
            fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        else:
            fig.tight_layout()

        safe_name = sanitize_filename(str(composition))
        png_path = outdir / f"{safe_name}.png"
        fig.savefig(png_path, dpi=args.dpi)
        plt.close(fig)
        saved_figures += 1

        if args.split_csv:
            csv_path = csv_outdir / f"{safe_name}.csv"
            group.to_csv(csv_path, index=False)
        if args.save_curve_map and curve_map_rows:
            curve_map_path = curve_map_outdir / f"{safe_name}_curve_map.csv"
            pd.DataFrame(curve_map_rows).to_csv(curve_map_path, index=False)
        if args.save_interactive_html and interactive_curves:
            html_path = html_outdir / f"{safe_name}.html"
            save_interactive_html(
                composition,
                interactive_curves,
                title,
                html_path,
                log_y=args.log_y,
            )

    print(f"rows_total: {len(df)}")
    print(f"groups_total: {total_groups}")
    print(f"figures_saved: {saved_figures}")
    print(f"groups_skipped_no_curves: {skipped_groups}")
    print(f"outdir: {outdir}")
    if args.split_csv:
        print(f"split_csv_outdir: {csv_outdir}")
    if args.save_curve_map:
        print(f"curve_map_outdir: {curve_map_outdir}")
    if args.save_interactive_html:
        print(f"interactive_html_outdir: {html_outdir}")


if __name__ == "__main__":
    main()
