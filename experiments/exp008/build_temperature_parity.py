"""Color existing all-material DF/ML/FML parity points by temperature."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data/output/df_ml_fml_material_parity_20261025"
METHODS = ("DF", "ML", "FML")
QUANTITIES = ("sigma", "zt")
FIXED_LIMITS = {"sigma": (1e2, 1e8), "zt": (0.0, 3.0)}
CMAP = "plasma"
HIGHLIGHT_COLOR = "#0072B2"


def paired(frame, quantity):
    x, y = frame[f"{quantity}_obs"], frame[f"{quantity}_pred"]
    valid = np.isfinite(x) & np.isfinite(y)
    if quantity == "sigma":
        valid &= x.gt(0) & y.gt(0)
    return frame.loc[valid].copy()


def full_limits(tables, quantity):
    values = np.concatenate([
        table[[f"{quantity}_obs", f"{quantity}_pred"]].to_numpy().ravel()
        for table in tables.values() if len(table)
    ])
    if quantity == "sigma":
        return (10.0 ** math.floor(np.log10(values.min())),
                10.0 ** math.ceil(np.log10(values.max())))
    return min(0.0, float(values.min()) * 1.08), max(0.1, float(values.max()) * 1.05)


def outside_count(rows, quantity, limits):
    return int((~(rows[f"{quantity}_obs"].between(*limits)
                   & rows[f"{quantity}_pred"].between(*limits))).sum())


def draw(ax, rows, method, quantity, limits, norm, color_max_temperature=None,
         single_color=False):
    # Shuffle all rows (without subsampling) to avoid a temperature or material
    # always being drawn last. Opaque colors match the colorbar exactly.
    points = rows.sample(frac=1, random_state=8)
    background_count = 0
    if color_max_temperature is not None:
        background = points[points.temperature_K.gt(color_max_temperature)]
        points = points[points.temperature_K.le(color_max_temperature)]
        background_scatter = ax.scatter(
            background[f"{quantity}_obs"], background[f"{quantity}_pred"],
            color="#d0d0d0", s=8, alpha=1, linewidths=0, rasterized=True, zorder=1.5)
        background_count = len(background_scatter.get_offsets())
        assert background_count == len(background)
        assert points.temperature_K.le(color_max_temperature).all()
    color_options = {"color": HIGHLIGHT_COLOR} if single_color else {
        "c": points.temperature_K, "cmap": CMAP, "norm": norm}
    scatter = ax.scatter(points[f"{quantity}_obs"], points[f"{quantity}_pred"],
                         **color_options,
                         s=8, alpha=1, linewidths=0, rasterized=True, zorder=2)
    lo, hi = limits
    line = np.geomspace(lo, hi, 200) if quantity == "sigma" else np.linspace(lo, hi, 200)
    ax.plot(line, line, color="#222222", lw=1.0, zorder=3)
    for factor, style in [(2, "--"), (10, ":")]:
        for multiplier in (factor, 1 / factor):
            ax.plot(line, line * multiplier, color="#777777", lw=.65,
                    ls=style, alpha=.7, zorder=1)
    if quantity == "sigma":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axisbelow(True)
    ax.grid(which="major", alpha=.18, lw=.5)
    label = "σ (S/m)" if quantity == "sigma" else "ZT"
    ax.set_xlabel(f"実測値 {label}", fontsize=11)
    ax.set_ylabel(f"予測値 {label}", fontsize=11)
    panel_title = f"{method}法  |  N = {len(rows):,}"
    if color_max_temperature is not None:
        panel_title += f"\n{color_max_temperature:g} K以下: {len(points):,}点"
    ax.set_title(panel_title, fontsize=13, pad=12)
    outside = outside_count(rows, quantity, limits)
    if outside:
        ax.text(.025, .975, f"軸範囲外: {outside:,}点", transform=ax.transAxes,
                va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .9, "edgecolor": "none"})
    assert len(scatter.get_offsets()) + background_count == len(rows)
    if single_color:
        np.testing.assert_array_equal(scatter.get_facecolors(), [to_rgba(HIGHLIGHT_COLOR)])
        handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                          markersize=5, label=label) for color, label in (
            (HIGHLIGHT_COLOR, f"T ≤ {color_max_temperature:g} K"),
            ("#d0d0d0", f"T > {color_max_temperature:g} K"))]
        ax.legend(handles=handles, loc="lower right", fontsize=10,
                  framealpha=.95, edgecolor="none")
    else:
        np.testing.assert_array_equal(scatter.get_array(), points.temperature_K.to_numpy())
    assert tuple(ax.get_xlim()) == tuple(ax.get_ylim()) == tuple(limits)
    return scatter


def save(fig, stem, dpi):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=dpi, facecolor="white")
    plt.close(fig)


def render(tables, quantity, variant, limits, norm, output, dpi, color_max_temperature=None,
           selection_note="", single_color=False):
    title = "電気伝導率 σ" if quantity == "sigma" else "ZT"
    variant_label = "既存図と同じ軸範囲" if variant == "fixed" else "全点を含む軸範囲"
    if selection_note:
        variant_label += f"  |  {selection_note}"
    color_note = "色: 温度 (K)"
    if color_max_temperature is not None:
        variant_label += f"  |  {color_max_temperature:g} K以下を色分け"
        color_note = f"色: {color_max_temperature:g} K以下の温度  /  灰色: {color_max_temperature:g} K超"
        if single_color:
            variant_label = variant_label.replace("以下を色分け", "以下を青一色で表示")
            color_note = f"青: {color_max_temperature:g} K以下  /  灰色: {color_max_temperature:g} K超"
    note = f"全材料を重ねて表示  /  {color_note}  /  実線: y=x  /  破線: 2倍・1/2倍  /  点線: 10倍・1/10倍"
    height = 7.4 if color_max_temperature is not None else 6.3
    fig, axes = plt.subplots(1, 3, figsize=(17, height), layout="constrained")
    if color_max_temperature is not None:
        fig.get_layout_engine().set(h_pad=.12)
    fig.suptitle(f"{title}：DF・ML・FML法の温度別比較\n{variant_label}", fontsize=17)
    for ax, method in zip(axes, METHODS):
        scatter = draw(ax, tables[method], method, quantity, limits, norm,
                       color_max_temperature, single_color)
    if not single_color:
        bar = fig.colorbar(scatter, ax=list(axes), fraction=.025, pad=.025, shrink=.78)
        bar.set_label("温度 T (K)", fontsize=12)
    fig.supxlabel(note + "\nNは温度点数。手法間で評価対象データが異なります。", fontsize=10)
    save(fig, output / "figures" / f"{quantity}_DF_ML_FML_temperature_{variant}", dpi)

    for method in METHODS:
        fig, ax = plt.subplots(figsize=(8, 7.1), layout="constrained")
        if color_max_temperature is not None:
            fig.get_layout_engine().set(h_pad=.10)
        fig.suptitle(f"{title}：全材料の予測–実測散布図\n{variant_label}", fontsize=15)
        scatter = draw(ax, tables[method], method, quantity, limits, norm,
                       color_max_temperature, single_color)
        if not single_color:
            bar = fig.colorbar(scatter, ax=ax, fraction=.045, pad=.035, shrink=.85)
            bar.set_label("温度 T (K)", fontsize=12)
        fig.supxlabel(color_note + "\n実線: y=x  /  破線: 2倍・1/2倍  /  点線: 10倍・1/10倍", fontsize=9)
        save(fig, output / "figures" / method / f"{quantity}_temperature_{variant}", dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--color-max-temperature", type=float,
                        help="Color only T <= this value in K; show hotter points in gray.")
    parser.add_argument("--single-color", action="store_true",
                        help="Use solid blue for points at or below --color-max-temperature.")
    args = parser.parse_args()
    if args.single_color and args.color_max_temperature is None:
        parser.error("--single-color requires --color-max-temperature")
    if args.color_max_temperature is not None:
        if not np.isfinite(args.color_max_temperature) or args.color_max_temperature <= 0:
            parser.error("--color-max-temperature must be a positive finite value in K")
        suffix = "_single_color" if args.single_color else ""
        args.output = args.output / f"temperature_le_{args.color_max_temperature:g}K{suffix}"
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False})

    data, sources = {}, []
    for method in METHODS:
        path = args.input_dir / f"plot_source_{method}.csv"
        frame = pd.read_csv(path, low_memory=False)
        required = {"sample_key", "temperature_K", "material_group",
                    "sigma_obs", "sigma_pred", "zt_obs", "zt_pred"}
        assert required <= set(frame.columns), f"Missing columns: {path}"
        assert not frame.duplicated(["sample_key", "temperature_K"]).any()
        # A missing temperature must not silently remove an original parity point.
        assert (np.isfinite(frame.temperature_K) & frame.temperature_K.gt(0)).all()
        data[method] = frame
        sources.append({"method": method, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(frame)})

    tables = {q: {m: paired(d, q) for m, d in data.items()} for q in QUANTITIES}
    temperatures = pd.concat([d.temperature_K for group in tables.values() for d in group.values()])
    vmin = 100.0 * math.floor(temperatures.min() / 100.0)
    vmax = 100.0 * math.ceil(temperatures.max() / 100.0)
    norm = Normalize(vmin=vmin, vmax=max(vmin + 100.0, vmax))
    if args.color_max_temperature is not None:
        norm = Normalize(vmin=0, vmax=args.color_max_temperature)
    # Check the point count against the original material-wise figure index.
    source_index = pd.read_csv(args.input_dir / "figure_index.csv")
    original_counts = source_index.groupby(["method", "quantity"]).paired_points.sum()
    records = []
    for quantity, group in tables.items():
        for variant in ("fixed", "full_range"):
            limits = FIXED_LIMITS[quantity] if variant == "fixed" else full_limits(group, quantity)
            for method, rows in group.items():
                assert len(rows) == original_counts.loc[(method, quantity)]
                outside = outside_count(rows, quantity, limits)
                if variant == "full_range":
                    assert outside == 0, f"Full-range axes clip {method}/{quantity}"
                colored = rows if args.color_max_temperature is None else rows[rows.temperature_K.le(args.color_max_temperature)]
                records.append({"method": method, "quantity": quantity, "variant": variant,
                                "points": len(rows), "samples": rows.sample_key.nunique(),
                                "colored_points": len(colored), "gray_points": len(rows) - len(colored),
                                "colored_outside_axes": outside_count(colored, quantity, limits),
                                "material_groups": rows.material_group.nunique(),
                                "temperature_min_K": rows.temperature_K.min(),
                                "temperature_max_K": rows.temperature_K.max(),
                                "color_min_K": None if args.single_color else norm.vmin,
                                "color_max_K": None if args.single_color else norm.vmax,
                                "axis_min": limits[0], "axis_max": limits[1],
                                "outside_axes": outside})
            render(group, quantity, variant, limits, norm, args.output, args.dpi,
                   args.color_max_temperature, single_color=args.single_color)
            print(f"Rendered {quantity}/{variant}: " + ", ".join(f"{m}={len(d):,}" for m, d in group.items()), flush=True)

    pd.DataFrame(records).to_csv(args.output / "figure_counts.csv", index=False, encoding="utf-8-sig")
    manifest = {"sources": sources, "colormap": None if args.single_color else CMAP,
                "highlight_color": HIGHLIGHT_COLOR if args.single_color else None,
                "temperature_color_limits_K": None if args.single_color else [norm.vmin, norm.vmax],
                "color_max_temperature_K": args.color_max_temperature,
                "hotter_points": "gray background" if args.color_max_temperature is not None else "temperature color",
                "shuffle_seed": 8, "subsampling": False,
                "validation": "Original point counts, per-point temperatures, shared axes, and full-range coverage checked."}
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
