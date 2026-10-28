"""Stack per-material ML/FML parity pairs with shared, material-specific axes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from PIL import Image

from build_df_ml_fml_material_parity import GROUPS, OUT, paired, draw, legend, full_limits

DEFAULT_OUTPUT = OUT / "ML_vs_FML_vertical"


def material_limits(frames, quantity, zoom=False):
    nonempty = [f for f in frames if len(f)]
    if not nonempty:
        return (100., 1e8) if quantity == "sigma" else (0., 1.)
    if quantity == "sigma" and zoom:
        # Separate quantiles for each method and each coordinate prevent a
        # larger cohort from overwhelming the smaller cohort's display range.
        low, high = [], []
        for frame in nonempty:
            for column in ["sigma_obs", "sigma_pred"]:
                logs = np.log10(frame[column].to_numpy())
                q = np.quantile(logs, [.005, .995]) if len(logs) >= 100 else [logs.min(), logs.max()]
                low.append(q[0])
                high.append(q[1])
        return 10.**(np.floor((min(low)-.05)*2)/2), 10.**(np.ceil((max(high)+.05)*2)/2)
    if quantity == "sigma":
        return full_limits(nonempty, quantity)
    values = np.concatenate([f[["zt_obs", "zt_pred"]].to_numpy().ravel() for f in nonempty])
    lo = min(0., float(values.min()) * 1.08)
    hi = max(1e-12, float(values.max()) * 1.05)
    # Keep small negative reported ZT visible and round positive bounds to a
    # readable value. Limits depend only on this material's ML/FML values.
    tick_step = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(0, hi)
    upper = float(tick_step[-1])
    return lo, upper


def compose_vertical(paths, destination):
    sizes = []
    for path in paths:
        with Image.open(path) as im:
            sizes.append(im.size)
    assert len({w for w, _ in sizes}) == 1
    canvas = Image.new("RGB", (sizes[0][0], sum(h for _, h in sizes)), "white")
    y = 0
    for path in paths:
        with Image.open(path) as im:
            canvas.paste(im.convert("RGB"), (0, y))
            y += im.height
    canvas.save(destination)
    canvas.close()
    return sizes[0][0], y


def render(output, quantity, tables, dpi, zoom=False):
    variant = "zoom" if zoom else "full_range"
    folder = output / quantity / variant
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{quantity}_ML_vs_FML_vertical" + ("_zoom" if zoom else "")
    records, paths = [], []
    with PdfPages(output / f"{stem}.pdf") as pdf:
        for number, (title, key, slug) in enumerate(GROUPS, 1):
            subsets = {m: t[t.material_group.eq(key)] for m, t in tables.items()}
            bounds = material_limits(list(subsets.values()), quantity, zoom)
            fig, axes = plt.subplots(1, 2, figsize=(11.8, 6.0))
            symbol = "電気伝導率 σ" if quantity == "sigma" else "ZT"
            fig.suptitle(f"{number:02d}  {title}  |  {symbol}", fontsize=17, y=.965, fontweight="bold")
            outside_total = 0
            for ax, method in zip(axes, ["ML", "FML"]):
                rows = subsets[method]
                outside = draw(ax, rows, f"{method}法", quantity, bounds)
                outside_total += outside
                ax.set_xlabel("実測値 σ (S/m)" if quantity == "sigma" else "実測値 ZT", fontsize=11)
                ax.set_ylabel("予測値 σ (S/m)" if quantity == "sigma" else "予測値 ZT", fontsize=11)
                if quantity == "zt":
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
                assert tuple(ax.get_xlim()) == tuple(ax.get_ylim()) == bounds
                records.append({"quantity": quantity, "variant": variant, "material_group": key,
                                "material_name": title, "method": method, "axis_min": bounds[0],
                                "axis_max": bounds[1], "paired_points": len(rows), "outside_axes": outside,
                                "pair_png": f"{quantity}/{variant}/{number:02d}_{slug}.png"})
            assert axes[0].get_xlim() == axes[1].get_xlim()
            assert axes[0].get_ylim() == axes[1].get_ylim()
            if not zoom:
                assert outside_total == 0, f"Full-range panel clips {key}"
            note = "材料ごとに軸を調整・左右と縦横は同一範囲"
            if zoom:
                note += "  |  分布中心の拡大版（全点版も別途保存）"
            fig.text(.5, .885, note, ha="center", fontsize=9, color="#505864")
            legend(fig)
            fig.subplots_adjust(left=.085, right=.975, bottom=.15, top=.79, wspace=.30)
            path = folder / f"{number:02d}_{slug}.png"
            fig.savefig(path, dpi=dpi, facecolor="white")
            pdf.savefig(fig, dpi=dpi, facecolor="white")
            plt.close(fig)
            paths.append(path)
    dimensions = compose_vertical(paths, output / f"{stem}.png")
    print(f"{stem}: {dimensions[0]} x {dimensions[1]} px, {len(paths)} material rows", flush=True)
    return records


def write_gallery(output):
    sections = []
    for title, quantity, variant, stem in [
        ("電気伝導率 σ：全点版", "sigma", "full_range", "sigma_ML_vs_FML_vertical"),
        ("ZT：全点版", "zt", "full_range", "zt_ML_vs_FML_vertical"),
        ("電気伝導率 σ：分布中心の拡大版", "sigma", "zoom", "sigma_ML_vs_FML_vertical_zoom"),
    ]:
        imgs = ''.join(f'<img loading="lazy" src="{quantity}/{variant}/{i:02d}_{slug}.png" alt="{name} ML・FML比較">' for i, (name, _, slug) in enumerate(GROUPS, 1))
        sections.append(f'<section id="{quantity}-{variant}"><h2>{title}</h2><p><a href="{stem}.png">縦長画像PNG</a> ／ <a href="{stem}.pdf">PDF（材料ごとに1ページ）</a></p>{imgs}</section>')
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>材料別 ML・FML 比較</title>
<style>body{font:16px sans-serif;background:#eef2f6;color:#243044;margin:0}header,section{max-width:1180px;margin:24px auto;padding:22px;background:white}nav{position:sticky;top:0;background:#213348;color:white;padding:14px;text-align:center;z-index:1}nav a{color:white;margin:0 18px}a{color:#145ca0}img{width:100%;height:auto;display:block;border-bottom:1px solid #d9e1e9}p{line-height:1.8}section{scroll-margin-top:65px}</style>
<nav><a href="#sigma-full_range">σ 全点</a><a href="#zt-full_range">ZT 全点</a><a href="#sigma-zoom">σ 拡大</a></nav>
<header><h1>材料系ごとのML法・FML法比較</h1><p>左：ML法 ／ 右：FML法。同じ材料では左右・縦横の軸範囲を統一し、材料ごとに両手法のデータから範囲を設定しています。σは対数軸、ZTは線形軸です。全点版はすべての比較可能な点を含みます。</p><p>σ拡大版は100点以上ある系列の0.5～99.5パーセンタイルを基に表示範囲を設定し、範囲外の点数を表示しています。元データ・予測値は前回作成したものを使用しています。手法間で評価対象データは異なります。<a href="axis_ranges_and_counts.csv">軸範囲・点数CSV</a></p></header>
'''
    (output / "index.html").write_text(page + ''.join(sections) + '</html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False})
    data = {m: pd.read_csv(args.input_dir / f"plot_source_{m}.csv", low_memory=False) for m in ["ML", "FML"]}
    records = []
    for quantity, zoom in [("sigma", False), ("zt", False), ("sigma", True)]:
        records += render(args.output, quantity, {m: paired(d, quantity) for m, d in data.items()}, args.dpi, zoom)
    audit = pd.DataFrame(records)
    assert len(audit) == 16*2*3
    for _, group in audit.groupby(["quantity", "variant", "material_group"]):
        assert len(group) == 2 and group.axis_min.nunique() == group.axis_max.nunique() == 1
    audit.to_csv(args.output / "axis_ranges_and_counts.csv", index=False, encoding="utf-8-sig")
    write_gallery(args.output)
    manifest = {m: {"path": str(args.input_dir / f"plot_source_{m}.csv"),
                    "sha256": hashlib.sha256((args.input_dir / f"plot_source_{m}.csv").read_bytes()).hexdigest()}
                for m in data}
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
