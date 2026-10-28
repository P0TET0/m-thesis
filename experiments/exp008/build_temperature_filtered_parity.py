"""Add DF/ML/FML pooled and material parity plots using only T > 300 K."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from scripts import build_df_ml_fml_material_parity as material
from experiments.exp008 import build_temperature_parity as temperature

METHODS = temperature.METHODS
QUANTITIES = temperature.QUANTITIES


def summary_row(original, selected, method, quantity, key, name, cutoff):
    before = material.paired(original, quantity)
    after = material.paired(selected, quantity)
    assert len(before) == len(after) + before.temperature_K.le(cutoff).sum()
    error = after[f"{quantity}_pred"] - after[f"{quantity}_obs"]
    positive = after[f"{quantity}_obs"].gt(0) & after[f"{quantity}_pred"].gt(0)
    logs = np.log10(after.loc[positive, f"{quantity}_pred"]) - np.log10(after.loc[positive, f"{quantity}_obs"])
    return {
        "method": method, "quantity": quantity, "material_group": key,
        "material_name": name, "paired_points_before": len(before),
        "excluded_below_cutoff": int(before.temperature_K.lt(cutoff).sum()),
        "excluded_at_cutoff": int(before.temperature_K.eq(cutoff).sum()),
        "excluded_at_or_below_cutoff": int(before.temperature_K.le(cutoff).sum()),
        "paired_points_after": len(after), "samples_after": after.sample_key.nunique(),
        "temperature_min_K": after.temperature_K.min(),
        "temperature_max_K": after.temperature_K.max(),
        "rmse_all_paired_points": float(np.sqrt((error**2).mean())),
        "mae_all_paired_points": float(error.abs().mean()),
        "positive_pairs_for_log10": int(positive.sum()),
        "rmse_log10_positive_pairs": float(np.sqrt((logs**2).mean())),
        "mae_log10_positive_pairs": float(logs.abs().mean()),
    }


def render_comparisons(tables, output, dpi, selection_note):
    """Place DF/ML/FML side by side with the original carrier colors."""
    axes_records = []
    for quantity, methods in tables.items():
        for title, key, slug in [("全材料を重ねた散布図", "all", "all_materials"), *material.GROUPS]:
            subsets = {m: d if key == "all" else d[d.material_group.eq(key)] for m, d in methods.items()}
            for variant in ("fixed", "full_range"):
                limits = temperature.FIXED_LIMITS[quantity] if variant == "fixed" else material.full_limits(list(subsets.values()), quantity)
                fig, axes = plt.subplots(1, 3, figsize=(17, 6.3))
                symbol = "電気伝導率 σ" if quantity == "sigma" else "ZT"
                fig.suptitle(f"{title}：{symbol}  |  {selection_note}", fontsize=17, y=.965)
                for ax, method in zip(axes, METHODS):
                    rows = subsets[method]
                    outside = material.draw(ax, rows, f"{method}法", quantity, limits)
                    # Check every submitted scatter coordinate, including off-axis points.
                    offsets = np.concatenate([np.asarray(c.get_offsets()) for c in ax.collections])
                    expected = pd.concat([rows[rows.carrier.eq(c)] for c in material.COLORS])
                    np.testing.assert_array_equal(offsets, expected[[f"{quantity}_obs", f"{quantity}_pred"]].to_numpy())
                    assert tuple(ax.get_xlim()) == tuple(ax.get_ylim()) == tuple(limits)
                    if variant == "full_range":
                        assert outside == 0
                    label = "σ (S/m)" if quantity == "sigma" else "ZT"
                    ax.set_xlabel(f"実測値 {label}")
                    ax.set_ylabel(f"予測値 {label}")
                    axes_records.append({"method": method, "quantity": quantity,
                                         "material_group": key, "variant": variant,
                                         "points": len(rows), "outside_axes": outside,
                                         "axis_min": limits[0], "axis_max": limits[1]})
                material.legend(fig)
                fig.text(.5, .065, "Nは温度点数。手法間で評価対象データが異なります。", ha="center", fontsize=9)
                fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.19, wspace=.30)
                stem = f"{quantity}_DF_ML_FML_carrier_{variant}" if key == "all" else f"{slug}_{variant}"
                folder = output / "figures" if key == "all" else output / "comparisons" / quantity
                material.save(fig, folder / stem, dpi)
            print(f"Comparison {quantity}/{slug}", flush=True)
    return pd.DataFrame(axes_records)


def write_gallery(output, selection_note):
    sections = []
    for quantity, label in (("sigma", "電気伝導率 σ"), ("zt", "ZT")):
        for color, color_label in (("carrier", "p型・n型の色分け"), ("temperature", "温度の色分け")):
            stem = f"figures/{quantity}_DF_ML_FML_{color}"
            sections.append(f'<section><h2>{label}：全材料（{color_label}）</h2><a href="{stem}_fixed.png"><img src="{stem}_fixed.png"></a><p><a href="{stem}_fixed.pdf">PDF</a> / <a href="{stem}_full_range.png">全点範囲 PNG</a> / <a href="{stem}_full_range.pdf">全点範囲 PDF</a></p></section>')
    rows = []
    for name, _, slug in material.GROUPS:
        cells = []
        for quantity in QUANTITIES:
            stem = f"comparisons/{quantity}/{slug}"
            cells.append(f'<td><a href="{stem}_fixed.png">3手法 PNG</a> / <a href="{stem}_fixed.pdf">PDF</a> / <a href="{stem}_full_range.png">全点範囲 PNG</a> / <a href="{stem}_full_range.pdf">PDF</a></td>')
        rows.append(f'<tr><th>{html.escape(name)}</th>{"".join(cells)}</tr>')
    overviews = []
    for quantity in QUANTITIES:
        for method in METHODS:
            base = f"{quantity}/{method}"
            overviews.append(f'<li>{quantity} / {method}：<a href="{base}/overview_fixed.png">材料別一覧</a> / <a href="{base}/overview_full_range.png">全点範囲一覧</a> / <a href="{base}/all_materials_pooled_fixed.png">全材料を重ねた図</a></li>')
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>DF・ML・FML：{html.escape(selection_note)}</title>
<style>body{{font:16px sans-serif;background:#f3f6fa;color:#243044;max-width:1300px;margin:24px auto;padding:20px}}section{{background:white;padding:18px;margin:20px 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #d7dfe8;padding:12px;text-align:left}}a{{color:#145ba0}}p,li{{line-height:1.8}}</style>
<h1>DF・ML・FML法の比較：{html.escape(selection_note)}</h1>
<p>横軸＝実測値、縦軸＝予測値。温度条件を満たす全点を表示しています。固定軸はσが10²～10⁸ S/m、ZTが0～3です。全点範囲版では材料ごとに3手法共通の軸を使います。<a href="report.txt">比較条件・集計</a> / <a href="filter_summary.csv">除外前後の点数・誤差</a></p>
{''.join(sections)}<h2>材料別の3手法比較</h2><table><tr><th>材料系</th><th>電気伝導率 σ</th><th>ZT</th></tr>{''.join(rows)}</table>
<h2>手法ごとの材料別一覧</h2><ul>{''.join(overviews)}</ul><p>個別画像は sigma または zt / 手法名 / materials に保存しています。<a href="figure_index.csv">個別画像の一覧CSV</a></p></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=temperature.DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-temperature", type=float, default=300.0,
                        help="Exclusive lower bound: keep only temperature_K > this value.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    cutoff = args.min_temperature
    if not np.isfinite(cutoff) or cutoff < 0:
        parser.error("--min-temperature must be finite and nonnegative")
    output = args.output or HERE / f"temperature_gt_{cutoff:g}K"
    if output.resolve() == args.input_dir.resolve():
        parser.error("Output must differ from the source directory")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False})
    selection_note = f"T > {cutoff:g} K（{cutoff:g} K以下を除外）"
    original_index = pd.read_csv(args.input_dir / "figure_index.csv")
    original_counts = original_index.groupby(["method", "quantity"]).paired_points.sum()
    data, sources, counts = {}, [], []
    for method in METHODS:
        path = args.input_dir / f"plot_source_{method}.csv"
        original = pd.read_csv(path, low_memory=False)
        required = {"sample_key", "temperature_K", "material_group", "carrier", "sigma_obs", "sigma_pred", "zt_obs", "zt_pred"}
        assert required <= set(original.columns)
        assert not original.duplicated(["sample_key", "temperature_K"]).any()
        assert (np.isfinite(original.temperature_K) & original.temperature_K.gt(0)).all()
        assert set(original.material_group) <= {g[1] for g in material.GROUPS}
        assert set(original.carrier) <= set(material.COLORS)
        selected = original.loc[original.temperature_K.gt(cutoff)].copy()
        assert selected.temperature_K.gt(cutoff).all()
        data[method] = selected
        selected.to_csv(output / f"plot_source_{method}.csv", index=False, encoding="utf-8-sig")
        sources.append({"method": method, "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows_before": len(original), "rows_after": len(selected),
                        "excluded_at_cutoff": int(original.temperature_K.eq(cutoff).sum())})
        for quantity in QUANTITIES:
            assert len(material.paired(original, quantity)) == original_counts.loc[(method, quantity)]
            for name, key, _ in [("全材料", "all", "all"), *material.GROUPS]:
                before = original if key == "all" else original[original.material_group.eq(key)]
                after = selected if key == "all" else selected[selected.material_group.eq(key)]
                counts.append(summary_row(before, after, method, quantity, key, name, cutoff))
    summary = pd.DataFrame(counts)
    summary.to_csv(output / "filter_summary.csv", index=False, encoding="utf-8-sig")
    pooled = summary[summary.material_group.eq("all")]
    print(pooled[["method", "quantity", "paired_points_before", "excluded_at_or_below_cutoff", "paired_points_after"]].to_string(index=False), flush=True)

    tables = {q: {m: material.paired(d, q) for m, d in data.items()} for q in QUANTITIES}
    # Preserve the existing all-temperature color scale for direct visual comparison.
    reference_manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    norm = Normalize(*reference_manifest["temperature_color_limits_K"])
    for quantity, group in tables.items():
        for variant in ("fixed", "full_range"):
            limits = temperature.FIXED_LIMITS[quantity] if variant == "fixed" else material.full_limits(list(group.values()), quantity)
            if variant == "full_range":
                assert all(temperature.outside_count(d, quantity, limits) == 0 for d in group.values())
            temperature.render(group, quantity, variant, limits, norm, output, args.dpi, selection_note=selection_note)
        print(f"Temperature plots: {quantity}", flush=True)

    index = material.render(data, output, args.dpi, selection_note=selection_note)
    assert len(index) == len(METHODS) * len(QUANTITIES) * len(material.GROUPS)
    for (method, quantity), rows in index.groupby(["method", "quantity"]):
        assert rows.paired_points.sum() == len(tables[quantity][method])
    for record in index.itertuples():
        for name in (record.fixed_png, record.full_range_png):
            assert (output / name).stat().st_size > 0
            assert (output / name).with_suffix(".pdf").stat().st_size > 0
    index.to_csv(output / "figure_index.csv", index=False, encoding="utf-8-sig")
    axes = render_comparisons(tables, output, args.dpi, selection_note)
    axes.to_csv(output / "axis_ranges_and_counts.csv", index=False, encoding="utf-8-sig")
    manifest = {"sources": sources, "filter": {"column": "temperature_K", "operator": ">", "threshold_K": cutoff},
                "prediction_values": "Unchanged subset of existing predictions; no refitting or training",
                "temperature_color_limits_K": [norm.vmin, norm.vmax], "subsampling": False,
                "validation": "Strict cutoff, source parity counts, material partitions, plotted coordinates, shared axes and full-range coverage checked"}
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = f"""DF・ML・FML法の比較：{selection_note}

既存のplot_source_DF/ML/FML.csvのtemperature_Kに厳密な > 条件を適用しました。
全体図・材料別図・図中のN・誤差指標は、すべて除外後のデータから作成しています。
元の予測値をそのまま使い、モデルの再学習や再フィッティングは行っていません。
σは正の有限値の組、ZTは有限値の組を使います。全点範囲版は負の実測ZTも含みます。
Nは試料数ではなく温度点数です。固定軸の範囲外の点も誤差指標に含めています。
RMSE・MAEは元の単位で、log10の指標は実測値と予測値がともに正の点に限り計算します。

DFは評価対象試料へのフィッティング結果、MLとFMLは既存の予測結果です。
手法間で試料・温度点・評価分割が異なるため、同一条件での性能順位は決められません。
ZT予測値は実測S・κと予測σから計算した既存値（S²σT/κ）です。
FMLのκ・実測ZTには元処理で最大±2 Kの近傍対応があります。
今回の温度条件は既存図と同じtemperature_K（σの評価温度）に適用しています。
データのない材料分類も「比較可能なデータなし」として残しています。

保存先：index.htmlから全体図・材料別比較図を閲覧できます。
全体の3手法比較：figures/（carrierはp型・n型、temperatureは温度の色分け）
材料別の3手法比較：comparisons/sigma/、comparisons/zt/
手法別の全体・材料別一覧・個別図：sigma/、zt/
各図は固定軸版・全点範囲版をPNGとPDFで保存しています。
温度色分けのスケールは全温度版と同じ{norm.vmin:g}～{norm.vmax:g} Kです。

除外前後の点数・誤差：filter_summary.csv
手法別・材料別の画像一覧：figure_index.csv
比較図の軸範囲・点数：axis_ranges_and_counts.csv
入力ファイルとハッシュ・温度条件：source_manifest.json
再作成：python experiments/exp008/build_temperature_filtered_parity.py --min-temperature {cutoff:g}

"""
    report += pooled[["method", "quantity", "paired_points_before", "excluded_at_or_below_cutoff", "paired_points_after", "rmse_all_paired_points", "rmse_log10_positive_pairs"]].to_string(index=False) + "\n"
    (output / "report.txt").write_text(report, encoding="utf-8")
    write_gallery(output, selection_note)
    print(f"Completed: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
