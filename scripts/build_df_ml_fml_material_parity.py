"""Export material-wise sigma/ZT parity figures from existing predictions.

FML sigma is exactly the data used by the exp006 focus overview. Its ZT is
recomputed from that sigma, not copied from the older Step5B downstream model.
Run from any directory; no training or source-file changes are performed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from experiments.exp005.predict_sigma_ml_by_material_system_step32 import classify_samples
from experiments.exp006.build_focus_broad_families_uniform_axis_scatter import (
    GROUP_SPECS, DEFAULT_BROAD_DATA, DEFAULT_SIGE_DATA, source_path, clean_rows,
)

OUT = ROOT / "data/output/df_ml_fml_material_parity_20261025"
STEP22 = ROOT / "data/output/starrydata2_step22_fitting_vs_ml_comparison/step22_row_level_comparison.csv"
STEP32 = ROOT / "data/output/starrydata2_step32_sigma_ml_by_material_system/sigma_ml_doi_holdout_predictions_by_material_system_step32.csv"
CLASS32 = ROOT / "data/output/starrydata2_step32_sigma_ml_by_material_system/sample_material_system_classification_step32.csv"
ZT_SOURCE = ROOT / "data/output/exp006_pf_zt_from_sigma_prediction_full_curves/exp006_pf_zt_row_predictions.csv"
GROUPS = [(title, key, key.replace("::", "_")) for title, key, _, _ in GROUP_SPECS]
GROUPS.append(("ハーフホイスラー系", "broad::half_heusler", "broad_half_heusler"))
COLORS = {"p": "#1f77b4", "n": "#d62728", "other": "#777777"}
MARKERS = {"p": "o", "n": "^", "other": "s"}
SOURCES: set[Path] = set()


def read(path: Path) -> pd.DataFrame:
    SOURCES.add(path)
    return pd.read_csv(path, low_memory=False)


def numerical(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def standardized(d, group, sigma_obs, sigma_pred, seebeck, kappa, zt_obs, carrier, temperature="temperature_K"):
    out = pd.DataFrame({
        "sample_key": d.sample_key,
        "temperature_K": numerical(d[temperature]),
        "material_group": d[group],
        "carrier": d[carrier].where(d[carrier].isin(["p", "n"]), "other"),
        "sigma_obs": numerical(d[sigma_obs]),
        "sigma_pred": numerical(d[sigma_pred]),
        "S_obs_V_per_K": numerical(d[seebeck]),
        "kappa_obs_W_per_mK": numerical(d[kappa]),
        "zt_obs": numerical(d[zt_obs]),
    })
    out["zt_pred"] = out.S_obs_V_per_K**2 * out.sigma_pred * out.temperature_K / out.kappa_obs_W_per_mK
    valid = out.sigma_pred.gt(0) & out.temperature_K.gt(0) & out.kappa_obs_W_per_mK.gt(0)
    out.loc[~valid, "zt_pred"] = np.nan
    out["zt_pred"] = numerical(out.zt_pred)
    assert out.material_group.notna().all()
    assert not out.duplicated(["sample_key", "temperature_K"]).any()
    assert set(out.material_group) <= {g[1] for g in GROUPS}
    return out


def load_data():
    old = read(STEP22)
    # Keep the saved Step32 classification where available; apply its existing
    # classifier only to samples absent from that saved mapping.
    classification = read(CLASS32)[["sample_key", "material_group_key_broad_family"]]
    assert not classification.sample_key.duplicated().any()
    missing = old[~old.sample_key.isin(classification.sample_key)]
    if len(missing):
        extra = classify_samples(missing)[classification.columns]
        classification = pd.concat([classification, extra], ignore_index=True)
    old = old.merge(classification, on="sample_key", validate="many_to_one")
    df = standardized(old, "material_group_key_broad_family", "sigma_obs_S_per_m_step11",
                      "sigma_pred_S_per_m_step12", "seebeck_obs_V_per_K_step11",
                      "kappa_obs_W_per_mK_step11", "zt_obs_dimensionless_step11", "n_or_p")
    np.testing.assert_allclose(df.zt_pred, old.zt_pred_fitting_step21, equal_nan=True, rtol=1e-10)

    ml_raw = read(STEP32)
    # Exact sample/temperature join; no interpolation or cross-sample matching.
    observation_cols = ["sample_key", "temperature_K", "seebeck_obs_V_per_K_step11",
                        "kappa_obs_W_per_mK_step11", "zt_obs_dimensionless_step11",
                        "sigma_pred_ML_S_per_m_step20", "zt_pred_ML_step21"]
    ml_raw = ml_raw.merge(old[observation_cols], on=["sample_key", "temperature_K"],
                          how="left", validate="one_to_one", indicator=True)
    assert ml_raw._merge.eq("both").all(), "Unmatched ML observation rows"
    np.testing.assert_allclose(ml_raw.sigma_pred_ML_S_per_m_step32, ml_raw.sigma_pred_ML_S_per_m_step20, rtol=1e-10)
    ml = standardized(ml_raw, "material_group_key_broad_family", "sigma_obs_S_per_m_step11",
                      "sigma_pred_ML_S_per_m_step32", "seebeck_obs_V_per_K_step11",
                      "kappa_obs_W_per_mK_step11", "zt_obs_dimensionless_step11", "n_or_p")
    np.testing.assert_allclose(ml.zt_pred, ml_raw.zt_pred_ML_step21, equal_nan=True, rtol=1e-10)

    parts = []
    for _, group, safe, kind in GROUP_SPECS:
        part = clean_rows(read(source_path(DEFAULT_BROAD_DATA, DEFAULT_SIGE_DATA, safe, kind)))
        part["group"] = group
        parts.append(part)
    focus = pd.concat(parts, ignore_index=True)
    assert not focus.row_id.duplicated().any()
    downstream = read(ZT_SOURCE)
    cols = ["row_id", "sample_key", "temperature_K", "seebeck_obs_V_per_K",
            "kappa_obs_W_per_mK", "ZT_obs", "sigma_obs_S_per_m",
            "kappa_temperature_delta_K", "zt_temperature_delta_K"]
    merged = focus.merge(downstream[cols], on="row_id", how="left", suffixes=("", "_downstream"),
                         validate="one_to_one", indicator=True)
    assert merged._merge.eq("both").all(), "Unmatched FML observation rows"
    assert merged.sample_key.eq(merged.sample_key_downstream).all()
    np.testing.assert_allclose(merged.T_K, merged.temperature_K)
    np.testing.assert_allclose(merged.sigma_S_per_m, merged.sigma_obs_S_per_m)
    np.testing.assert_allclose(merged.S_uV_per_K * 1e-6, merged.seebeck_obs_V_per_K)
    for column in ["kappa_temperature_delta_K", "zt_temperature_delta_K"]:
        assert merged[column].dropna().abs().le(2.0 + 1e-9).all()
    fml = standardized(merged, "group", "sigma_S_per_m", "sigma_pred_S_per_m",
                       "seebeck_obs_V_per_K", "kappa_obs_W_per_mK", "ZT_obs", "carrier_type")
    fml["row_id"] = merged.row_id
    fml["kappa_temperature_delta_K"] = merged.kappa_temperature_delta_K
    fml["zt_temperature_delta_K"] = merged.zt_temperature_delta_K
    return {"DF": df, "ML": ml, "FML": fml}


def paired(d, quantity):
    x, y = d[f"{quantity}_obs"], d[f"{quantity}_pred"]
    valid = np.isfinite(x) & np.isfinite(y)
    if quantity == "sigma":
        valid &= x.gt(0) & y.gt(0)
    return d.loc[valid].copy()


def full_limits(frames, quantity):
    arrays = [f[[f"{quantity}_obs", f"{quantity}_pred"]].to_numpy().ravel() for f in frames if len(f)]
    if not arrays:
        return (1e2, 1e8) if quantity == "sigma" else (0, 3)
    values = np.concatenate(arrays)
    if quantity == "sigma":
        return 10.0**math.floor(np.log10(values.min())), 10.0**math.ceil(np.log10(values.max()))
    return min(0., float(values.min()) * 1.08), max(0.1, float(values.max()) * 1.05)


def draw(ax, rows, title, quantity, limits, small=False):
    xcol, ycol = f"{quantity}_obs", f"{quantity}_pred"
    lo, hi = limits
    for carrier in COLORS:
        d = rows[rows.carrier.eq(carrier)]
        ax.scatter(d[xcol], d[ycol], s=7 if small else 14, alpha=.55,
                   color=COLORS[carrier], marker=MARKERS[carrier], edgecolors="none", rasterized=True)
    line = np.geomspace(lo, hi, 100) if quantity == "sigma" else np.linspace(max(0, lo), hi, 100)
    ax.plot(line, line, color="black", lw=.9)
    for factor, style in [(2, "--"), (10, ":")]:
        for multiplier in [factor, 1/factor]:
            ax.plot(line, line*multiplier, color="gray", lw=.55, ls=style, alpha=.6)
    if quantity == "sigma":
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(which="both", alpha=.22, lw=.3)
    ax.tick_params(labelsize=7 if small else 10)
    ax.set_title(f"{title}  (N={len(rows):,})", fontsize=8.5 if small else 12, pad=7)
    outside = int((~(rows[xcol].between(lo, hi) & rows[ycol].between(lo, hi))).sum())
    if rows.empty:
        ax.text(.5, .5, "比較可能なデータなし", transform=ax.transAxes, ha="center", fontsize=9, color="#666666")
    elif outside:
        ax.text(.03, .96, f"軸範囲外: {outside:,}点", transform=ax.transAxes, va="top", fontsize=7 if small else 9,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 2})
    return outside


def legend(fig):
    handles = [Line2D([], [], marker=MARKERS[c], linestyle="none", color=COLORS[c],
                      markersize=5, label=label) for c, label in [("p", "p-type"), ("n", "n-type"), ("other", "mixed / unknown")]]
    handles += [Line2D([], [], color="black", lw=1, label="y=x"),
                Line2D([], [], color="gray", ls="--", label="factor 2"),
                Line2D([], [], color="gray", ls=":", label="factor 10")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9)


def save(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=dpi, facecolor="white")
    plt.close(fig)


def render(data, output, dpi, selection_note=""):
    title_suffix = f"  |  {selection_note}" if selection_note else ""
    records = []
    for quantity in ["sigma", "zt"]:
        tables = {method: paired(d, quantity) for method, d in data.items()}
        fixed = (1e2, 1e8) if quantity == "sigma" else (0., 3.)
        full = {key: full_limits([t[t.material_group.eq(key)] for t in tables.values()], quantity) for _, key, _ in GROUPS}
        pooled_full = full_limits(list(tables.values()), quantity)
        xname = "Observed σ (S/m)" if quantity == "sigma" else "Observed ZT"
        yname = "Predicted σ (S/m)" if quantity == "sigma" else "Predicted ZT"
        for method, t in tables.items():
            folder = output / quantity / method
            for variant in ["fixed", "full_range"]:
                fig, axes = plt.subplots(4, 4, figsize=(14.5, 14.2))
                fig.suptitle(f"{method}  |  {'σ' if quantity == 'sigma' else 'ZT'}  |  材料系別・実測値と予測値{title_suffix}", fontsize=18, y=.987)
                for ax, (title, key, _) in zip(axes.flat, GROUPS):
                    draw(ax, t[t.material_group.eq(key)], title, quantity,
                         fixed if variant == "fixed" else full[key], small=True)
                fig.supxlabel(xname, fontsize=13, y=.039)
                fig.supylabel(yname, fontsize=13, x=.012)
                legend(fig)
                fig.subplots_adjust(left=.07, right=.982, bottom=.095, top=.94, wspace=.28, hspace=.30)
                save(fig, folder / f"overview_{variant}", dpi)

            for title, key, slug in GROUPS:
                d = t[t.material_group.eq(key)]
                outside = 0
                panel_title = f"{method} | {title}{title_suffix}"
                if selection_note and len(title) > 12:
                    panel_title = f"{method} | {title}\n{selection_note}"
                for variant in ["fixed", "full_range"]:
                    fig, ax = plt.subplots(figsize=(6.3, 6.0))
                    n = draw(ax, d, panel_title, quantity, fixed if variant == "fixed" else full[key])
                    if variant == "fixed":
                        outside = n
                    ax.set_xlabel(xname)
                    ax.set_ylabel(yname)
                    fig.subplots_adjust(left=.16, right=.95, top=.9, bottom=.13)
                    save(fig, folder / "materials" / f"{slug}_{variant}", dpi)
                error = d[f"{quantity}_pred"] - d[f"{quantity}_obs"]
                records.append({"method": method, "quantity": quantity, "material_group": key,
                                "material_name": title, "paired_points": len(d), "samples": d.sample_key.nunique(),
                                "outside_fixed_axes": outside, "rmse_all_paired_points": np.sqrt((error**2).mean()),
                                "fixed_png": f"{quantity}/{method}/materials/{slug}_fixed.png",
                                "full_range_png": f"{quantity}/{method}/materials/{slug}_full_range.png"})
            for variant, bounds in [("fixed", fixed), ("full_range", pooled_full)]:
                fig, ax = plt.subplots(figsize=(7, 6.8))
                draw(ax, t, f"{method} | 全材料を重ねた散布図{title_suffix}", quantity, bounds)
                ax.set_xlabel(xname)
                ax.set_ylabel(yname)
                legend(fig)
                fig.subplots_adjust(left=.16, right=.95, top=.92, bottom=.16)
                save(fig, folder / f"all_materials_pooled_{variant}", dpi)
            print(f"{method} {quantity}: {len(t):,} paired points; 16 individual panels + overview + pooled", flush=True)
    return pd.DataFrame(records)


def write_index(output):
    cards = []
    for quantity in ["sigma", "zt"]:
        for method in ["DF", "ML", "FML"]:
            base = f"{quantity}/{method}"
            cards.append(f'<section><h2>{method} / {quantity}</h2><a href="{base}/overview_fixed.png"><img src="{base}/overview_fixed.png"></a>'
                         f'<p><a href="{base}/overview_fixed.pdf">PDF</a> · <a href="{base}/overview_full_range.png">全範囲一覧</a> · '
                         f'<a href="{base}/all_materials_pooled_fixed.png">全材料を重ねた図</a></p></section>')
    rows = []
    for title, _, slug in GROUPS:
        cells = []
        for quantity in ["sigma", "zt"]:
            for method in ["DF", "ML", "FML"]:
                base = f"{quantity}/{method}/materials/{slug}"
                cells.append(f'<td><a href="{base}_fixed.png">PNG</a> / <a href="{base}_fixed.pdf">PDF</a> / <a href="{base}_full_range.png">全範囲</a></td>')
        rows.append(f'<tr><th>{html.escape(title)}</th>{"".join(cells)}</tr>')
    content = '<!doctype html><html lang="ja"><meta charset="utf-8"><title>DF・ML・FML 材料系別予測結果</title>'
    content += '<style>body{font:16px sans-serif;margin:32px;color:#243044;background:#f6f7fa}a{color:#145ba0} .cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}section{background:white;padding:16px;border-radius:12px}img{width:100%}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;border:1px solid #dce1e8;text-align:left;font-size:14px}h1{font-size:26px}p{line-height:1.7}</style>'
    content += '<h1>DF・ML・FML：材料系別の予測値 vs 実測値</h1><p>横軸＝実測値、縦軸＝予測値。σは10²〜10⁸ S/m、ZTは0〜3の共通軸。範囲外の点数は図内に表示しています。全範囲版は外れ値・負の実測ZTも含み、材料系ごとに3手法共通の軸を使います。</p>'
    content += '<p>ハーフホイスラー系を含む16分類。比較可能なデータがないパネルはその旨を表示しています。ZTは予測σと実測S・κから計算しています。手法間の評価行・分割は同一ではありません。DFはフィッティング参照です。<a href="report.txt">出典・計算条件</a> / <a href="figure_index.csv">点数・画像一覧CSV</a></p>'
    content += '<div class="cards">' + ''.join(cards) + '</div><h2>材料系別の個別画像</h2><table><tr><th>材料系</th>'
    content += ''.join(f'<th>{q} / {m}</th>' for q in ['σ', 'ZT'] for m in ['DF','ML','FML']) + '</tr>' + ''.join(rows) + '</table></html>'
    (output / "index.html").write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False})
    data = load_data()
    for method, frame in data.items():
        frame.to_csv(args.output / f"plot_source_{method}.csv", index=False, encoding="utf-8-sig")
    index = render(data, args.output, args.dpi)
    index.to_csv(args.output / "figure_index.csv", index=False, encoding="utf-8-sig")
    assert len(index) == 3 * 2 * 16
    for _, row in index.iterrows():
        for col in ["fixed_png", "full_range_png"]:
            assert (args.output / row[col]).stat().st_size > 0
    for method, frame in data.items():
        for quantity in ["sigma", "zt"]:
            n = int(index.loc[index.method.eq(method) & index.quantity.eq(quantity), "paired_points"].sum())
            assert n == len(paired(frame, quantity)), "Material partition lost or duplicated rows"
    manifest = [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SOURCES)]
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = """DF / ML / FML material-wise parity figures (2026-10-25)

DF: Step22 direct-fit sigma, all saved primary DOI-test comparison rows. This
uses the evaluated sample's fitted tau: a fitting reference, not held-out DF.
ML: Step32 DOI-holdout sigma rows, matching the source of Step38 material plots.
The Step22 S/kappa/reported-ZT join is exact on sample_key and temperature_K.
All ML sigma values and reconstructed ZT are checked against saved Step22 values.
DF/ML classification: saved Step32 sample map; missing samples use the existing
Step32 adapter to the exp006 broad_family classifier.
FML: the exact cleaned 15 input tables used by the requested exp006 focus overview.
Groups and sigma are retained from those tables. S/kappa/reported ZT are joined
from the full-curves downstream file by unique row_id, with sample, temperature,
sigma-observation and S agreement checks. Original nearest-temperature joins
are within +/-2 K; their deltas are included in plot_source_FML.csv.

ZT = S_observed^2 * sigma_predicted * T / kappa_observed. S and kappa are NOT
independent predictions. Only finite observed/predicted pairs are plotted.
Reported ZT is used on the x axis, never ZT calculated from observed sigma.
The older downstream FML ZT column belongs to a different sigma prediction
version: it is intentionally recomputed from the requested focus sigma here.
There is no interpolation, retraining, new fitting, or source-file replacement.

Sigma excludes nonpositive/nonfinite pairs. ZT includes zero/negative measured
values in full-range figures. Fixed axes: sigma 1e2--1e8 S/m, ZT 0--3.
Every fixed panel explicitly counts off-axis points; full-range panels include
all paired points and use identical per-material limits across the 3 methods.
Metrics use all paired points, including points outside fixed display limits.
No random point subsampling is used. N denotes temperature points, not samples.
The 16th classification (half-Heusler) has screening rows in the saved index
but no held-out pairs in these inputs. Empty classifications remain visible.

Each method/quantity has 16 individual panels, a 4x4 overview, and a pooled
all-material scatter, each as fixed/full-range PNG and PDF. The 3 methods use
different native cohorts/splits; these are NOT matched-cohort benchmarks.

Reproduce: python scripts/build_df_ml_fml_material_parity.py
Sources and SHA256: source_manifest.json
All plotted values: plot_source_DF.csv, plot_source_ML.csv, plot_source_FML.csv
Counts, all-point RMSE and image paths: figure_index.csv
"""
    report += "\nPaired counts by method / quantity:\n" + index.groupby(["method", "quantity"]).paired_points.sum().to_string() + "\n"
    (args.output / "report.txt").write_text(report, encoding="utf-8")
    write_index(args.output)
    print(f"Completed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
