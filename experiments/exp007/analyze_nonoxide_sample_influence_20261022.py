"""One-sample deletion diagnostics on frozen predictions (no refit or deletion).

Run --analyze with bundled Python; --plot with a matplotlib-capable Python.
The removed sample is selected post hoc from evaluation errors, not a QC rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parent
SOURCE = EXP / 'data/processed/phase1_sigma_residual_rows.csv'
BASELINE = EXP / 'data/processed/phase2_material_visual_summary.csv'
DATA = EXP / 'data/processed/nonoxide_sample_influence_20261022'
FIG = EXP / 'figures/nonoxide_sample_influence_20261022'
REPORT = EXP / 'reports/nonoxide_sample_influence_20261022.txt'
GROUP = 'material_group_key_for_prediction'
SAMPLE = 'validation_sample_group_id'
OBS = 'sigma_observed_S_per_m'
PRED = 'sigma_predicted_S_per_m'
RES = 'residual_log10_pred_over_observed'
GETE = 'broad::GeTe_like'
TARGET = '570::13754'
LOG2 = math.log10(2)
LABELS = {
    'broad::GeTe_like': ('Ge-Te', 'Ge–Te系'),
    'broad::sulfide': ('Sulfides', '硫化物系'),
    'broad::SiGe_like': ('Si-Ge', 'Si–Ge系'),
    'broad::Mg2SiSn_like': ('Mg-Si/Sn', 'Mg–Si/Sn系'),
    'broad::CoSb_skutterudite_like': ('Co-Sb skutterudite-like', 'Co–Sb系'),
    'broad::BiSbTe_tetradymite_like': ('Bi-Sb-Te', 'Bi–Sb–Te系'),
    'broad::selenide': ('Selenides', 'セレン化物系'),
    'broad::telluride': ('Other tellurides', 'テルル化物系'),
    'broad::BiTe_like': ('Bi-Te', 'Bi–Te系'),
    'broad::SbTe_like': ('Sb-Te', 'Sb–Te系'),
    'broad::PbTe_like': ('Pb-Te', 'Pb–Te系'),
    'broad::SnTe_like': ('Sn-Te', 'Sn–Te系'),
    'broad::other_formula_system': ('Other formula systems', 'その他の化学式系'),
    'unknown_material_group': ('Unknown (label category)', '分類不能（材料系ではない）'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def joined(series):
    return '; '.join(sorted(set(series.dropna().astype(str))))


def load():
    raw = pd.read_csv(SOURCE, low_memory=False)
    valid = raw.is_evaluable.astype(str).str.lower().eq('true')
    d = raw.loc[valid & raw[GROUP].ne('broad::oxide')].copy()
    assert len(raw) == 19008 and valid.sum() == 18968
    assert len(d) == 15531 and d.row_id.is_unique
    assert d[SAMPLE].notna().all()
    assert d.groupby(SAMPLE)[GROUP].nunique().max() == 1
    assert (d[[OBS, PRED]] > 0).all().all()
    assert np.isfinite(d[[OBS, PRED]].to_numpy()).all()
    r = np.log10(d[PRED].to_numpy()) - np.log10(d[OBS].to_numpy())
    np.testing.assert_allclose(r, d[RES], atol=1e-12, rtol=1e-12)
    d[RES] = r
    d['_abs'] = abs(r)
    d['_sq'] = r*r
    return raw, d


def metrics(g):
    r = g[RES].to_numpy()
    sample = g.groupby(SAMPLE).agg(mae=('_abs', 'mean'), mse=('_sq', 'mean'))
    return dict(n_rows=len(g), n_samples=len(sample), mae_log10=float(abs(r).mean()),
                rmse_log10=float(np.sqrt(np.mean(r*r))), mean_residual_log10=float(r.mean()),
                factor2=float(np.mean(abs(r) <= LOG2)), factor10=float(np.mean(abs(r) <= 1)),
                sample_equal_mae=float(sample.mae.mean()),
                sample_equal_rmse=float(np.sqrt(sample.mse.mean())))


def sample_stats(g):
    result = []
    for sid, s in g.groupby(SAMPLE, sort=True):
        r = s[RES].to_numpy()
        result.append(dict(sample_id=sid, formula=joined(s.formula_raw), doi=joined(s.doi),
                           paper_id=joined(s.paper_id), carrier_types=joined(s.carrier_type),
                           rows=len(s), t_min_K=float(s.T_K.min()), t_max_K=float(s.T_K.max()),
                           sae=float(abs(r).sum()), sse=float((r*r).sum()),
                           sample_mae=float(abs(r).mean()), sample_mse=float((r*r).mean()),
                           mean_residual=float(r.mean()), median_residual=float(np.median(r)),
                           beyond10=int((abs(r)>1).sum()), over10=int((r>1).sum()), under10=int((r < -1).sum()),
                           factor2_count=int((abs(r)<=LOG2).sum()), factor10_count=int((abs(r)<=1).sum()),
                           observed_min_S_per_m=float(s[OBS].min()), observed_max_S_per_m=float(s[OBS].max()),
                           predicted_min_S_per_m=float(s[PRED].min()), predicted_max_S_per_m=float(s[PRED].max()),
                           min_reference_samples=float(s.train_sample_count.min()),
                           min_reference_papers=float(s.train_paper_count.min())))
    return pd.DataFrame(result)


def analyze():
    DATA.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(EXP)): sha(p) for p in (SOURCE, BASELINE)}
    raw, d = load()
    baseline = pd.read_csv(BASELINE).set_index(GROUP)
    all_records, summaries = [], []
    for key, g in d.groupby(GROUP, sort=True):
        m = metrics(g)
        np.testing.assert_allclose([m['mae_log10'], m['rmse_log10']], baseline.loc[key, ['mae_log10', 'rmse_log10']].to_numpy(float), atol=1e-12)
        ss = sample_stats(g)
        n, ns = len(g), len(ss)
        sae, sse = g._abs.sum(), g._sq.sum()
        records = []
        for s in ss.to_dict('records'):
            nr = n - s['rows']
            assert nr > 0 and ns > 1
            after_mae = (sae - s['sae']) / nr
            after_rmse = math.sqrt(max(0, (sse - s['sse']) / nr))
            r = {GROUP:key, 'material_label_en': LABELS[key][0], 'material_label_ja': LABELS[key][1], **s,
                 'rows_before': n, 'samples_before': ns, 'rows_after': nr, 'samples_after': ns-1,
                 'mae_before': m['mae_log10'], 'mae_after': after_mae,
                 'rmse_before': m['rmse_log10'], 'rmse_after': after_rmse,
                 'mae_reduction_pct': 100*(1-after_mae/m['mae_log10']),
                 'rmse_reduction_pct': 100*(1-after_rmse/m['rmse_log10']),
                 'removed_row_fraction': s['rows']/n, 'removed_sample_fraction': 1/ns,
                 'sae_share': s['sae']/sae, 'sse_share': s['sse']/sse,
                 'factor2_before': m['factor2'],
                 'factor2_after': (int((g._abs<=LOG2).sum())-s['factor2_count'])/nr,
                 'factor10_before': m['factor10'],
                 'factor10_after': (int((g._abs<=1).sum())-s['factor10_count'])/nr,
                 'sample_equal_mae_before': m['sample_equal_mae'],
                 'sample_equal_mae_after': (ss.sample_mae.sum()-s['sample_mae'])/(ns-1),
                 'sample_equal_rmse_before': m['sample_equal_rmse'],
                 'sample_equal_rmse_after': math.sqrt(max(0,(ss.sample_mse.sum()-s['sample_mse'])/(ns-1)))}
            r['sample_equal_mae_reduction_pct'] = 100*(1-r['sample_equal_mae_after']/r['sample_equal_mae_before'])
            r['sample_equal_rmse_reduction_pct'] = 100*(1-r['sample_equal_rmse_after']/r['sample_equal_rmse_before'])
            # Independent direct filtering checks every one-sample deletion.
            rr = g.loc[g[SAMPLE].ne(s['sample_id']), RES].to_numpy()
            np.testing.assert_allclose([after_mae, after_rmse], [abs(rr).mean(), np.sqrt((rr*rr).mean())], atol=2e-12)
            records.append(r)
        table = pd.DataFrame(records)
        table['rank_rmse_reduction'] = table.rmse_reduction_pct.rank(method='min', ascending=False).astype(int)
        table['rank_mae_reduction'] = table.mae_reduction_pct.rank(method='min', ascending=False).astype(int)
        all_records.append(table)
        best = table.sort_values(['rmse_reduction_pct','sample_id'], ascending=[False,True]).iloc[0].to_dict()
        best_mae = table.sort_values(['mae_reduction_pct','sample_id'], ascending=[False,True]).iloc[0]
        ordered = np.sort(ss.sse.to_numpy())[::-1]
        best.update(best_mae_sample_id=best_mae.sample_id, best_mae_formula=best_mae.formula,
                    best_mae_after=best_mae.mae_after, best_mae_reduction_pct=best_mae.mae_reduction_pct,
                    same_top_sample_for_both=best['sample_id']==best_mae.sample_id,
                    samples_to_cover_half_sse=int(np.searchsorted(np.cumsum(ordered), 0.5*sse)+1),
                    top3_sse_fraction=float(ordered[:3].sum()/sse))
        summaries.append(best)
    candidates = pd.concat(all_records, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values('rmse_reduction_pct', ascending=False)
    assert len(summary)==14 and len(candidates)==d[SAMPLE].nunique()==2579
    ge = summary[summary[GROUP].eq(GETE)].iloc[0]
    assert ge.sample_id==TARGET and ge.formula=='Tl2HgGeTe4' and ge.rows==23
    assert ge.rows_before==183 and ge.rows_after==160 and ge.samples_after==29
    candidates.to_csv(DATA/'all_sample_deletion_effects.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(DATA/'material_summary.csv', index=False, encoding='utf-8-sig')
    selected = d.merge(summary[[GROUP,'sample_id']].rename(columns={'sample_id':'selected_validation_sample_id'}), on=GROUP, validate='many_to_one')
    selected = selected[selected[SAMPLE].eq(selected.selected_validation_sample_id)].copy()
    assert len(selected)==summary.rows.sum() and selected[SAMPLE].nunique()==14
    selected.drop(columns=['_abs','_sq']).to_csv(DATA/'top_influence_samples_rows.csv', index=False, encoding='utf-8-sig')
    gete = d[d[GROUP].eq(GETE)].copy()
    gete['excluded_in_diagnostic'] = gete[SAMPLE].eq(TARGET)
    gete.drop(columns=['_abs','_sq']).to_csv(DATA/'gete_plot_source_all_with_exclusion_flag.csv', index=False, encoding='utf-8-sig')
    gete[~gete.excluded_in_diagnostic].drop(columns=['_abs','_sq']).to_csv(DATA/'gete_retained_rows.csv', index=False, encoding='utf-8-sig')
    write_report(summary, len(d), len(candidates))
    for p in (SOURCE, BASELINE):
        assert sha(p)==hashes[str(p.relative_to(EXP))]
    validation = dict(status='PASS', input_sha256=hashes, original_test_rows=len(raw),
                      evaluable_nonoxide_rows=len(d), nonoxide_categories=14, sample_deletions_checked=len(candidates),
                      direct_filter_metric_checks='all candidates', baseline_reconciled='all 14 categories',
                      original_inputs_unchanged=True, refitted=False,
                      selection='maximum row-weighted RMSE reduction within each category; MAE maximum also retained',
                      sample_equal_definition='mean of within-sample MAE; square root of mean within-sample MSE (not median signed residual)',
                      formula_note='post-hoc influence diagnostics, not validated outlier removal or improved model performance')
    (DATA/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2), encoding='utf-8')
    print(summary[['material_label_en','sample_id','formula','rows','mae_before','mae_after','rmse_before','rmse_after','sse_share','mae_reduction_pct','rmse_reduction_pct','samples_to_cover_half_sse']].to_string(index=False))


def write_report(summary, rows, samples):
    lines = ['非酸化物の1試料除外による影響度分析（2026-10-22）', '',
             '目的：Ge–Te系の影響試料を外した予測―実測図と、他材料系で同様の誤差集中があるかを確認する。',
             f'対象：既存のbroad_family_100K / sample holdoutの評価可能な非酸化物{rows:,}測定点・{samples:,}試料。13材料分類＋分類不能の14区分。',
             '方法：各区分内の全試料を1つずつ仮に除外し、その試料に属する全温度点を評価から外して指標を再計算した。',
             '再学習、参照値の変更、元データの削除、予測値の変更は行っていない。これはLOOCVではない。',
             '残差r=log10(予測/実測)。MAE=mean(|r|)、RMSE=sqrt(mean(r^2))。すべてlog10単位。',
             '主一覧の候補は各区分でRMSEの低下率が最大となる試料。MAEの低下率最大の試料は別列にも記録し、異なる場合を明示する。',
             'SSE寄与率はその試料のsum(r^2)/分類全体のsum(r^2)。低下率は100*(除外前−除外後)/除外前。',
             '試料均等では、各試料のMAEを平均し、各試料のMSEを平均してから平方根を取る。以前の試料中央値方式とは区別する。', '',
             '分類別結果（RMSE低下率最大の1試料を除外）']
    for r in summary.to_dict('records'):
        lines += [f"\n{r['material_label_ja']}：{r['formula']}（ID {r['sample_id']}）",
                  f"  DOI：{r['doi']}。温度 {r['t_min_K']:.3f}–{r['t_max_K']:.3f} K、carrier={r['carrier_types']}。",
                  f"  実測欄の範囲 {r['observed_min_S_per_m']:.6g}–{r['observed_max_S_per_m']:.6g} S/m。予測範囲 {r['predicted_min_S_per_m']:.6g}–{r['predicted_max_S_per_m']:.6g} S/m。",
                  f"  {int(r['rows'])}/{int(r['rows_before'])}行、1/{int(r['samples_before'])}試料を除外。残り{int(r['rows_after'])}行・{int(r['samples_after'])}試料。",
                  f"  MAE {r['mae_before']:.6f} → {r['mae_after']:.6f}（{r['mae_reduction_pct']:.2f}%低下）。RMSE {r['rmse_before']:.6f} → {r['rmse_after']:.6f}（{r['rmse_reduction_pct']:.2f}%低下）。",
                  f"  SSE寄与率 {100*r['sse_share']:.2f}%。除外点割合 {100*r['removed_row_fraction']:.2f}%。10倍超の過大/過小点 {int(r['over10'])}/{int(r['under10'])}。",
                  f"  試料均等MAE {r['sample_equal_mae_before']:.6f} → {r['sample_equal_mae_after']:.6f}。試料均等RMSE {r['sample_equal_rmse_before']:.6f} → {r['sample_equal_rmse_after']:.6f}。",
                  f"  SSEの50%に達する最小試料数：{int(r['samples_to_cover_half_sse'])}。上位3試料のSSE寄与：{100*r['top3_sse_fraction']:.1f}%。"]
        if not r['same_top_sample_for_both']:
            lines += [f"  MAE低下最大は別試料：{r['best_mae_formula']}（{r['best_mae_sample_id']}）、MAE={r['best_mae_after']:.6f}、{r['best_mae_reduction_pct']:.2f}%低下。"]
    lines += ['', '結果の要約',
              '・Ge–Te：Tl2HgGeTe4の23/183行がSSEの99.49%を占める。除外後160行・29試料、MAE 0.109、RMSE 0.156、Factor-2 98.75%。1試料集中が最も顕著。',
              '・硫化物：CoS2の88/892行がSSEの60.72%。RMSEは33.99%低下するが、試料均等RMSEの低下は4.50%。温度点数の多さが行均等指標への影響を増幅している。',
              '・Sn–Te：Ir(Sn1.4In0.1)Te1.5の8行がSSEの57.26%。Bi–Sb–Te：(Bi0.48Sb1.52)Te3の21行が52.11%。この2群でも1試料に誤差が集中する例が確認できる。',
              '・Co–Sb：In0.5Co4Sb12の4行がSSEの42.92%。RMSEは24.34%低下するがMAEは6.36%低下。最大影響試料は極端な過小予測であり、群内で過大予測点が多いこととは別の現象。',
              '  この4行は実測電気伝導率欄が約1.9～2.3×10^19 S/m、予測は約6.1×10^4～1.1×10^5 S/m。元の物性名、単位、変換処理の確認を優先する候補。ただし原論文未照合のため誤記や物性の取り違えとは断定しない。',
              '・Si–Ge：YbCu2Si0.50Ge1.50の27行除外でRMSEは17.09%低下するが、残る誤差も大きい。',
              '・Bi–Te、Sb–Te、Mg–Si/Sn、Pb–Te、その他、セレン化物、テルル化物では最大影響の1試料除外でもRMSE低下は約3.9～8.1%。Ge–Teほど1試料に支配されていない。',
              '・その他の化学式系はSSEの半分に達するのに20試料、セレン化物は9試料を要する。少数1試料の除外だけでは説明できない。',
              '・分類不能はundefinedの1試料で変化が大きいが、まずラベル/材料同定を確認すべき別枠。',
              '', '解釈と限界',
              '・Ge–Te系で1試料の寄与が極端に大きいかを、測定点割合とSSE寄与率の両方で確認する。',
              '・他材料系でも1試料の影響はあり得るが、除外後の誤差、残る試料数、試料均等評価も併記する。',
              '・MAEとRMSEで最大影響試料が違う場合がある。少数の極端値に敏感なRMSEだけで「群全体が改善」と断定しない。',
              '・外れた試料を評価結果から事後的に選んでいるため、除外後の値は楽観的な診断値であり、新しいモデル性能ではない。',
              '・測定誤り、単位誤り、誤分類、モデルの適用範囲外かは、原論文や元データの確認なしに確定できない。',
              '・分類不能は材料系ではなくラベル上の区分で、少数試料の結果を一般化しない。',
              '・図は表示対象の全点を含み、対数軸で縦横を等スケールにする。比較図は除外前後で共通範囲、単独の除外後図は残存全点に合わせた範囲。', '',
              '成果物', 'data/processed/nonoxide_sample_influence_20261022/material_summary.csv：14区分の代表候補と前後指標',
              'data/processed/nonoxide_sample_influence_20261022/all_sample_deletion_effects.csv：全試料の除外効果',
              'data/processed/nonoxide_sample_influence_20261022/top_influence_samples_rows.csv：代表候補の元の測定行',
              'figures/nonoxide_sample_influence_20261022/：Ge–Te図、全分類の比較図、影響度一覧図',
              '出典：data/processed/phase1_sigma_residual_rows.csv（元の予測と実測）、phase2_material_visual_summary.csv（基準指標の照合）']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8-sig')


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    FIG.mkdir(parents=True, exist_ok=True)
    _, d = load()
    summary = pd.read_csv(DATA/'material_summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,
                         'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    plot_records = []

    def bounds(g):
        logs = np.log10(g[[OBS,PRED]].to_numpy())
        lo, hi = logs.min(), logs.max()
        pad = max(0.12, (hi-lo)*0.07)
        return 10**(lo-pad),10**(hi+pad)

    def panel(ax, g, title, limits, highlight=None):
        lo, hi = limits
        xx = np.geomspace(lo,hi,300)
        ax.fill_between(xx,xx/2,xx*2,color='#BBC7D5',alpha=.25,zorder=0)
        ax.plot(xx,xx,ls='--',color='#424B57',lw=1.3,zorder=1)
        highlight_mask = g[SAMPLE].eq(highlight) if highlight else pd.Series(False,index=g.index)
        displayed = 0
        for typ,color in [('p','#D55E00'),('n','#0072B2')]:
            v = g[g.carrier_type.eq(typ) & ~highlight_mask]
            ax.scatter(v[OBS],v[PRED],s=27,color=color,alpha=.58,edgecolors='none',zorder=2)
            displayed += len(v)
        unknown = g[~g.carrier_type.isin(['p','n']) & ~highlight_mask]
        ax.scatter(unknown[OBS],unknown[PRED],s=27,color='#777777',alpha=.6,zorder=2)
        displayed += len(unknown)
        v = g[highlight_mask]
        if len(v):
            ax.scatter(v[OBS],v[PRED],s=43,marker='x',color='#9C2042',linewidths=1.5,zorder=3)
        displayed += len(v)
        assert displayed==len(g)
        assert ((g[[OBS,PRED]]>=lo)&(g[[OBS,PRED]]<=hi)).all().all()
        ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,
               xlabel='Observed conductivity (S/m)',ylabel='Predicted conductivity (S/m)')
        ax.set_aspect('equal', adjustable='box')
        for axis in (ax.xaxis,ax.yaxis):
            axis.set_major_locator(LogLocator(base=10,numticks=7))
            axis.set_minor_formatter(NullFormatter())
        ax.grid(which='major',color='#DDE2E7',lw=.6,zorder=0)
        m=metrics(g)
        # Put metrics above the axes so they cannot hide the influential points.
        ax.set_title(f"{title}\n{m['n_rows']:,} points / {m['n_samples']:,} samples\nlog10 MAE = {m['mae_log10']:.3f}    RMSE = {m['rmse_log10']:.3f}\nFactor-2 = {m['factor2']*100:.1f}%",fontsize=11,pad=12)
        plot_records.append({'panel':title,'n_points':len(g),'n_highlighted':len(v),
                             'axis_min':lo,'axis_max':hi,'all_points_in_range':True})

    handles = [Line2D([],[],marker='o',ls='',color='#D55E00',label='p-type'),
               Line2D([],[],marker='o',ls='',color='#0072B2',label='n-type'),
               Line2D([],[],marker='x',ls='',color='#9C2042',label='Sample selected for exclusion'),
               Line2D([],[],ls='--',color='#424B57',label='Prediction = observation')]

    def save(fig, stem):
        fig.savefig(FIG/(stem+'.png'),dpi=220,bbox_inches='tight')
        plt.close(fig)

    for row in summary.to_dict('records'):
        key,sid=row[GROUP],row['sample_id']
        g=d[d[GROUP].eq(key)]
        kept=g[g[SAMPLE].ne(sid)]
        fig,axes=plt.subplots(1,2,figsize=(12.6,6.3))
        lim=bounds(g)
        panel(axes[0],g,'All evaluation samples',lim,sid)
        panel(axes[1],kept,'One selected sample excluded',lim)
        fig.suptitle(f"{LABELS[key][0]}: single-sample influence",fontsize=17,y=.99)
        fig.text(.5,.925,f"Excluded: {row['formula']} | ID {sid} | {int(row['rows'])} points",ha='center',fontsize=11)
        fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,frameon=False,fontsize=9)
        fig.text(.5,.025,'Frozen predictions; post-hoc diagnostic only. Same full-range axes; shading indicates factor 2.',ha='center',fontsize=10,color='#555555')
        fig.subplots_adjust(top=.73,bottom=.2,wspace=.28)
        stem = 'gete_before_after_same_axes' if key==GETE else key.replace('broad::','').replace('::','_')+'_before_after_same_axes'
        save(fig,stem)
        if key==GETE:
            fig,ax=plt.subplots(figsize=(7.6,7.4))
            panel(ax,kept,'Ge-Te: after excluding Tl2HgGeTe4',bounds(kept))
            fig.legend(handles=[handles[0],handles[1],handles[3]],loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=10)
            fig.text(.5,.025,'160 retained points; unchanged predictions. Full retained range, no point clipping.',ha='center',fontsize=9)
            fig.subplots_adjust(left=.14,right=.95,top=.80,bottom=.18)
            save(fig,'gete_parity_excluding_Tl2HgGeTe4')

    fig,axes=plt.subplots(1,2,figsize=(13.5,8.8),sharey=True)
    s=summary.sort_values('rmse_reduction_pct',ascending=True)
    y=np.arange(len(s))
    colors=['#A0A0A0' if k=='unknown_material_group' else '#D55E00' if k==GETE else '#287FA6' for k in s[GROUP]]
    for ax,metric,title in zip(axes,['mae_reduction_pct','rmse_reduction_pct'],['log10 MAE reduction','log10 RMSE reduction']):
        ax.barh(y,s[metric],color=colors,height=.65)
        for yy,val in zip(y,s[metric]): ax.text(val+1,yy,f'{val:.1f}%',va='center',fontsize=10)
        ax.set_xlim(min(-1,float(s[metric].min())-2),max(105,float(s[metric].max())+15))
        ax.set_title(title)
        ax.set_xlabel('Reduction after excluding one sample (%)')
        ax.grid(axis='x',alpha=.22)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y,s.material_label_en)
    fig.suptitle('Non-oxide categories: influence of one evaluation sample',fontsize=17,y=.98)
    fig.text(.5,.035,'Each category removes its RMSE-most-influential sample. MAE may favor a different sample. No model refit.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.94))
    save(fig,'nonoxide_single_sample_influence_summary')
    pd.DataFrame(plot_records).to_csv(DATA/'figure_point_validation.csv',index=False,encoding='utf-8-sig')
    print(f'Plots: {len(list(FIG.glob("*.png")))} PNG files; {len(plot_records)} panels verified with no point clipping.')


def bundle():
    import zipfile
    target=EXP/'exports/nonoxide_sample_influence_20261022.zip'
    target.parent.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__).resolve(),REPORT]+sorted(DATA.glob('*'))+sorted(FIG.glob('*.png'))
    assert all(p.is_file() for p in files)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,path.relative_to(EXP).as_posix())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None and len(z.infolist())==len(files)
    print(f'ZIP validated: {target} ({target.stat().st_size} bytes, {len(files)} files)')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analyze',action='store_true')
    parser.add_argument('--plot',action='store_true')
    parser.add_argument('--bundle',action='store_true')
    args=parser.parse_args()
    if not any((args.analyze,args.plot,args.bundle)): parser.error('Select --analyze, --plot and/or --bundle')
    if args.analyze: analyze()
    if args.plot: plot()
    if args.bundle: bundle()
