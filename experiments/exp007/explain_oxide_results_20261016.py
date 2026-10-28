"""Read-only diagnostic of frozen exp007 predictions; no model fitting."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP_DIR = Path(__file__).resolve().parent
ROOT = EXP_DIR.parents[1]
DATA = EXP_DIR / 'data' / 'processed'
OUT = EXP_DIR / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = DATA / 'phase6_prediction_comparison_rows.csv'
digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
df = pd.read_csv(SOURCE, low_memory=False)
def truth(s):
    return s.astype(str).str.lower().eq('true')
oxide = df[truth(df.is_phase0_broad_oxide)].copy()
paired = oxide[truth(oxide.oxide_v1_default_model_eligible) & truth(oxide.paired_prediction_available)].copy()
assert len(oxide) == 3437
assert len(paired) == 2984 and paired.validation_sample_group_id.nunique() == 552

def metrics(g, prefix):
    x = np.log10(g.sigma_S_per_m.to_numpy(float))
    y = np.log10(g[f'{prefix}_sigma_pred_S_per_m'].to_numpy(float))
    assert np.isfinite(x).all() and np.isfinite(y).all()
    r = y - x
    mae = np.mean(np.abs(r))
    ccc = 2*np.mean((x-x.mean())*(y-y.mean()))/(x.var()+y.var()+(x.mean()-y.mean())**2)
    return dict(n=len(g), samples=g.validation_sample_group_id.nunique(),
                mae=float(mae), rmse=float(np.sqrt(np.mean(r*r))),
                orthogonal=float(mae/math.sqrt(2)), error_factor=float(10**mae),
                bias=float(10**r.mean()), mean_residual=float(r.mean()), ccc=float(ccc),
                pearson=float(np.corrcoef(x,y)[0,1]),
                factor2=float(np.mean(np.abs(r)<=np.log10(2))),
                factor5=float(np.mean(np.abs(r)<=np.log10(5))),
                factor10=float(np.mean(np.abs(r)<=1)),
                over=float(np.mean(r>0)))

old = metrics(paired, 'baseline')
new = metrics(paired, 'phase6')
assert abs(old['mae']-1.0234964425762758)<1e-12
assert abs(new['mae']-0.8754094404717878)<1e-12
original = metrics(oxide, 'baseline')
orig = pd.read_csv(ROOT/'experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_metrics.csv')
orig = orig[(orig.material_group_key=='broad::oxide') & (orig.carrier_subset=='all')].iloc[0]
assert abs(original['ccc']-orig.lins_ccc_log10)<1e-12
assert abs(original['mae']-orig.mae_log10_pred_over_exp)<1e-12
print('OVERALL', json.dumps(dict(exp006_original_3437=original, paired_old=old, paired_new=new)))

group_results=[]
groups=[]
for name,g in paired.groupby('oxide_v1_display_name_en'):
    a,b=metrics(g,'baseline'),metrics(g,'phase6')
    group_results.append(dict(group=name,old=a,new=b))
    groups.append((name,g,a,b))
print('GROUPS',json.dumps(group_results))

# Descriptive bands are diagnostic only and do not change classification or evaluation.
for name,g in [('oxide_all',oxide),('primary_paired',paired)]:
    results=[]
    for lo,hi in [(-np.inf,1),(1,100),(100,10000),(10000,np.inf)]:
        band=g[(g.sigma_S_per_m>=lo)&(g.sigma_S_per_m<hi)]
        if len(band):
            row=dict(sigma_band=f'[{lo},{hi}) S/m',old=metrics(band,'baseline'))
            if name=='primary_paired': row['new']=metrics(band,'phase6')
            results.append(row)
    print('SIGMA_BANDS',name,json.dumps(results))

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
OLD='#87919e'
NEW='#087f8c'
x=np.log10(paired.sigma_S_per_m.to_numpy(float))
ys=[np.log10(paired[f'{p}_sigma_pred_S_per_m'].to_numpy(float)) for p in ['baseline','phase6']]
low=math.floor(min(x.min(),*(y.min() for y in ys)))
high=math.ceil(max(x.max(),*(y.max() for y in ys)))
limits=(low-.3,high+.3)

def axes_lines(ax):
    z=np.array(limits)
    ax.plot(z,z,color='#222222',lw=1,label='y = x')
    for delta in [-1,1]:
        ax.plot(z,z+delta,color='#a23b32',ls='--',lw=.8)
    ax.set(xlim=limits,ylim=limits)
    ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.15)

fig,axs=plt.subplots(1,2,figsize=(12.6,7.8))
fig.subplots_adjust(left=.075,right=.98,bottom=.14,top=.83,wspace=.22)
for ax,y,m,title,color in zip(axs,ys,[old,new],['Original broad oxide','Composition-based oxide_v1'],[OLD,NEW]):
    axes_lines(ax)
    ax.scatter(x,y,s=10,alpha=.27,c=color,edgecolors='none',rasterized=True)
    ax.set_title(title,pad=12)
    ax.set_xlabel('Observed log10 conductivity (S/m)')
    ax.set_ylabel('Predicted log10 conductivity (S/m)')
    ax.text(.035,.96,f"MAE = {m['mae']:.3f}   RMSE = {m['rmse']:.3f}\nLin's CCC = {m['ccc']:.3f}\nWithin factor 10 = {m['factor10']:.1%}",
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
fig.suptitle('Electrical conductivity: same 2,984 rows / 552 samples\nFixed sample holdout; full data range, no point clipping',fontsize=15,y=.97)
fig.text(.5,.035,'Black: exact agreement. Red dashed: factor-10 bounds. No refitting was performed for this diagnostic.',fontsize=10,ha='center')
fig.savefig(OUT/'oxide_parity_before_after_20261016.png',dpi=180)
plt.close(fig)

groups.sort(key=lambda t:t[3]['mae']-t[2]['mae'])
fig,axs=plt.subplots(4,3,figsize=(15,18),layout='constrained')
short={'Co-Mn mixed oxide':'Co + Mn','Mn-containing oxide':'Mn','Mo/W/V/Nb-containing oxide':'Mo / W / V / Nb',
       'In/Sn/Cd-containing oxide':'In / Sn / Cd','Mixed-anion O + S/Se/Te':'O + S / Se / Te',
       'Cu-containing oxide':'Cu','Fe/Ni/Cr-containing oxide':'Fe / Ni / Cr','Ti-containing oxide':'Ti',
       'Other inorganic O-containing':'Other inorganic O','Co-containing oxide':'Co','Zn-containing oxide':'Zn',
       'Mixed-anion O + As/P':'O + As / P'}
for ax,(name,g,a,b) in zip(axs.flat,groups):
    axes_lines(ax)
    xx=np.log10(g.sigma_S_per_m.to_numpy(float))
    ax.scatter(xx,np.log10(g.baseline_sigma_pred_S_per_m),s=16,c=OLD,alpha=.5,label='Before',edgecolors='none')
    ax.scatter(xx,np.log10(g.phase6_sigma_pred_S_per_m),s=11,c=NEW,alpha=.55,label='After',edgecolors='none')
    ax.set_title(f"{short[name]} | {len(g)} rows / {a['samples']} samples",fontsize=12,pad=9)
    ax.text(.035,.96,f"MAE {a['mae']:.3f} -> {b['mae']:.3f}\nRMSE {a['rmse']:.3f} -> {b['rmse']:.3f}",
            transform=ax.transAxes,va='top',fontsize=10,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
axs[0,0].legend(loc='lower right',fontsize=9)
fig.suptitle('Oxide subgroups: same rows before and after classification change\nGrey = broad oxide; teal = oxide_v1. Full range; all panels use identical axes.',fontsize=17)
fig.supxlabel('Observed log10 conductivity (S/m)',fontsize=14)
fig.supylabel('Predicted log10 conductivity (S/m)',fontsize=14)
fig.savefig(OUT/'oxide_parity_by_group_20261016.png',dpi=150)
plt.close(fig)

# Additional view: color identifies carrier type, marker identifies prediction
# variant. Reuse exactly the same paired rows, group order, and axis limits.
carrier_colors = {'p': '#D55E00', 'n': '#0072B2'}
carrier_labels = paired.carrier_type.astype(str).str.strip().str.lower()
assert carrier_labels.isin(carrier_colors).all(), 'Unexpected carrier type in paired rows'
assert len(groups) == 12
fig, axs = plt.subplots(4, 3, figsize=(15.5, 20.4))
fig.subplots_adjust(left=.065, right=.985, bottom=.06, top=.89,
                    wspace=.15, hspace=.30)
plotted_counts = {'baseline': 0, 'phase6': 0}
carrier_summary = []
for ax, (name, g, a, b) in zip(axs.flat, groups):
    axes_lines(ax)
    labels = g.carrier_type.astype(str).str.strip().str.lower()
    counts = {carrier: int(labels.eq(carrier).sum()) for carrier in carrier_colors}
    assert sum(counts.values()) == len(g)
    carrier_summary.append({'group': name, **counts})
    # Draw all Before points first so After visibility does not depend on type.
    for prefix, marker, size, alpha in [('baseline', 'x', 23, .40),
                                        ('phase6', 'o', 15, .65)]:
        for carrier, color in carrier_colors.items():
            part = g.loc[labels.eq(carrier)]
            if part.empty:
                continue
            options = {'linewidths': .7} if marker == 'x' else {'edgecolors': 'none'}
            ax.scatter(np.log10(part.sigma_S_per_m),
                       np.log10(part[f'{prefix}_sigma_pred_S_per_m']),
                       marker=marker, s=size, color=color, alpha=alpha,
                       rasterized=True, **options)
            plotted_counts[prefix] += len(part)
    ax.set_title(f"{short[name]} | {len(g)} rows / {a['samples']} samples\n"
                 f"p-type: {counts['p']} rows; n-type: {counts['n']} rows",
                 fontsize=11, pad=9)
    ax.text(.035, .96,
            f"MAE {a['mae']:.3f} -> {b['mae']:.3f}\n"
            f"RMSE {a['rmse']:.3f} -> {b['rmse']:.3f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', alpha=.9, edgecolor='none'))
handles = [Line2D([], [], linestyle='none', color=carrier_colors[carrier],
                  marker=marker, markersize=7,
                  label=f'{carrier}-type / {variant}')
           for carrier in ('p', 'n')
           for variant, marker in [('Before (broad oxide)', 'x'),
                                   ('After (oxide_v1)', 'o')]]
fig.suptitle('Oxide subgroups: p-type and n-type predictions before and after classification',
             fontsize=16, y=.98)
fig.text(.5, .960,
         'Same 2,984 rows / 552 held-out samples. Full range; identical axes; no point clipping.',
         ha='center', fontsize=11)
fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .947),
           ncol=4, frameon=False, fontsize=10, columnspacing=1.2, handletextpad=.5)
fig.supxlabel('Observed log10 conductivity (S/m)', fontsize=14, y=.033)
fig.supylabel('Predicted log10 conductivity (S/m)', fontsize=14, x=.012)
fig.text(.5, .015,
         'Black: y = x. Red dashed: factor-10 bounds. MAE/RMSE combine p and n; values are Before -> After.',
         ha='center', fontsize=10)
assert plotted_counts == {'baseline': len(paired), 'phase6': len(paired)}
pn_figure = OUT / 'oxide_parity_by_group_pn_20261017.png'
fig.savefig(pn_figure, dpi=170)
plt.close(fig)
print('PN_COUNTS', json.dumps(carrier_summary))
print('PN_VERIFIED', json.dumps(plotted_counts), 'carrier types use source carrier_type column')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==digest
print('VERIFIED: 2,984 paired rows; 552 samples; exp006 metrics reproduced; Phase 7 MAE reproduced; source unchanged.')
print('FIGURES',str(OUT/'oxide_parity_before_after_20261016.png'),str(OUT/'oxide_parity_by_group_20261016.png'),str(pn_figure))
