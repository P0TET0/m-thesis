"""Controlled broad/fine oxide DF/ML refits, with frozen exp007 FML predictions."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LOCAL_PACKAGES = ROOT / '.runtime/python-packages'
if LOCAL_PACKAGES.is_dir():
    sys.path.insert(0, str(LOCAL_PACKAGES))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
import joblib

from experiments.exp005.fit_tau_eff_step12 import median_excluding
from experiments.exp005.train_tau_eff_ml_step19 import prepare_features, is_leakage_column
from experiments.exp007.oxide_taxonomy_v1 import classify_oxide_v1, RULES, rules_sha256
from experiments.exp006.build_focus_broad_families_uniform_axis_scatter import GROUP_SPECS

SOURCE = ROOT / 'experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.csv'
FEATURES = ROOT / 'data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_feature_matrix_step18.csv'
FML_PATHS = {
    'broad': ROOT / 'experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_default.csv',
    'fine': ROOT / 'experiments/exp007/data/processed/phase6_oxide_v1_test_predictions.csv',
}
ASSIGNMENTS = ROOT / 'experiments/exp007/data/processed/phase6_oxide_v1_all_row_assignments.csv'
DOWNSTREAM = ROOT / 'data/output/exp006_pf_zt_from_sigma_prediction_full_curves/exp006_pf_zt_row_predictions.csv'
PRIMARY_COHORT = ROOT / 'experiments/exp007/data/processed/phase6_prediction_comparison_rows.csv'
REFERENCE_PLOT_SCRIPT = ROOT / 'experiments/exp007/explain_oxide_results_20261016.py'
DATA = HERE / 'data'
METHODS = ['DF', 'ML', 'FML']
GROUPS = [(key, name) for name, key, _, _ in GROUP_SPECS if key != 'broad::oxide']
GROUPS.insert(-1, ('broad::half_heusler', 'ハーフホイスラー系'))
GROUPS += [(r['group_key'], r['display_name_ja']) for r in RULES if r['rule_id'] != 'OXV1_R999']
NAMES = dict(GROUPS)
NAMES.update(all_materials='全材料（酸化物はexp007対象）', all_oxides='酸化物全体（exp007対象）', eligible_oxides='酸化物・exp007と同じ対象')
GROUPS = [(key, name) for key, name in GROUPS if key not in {
    'oxide_v1::excluded_organic_polymer_or_carbon_composite',
    'oxide_v1::mixed_anion_O_N_or_halogen',
}]
# Match exp007's latest carrier-colored oxide comparison exactly.
CARRIER_COLORS = {'p': '#D55E00', 'n': '#0072B2', 'other': '#777777'}
SOURCES = [SOURCE, FEATURES, ASSIGNMENTS, DOWNSTREAM, *FML_PATHS.values(),
           ROOT / 'experiments/exp007/oxide_taxonomy_v1.py',
           ROOT / 'experiments/exp005/fit_tau_eff_step12.py',
           ROOT / 'experiments/exp005/train_tau_eff_ml_step19.py']


def read(path):
    return pd.read_csv(path, low_memory=False)


def write_csv(frame, name):
    DATA.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA / name, index=False, encoding='utf-8-sig')


def positive(values):
    return np.isfinite(values) & values.gt(0)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def exp007_primary_rows():
    source = read(PRIMARY_COHORT)
    truth = lambda s: s.astype(str).str.lower().eq('true')
    selected = source[truth(source.is_phase0_broad_oxide)
                      & truth(source.oxide_v1_default_model_eligible)
                      & truth(source.paired_prediction_available)].copy()
    assert len(selected) == 2984 and selected.validation_sample_group_id.nunique() == 552
    assert not selected.row_id.duplicated().any()
    assert selected.oxide_v1_group_key.nunique() == 12
    return selected


def align_display_scope(output):
    """Keep exact exp007 primary oxide row IDs for every method; retain nonoxides."""
    reference = exp007_primary_rows()
    ids = set(reference.row_id)
    aligned = {}
    for variant, rows in output.items():
        keep = ~rows.broad_group.eq('broad::oxide') | rows.row_id.isin(ids)
        aligned[variant] = rows.loc[keep].copy()
        for method in METHODS:
            oxide = aligned[variant][aligned[variant].method.eq(method) & aligned[variant].broad_group.eq('broad::oxide')]
            assert set(oxide.row_id) == ids
            assert oxide.oxide_default_eligible.eq(True).all()
        write_csv(aligned[variant], f'plot_source_{variant}.csv')
    fml = aligned['fine'].query("method == 'FML'").set_index('row_id').loc[reference.row_id]
    np.testing.assert_allclose(fml.sigma_pred, reference.phase6_sigma_pred_S_per_m, rtol=1e-12)
    np.testing.assert_allclose(fml.sigma_obs, reference.sigma_S_per_m, rtol=1e-12)
    np.testing.assert_array_equal(fml.carrier, reference.carrier_type)
    counts = []
    for key in ['all_oxides', *reference.oxide_v1_group_key.unique()]:
        old = reference if key == 'all_oxides' else reference[reference.oxide_v1_group_key.eq(key)]
        for method in METHODS:
            current = select(aligned['fine'][aligned['fine'].method.eq(method)], key)
            for quantity in ['sigma', 'zt']:
                valid = paired(current, quantity)
                counts.append({'method': method, 'quantity': quantity, 'group': key, 'group_name': NAMES[key],
                               'exp007_reference_sigma_points': len(old), 'selected_target_rows': len(current),
                               'plotted_points': len(valid), 'samples': valid.sample_key.nunique(),
                               'p_points': int(valid.carrier.eq('p').sum()),
                               'n_points': int(valid.carrier.eq('n').sum()),
                               'missing_prediction_or_observation': len(current) - len(valid)})
    write_csv(pd.DataFrame(counts), 'exp007_alignment_counts.csv')
    manifest = {
        'oxide_scope': 'exact exp007 default-eligible paired row IDs',
        'oxide_target_rows': len(reference), 'oxide_target_samples': 552, 'oxide_groups': 12,
        'nonoxide_scope': 'unchanged', 'carrier_colors': CARRIER_COLORS,
        'prediction_values_changed': False,
        'reference_sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': file_sha256(p)}
                              for p in [PRIMARY_COHORT, REFERENCE_PLOT_SCRIPT]],
        'raw_prediction_hashes': {v: file_sha256(DATA / f'predictions_{v}.csv') for v in output},
    }
    (HERE / 'display_scope_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return aligned


def initialize_rows():
    raw = read(SOURCE)
    cols = ['row_id', 'sample_key', 'doi', 'formula_raw', 'parsed_elements', 'sample_holdout_split',
            'material_group_key', 'T_K', 'sigma_S_per_m', 'S_V_per_K', 'carrier_type', 'sigma_source']
    rows = raw[cols].rename(columns={
        'material_group_key': 'broad_group', 'T_K': 'temperature_K',
        'sigma_S_per_m': 'sigma_obs', 'S_V_per_K': 'S_obs_V_per_K',
        'carrier_type': 'carrier', 'sigma_source': 'measurement_source',
    }).copy()
    assert not rows.row_id.duplicated().any()
    assert rows.groupby('sample_key').sample_holdout_split.nunique().max() == 1
    assert rows.groupby('sample_key').broad_group.nunique().max() == 1
    rows['material_group'] = rows.broad_group
    rows['oxide_status'] = 'nonoxide'
    rows['oxide_default_eligible'] = False
    samples = rows.drop_duplicates('sample_key')
    classifications = []
    for rec in samples[samples.broad_group.eq('broad::oxide')].itertuples():
        c = classify_oxide_v1(rec.formula_raw, rec.parsed_elements)
        classifications.append({'sample_key': rec.sample_key, **c})
    cls = pd.DataFrame(classifications).set_index('sample_key')
    mask = rows.broad_group.eq('broad::oxide')
    for col, source_col in [('material_group', 'oxide_v1_group_key'),
                            ('oxide_status', 'oxide_v1_inclusion_status'),
                            ('oxide_default_eligible', 'oxide_v1_default_model_eligible')]:
        rows.loc[mask, col] = rows.loc[mask, 'sample_key'].map(cls[source_col])
    assert set(rows.material_group) <= set(NAMES), set(rows.material_group) - set(NAMES)
    frozen = read(ASSIGNMENTS).set_index('row_id')
    np.testing.assert_array_equal(rows.loc[mask].set_index('row_id').material_group.reindex(frozen.index),
                                  frozen.material_group_key_phase6)
    rows['temperature_bin'] = np.round(rows.temperature_K / 25.) * 25.
    assert (positive(rows.sigma_obs) & positive(rows.temperature_K)).all()
    write_csv(rows[['row_id', 'sample_key', 'sample_holdout_split', 'broad_group', 'material_group',
                    'oxide_status', 'oxide_default_eligible']], 'classification_and_splits.csv')
    return rows


def empirical_prefactors(rows, group_column):
    """Step12 hierarchy; only train references, with own-sample exclusion in train.

    The sigma values of test rows are never accessed by this function.
    """
    rows = rows.reset_index(drop=True)
    train_mask = rows.sample_holdout_split.eq('train')
    train = rows.loc[train_mask]
    levels = [[group_column, 'carrier', 'measurement_source', 'temperature_bin'],
              [group_column, 'carrier', 'temperature_bin'],
              ['carrier', 'temperature_bin'], ['temperature_bin'], []]
    result = pd.DataFrame(index=rows.index)
    result['prefactor_C'] = np.nan
    result['reference_level'] = ''
    result['reference_samples'] = 0
    for level, keys in enumerate(levels):
        reference = train.groupby(keys, sort=False, dropna=False) if keys else [((), train)]
        lookups = {}
        for key, ref in reference:
            key = key if isinstance(key, tuple) else (key,)
            count = ref.sample_key.nunique()
            if count < 3:
                continue
            values = np.sort(ref.sigma_obs.to_numpy(float))
            own = {sample: np.sort(g.sigma_obs.to_numpy(float))
                   for sample, g in ref.groupby('sample_key', sort=False)}
            lookups[key] = (values, own, count, float(np.median(values)))
        pending = rows.loc[result.prefactor_C.isna()]
        targets = pending.groupby(keys, sort=False, dropna=False) if keys else [((), pending)]
        for key, target in targets:
            key = key if isinstance(key, tuple) else (key,)
            if key not in lookups:
                continue
            values, own, count, median = lookups[key]
            test_idx = target.index[target.sample_holdout_split.eq('test')]
            result.loc[test_idx, ['prefactor_C', 'reference_samples', 'reference_level']] = [median, count, str(level)]
            for sample, g in target[target.sample_holdout_split.eq('train')].groupby('sample_key', sort=False):
                if count - 1 < 3:
                    continue
                value = median_excluding(values, own[sample])
                result.loc[g.index, ['prefactor_C', 'reference_samples', 'reference_level']] = [value, count - 1, str(level)]
        print(f'{group_column}: reference level {level}, {result.prefactor_C.notna().sum():,}/{len(rows):,}', flush=True)
    assert positive(result.prefactor_C).all()
    assert result.reference_samples.ge(3).all()
    return result


def fit_df_ml(rows, feature_matrix, variant):
    group_col = 'broad_group' if variant == 'broad' else 'material_group'
    p = empirical_prefactors(rows, group_col)
    work = pd.concat([rows.reset_index(drop=True), p], axis=1)
    # Step12 log-space least-squares intercept, minimum five points per sample.
    work['log_ratio'] = np.log(work.sigma_obs) - np.log(work.prefactor_C)
    tau = work.groupby('sample_key', sort=True).agg(
        log_tau=('log_ratio', 'mean'), fit_points=('log_ratio', 'size'),
        split=('sample_holdout_split', 'first'), group=(group_col, 'first'))
    tau['fit_valid'] = tau.fit_points.ge(5) & np.isfinite(tau.log_tau)
    work['tau_DF'] = work.sample_key.map(np.exp(tau.log_tau.where(tau.fit_valid)))
    test = work[work.sample_holdout_split.eq('test')].copy()
    df = test.copy()
    df['sigma_pred'] = df.prefactor_C * df.tau_DF
    df['prediction_status'] = np.where(positive(df.sigma_pred), 'ok', 'fewer_than_5_fit_points')
    df['method'] = 'DF'

    # Fixed composition/process features from exp005; no old tau targets or predictions.
    metadata = rows.drop_duplicates('sample_key')[['sample_key', 'sample_holdout_split', group_col]]
    samples = metadata.merge(feature_matrix, on='sample_key', how='left', validate='one_to_one', indicator=True)
    samples['features_available'] = samples['_merge'].eq('both')
    samples['target_log_tau'] = samples.sample_key.map(tau.log_tau.where(tau.fit_valid))
    train = samples.sample_holdout_split.eq('train') & samples.features_available & samples.target_log_tau.notna()
    test_mask = samples.sample_holdout_split.eq('test') & samples.features_available
    assert not (set(samples.loc[train, 'sample_key']) & set(samples.loc[test_mask, 'sample_key']))
    feature_cols = [c for c in feature_matrix if c != 'sample_key'
                    and not c.startswith('material_system__') and not is_leakage_column(c)]
    # Categories are fitted on train samples, so unseen fine groups remain all-zero.
    for group in sorted(samples.loc[train, group_col].unique()):
        name = f'material_category::{group}'
        samples[name] = samples[group_col].eq(group).astype(float)
        feature_cols.append(name)
    X, used, medians, audit = prepare_features(samples, feature_cols, train)
    model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=1)
    model.fit(X.loc[train], samples.loc[train, 'target_log_tau'])
    log_prediction = pd.Series(model.predict(X.loc[test_mask]), index=samples.loc[test_mask, 'sample_key'])
    ml = test.copy()
    ml['tau_ML'] = ml.sample_key.map(np.exp(log_prediction.clip(-50, 50)))
    ml['sigma_pred'] = ml.prefactor_C * ml.tau_ML
    ml['prediction_status'] = np.where(positive(ml.sigma_pred), 'ok', 'missing_existing_feature_vector')
    ml['method'] = 'ML'
    models = HERE / 'models'
    models.mkdir(exist_ok=True)
    joblib.dump({'model': model, 'features': used, 'imputation': medians}, models / f'ml_{variant}.joblib')
    write_csv(audit, f'feature_audit_{variant}.csv')
    write_csv(pd.DataFrame({'feature': used, 'importance': model.feature_importances_}), f'feature_importance_{variant}.csv')
    sample_audit = samples[['sample_key', 'sample_holdout_split', group_col, 'features_available', 'target_log_tau']].copy()
    sample_audit['used_to_train_ML'] = train.to_numpy()
    sample_audit['ML_log_tau_prediction'] = samples.sample_key.map(log_prediction)
    write_csv(sample_audit, f'ml_sample_audit_{variant}.csv')
    write_csv(tau.reset_index(), f'df_fitted_tau_{variant}.csv')
    write_csv(work[['row_id', 'sample_key', 'sample_holdout_split', 'prefactor_C', 'reference_level', 'reference_samples']],
              f'prefactors_{variant}.csv')
    print(f'{variant}: ML trained on {train.sum():,} samples, predicts {test_mask.sum():,}', flush=True)
    return df, ml


def attach_zt(frame, downstream):
    cols = ['row_id', 'sample_key', 'temperature_K', 'seebeck_obs_V_per_K', 'sigma_obs_S_per_m',
            'kappa_obs_W_per_mK', 'ZT_obs', 'kappa_temperature_delta_K', 'zt_temperature_delta_K']
    d = frame.merge(downstream[cols], on='row_id', how='left', validate='one_to_one',
                    suffixes=('', '_zt_source'), indicator=True)
    assert d['_merge'].eq('both').all()
    assert d.sample_key.eq(d.sample_key_zt_source).all()
    np.testing.assert_allclose(d.temperature_K, d.temperature_K_zt_source)
    np.testing.assert_allclose(d.sigma_obs, d.sigma_obs_S_per_m)
    np.testing.assert_allclose(d.S_obs_V_per_K, d.seebeck_obs_V_per_K)
    for c in ['kappa_temperature_delta_K', 'zt_temperature_delta_K']:
        assert d[c].dropna().abs().le(2 + 1e-9).all()
    d['zt_obs'] = d.ZT_obs
    d['zt_pred'] = d.S_obs_V_per_K.pow(2) * d.sigma_pred * d.temperature_K / d.kappa_obs_W_per_mK
    valid = positive(d.sigma_pred) & positive(d.kappa_obs_W_per_mK) & positive(d.temperature_K)
    d.loc[~valid, 'zt_pred'] = np.nan
    d['zt_pred'] = d.zt_pred.replace([np.inf, -np.inf], np.nan)
    keep = ['row_id', 'sample_key', 'doi', 'formula_raw', 'material_group', 'broad_group',
            'oxide_status', 'oxide_default_eligible', 'carrier', 'temperature_K',
            'method', 'prediction_status', 'sigma_obs', 'sigma_pred', 'S_obs_V_per_K',
            'kappa_obs_W_per_mK', 'zt_obs', 'zt_pred', 'kappa_temperature_delta_K', 'zt_temperature_delta_K']
    return d[keep]


def calculate():
    rows = initialize_rows()
    feature_matrix = read(FEATURES)
    assert not feature_matrix.sample_key.duplicated().any()
    downstream = read(DOWNSTREAM)
    output = {}
    for variant in ['broad', 'fine']:
        df, ml = fit_df_ml(rows, feature_matrix, variant)
        frozen = read(FML_PATHS[variant])
        test = rows[rows.sample_holdout_split.eq('test')].copy()
        assert set(frozen.row_id) == set(test.row_id)
        fml = test.merge(frozen[['row_id', 'prediction_status', 'sigma_pred_S_per_m', 'sigma_S_per_m']],
                         on='row_id', validate='one_to_one')
        np.testing.assert_allclose(fml.sigma_obs, fml.sigma_S_per_m)
        fml['sigma_pred'] = fml.sigma_pred_S_per_m.where(fml.prediction_status.eq('ok'))
        fml['method'] = 'FML'
        output[variant] = pd.concat([attach_zt(d, downstream) for d in [df, ml, fml]], ignore_index=True)
        output[variant]['variant'] = variant
        assert not output[variant].duplicated(['method', 'row_id']).any()
        write_csv(output[variant], f'predictions_{variant}.csv')
    write_manifest(rows)
    return output


def paired(rows, q):
    ok = np.isfinite(rows[f'{q}_obs']) & np.isfinite(rows[f'{q}_pred'])
    if q == 'sigma':
        ok &= positive(rows.sigma_obs) & positive(rows.sigma_pred)
    return rows.loc[ok]


def select(rows, group):
    if group == 'all_materials':
        return rows
    if group == 'all_oxides':
        return rows[rows.broad_group.eq('broad::oxide')]
    if group == 'eligible_oxides':
        return rows[rows.oxide_default_eligible.eq(True)]
    return rows[rows.material_group.eq(group)]


def scores(rows, q, weighting='row_equal'):
    d = paired(rows, q)
    x, y = d[f'{q}_obs'], d[f'{q}_pred']
    logmask = positive(x) & positive(y)
    residual = y - x
    log_error = np.log10(y[logmask]) - np.log10(x[logmask])

    def mean(values, table):
        if len(values) == 0:
            return float('nan')
        weights = (1 / table.groupby('sample_key').sample_key.transform('size')).to_numpy() if weighting == 'sample_equal' else None
        return float(np.average(values, weights=weights))

    return {'points': len(d), 'samples': d.sample_key.nunique(), 'log10_points': int(logmask.sum()),
            'mae': mean(abs(residual), d), 'rmse': math.sqrt(mean(residual**2, d)),
            'mae_log10': mean(abs(log_error), d.loc[logmask]),
            'rmse_log10': math.sqrt(mean(log_error**2, d.loc[logmask])),
            'bias_log10': mean(log_error, d.loc[logmask]),
            'within_factor_2': mean(abs(log_error) <= math.log10(2), d.loc[logmask])}


def common_rows(data, q):
    sets = [set(paired(data[data.method.eq(m)], q).row_id) for m in METHODS]
    return set.intersection(*sets)


def summarize(output):
    records, changes = [], []
    groups = ['all_materials', 'all_oxides', 'eligible_oxides', *dict(GROUPS)]
    for variant, frame in output.items():
        for q in ['sigma', 'zt']:
            common = common_rows(frame, q)
            for cohort in ['available', 'common_methods']:
                subset = frame if cohort == 'available' else frame[frame.row_id.isin(common)]
                for method in METHODS:
                    method_rows = subset[subset.method.eq(method)]
                    for group in groups:
                        selected = select(method_rows, group)
                        for weighting in ['row_equal', 'sample_equal']:
                            records.append({'variant': variant, 'quantity': q, 'cohort': cohort,
                                            'method': method, 'group': group, 'group_name': NAMES[group],
                                            'weighting': weighting, **scores(selected, q, weighting)})
    metrics = pd.DataFrame(records)
    write_csv(metrics, 'metrics.csv')
    for method in METHODS:
        for q in ['sigma', 'zt']:
            a = paired(output['broad'][output['broad'].method.eq(method)], q)
            b = paired(output['fine'][output['fine'].method.eq(method)], q)
            common = set(a.row_id) & set(b.row_id)
            a, b = a[a.row_id.isin(common)], b[b.row_id.isin(common)]
            for group in groups:
                for weighting in ['row_equal', 'sample_equal']:
                    before, after = scores(select(a, group), q, weighting), scores(select(b, group), q, weighting)
                    assert before['points'] == after['points']
                    changes.append({'method': method, 'quantity': q, 'group': group, 'group_name': NAMES[group],
                                    'weighting': weighting, 'paired_points': before['points'],
                                    **{f'broad_{k}': v for k, v in before.items()},
                                    **{f'fine_{k}': v for k, v in after.items()},
                                    **{f'delta_{k}': after[k] - before[k] for k in ['mae', 'rmse', 'mae_log10', 'rmse_log10']}})
    write_csv(pd.DataFrame(changes), 'classification_effect.csv')
    statuses = output['fine'].groupby(['method', 'prediction_status'], dropna=False).agg(
        points=('row_id', 'size'), samples=('sample_key', 'nunique')).reset_index()
    write_csv(statuses, 'prediction_status_counts.csv')
    return metrics, statuses


def write_manifest(rows):
    def digest(p):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    train = rows[rows.sample_holdout_split.eq('train')]
    test = rows[rows.sample_holdout_split.eq('test')]
    manifest = {
        'taxonomy': 'oxide_v1', 'taxonomy_sha256': rules_sha256(),
        'input_rows': len(rows), 'train_rows': len(train), 'test_rows': len(test),
        'train_samples': train.sample_key.nunique(), 'test_samples': test.sample_key.nunique(),
        'train_test_sample_overlap': len(set(train.sample_key) & set(test.sample_key)),
        'train_test_doi_overlap': len(set(train.doi.dropna()) & set(test.doi.dropna())),
        'df_ml_bin_width_K': 25, 'fml_bin_width_K': 100, 'min_reference_samples': 3,
        'min_df_fit_points': 5, 'temperature_filter': 'all_temperatures',
        'model': {'name': 'ExtraTreesRegressor', 'n_estimators': 300, 'min_samples_leaf': 3, 'random_state': 42},
        'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                     'sklearn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': digest(p)} for p in SOURCES],
    }
    (HERE / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def limits(tables, q):
    arrays = [t[[f'{q}_obs', f'{q}_pred']].to_numpy().ravel() for t in tables if len(t)]
    if not arrays:
        return (1e-2, 1e8) if q == 'sigma' else (0., 3.)
    values = np.concatenate(arrays)
    if q == 'sigma':
        return 10.**math.floor(np.log10(values.min())), 10.**math.ceil(np.log10(values.max()) + .001)
    return min(0., values.min() * 1.08), max(.1, values.max() * 1.08)


def draw(ax, rows, q, bounds, title, small=False):
    xcol, ycol = f'{q}_obs', f'{q}_pred'
    carrier_labels = rows.carrier.where(rows.carrier.isin(['p', 'n']), 'other')
    for carrier, color in CARRIER_COLORS.items():
        group = rows[carrier_labels.eq(carrier)]
        ax.scatter(group[xcol], group[ycol], color=color, s=5 if small else 15,
                   alpha=.65, marker='o', edgecolors='none', rasterized=True,
                   label=carrier)
    assert sum(len(collection.get_offsets()) for collection in ax.collections) == len(rows)
    lo, hi = bounds
    line = np.geomspace(lo, hi, 120) if q == 'sigma' else np.linspace(max(lo, 0), hi, 120)
    ax.plot(line, line, color='#222222', lw=.9)
    for factor, style, color in [(2, ':', '#777777'), (10, '--', '#a23b32')]:
        for k in [factor, 1/factor]:
            ax.plot(line, line*k, color=color, lw=.6, ls=style, alpha=.7)
    if q == 'sigma':
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(which='major', alpha=.2)
    ax.set_title(f'{title}  N={len(rows):,}', fontsize=8 if small else 13)
    ax.tick_params(labelsize=7 if small else 10)
    outside = int((~(rows[xcol].between(lo, hi) & rows[ycol].between(lo, hi))).sum())
    if rows.empty:
        ax.text(.5, .5, '評価可能なデータなし', ha='center', transform=ax.transAxes, fontsize=8 if small else 10)
    elif not small:
        m = scores(rows, q)
        ax.text(.03, .97, f"試料数={m['samples']:,}\nlog10 RMSE={m['rmse_log10']:.3f}\nlog10 MAE={m['mae_log10']:.3f}"
                + (f'\n軸範囲外={outside:,}点' if outside else ''), transform=ax.transAxes, va='top',
                fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .84})
    return outside


def carrier_legend(fig):
    handles = [Line2D([], [], marker='o', ls='', color=CARRIER_COLORS[k], markersize=5, label=label)
               for k, label in [('p', 'p-type'), ('n', 'n-type'), ('other', 'unknown / other')]]
    handles += [Line2D([], [], color='#222222', label='y = x'),
                Line2D([], [], color='#777777', ls=':', label='factor 2'),
                Line2D([], [], color='#a23b32', ls='--', label='factor 10')]
    fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=9, frameon=False,
               bbox_to_anchor=(.5, .01))


def save(fig, stem, dpi):
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(stem.with_suffix('.' + suffix), dpi=dpi, facecolor='white')
    plt.close(fig)


def render(frame, dpi):
    plt.rcParams.update({'font.family': 'Yu Gothic', 'axes.unicode_minus': False})
    index = []
    for q in ['sigma', 'zt']:
        available = {m: paired(frame[frame.method.eq(m)], q) for m in METHODS}
        all_common = common_rows(frame, q)
        jobs = [('available', g) for g in ['all_materials', 'all_oxides', *dict(GROUPS)]]
        jobs += [('common_methods', g) for g in ['all_materials', 'all_oxides']]
        for cohort, group in jobs:
            tables = {m: select(d if cohort == 'available' else d[d.row_id.isin(all_common)], group)
                      for m, d in available.items()}
            for scale in ['fixed', 'full_range']:
                bounds = ((1e2, 1e8) if q == 'sigma' else (0., 3.)) if scale == 'fixed' else limits(list(tables.values()), q)
                fig, axes = plt.subplots(1, 3, figsize=(17, 6.8))
                for ax, m in zip(axes, METHODS):
                    outside = draw(ax, tables[m], q, bounds, f'{m}法' + ('（フィット）' if m == 'DF' else ''))
                    if scale == 'full_range':
                        assert outside == 0
                    index.append({'quantity': q, 'method': m, 'group': group, 'cohort': cohort,
                                  'scale': scale, 'points': len(tables[m]), 'outside_axes': outside,
                                  'p_points': int(tables[m].carrier.eq('p').sum()),
                                  'n_points': int(tables[m].carrier.eq('n').sum()),
                                  'other_points': int((~tables[m].carrier.isin(['p', 'n'])).sum()),
                                  'axis_min': bounds[0], 'axis_max': bounds[1]})
                scope = '同一評価点' if cohort == 'common_methods' else '手法ごとの評価可能点'
                fig.suptitle(f"{NAMES[group]}：{'電気伝導率 σ' if q == 'sigma' else 'ZT'}  |  細分類後・全温度", y=.96, fontsize=17)
                fig.text(.5, .91, f'{scope}  /  酸化物の対象行・p/nの色はexp007と共通  /  DFは評価試料へのフィット', ha='center', fontsize=10)
                carrier_legend(fig)
                fig.subplots_adjust(left=.065, right=.98, top=.85, bottom=.17, wspace=.31)
                slug = group.replace('::', '_')
                folder = HERE / 'figures' / q if cohort == 'available' else HERE / 'figures/common' / q
                stem = folder / f'{slug}_{scale}'
                save(fig, stem, dpi)
                for record in index[-3:]:
                    record['png'] = stem.with_suffix('.png').relative_to(HERE).as_posix()
            print(f'Plot {q}/{group}/{cohort}', flush=True)
        # One compact material overview per method and quantity.
        for m, table in available.items():
            columns = 5
            nrows = math.ceil(len(GROUPS) / columns)
            fig, axes = plt.subplots(nrows, columns, figsize=(17, nrows*3.1))
            bounds = (1e2, 1e8) if q == 'sigma' else (0., 3.)
            for ax, (key, name) in zip(axes.flat, GROUPS):
                draw(ax, select(table, key), q, bounds, name, small=True)
            for ax in axes.flat[len(GROUPS):]:
                ax.set_visible(False)
            fig.suptitle(f'{m}法 | {q} | 酸化物の対象・色をexp007に統一（固定軸）', fontsize=18, y=.994)
            carrier_legend(fig)
            fig.subplots_adjust(left=.06, right=.99, top=.96, bottom=.065, hspace=.38, wspace=.40)
            save(fig, HERE / 'figures' / q / f'overview_{m}', dpi)
    write_csv(pd.DataFrame(index), 'figure_index.csv')


def write_reports(metrics, statuses):
    report = ['# exp011 実行結果', '', 'DF・MLの再計算結果とexp007のFML結果について、酸化物の表示・集計対象をexp007の比較図と同じ2,984温度点・552試料にそろえました。酸化物以外の対象は維持しています。予測値は変更していません。', '',
              'p型はオレンジ（#D55E00）、n型は青（#0072B2）で、exp007の `oxide_parity_by_group_pn_20261017.png` と同じ色です。灰色はキャリア型不明を表します。', '',
              'FMLの酸化物σは前回と同じ2,984点です。DF・MLにも同じ行IDの集合を適用し、予測値がない点は描きません。ZTではさらに実測κ・ZTの対応が必要です。3手法共通点の図は各物性で全手法の値が得られる行に限定します。', '',
              '## 細分類後：3手法共通の評価点', '', '| 物性 | 手法 | 温度点数 | 試料数 | log10 RMSE | log10 MAE |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    selected = metrics.query("variant == 'fine' and cohort == 'common_methods' and group == 'all_materials' and weighting == 'row_equal'")
    for r in selected.itertuples():
        report.append(f'| {r.quantity} | {r.method} | {r.points:,} | {r.samples:,} | {r.rmse_log10:.4f} | {r.mae_log10:.4f} |')
    report += ['', 'DFは評価試料自身へのフィッティングです。ML・FMLの未知試料への予測性能と同じ意味ではありません。', '',
               '## 再計算の条件', '',
               'exp007と同じexp006の97,086温度点と試料分割を使用します。DF・MLの基準伝導率C(T)はexp005の経験的基準法に従い、25 K幅・最低3参照試料とします。材料＋キャリア型＋測定元＋温度、材料＋キャリア型＋温度、キャリア型＋温度、温度、全体の順で参照し、学習側だけから作成します。学習試料自身のC(T)では当該試料を除きます。', '',
               'DFは5点以上の各試料についてlog(σ/C)の平均をlog(τ)として再フィットし、σ=C(T)τとします。MLは既存の組成等の特徴量と材料分類でlog(τ)を学習します。ExtraTreesは300本・最小葉3・seed=42に固定し、今回の評価値でモデル選択は行いません。欠損値の補完と定数特徴の除去は学習側だけで決めます。既存特徴量のキャリア型等は実測情報を含みます。', '',
               'FMLはexp007の100 K幅・学習試料のみの参照による保存済み結果です。3手法は入力の評価点と試料分割を共有しますが、学習に使える試料数・温度幅・入力特徴は手法ごとに異なります。DF・MLは今回の共通入力から再計算しており、exp005の旧出力の再現ではありません。試料分割のため、同じ論文の異なる試料が学習側と評価側に含まれます。', '',
               'ZT=S²σT/κでは実測S・κと予測σを使います。κ・実測ZTは既存の最大±2 K対応を引き継ぎます。σの図は正の有限値、ZTの図は有限値の組を描画します。元の学習・予測データには酸化物の要確認・除外候補を保持していますが、現在の図と集計からはexp007の主解析と同様に除いています。', '',
               '## 出力の範囲', '', f'材料別の図は{len(GROUPS)}分類です。各分類・酸化物全体・全材料についてσとZTの3手法比較を保存しています。固定軸版と全点範囲版を併記し、全点範囲版の軸は同じ図の3手法で統一しています。', '',
               'Nは温度点数です。該当点がない分類も空欄にせず「評価可能なデータなし」と表示します。図の各点は間引いていません。固定軸外の点も誤差計算に含めます。', '',
               '## 計算可否', '', '| 手法 | 状態 | 温度点数 | 試料数 |', '| --- | --- | ---: | ---: |']
    for r in statuses.itertuples():
        report.append(f'| {r.method} | {r.prediction_status} | {r.points:,} | {r.samples:,} |')
    report += ['', 'DFの不足点数は試料当たり5点未満、MLの欠落はexp005の既存特徴量に当該試料がないことを示します。FMLはexp007の参照不足の状態を引き継ぎます。ZTにはさらにκと実測ZTの取得が必要です。', '',
               '## 分類変更の評価', '', '`data/classification_effect.csv` は同一手法の細分類前後で共通する点を使います。deltaは細分類後−細分類前で、RMSE・MAEは負なら改善です。比較群のラベルは両版とも細分類後で統一しています。', '',
               '`data/metrics.csv` のavailableはexp007に合わせた対象内での各手法の利用可能点、common_methodsはその3手法共通点です。row_equalは温度点を等重み、sample_equalは各試料内の重み合計を等しくします。log10指標は正の値だけを対象とします。', '',
               '参照値の作成には評価試料のσを使っていません。学習試料のτ算出に使う参照値からはその試料自身も除いています。DFのτだけは評価試料の実測値でフィットします。細分類で参照試料が3未満の場合、DF・MLでは従来のキャリア型・温度などの代替参照を使うため、細分類の効果が全点に現れるとは限りません。', '',
               '入力ファイルのSHA-256と使用ライブラリは `source_manifest.json`、学習対象と特徴量は `data/ml_sample_audit_*.csv` と `data/feature_audit_*.csv` に保存しています。表示条件と元予測値のハッシュは `display_scope_manifest.json`、exp007との分類別点数の対応は `data/exp007_alignment_counts.csv`、描画対象の行は `data/plot_source_*.csv` に保存しています。旧条件の図と集計は `archive/unfiltered_material_colors_20261026/` にあります。']
    effect = read(DATA / 'classification_effect.csv')
    report += ['', '## 既定の酸化物対象：細分類前後の比較', '',
               '各手法の細分類前後で共通する点のlog10 RMSEを示します。負の差は誤差の低下です。', '',
               '| 物性 | 手法 | 共通点数 | 細分類前 | 細分類後 | 後−前 |',
               '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in effect.query("group == 'eligible_oxides' and weighting == 'row_equal'").itertuples():
        report.append(f'| {r.quantity} | {r.method} | {r.paired_points:,} | {r.broad_rmse_log10:.4f} | {r.fine_rmse_log10:.4f} | {r.delta_rmse_log10:+.4f} |')
    folder = HERE / 'reports'
    folder.mkdir(exist_ok=True)
    (folder / 'results.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    cards = []
    for q, name in [('sigma', '電気伝導率 σ'), ('zt', 'ZT')]:
        for key in ['all_materials', 'all_oxides']:
            stem = f'figures/{q}/{key}'
            cards.append(f'<section><h2>{name}：{NAMES[key]}</h2><a href="{stem}_fixed.png"><img src="{stem}_fixed.png"></a><p><a href="{stem}_fixed.pdf">PDF</a> · <a href="{stem}_full_range.png">全点範囲 PNG</a> · <a href="{stem}_full_range.pdf">全点範囲 PDF</a> · <a href="figures/common/{q}/{key}_full_range.png">同一評価点の比較</a></p></section>')
    table = []
    for key, name in GROUPS:
        cells = []
        for q in ['sigma', 'zt']:
            stem = f'figures/{q}/{key.replace("::", "_")}'
            cells.append(f'<td><a href="{stem}_fixed.png">PNG</a> / <a href="{stem}_fixed.pdf">PDF</a> / <a href="{stem}_full_range.png">全点範囲</a></td>')
        table.append(f'<tr><th>{html.escape(name)}</th>{"".join(cells)}</tr>')
    overviews = ''.join(f'<li>{q} / {m}：<a href="figures/{q}/overview_{m}.png">材料別一覧</a></li>' for q in ['sigma', 'zt'] for m in METHODS)
    page = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><title>exp011 酸化物細分類比較</title>
<style>body{{font:16px sans-serif;color:#26364a;background:#f3f6fa;max-width:1400px;margin:auto;padding:30px}}section{{background:white;padding:18px;margin:24px 0}}img{{width:100%}}table{{border-collapse:collapse;background:white;width:100%}}th,td{{border:1px solid #ccd6df;padding:12px;text-align:left}}a{{color:#155d9b}}p,li{{line-height:1.8}}</style>
<h1>exp011：酸化物の細分類によるDF・ML・FML法の比較</h1>
<p>酸化物はexp007と同じ2,984温度点・552試料を対象とし、要確認・除外候補を含めません。p型はオレンジ、n型は青でexp007と共通です。酸化物以外の対象は維持しています。横軸は実測値、縦軸は予測値です。DFは評価試料へのフィット、MLとFMLは学習試料からの予測です。DF・MLとZTの点数は利用可能な予測・観測に応じて減ります。<a href="README.md">目的・再実行方法</a> / <a href="reports/results.md">結果・条件</a> / <a href="data/metrics.csv">誤差指標CSV</a> / <a href="data/classification_effect.csv">細分類前後の差</a> / <a href="data/exp007_alignment_counts.csv">exp007との点数対応</a></p>
{''.join(cards)}<h2>材料分類別の3手法比較</h2><table><tr><th>材料分類</th><th>σ</th><th>ZT</th></tr>{''.join(table)}</table>
<h2>手法ごとの一覧</h2><ul>{overviews}</ul></html>'''
    (HERE / 'index.html').write_text(page, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--dpi', type=int, default=180)
    args = parser.parse_args()
    output = ({v: read(DATA / f'predictions_{v}.csv') for v in ['broad', 'fine']}
              if args.plots_only else calculate())
    output = align_display_scope(output)
    metrics, statuses = summarize(output)
    if not args.skip_plots:
        render(output['fine'], args.dpi)
    write_reports(metrics, statuses)
    print(f'Completed: {HERE}', flush=True)


if __name__ == '__main__':
    main()
