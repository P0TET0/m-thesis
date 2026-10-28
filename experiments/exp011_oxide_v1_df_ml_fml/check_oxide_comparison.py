"""Independent numerical, split, reference-leakage and figure coverage checks."""
from __future__ import annotations

import hashlib
import argparse
import ast
import json
from pathlib import Path
import re

import build_oxide_comparison as build
import numpy as np
import pandas as pd


def check_reference_isolation():
    rows = []
    for sample in range(5):
        for temperature in [300., 301.]:
            rows.append({'sample_key': f's{sample}', 'sample_holdout_split': 'train' if sample < 4 else 'test',
                         'broad_group': 'broad::oxide', 'carrier': 'n', 'measurement_source': 'direct',
                         'temperature_bin': 300., 'sigma_obs': float(10 * sample + temperature - 290)})
    frame = pd.DataFrame(rows)
    reference = build.empirical_prefactors(frame, 'broad_group').prefactor_C
    for sample in range(5):
        expected = frame[frame.sample_holdout_split.eq('train') & frame.sample_key.ne(f's{sample}')].sigma_obs.median()
        np.testing.assert_allclose(reference[frame.sample_key.eq(f's{sample}')], expected)
    changed = frame.copy()
    changed.loc[changed.sample_holdout_split.eq('test'), 'sigma_obs'] = 1e30
    np.testing.assert_array_equal(reference, build.empirical_prefactors(changed, 'broad_group').prefactor_C)
    changed = frame.copy()
    changed.loc[changed.sample_key.eq('s0'), 'sigma_obs'] = 1e20
    other = build.empirical_prefactors(changed, 'broad_group').prefactor_C
    np.testing.assert_array_equal(reference[frame.sample_key.eq('s0')], other[frame.sample_key.eq('s0')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    checks = []
    check_reference_isolation()
    checks.append('PASS: train-only references and own-sample exclusion on a hand-checkable fixture')
    manifest = json.loads((build.HERE / 'source_manifest.json').read_text(encoding='utf-8'))
    assert manifest['train_test_sample_overlap'] == 0
    for source in manifest['sources']:
        h = hashlib.sha256()
        with (build.ROOT / source['path']).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        assert h.hexdigest() == source['sha256'], source['path']
    checks.append('PASS: original input hashes unchanged and train/test sample sets disjoint')
    metrics = build.read(build.DATA / 'metrics.csv')
    saved = {}
    displayed = {}
    reference = build.read(build.PRIMARY_COHORT)
    truth = lambda s: s.astype(str).str.lower().eq('true')
    reference = reference[truth(reference.is_phase0_broad_oxide)
                          & truth(reference.oxide_v1_default_model_eligible)
                          & truth(reference.paired_prediction_available)]
    expected_ids = set(reference.row_id)
    assert len(expected_ids) == 2984
    scope_manifest = json.loads((build.HERE / 'display_scope_manifest.json').read_text(encoding='utf-8'))
    for variant in ['broad', 'fine']:
        data = build.read(build.DATA / f'predictions_{variant}.csv')
        saved[variant] = data
        display = build.read(build.DATA / f'plot_source_{variant}.csv')
        displayed[variant] = display
        expected = data[~data.broad_group.eq('broad::oxide') | data.row_id.isin(expected_ids)]
        assert set(zip(display.method, display.row_id)) == set(zip(expected.method, expected.row_id))
        display_sorted = display.sort_values(['method', 'row_id'])
        expected_sorted = expected.sort_values(['method', 'row_id'])
        np.testing.assert_allclose(display_sorted[['sigma_obs', 'sigma_pred', 'zt_obs', 'zt_pred']],
                                   expected_sorted[['sigma_obs', 'sigma_pred', 'zt_obs', 'zt_pred']],
                                   equal_nan=True, rtol=1e-12)
        assert build.file_sha256(build.DATA / f'predictions_{variant}.csv') == scope_manifest['raw_prediction_hashes'][variant]
        assert len(data) == 3 * manifest['test_rows']
        assert not data.duplicated(['method', 'row_id']).any()
        assert set(data.method) == set(build.METHODS)
        assert not np.isinf(data[['sigma_obs', 'sigma_pred', 'zt_obs', 'zt_pred']].to_numpy()).any()
        ids = [set(data[data.method.eq(m)].row_id) for m in build.METHODS]
        assert ids[0] == ids[1] == ids[2]
        fml = data[data.method.eq('FML')].set_index('row_id')
        frozen = build.read(build.FML_PATHS[variant]).set_index('row_id').reindex(fml.index)
        np.testing.assert_allclose(fml.sigma_pred, frozen.sigma_pred_S_per_m.where(frozen.prediction_status.eq('ok')), equal_nan=True)
        expected_zt = data.S_obs_V_per_K**2 * data.sigma_pred * data.temperature_K / data.kappa_obs_W_per_mK
        expected_zt = expected_zt.where(build.positive(data.kappa_obs_W_per_mK) & build.positive(data.sigma_pred)).replace([np.inf, -np.inf], np.nan)
        np.testing.assert_allclose(data.zt_pred, expected_zt, equal_nan=True, rtol=1e-12)
        audit = build.read(build.DATA / f'ml_sample_audit_{variant}.csv')
        train = audit[audit.used_to_train_ML.eq(True)]
        assert train.sample_holdout_split.eq('train').all()
        assert audit.loc[audit.ML_log_tau_prediction.notna(), 'sample_holdout_split'].eq('test').all()
        assert not set(train.sample_key) & set(data.sample_key)
        # Direct recomputation from arrays, independent of the summary helpers.
        for q in ['sigma', 'zt']:
            valid_sets = []
            for m in build.METHODS:
                d = display[display.method.eq(m)]
                ok = np.isfinite(d[f'{q}_obs']) & np.isfinite(d[f'{q}_pred'])
                if q == 'sigma':
                    ok &= d.sigma_obs.gt(0) & d.sigma_pred.gt(0)
                valid_sets.append(set(d.loc[ok, 'row_id']))
            common = set.intersection(*valid_sets)
            for m in build.METHODS:
                d = display[display.method.eq(m) & display.row_id.isin(common)]
                positive = d[f'{q}_obs'].gt(0) & d[f'{q}_pred'].gt(0)
                logs = np.log10(d.loc[positive, f'{q}_pred'].to_numpy()) - np.log10(d.loc[positive, f'{q}_obs'].to_numpy())
                item = metrics[(metrics.variant == variant) & (metrics.method == m) & (metrics.quantity == q)
                               & (metrics.cohort == 'common_methods') & (metrics.group == 'all_materials')
                               & (metrics.weighting == 'row_equal')].iloc[0]
                assert int(item.points) == len(common)
                np.testing.assert_allclose(item.rmse_log10, np.sqrt(np.mean(logs**2)), rtol=1e-12)
                np.testing.assert_allclose(item.mae_log10, np.mean(abs(logs)), rtol=1e-12)
        checks.append(f'PASS: {variant} predictions, ZT formula, frozen FML values, ML split, common-row metrics')
    a = saved['broad'].sort_values(['method', 'row_id']).reset_index(drop=True)
    b = saved['fine'].sort_values(['method', 'row_id']).reset_index(drop=True)
    for c in ['row_id', 'sample_key', 'material_group', 'method']:
        pd.testing.assert_series_equal(a[c], b[c])
    np.testing.assert_allclose(a[['sigma_obs', 'zt_obs']], b[['sigma_obs', 'zt_obs']], equal_nan=True)
    for m in ['DF', 'ML']:
        mask = a.method.eq(m) & a.broad_group.eq('broad::oxide') & a.sigma_pred.notna() & b.sigma_pred.notna()
        assert (~np.isclose(a.loc[mask, 'sigma_pred'], b.loc[mask, 'sigma_pred'], rtol=1e-10)).any()
    checks.append('PASS: before/after observations and classifications match; both DF and ML actually changed')
    fml = displayed['fine'].query("method == 'FML' and broad_group == 'broad::oxide'")
    assert len(fml) == 2984 and fml.sample_key.nunique() == 552
    assert fml.oxide_default_eligible.eq(True).all()
    assert fml.material_group.nunique() == 12
    joined = fml.merge(reference, on='row_id', validate='one_to_one')
    np.testing.assert_allclose(joined.sigma_pred, joined.phase6_sigma_pred_S_per_m, rtol=1e-12)
    assert joined.carrier.eq(joined.carrier_type).all()
    for key, rows in fml.groupby('material_group'):
        old = reference[reference.oxide_v1_group_key.eq(key)]
        assert set(rows.row_id) == set(old.row_id)
        for carrier in ['p', 'n']:
            assert rows.carrier.eq(carrier).sum() == old.carrier_type.eq(carrier).sum()
    nodes = ast.parse(build.REFERENCE_PLOT_SCRIPT.read_text(encoding='utf-8')).body
    old_colors = [ast.literal_eval(node.value) for node in nodes if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'carrier_colors' for t in node.targets)][0]
    assert {k: build.CARRIER_COLORS[k] for k in ['p', 'n']} == old_colors
    fig, ax = build.plt.subplots()
    build.draw(ax, fml, 'sigma', (1e-10, 1e10), 'verification')
    for collection, carrier in zip(ax.collections, build.CARRIER_COLORS):
        expected_color = build.matplotlib.colors.to_rgba(build.CARRIER_COLORS[carrier], alpha=.65)
        np.testing.assert_allclose(collection.get_facecolors()[0], expected_color)
        expected_coords = fml[fml.carrier.eq(carrier)][['sigma_obs', 'sigma_pred']].to_numpy()
        np.testing.assert_allclose(np.asarray(collection.get_offsets()), expected_coords)
    build.plt.close(fig)
    checks.append('PASS: exact exp007 2,984 oxide rows / 552 samples / 12 groups; p/n counts, point colors and coordinates match')
    if args.data_only:
        print('\n'.join(checks))
        print('data_status=PASS')
        return
    figure_path = build.DATA / 'figure_index.csv'
    assert figure_path.is_file(), 'Run plotting before the final check'
    figures = build.read(figure_path)
    assert figures.loc[figures.scale.eq('full_range'), 'outside_axes'].eq(0).all()
    for path in figures.png.unique():
        png = build.HERE / path
        assert png.is_file() and png.stat().st_size > 1000
        assert png.with_suffix('.pdf').is_file()
    assert (figures.p_points + figures.n_points + figures.other_points).eq(figures.points).all()
    assert not figures.group.isin(['oxide_v1::excluded_organic_polymer_or_carbon_composite',
                                  'oxide_v1::mixed_anion_O_N_or_halogen']).any()
    for q in ['sigma', 'zt']:
        for cohort in ['available', 'common_methods']:
            for method in build.METHODS:
                part = figures[(figures.quantity == q) & (figures.method == method) & (figures.cohort == cohort)
                               & (figures.group == 'all_materials') & (figures.scale == 'full_range')].iloc[0]
                metric = metrics[(metrics.variant == 'fine') & (metrics.quantity == q) & (metrics.method == method)
                                 & (metrics.cohort == cohort) & (metrics.group == 'all_materials')
                                 & (metrics.weighting == 'row_equal')].iloc[0]
                assert part.points == metric.points
    for link in re.findall(r'(?:href|src)="([^"]+)"', (build.HERE / 'index.html').read_text(encoding='utf-8')):
        assert (build.HERE / link).is_file(), link
    checks.append(f'PASS: {figures.png.nunique()} comparison PNG/PDF pairs, full-range axes, counts and gallery links')
    (build.HERE / 'reports/validation.txt').write_text('\n'.join(checks) + '\nstatus=PASS\n', encoding='utf-8')
    print('\n'.join(checks))
    print('status=PASS')


if __name__ == '__main__':
    main()
