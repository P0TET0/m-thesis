# exp005 / exp006 comparison handoff summary

This folder is a handoff package for continuing the discussion in another chat.

## Main context

The work compared exp005 and exp006 thermoelectric-property predictions, especially electrical conductivity sigma and ZT.

The central comparison cohort used in recent figures is:

- 383 rows
- 109 samples
- same sample/DOI common cohort
- reported ZT or sigma matched within +/-2 K
- same evaluation rows for exp005 DF, exp005 ML, and exp006

## Important conclusions

1. exp005 ML model

The exp005 ML method shown in the three-method parity figures is not XGBoost.

It uses scikit-learn:

```python
sklearn.ensemble.GradientBoostingRegressor
```

with approximately:

```python
n_estimators=300
learning_rate=0.05
max_depth=3
random_state=42
```

The target is `target_log_tau_eff_step18`, meaning the model predicts log effective tau, then reconstructs:

- tau_eff = exp(predicted log tau_eff)
- sigma from the Step12 empirical prefactor times predicted tau_eff
- ZT from S^2 sigma T / kappa

This is different from `xgboost.XGBRegressor`.

2. Difference between Gradient Boosting and XGBoost

Both are gradient-boosted decision-tree methods. However:

- exp005 uses the classic scikit-learn `GradientBoostingRegressor`
- XGBoost is a separate optimized implementation with stronger regularization, faster training in many cases, and built-in missing-value handling

So the correct poster wording is:

> Gradient Boosting regression implemented with scikit-learn `GradientBoostingRegressor`

3. ZT comparison interpretation

The common-cohort ZT figure showed exp005 DF as very good because DF is direct in-sample curve fitting, not a held-out prediction method.

The fairer method comparison is between:

- exp005 ML
- exp006

Both are held-out prediction methods in the plotted comparison.

4. Why the ZT MAE looked small

In the originally discussed ZT parity figure, MAE around 0.20 can look small because the plotted/evaluated ZT range is mostly low ZT, especially ZT < 1. Absolute ZT MAE is therefore not enough by itself.

For poster comparison, the metrics were recalculated using unified log10 multiplicative errors:

- RMSE(log10(pred / observed))
- MAE(log10(pred / observed))
- factor-of-2 accuracy
- factor-of-10 accuracy

This puts sigma and ZT on a comparable multiplicative-error scale.

5. exp006 sigma source-version caution

There are two exp006 sigma prediction variants involved:

- old/default source:
  `experiments/exp006/data/processed/step5b_test_predictions_default.csv`
- current broad-family source:
  `experiments/exp006/data/processed/step6b_broad_family/step5b_test_predictions_default.csv`

The common-cohort ZT/sigma comparison figures in this package use the old/default exp006 source unless explicitly labeled otherwise.

The broader poster sigma metric used the current broad-family source.

This explains why some exp006 sigma metrics differed between earlier outputs.

## Key metric values

### Common 383-row sigma parity cohort

Source:

`outputs/common_cohort_sigma_zt_unified_log10_metrics/sigma_common_cohort_unified_log10_metrics.csv`

Metrics:

| method | valid n | RMSE log10 | MAE log10 | factor-2 | factor-10 |
|---|---:|---:|---:|---:|---:|
| exp005 DF | 383 | 0.119 | 0.089 | 97.9% | 100.0% |
| exp005 ML | 383 | 0.803 | 0.537 | 33.4% | 94.3% |
| exp006 old/default | 383 | 0.568 | 0.430 | 41.5% | 94.0% |

### Common 383-row ZT parity cohort

Source:

`outputs/common_cohort_sigma_zt_unified_log10_metrics/zt_common_cohort_unified_log10_metrics.csv`

Metrics:

| method | valid n | RMSE log10 | MAE log10 | factor-2 | factor-10 |
|---|---:|---:|---:|---:|---:|
| exp005 DF | 379 | 0.447 | 0.182 | 91.6% | 95.3% |
| exp005 ML | 379 | 0.859 | 0.585 | 31.4% | 90.0% |
| exp006 old/default | 379 | 0.563 | 0.435 | 40.6% | 92.9% |

ZT has 379 valid log-metric rows out of 383 because 4 rows have nonpositive observed ZT and cannot be evaluated using log10(pred / observed).

### Broad-family exp006 sigma overall metric

Source:

`outputs/poster_unified_log10_metrics/sigma_unified_log10_metrics.csv`

Approximate overall values:

- n total: 19008
- n valid: 18968
- excluded: 40 missing predictions
- RMSE log10: 1.251
- MAE log10: 0.724
- factor-2: 42.7%
- factor-10: 78.4%

## Included output folders

- `outputs/exp005_ml_df_exp006_common_cohort_zt_comparison`
  - three-method ZT and sigma parity figures
  - row-level and method-level comparison CSVs
  - notes for the exp005 DF / exp005 ML / exp006 comparison

- `outputs/common_cohort_sigma_zt_unified_log10_metrics`
  - unified log10 sigma and ZT metrics for the exact 383-row cohort
  - validation log

- `outputs/poster_unified_log10_metrics`
  - poster-oriented unified log10 metrics
  - sigma and ZT summaries

- `outputs/exp005_exp006_common_doi_group_cv`
  - common DOI-group cross-validation reconstruction
  - fold metrics and figures
  - material-system-colored exp006 ZT figure outputs

## Included scripts

- `scripts/compute_common_cohort_sigma_zt_log10_metrics.py`
  - recalculates sigma and ZT metrics for the exact common 383-row cohort

- `scripts/compute_poster_unified_log10_metrics.py`
  - recalculates poster-oriented unified log10 metrics

## Recommended phrasing for poster/discussion

For exp005 ML:

> exp005 ML used scikit-learn Gradient Boosting regression to predict log effective tau, followed by physical reconstruction of sigma and ZT.

For comparison caveat:

> exp005 DF is a direct in-sample curve-fitting baseline, whereas exp005 ML and exp006 are held-out prediction methods.

For metric caveat:

> Sigma and ZT were compared using log10 multiplicative error metrics to avoid overinterpreting small absolute ZT errors in the low-ZT regime.

For exp006 source caveat:

> Some exp006 sigma metrics differ depending on whether the old/default prediction file or the newer broad-family prediction file is used.

