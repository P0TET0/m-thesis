# exp007 Phase 2 residual visualization

## Scope

Phase 2 visualizes how the frozen broad-family 100 K conductivity prediction misses the observed values. It does not change the model or the material taxonomy.

- Evaluable rows: 18,968
- Evaluable samples: 3,189
- Broad-oxide rows: 3,437
- Factor-10 outlier rows: 4,101

## Figure set

1. Global residual distribution, parity, residual-versus-observed relationship, and error bands.
2. Broad material-group error, bias, and threshold accuracy.
3. Error, bias, support, and threshold accuracy across 100 K temperature bins.
4. Broad-oxide comparison and oxide residual relationships with conductivity, temperature, and |S|.

## Descriptive observations

- The median signed residual is 0.003, while the mean is 0.245; the large-error tail therefore affects the mean more strongly than the median.
- Overprediction and underprediction account for 50.3% and 49.7% of evaluable rows, respectively.
- Broad oxides have row-equal MAE 1.016 and factor-10 accuracy 67.3%, compared with 0.659 and 80.8% for non-oxides.
- The largest broad-group row-equal MAE is `broad::sulfide` at 1.443 across 892 rows.
- Among temperature bins with at least 50 rows, `-50_50K` has the largest row-equal MAE (1.312).

These are descriptive patterns, not causal conclusions. The next phase should inspect the underlying material labels and data quality before changing the oxide taxonomy.

## Plot-range policy

Residual-density panels show −4 to +4 log10 units so the main distribution remains readable. Counts outside that display range are retained in all metrics and are annotated in the global overview.

## Validation

- Figure files created: 8
- Phase 1 input unchanged: True
- All Phase 2 reconciliation checks: PASS
