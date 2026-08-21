# exp005 DF / exp005 ML / exp006 common-cohort ZT comparison

## Scope

- Identical comparable rows: 383.
- Identical samples: 109.
- DOI: 83.
- Reported ZT and thermal conductivity are matched from the same Step5 curves within +/-2 K.
- All three methods use the same observed S, kappa, T, and reported-ZT reference.

## Results

- exp005 DF sample-balanced MAE: 0.0533625; log RMSE: 1.02991.
- exp005 ML sample-balanced MAE: 0.167266; log RMSE: 1.97754.
- exp006 sample-balanced MAE: 0.180304; log RMSE: 1.29696.
- Paired exp006-minus-ML MAE difference: +0.0130383; sample-bootstrap 95% CI [-0.0100537, +0.0371044].
- The absolute-error interval includes zero, so this common cohort does not show a clear MAE improvement of exp006 over exp005 ML.
- exp006 has better log RMSE and factor-2 coverage than exp005 ML, but worse Spearman correlation on this subset.

## Fairness cautions

- exp005 DF is direct per-sample in-sample curve fitting and is expected to outperform held-out prediction methods; it is a fit-quality reference, not an equal generalization benchmark.
- exp005 ML is DOI-held-out, while exp006 is sample-held-out. The evaluation rows are identical, but training-group leakage conditions are not identical.
- This common cohort contains no sample with reported maximum ZT >= 1, so it cannot support conclusions about high-ZT discovery performance.
- The plots show the measured result; they should not be described as proof that exp006 is universally no better than ML.
