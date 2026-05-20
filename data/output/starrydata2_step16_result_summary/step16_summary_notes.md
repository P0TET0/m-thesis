# Step16 Summary

## Purpose
Step16 summarizes PF/ZT error analysis and ZT>=1 classification performance from Step15 for reporting and Step17 review prioritization.

## Data Used
The main input is `pf_zt_error_samples_step15.csv` with 18466 sample-level rows. Manual review, sintering-check, best-candidate, high-ZT classification, material, n/p, and feature-flag summaries from Step15 were also used when available.

## Main PF/ZT Error Results
- Median PF MAPE: 0.177281
- Median ZT pred-vs-observed MAPE: 0.184924
- Median ZT pred-vs-calc-from-observed MAPE: 0.169317

## ZT>=1 Classification Performance
For the ZT>=1 threshold, classification performance is precision 0.647739, recall 0.779794, F1 0.707659.

## Ranking Correlation
ZT predicted-vs-observed ranking correlation is Pearson 0.0414143, Spearman 0.803437. Top-k overlap values are recorded in `step16_ranking_correlation.csv`.

## Important Caveats
tau_eff is a relative effective scalar, not a physical relaxation time in seconds.

Seebeck coefficient and thermal conductivity were not predicted in Steps 14-16.

PF_pred and ZT_pred were calculated using sigma_pred and observed S/kappa.

Sintering methods are still unknown and will be checked only for prioritized samples in Step17.

## Manual Review Targets for Step17
`step16_next_step17_review_targets.csv` contains 300 prioritized samples. The target list combines manual-review candidates, sintering-check candidates, best-candidate samples, and high-ZT false positive/false negative samples, deduplicated by `sample_key`.

## Sintering Check Policy
Step16 does not investigate sintering methods. `sintering_method=unknown`, `sintering_checked=no`, and `record_checked=no` are preserved. Step17 should check sintering only for prioritized samples marked `step17_check_sintering=yes`.

## Next Step
Use `step16_next_step17_review_targets.csv` to inspect original papers for high-ZT samples, large-error samples, ZT>=1 false negatives/false positives, paper candidates, selected sintering-check candidates, and samples where additive/structure metadata should be improved.
