# Common DOI-group cross-validation (exp005 DF / exp005 ML / exp006)

## Primary result

- Common trainable universe: 9,764 samples / 2,855 DOI groups.
- Strict shared reported-ZT evaluation: 10,661 rows / 2,881 samples / 954 DOI groups.
- No ZT<1 cutoff is applied: the strict shared sample-max range is -0.0009637 to 2.276; high-ZT samples (max ZT >= 1) = 39.

| method | sample-max log10 RMSE | factor-2 rate | Spearman | high-ZT F1 |
|---|---:|---:|---:|---:|
| exp005 DF (per-sample fit reference) | 0.4266 | 0.9265 | 0.9340 | 0.3182 |
| exp005 ML (DOI-group OOF) | 0.5393 | 0.5418 | 0.6202 | 0.0833 |
| exp006 (DOI-group OOF) | 0.6066 | 0.5066 | 0.4819 | 0.1549 |

## What was made common

- The sample universe is the intersection of exp005 Step18 trainable samples and exp006 Step5A valid sigma0 rows, with normalized DOI agreement required.
- One DOI is assigned to exactly one of five folds; no DOI appears in both train and test within a fold.
- Reported high-ZT DOI groups were explicitly balanced across folds.
- exp005 ML uses the historical GradientBoostingRegressor settings and fold-local feature imputation/variance filtering.
- exp006 rebuilds material-family + carrier + 25 K-bin sample-median sigma0 references from fold training rows only. Eligibility remains >=3 rows, >=3 samples, >=1 DOI.
- Kappa and reported ZT are matched independently from the same original Step5 curves by same-sample nearest temperature within +/-2 K.
- PF/ZT for every method uses the same row temperature, observed S, and observed kappa.
- Primary metrics use only strict rows where reported ZT and all three method predictions are available.

## Important interpretation

- exp005 ML and exp006 are genuine out-of-fold predictions under the common DOI split.
- exp005 DF is a per-sample fitting reference, not a cross-sample predictive model: it uses the test sample's fitted Step12 tau_eff. Fold labels only partition its evaluation rows. Treat DF as an attainable fit/reference ceiling, not as a deployable OOF predictor.
- The Step12 empirical prefactor is rebuilt from the original Step5 electrical curves with the historical leave-one-sample-out hierarchy. This preserves exp005's original target definition; it is shared by exp005 DF and ML.
- Exact-bin exp006 prediction coverage is reported separately. Missing exp006 references are not silently replaced by a global fallback.

## Fold balance

| fold | DOI_count | sample_count | evaluable_sample_count | high_ZT_sample_count | max_ZT_obs |
|---|---|---|---|---|---|
| 0 | 581 | 1924 | 525 | 11 | 3.548 |
| 1 | 553 | 2005 | 652 | 10 | 1.214 |
| 2 | 584 | 1915 | 526 | 11 | 1.403 |
| 3 | 583 | 1923 | 525 | 11 | 2.276 |
| 4 | 554 | 1997 | 653 | 10 | 1.303 |

## exp006 fold reference coverage

| fold | train_rows | train_samples | train_DOIs | reference_bins_total | reference_bins_eligible | test_rows_all_valid | test_rows_prediction_ok | test_samples_prediction_ok |
|---|---|---|---|---|---|---|---|---|
| 0 | 43534 | 7840 | 2274 | 27 | 26 | 11017 | 11017 | 1924 |
| 1 | 43206 | 7759 | 2302 | 26 | 26 | 11345 | 11343 | 2004 |
| 2 | 44674 | 7849 | 2271 | 27 | 26 | 9877 | 9877 | 1915 |
| 3 | 43550 | 7841 | 2272 | 27 | 26 | 11001 | 11001 | 1923 |
| 4 | 43240 | 7767 | 2301 | 27 | 26 | 11311 | 11311 | 1997 |
