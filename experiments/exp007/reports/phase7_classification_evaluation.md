# Phase 7 classification-change evaluation

## Evaluation design

- Primary cohort: default-eligible oxide rows predicted by both the broad-oxide baseline and oxide_v1.
- Primary comparison: paired row-equal metrics. Sample-equal metrics prevent samples with many temperature rows from dominating.
- Uncertainty: deterministic 2,000-replicate paired bootstrap with validation samples as clusters.
- Coverage is evaluated separately on all available rows so missing reference bins are not hidden by paired filtering.
- This phase evaluates existing Phase 6 predictions; it does not refit or change the taxonomy.

## Decision summary

| item | result | definition |
| --- | --- | --- |
| primary_scope | oxide_default_eligible | Paired rows predicted by both variants; composition-only default-eligible oxide labels |
| overall_verdict | classification_improves_large_error_control_with_small_coverage_cost | MAE/RMSE, tolerance accuracy, sample-level changes, bootstrap intervals, and coverage considered together |
| row_mae_change | 1.023496 -> 0.875409 (-14.47%) | Lower is better |
| row_rmse_change | 1.586943 -> 1.359271 (-14.35%) | Lower is better |
| sample_equal_mae_change | 0.890046 -> 0.764757 | Each sample receives equal weight |
| factor_2_accuracy_change | 33.043% -> 34.517% (+1.47 pp) | Higher is better |
| factor_10_accuracy_change | 67.326% -> 71.716% (+4.39 pp) | Higher is better |
| sample_change_counts | 323 improved / 211 worsened / 18 unchanged | Based on absolute sample-median residual |
| coverage_change | 100.000% -> 98.417% (-1.58 pp) | Available-as-is default-eligible oxide rows |
| interpretation_limit | descriptive_group_results_need_support_checks | Groups with fewer than 10 paired samples are very limited; classification is formula-only and does not add structure labels |

## Primary paired metrics

| metric_name | baseline_value | oxide_v1_value | delta_oxide_v1_minus_baseline | delta_percentage_points | evaluation |
| --- | --- | --- | --- | --- | --- |
| mae_log10 | 1.0234964425762758 | 0.8754094404717878 | -0.148087002104488 |  | improved |
| rmse_log10 | 1.5869425168820255 | 1.3592712571629202 | -0.2276712597191053 |  | improved |
| factor_2_accuracy | 0.3304289544235925 | 0.34517426273458446 | 0.014745308310991967 | 1.4745308310991967 | improved |
| factor_5_accuracy | 0.5472520107238605 | 0.6243297587131368 | 0.07707774798927625 | 7.707774798927625 | improved |
| factor_10_accuracy | 0.6732573726541555 | 0.7171581769436998 | 0.04390080428954424 | 4.390080428954423 | improved |

## Primary bootstrap intervals

| metric_name | point_delta | ci95_lower | ci95_upper | bootstrap_probability_improved | interval_interpretation |
| --- | --- | --- | --- | --- | --- |
| mae_log10 | -0.148087002104488 | -0.2427982620273522 | -0.06574900384792776 | 1.0 | supported_improvement |
| rmse_log10 | -0.2276712597191053 | -0.36706723739189073 | -0.10639407592519053 | 1.0 | supported_improvement |
| factor_2_accuracy | 0.014745308310991967 | -0.020773063134989583 | 0.04993060182100823 | 0.8115 | inconclusive |
| factor_10_accuracy | 0.04390080428954424 | 0.00901115544182009 | 0.07758764944836817 | 0.993 | supported_improvement |

## Sample-level direction

| summary_level | scope_or_group | display_name_en | n_samples | improved_samples | worsened_samples | unchanged_samples | improved_fraction_excluding_ties | mean_delta_abs_median_error | median_delta_abs_median_error | two_sided_exact_sign_test_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_scope | oxide_default_eligible | Default-eligible oxide | 552 | 323 | 211 | 18 | 0.6048689138576779 | -0.12528884201173543 | -0.013744497135830758 | 1.433785578362726e-06 |

## Coverage trade-off

| summary_level | scope_or_group | display_name_en | test_rows | n_samples | baseline_prediction_ok_rows | oxide_v1_prediction_ok_rows | lost_prediction_rows | lost_prediction_samples | baseline_coverage_fraction | oxide_v1_coverage_fraction | coverage_delta_oxide_v1_minus_baseline | loss_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| analysis_scope | oxide_default_eligible | oxide default eligible | 3032 | 553 | 3032 | 2984 | 48 | 27 | 1.0 | 0.9841688654353562 | -0.0158311345646438 | unreliable_reference_bin | missing_reference_bin |

## Default-eligible oxide groups

| oxide_v1_display_name_en | paired_rows | paired_samples | row_delta_mae_log10 | sample_delta_mae_log10 | coverage_delta_oxide_v1_minus_baseline | improved_fraction_excluding_ties | support_note | descriptive_result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Co-Mn mixed oxide | 52 | 7 | -1.0864662709566006 | -0.28877574368280245 | -0.11864406779661017 | 0.5714285714285714 | very_limited | error_improved_with_material_coverage_loss |
| Mn-containing oxide | 489 | 89 | -0.5550924533240633 | -0.4543420273200047 | 0.0 | 0.6404494382022472 | adequate_for_descriptive_comparison | error_improved |
| Mo/W/V/Nb-containing oxide | 137 | 27 | -0.22862441570523973 | -0.14737654747171502 | -0.035211267605633804 | 0.6296296296296297 | limited | error_improved |
| In/Sn/Cd-containing oxide | 115 | 18 | -0.17457582910555147 | -0.17022310723923362 | 0.0 | 0.7222222222222222 | limited | error_improved |
| Mixed-anion O + S/Se/Te | 127 | 34 | -0.06473281289773702 | -0.0601311518487786 | 0.0 | 0.5294117647058824 | adequate_for_descriptive_comparison | error_improved |
| Cu-containing oxide | 170 | 23 | -0.057808115219797784 | -0.06411131071592369 | -0.017341040462427744 | 0.6521739130434783 | limited | error_improved |
| Fe/Ni/Cr-containing oxide | 19 | 2 | -0.04411065111773227 | -0.5768215305791986 | -0.05 | 1.0 | very_limited | error_improved |
| Ti-containing oxide | 210 | 61 | -0.03311216002549622 | -0.14234874038704282 | -0.027777777777777776 | 0.6721311475409836 | adequate_for_descriptive_comparison | error_improved |
| Other inorganic O-containing | 136 | 17 | -0.027543931423486656 | 0.03358907997411609 | -0.02857142857142857 | 0.5882352941176471 | limited | mixed_error_result |
| Co-containing oxide | 1210 | 216 | -0.025744557092949094 | -0.030459321552733787 | -0.00411522633744856 | 0.5909090909090909 | adequate_for_descriptive_comparison | error_improved |
| Zn-containing oxide | 268 | 52 | -0.021070865377913583 | -0.008081669591573526 | -0.014705882352941176 | 0.5 | adequate_for_descriptive_comparison | error_improved |
| Mixed-anion O + As/P | 51 | 6 | 0.07503118163702216 | 0.020999001251614025 | -0.203125 | 0.5 | very_limited | error_worsened_with_material_coverage_loss |

## Figures

- experiments/exp007/figures/figure_phase7_primary_metrics_before_after.png
- experiments/exp007/figures/figure_phase7_group_mae_coverage_tradeoff.png
- experiments/exp007/figures/figure_phase7_sample_error_changes.png

## Validation

| check | status | detail |
| --- | --- | --- |
| phase6_build_validation_passes | PASS | phase6_validation_checks.csv |
| phase6_independent_check_passes | PASS | phase6_check_log.txt |
| comparison_row_count_frozen | PASS | rows=19008 |
| paired_row_count_frozen | PASS | rows=18890 |
| primary_row_count | PASS | rows=2984 |
| primary_sample_count | PASS | samples=552 |
| paired_metric_table_complete | PASS | rows=70 |
| bootstrap_table_complete | PASS | rows=42 |
| bootstrap_intervals_ordered | PASS | lower <= point <= upper |
| group_count_reconciles | PASS | groups=12 |
| group_paired_rows_reconcile | PASS | rows=2984 |
| coverage_loss_count | PASS | rows=78 |
| primary_coverage_loss_count | PASS | rows=48 |
| coverage_scopes_complete | PASS | ['all_oxide_assigned', 'oxide_default_eligible', 'oxide_excluded_only', 'oxide_review_only'] |
| nonoxide_predictions_remain_unchanged | PASS | max_abs_delta=1.157e-15 |
| figures_created | PASS | figure_phase7_primary_metrics_before_after.png=104407, figure_phase7_group_mae_coverage_tradeoff.png=134561, figure_phase7_sample_error_changes.png=65394 |
| input_hashes_match_manifest | PASS | current SHA-256 versus recorded |

## Interpretation limits

The intervals quantify held-out-sample variation within this frozen data split; they do not represent performance on a new external database. Group-level results are descriptive, especially when fewer than 10 paired samples are available. oxide_v1 remains a composition-only taxonomy and does not add crystal-structure labels.
