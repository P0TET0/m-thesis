# Step7B Review Application Report

## Inputs

- predictions: experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet
- decision_template: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_review_decisions_template_reviewed.csv
- review_master: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_manual_review_master.csv
- source_trace: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_source_traceability_table.csv

## Application Summary

| item | value | comment |
| --- | --- | --- |
| input_prediction_rows | 64000 | Rows read from Step6B prediction valid table. |
| decision_template_rows | 400 | Rows read from Step7A decision template. |
| human_reviewed_decisions | 0 | Decision rows with non-pending review status/name/date. |
| pending_decisions | 400 | Decision rows still pending or unresolved. |
| applied_review_rows | 200 | Prediction rows touched by at least one review case. |
| rows_with_no_review_case | 63800 | Prediction rows with no applied review case. |
| rows_with_pending_review | 200 | Prediction rows with pending/unresolved applied review. |
| rows_with_conflicts | 0 | Prediction rows with conflicting review decisions. |
| primary_kept_rows | 63800 | Rows retained in primary analysis. |
| primary_excluded_rows | 200 | Rows excluded from primary analysis. |
| sensitivity_kept_rows | 64000 | Rows retained in sensitivity analysis. |
| sensitivity_excluded_rows | 0 | Rows excluded from sensitivity analysis. |
| pending_policy | exclude_from_primary | Policy for pending/unresolved decisions. |
| suspect_policy | exclude_primary_keep_sensitivity | Policy for suspect decisions. |

## Broad Material Family Default Metrics

| review_scenario | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | extreme_ge_10_count | n_rows |
| --- | --- | --- | --- | --- | --- | --- |
| all_predictions_no_review_filter | 0.8568205664943855 | 1.4699638977367877 | 0.3885 | 0.731 | 9.0 | 2000 |
| primary_review_applied | 0.7899381834320423 | 1.2211199615495998 | 0.3918305597579425 | 0.7372667675239536 | 0.0 | 1983 |
| sensitivity_review_applied | 0.8568205664943855 | 1.4699638977367877 | 0.3885 | 0.731 | 9.0 | 2000 |

## Review Effect Summary

| config_label | config_id | metric_weighting | metric_name | baseline_value | primary_review_applied_value | sensitivity_review_applied_value | delta_primary_minus_baseline | delta_sensitivity_minus_baseline | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_rows | 2000.0 | 1983.0 | 2000.0 | -17.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | mae_log10 | 0.8568205664943855 | 0.7899381834320423 | 0.8568205664943855 | -0.06688238306234329 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | rmse_log10 | 1.4699638977367877 | 1.2211199615495998 | 1.4699638977367877 | -0.24884393618718792 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | median_log10_error | 0.04972949862767547 | 0.04656113439657787 | 0.04972949862767547 | -0.0031683642310975993 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_2_accuracy | 0.3885 | 0.3918305597579425 | 0.3885 | 0.0033305597579424973 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_5_accuracy | 0.652 | 0.6575895108421583 | 0.652 | 0.005589510842158307 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_10_accuracy | 0.731 | 0.7372667675239536 | 0.731 | 0.006266767523953609 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | max_abs_log10_error | 11.212152574256098 | 5.2513719783650865 | 11.212152574256098 | -5.960780595891012 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | extreme_ge_10_count | 9.0 | 0.0 | 9.0 | -9.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | severe_ge_5_count | 35.0 | 18.0 | 35.0 | -17.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_rows | 317.0 | 313.0 | 317.0 | -4.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | mae_log10 | 0.7404230013289567 | 0.6490432986546834 | 0.7404230013289567 | -0.09137970267427331 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | rmse_log10 | 0.7595794538154401 | 0.6683856048485285 | 0.7595794538154401 | -0.09119384896691163 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | median_log10_error | 0.33374159142646365 | 0.23742132342666555 | 0.33374159142646365 | -0.09632026799979809 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_2_accuracy | 0.4731395743078588 | 0.4791860864395886 | 0.4731395743078588 | 0.006046512131729798 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_5_accuracy | 0.7372642579492513 | 0.746686165399082 | 0.7372642579492513 | 0.009421907449830624 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_10_accuracy | 0.8010666903073392 | 0.8113039643048772 | 0.8010666903073392 | 0.010237273997537932 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | max_abs_log10_error | 0.8907868262969764 | 0.7974358436221284 | 0.8907868262969764 | -0.09335098267484798 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | extreme_ge_10_count | 0.028391167192429023 | 0.0 | 0.028391167192429023 | -0.028391167192429023 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | severe_ge_5_count | 0.11041009463722397 | 0.05750798722044728 | 0.11041009463722397 | -0.052902107416776686 | 0.0 | lower_is_better |

## Readiness

| criterion | status | value | threshold_or_reason | comment |
| --- | --- | --- | --- | --- |
| decision_template_loaded | pass | 400 | > 0 | Step7A decision template was loaded. |
| human_review_completed | caution | 400 | pass if 0 | Pending decisions remain if value > 0. |
| primary_dataset_available | pass | 63800 | > 0 | Primary analysis rows after review policy. |
| sensitivity_dataset_available | pass | 64000 | > 0 | Sensitivity analysis rows after review policy. |
| primary_exclusion_documented | pass | 200 | count recorded | Excluded rows are written to CSV. |
| unresolved_rows_exist | caution | 200 | caution if > 0 | Pending/unresolved rows remain. |
| conflicts_exist | pass | 0 | caution if > 0 | Conflicting decisions are recorded. |
| extreme_outliers_remaining_in_primary | pass | 0.0 | caution if > 0 | Extreme outliers in primary broad material_family default. |
| broad_family_primary_mae_available | pass | 0.7899381834320423 | finite | Primary broad material_family default MAE. |
| recommended_next_action | caution | complete manual review then rerun Step7B | manual decision | Next action after applying review decisions. |

## Notes

- Step7B applies existing review decisions to existing prediction rows.
- Step7B does not compute new sigma predictions.
- Pending decisions may remain depending on pending policy.
- If pending rows are retained, final reporting should mention that unreviewed outliers remain.

## Sanity Checks

- prediction_input_exists: True
- normalized_decisions_created: True
- prediction_rows_with_review_flags_created: True
- row_count_matches_prediction_input: True
- row_id_not_missing: True
- config_id_not_missing: True
- keep_in_primary_not_missing: True
- keep_in_sensitivity_not_missing: True
- primary_only_kept: True
- sensitivity_only_kept: True
- excluded_primary_only_excluded: True
- excluded_sensitivity_only_excluded: True
- metrics_created: True
- default_metrics_created: True
- review_effect_created: True
- readiness_created: True
- conflict_table_created: True
- did_not_compute_new_sigma_pred: True
- did_not_read_step4_full_data_reference_curve: True
- did_not_read_raw_data: True
- report_created: True

- elapsed_seconds: 17.24
