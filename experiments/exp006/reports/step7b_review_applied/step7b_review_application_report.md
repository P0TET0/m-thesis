# Step7B Review Application Report

## Inputs

- predictions: experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet
- decision_template: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_review_decisions_template.csv
- review_master: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_manual_review_master.csv
- source_trace: experiments\exp006\data\processed\step7a_manual_review_packet\step7a_source_traceability_table.csv

## Application Summary

| item | value | comment |
| --- | --- | --- |
| input_prediction_rows | 604440 | Rows read from Step6B prediction valid table. |
| decision_template_rows | 400 | Rows read from Step7A decision template. |
| human_reviewed_decisions | 0 | Decision rows with non-pending review status/name/date. |
| pending_decisions | 400 | Decision rows still pending or unresolved. |
| applied_review_rows | 2664 | Prediction rows touched by at least one review case. |
| rows_with_no_review_case | 601776 | Prediction rows with no applied review case. |
| rows_with_pending_review | 2664 | Prediction rows with pending/unresolved applied review. |
| rows_with_conflicts | 0 | Prediction rows with conflicting review decisions. |
| primary_kept_rows | 604440 | Rows retained in primary analysis. |
| primary_excluded_rows | 0 | Rows excluded from primary analysis. |
| sensitivity_kept_rows | 604440 | Rows retained in sensitivity analysis. |
| sensitivity_excluded_rows | 0 | Rows excluded from sensitivity analysis. |
| pending_policy | keep_with_pending_flag | Policy for pending/unresolved decisions. |
| suspect_policy | exclude_primary_keep_sensitivity | Policy for suspect decisions. |

## Broad Material Family Default Metrics

| review_scenario | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | extreme_ge_10_count | n_rows |
| --- | --- | --- | --- | --- | --- | --- |
| all_predictions_no_review_filter | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.7837937579080557 | 13.0 | 18968 |
| primary_review_applied | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.7837937579080557 | 13.0 | 18968 |
| sensitivity_review_applied | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.7837937579080557 | 13.0 | 18968 |

## Review Effect Summary

| config_label | config_id | metric_weighting | metric_name | baseline_value | primary_review_applied_value | sensitivity_review_applied_value | delta_primary_minus_baseline | delta_sensitivity_minus_baseline | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_rows | 18968.0 | 18968.0 | 18968.0 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | mae_log10 | 0.7237456971201834 | 0.7237456971201834 | 0.7237456971201834 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | rmse_log10 | 1.2508679717579398 | 1.2508679717579398 | 1.2508679717579398 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | median_log10_error | 0.0030867094458137314 | 0.0030867094458137314 | 0.0030867094458137314 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_2_accuracy | 0.42666596372838467 | 0.42666596372838467 | 0.42666596372838467 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_5_accuracy | 0.6865773935048503 | 0.6865773935048503 | 0.6865773935048503 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_10_accuracy | 0.7837937579080557 | 0.7837937579080557 | 0.7837937579080557 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | max_abs_log10_error | 14.570733133464506 | 14.570733133464506 | 14.570733133464506 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | extreme_ge_10_count | 13.0 | 13.0 | 13.0 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | severe_ge_5_count | 246.0 | 246.0 | 246.0 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_rows | 3189.0 | 3189.0 | 3189.0 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | mae_log10 | 0.6413098859900249 | 0.6413098859900249 | 0.6413098859900249 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | rmse_log10 | 0.6617151467234016 | 0.6617151467234016 | 0.6617151467234016 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | median_log10_error | 0.2515415432463362 | 0.2515415432463362 | 0.2515415432463362 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_2_accuracy | 0.47480896340799145 | 0.47480896340799145 | 0.47480896340799145 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_5_accuracy | 0.7391348912446652 | 0.7391348912446652 | 0.7391348912446652 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_10_accuracy | 0.8215004309151508 | 0.8215004309151508 | 0.8215004309151508 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | max_abs_log10_error | 0.801301810953687 | 0.801301810953687 | 0.801301810953687 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | extreme_ge_10_count | 0.004076513013483851 | 0.004076513013483851 | 0.004076513013483851 | 0.0 | 0.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | severe_ge_5_count | 0.07714016933207903 | 0.07714016933207903 | 0.07714016933207903 | 0.0 | 0.0 | lower_is_better |

## Readiness

| criterion | status | value | threshold_or_reason | comment |
| --- | --- | --- | --- | --- |
| decision_template_loaded | pass | 400 | > 0 | Step7A decision template was loaded. |
| human_review_completed | caution | 400 | pass if 0 | Pending decisions remain if value > 0. |
| primary_dataset_available | pass | 604440 | > 0 | Primary analysis rows after review policy. |
| sensitivity_dataset_available | pass | 604440 | > 0 | Sensitivity analysis rows after review policy. |
| primary_exclusion_documented | pass | 0 | count recorded | Excluded rows are written to CSV. |
| unresolved_rows_exist | caution | 2664 | caution if > 0 | Pending/unresolved rows remain. |
| conflicts_exist | pass | 0 | caution if > 0 | Conflicting decisions are recorded. |
| extreme_outliers_remaining_in_primary | caution | 13.0 | caution if > 0 | Extreme outliers in primary broad material_family default. |
| broad_family_primary_mae_available | pass | 0.7237456971201834 | finite | Primary broad material_family default MAE. |
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

- elapsed_seconds: 118.21
