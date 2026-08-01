# Step7C Reviewed Decisions Report

## Decision Validation Summary

| item | value | comment |
| --- | --- | --- |
| decision_template_rows | 400 | Rows in reviewed decision template. |
| human_reviewed_decisions | 0 | Rows with human review evidence. |
| pending_decisions | 400 | Rows pending or unresolved. |
| unresolved_decisions | 0 | Rows marked unresolved. |
| keep_decisions | 0 | Rows marked keep. |
| keep_but_note_decisions | 0 | Rows marked keep_but_note. |
| suspect_decisions | 0 | Rows marked suspect. |
| exclude_from_primary_decisions | 0 | Rows marked exclude_from_primary. |
| exclude_from_all_decisions | 0 | Rows marked exclude_from_all. |
| invalid_decisions | 0 | Rows with invalid or incomplete reviewed decision values. |
| extreme_case_count | 17 | Extreme cases in reviewed template. |
| extreme_human_reviewed_count | 0 | Extreme cases with human review evidence. |
| extreme_pending_count | 17 | Extreme cases still pending. |
| extreme_exclude_count | 0 | Extreme cases excluded by review status. |
| full_run_ready | False | Full run readiness based on reviewed decisions. |

## Extreme Review Completion

| extreme_case_count | extreme_human_reviewed_count | extreme_pending_count | extreme_exclude_from_primary_count | extreme_exclude_from_all_count | extreme_keep_count | extreme_unresolved_count |
| --- | --- | --- | --- | --- | --- | --- |
| 17 | 0 | 17 | 0 | 0 | 0 | 0 |

## Policy Comparison

| config_label | config_id | metric_weighting | metric_name | keep_pending_value | exclude_pending_primary_value | delta_exclude_pending_minus_keep_pending | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_rows | 2000.0 | 1983.0 | -17.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_pending_rows | 17.0 | 0.0 | -17.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_excluded_from_primary | 0.0 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | mae_log10 | 0.8568205664943855 | 0.7899381834320423 | -0.06688238306234329 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | rmse_log10 | 1.4699638977367877 | 1.2211199615495998 | -0.24884393618718792 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | median_log10_error | 0.0497294986276754 | 0.0465611343965778 | -0.0031683642310975993 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_2_accuracy | 0.3885 | 0.3918305597579425 | 0.0033305597579424973 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_5_accuracy | 0.652 | 0.6575895108421583 | 0.005589510842158307 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_10_accuracy | 0.731 | 0.7372667675239536 | 0.006266767523953609 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | max_abs_log10_error | 11.212152574256098 | 5.251371978365087 | -5.960780595891011 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | extreme_ge_10_count | 9.0 | 0.0 | -9.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | severe_ge_5_count | 35.0 | 18.0 | -17.0 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_rows | 317.0 | 313.0 | -4.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_pending_rows | 0.0536277602523659 | 0.0 | -0.0536277602523659 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_excluded_from_primary | 0.0 | 0.0 | 0.0 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | mae_log10 | 0.7404230013289567 | 0.6490432986546834 | -0.09137970267427331 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | rmse_log10 | 0.7595794538154401 | 0.6683856048485285 | -0.09119384896691163 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | median_log10_error | 0.3337415914264636 | 0.2374213234266655 | -0.09632026799979809 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_2_accuracy | 0.4731395743078588 | 0.4791860864395886 | 0.006046512131729798 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_5_accuracy | 0.7372642579492513 | 0.746686165399082 | 0.009421907449830624 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_10_accuracy | 0.8010666903073392 | 0.8113039643048772 | 0.010237273997537932 | higher_is_better_or_count |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | max_abs_log10_error | 0.8907868262969764 | 0.7974358436221284 | -0.09335098267484798 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | extreme_ge_10_count | 0.028391167192429 | 0.0 | -0.028391167192429 | lower_is_better |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | severe_ge_5_count | 0.1104100946372239 | 0.0575079872204472 | -0.0529021074167767 | lower_is_better |

## Previous Baseline Comparison

| config_label | config_id | metric_weighting | metric_name | previous_pending_baseline_value | keep_pending_reviewed_value | exclude_pending_primary_reviewed_value | delta_keep_pending_minus_previous | delta_exclude_pending_minus_previous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | n_rows | 18968.0 | 2000.0 | 1983.0 | -16968.0 | -16985.0 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | mae_log10 | 0.7237456971201834 | 0.8568205664943855 | 0.7899381834320423 | 0.1330748693742021 | 0.06619248631185881 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | rmse_log10 | 1.2508679717579398 | 1.4699638977367877 | 1.2211199615495998 | 0.21909592597884786 | -0.02974801020834006 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_2_accuracy | 0.4266659637283846 | 0.3885 | 0.3918305597579425 | -0.0381659637283846 | -0.0348354039704421 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_5_accuracy | 0.6865773935048503 | 0.652 | 0.6575895108421583 | -0.03457739350485023 | -0.02898788266269192 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_10_accuracy | 0.7837937579080557 | 0.731 | 0.7372667675239536 | -0.05279375790805574 | -0.04652699038410213 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | max_abs_log10_error | 14.570733133464506 | 11.212152574256098 | 5.251371978365087 | -3.3585805592084075 | -9.319361155099418 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | extreme_ge_10_count | 13.0 | 9.0 | 0.0 | -4.0 | -13.0 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | n_rows | 3189.0 | 317.0 | 313.0 | -2872.0 | -2876.0 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | mae_log10 | 0.6413098859900249 | 0.7404230013289567 | 0.6490432986546834 | 0.09911311533893175 | 0.0077334126646584345 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | rmse_log10 | 0.6617151467234016 | 0.7595794538154401 | 0.6683856048485285 | 0.09786430709203853 | 0.006670458125126899 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_2_accuracy | 0.4748089634079914 | 0.4731395743078588 | 0.4791860864395886 | -0.00166938910013259 | 0.004377123031597208 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_5_accuracy | 0.7391348912446652 | 0.7372642579492513 | 0.746686165399082 | -0.001870633295413926 | 0.007551274154416698 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | factor_10_accuracy | 0.8215004309151508 | 0.8010666903073392 | 0.8113039643048772 | -0.020433740607811557 | -0.010196466610273625 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | max_abs_log10_error | 0.801301810953687 | 0.8907868262969764 | 0.7974358436221284 | 0.08948501534328934 | -0.0038659673315586396 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | extreme_ge_10_count | 0.0040765130134838 | 0.028391167192429 | 0.0 | 0.0243146541789452 | -0.0040765130134838 |

## Final Candidate Dataset Manifest

| dataset_role | dataset_path | policy | description | recommended_use |
| --- | --- | --- | --- | --- |
| primary predictions | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_primary_analysis_predictions_test.csv | keep_pending | Primary prediction rows after review policy. | Use for primary final tables if this policy is selected. |
| sensitivity predictions | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_sensitivity_analysis_predictions_test.csv | keep_pending | Sensitivity prediction rows after review policy. | Use for sensitivity comparisons. |
| pending/unresolved rows | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_pending_or_unresolved_rows_test.csv | keep_pending | Rows still pending or unresolved. | Review before final reporting. |
| excluded from primary | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_excluded_from_primary_test.csv | keep_pending | Rows excluded from primary by policy. | Document exclusion rule if used. |
| excluded from sensitivity | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_excluded_from_sensitivity_test.csv | keep_pending | Rows excluded from sensitivity by policy. | Document exclusion rule if used. |
| metrics by scenario | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_metrics_by_review_scenario_config_test.csv | keep_pending | Review scenario metrics for all configs. | Use for method appendix. |
| default metrics | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\keep_pending_test\step7b_default_metrics_by_review_scenario_test.csv | keep_pending | Review scenario metrics for default configs. | Use for final comparison table. |
| primary predictions | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_primary_analysis_predictions_test.csv | exclude_pending_primary | Primary prediction rows after review policy. | Use for primary final tables if this policy is selected. |
| sensitivity predictions | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_sensitivity_analysis_predictions_test.csv | exclude_pending_primary | Sensitivity prediction rows after review policy. | Use for sensitivity comparisons. |
| pending/unresolved rows | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_pending_or_unresolved_rows_test.csv | exclude_pending_primary | Rows still pending or unresolved. | Review before final reporting. |
| excluded from primary | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_excluded_from_primary_test.csv | exclude_pending_primary | Rows excluded from primary by policy. | Document exclusion rule if used. |
| excluded from sensitivity | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_excluded_from_sensitivity_test.csv | exclude_pending_primary | Rows excluded from sensitivity by policy. | Document exclusion rule if used. |
| metrics by scenario | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_metrics_by_review_scenario_config_test.csv | exclude_pending_primary | Review scenario metrics for all configs. | Use for method appendix. |
| default metrics | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_default_metrics_by_review_scenario_test.csv | exclude_pending_primary | Review scenario metrics for default configs. | Use for final comparison table. |

## Final Readiness

| criterion | status | value | threshold_or_reason | comment |
| --- | --- | --- | --- | --- |
| reviewed_template_exists | pass | True | file exists | Reviewed decision template was loaded. |
| human_reviewed_decisions_exist | fail | 0 | > 0 | At least one human reviewed decision is required for full Step7C. |
| extreme_cases_reviewed | fail | 0/17 | at least one extreme reviewed | Extreme outlier review completion. |
| invalid_decisions_absent | pass | 0 | 0 | Invalid decisions must be corrected. |
| keep_pending_policy_outputs_exist | pass | True | exists | keep_pending Step7B output. |
| exclude_pending_policy_outputs_exist | pass | True | exists | exclude_pending_primary Step7B output. |
| primary_dataset_available | pass | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_primary_analysis_predictions_test.csv | non-empty | Primary candidate dataset exists. |
| sensitivity_dataset_available | pass | experiments\exp006\data\processed\step7c_reviewed_decisions_applied\exclude_pending_primary_test\step7b_sensitivity_analysis_predictions_test.csv | non-empty | Sensitivity candidate dataset exists. |
| unresolved_or_pending_cases_remaining | caution | 400 | caution if > 0 | Pending is allowed but must be disclosed. |
| extreme_outliers_remaining_in_primary | pass | 0.0 | caution if > 0 | Extreme outliers in exclude_pending_primary primary. |
| recommended_policy_for_step8 | caution | exclude_pending_primary | manual decision | Recommended policy for Step8 candidate dataset. |
| ready_for_step8 | fail | False | no invalid decisions and reviewed evidence | Step8 readiness. |

## Notes

- Step7C does not compute new sigma predictions.
- Step7C applies reviewed decisions by rerunning the existing Step7B decision application script.
- No figures are created.
- If pending decisions remain, final reporting should disclose the policy used.

- elapsed_seconds: 44.55
