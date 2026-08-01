# Step6D Outlier Robustness Audit Report

## Inputs

- broad_prediction_rows: 604440
- audit_prediction_rows_after_optional_limit: 64000
- broad_material_family_default_rows: 2000
- broad_metrics_rows: 64
- broad_default_summary_rows: 8
- default_comparison_rows: 8
- original_prediction_rows: 604856

## Outputs

- outlier_rows_topN: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_rows_topN_test.csv
- outlier_summary_by_row_id: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_row_id_test.csv
- outlier_summary_by_sample: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_sample_test.csv
- outlier_summary_by_paper: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_paper_test.csv
- top_outlier_sample_context_rows: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_top_outlier_sample_context_rows_test.csv
- robust_metrics_by_filter: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_robust_metrics_by_filter_test.csv
- robust_metrics_by_config: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_robust_metrics_by_config_test.csv
- original_vs_broad_robust_metrics_comparison: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_original_vs_broad_robust_metrics_comparison_test.csv
- error_contribution_concentration: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_error_contribution_concentration_test.csv
- error_contribution_summary: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_error_contribution_summary_test.csv
- manual_review_shortlist: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_manual_review_shortlist_test.csv
- readiness_summary: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_broad_family_main_result_readiness_summary_test.csv

## Broad Material Family Default Metrics

| default_label | config_id | filter_label | n_rows | n_samples | n_papers | retained_row_fraction | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | max_abs_log10_error | extreme_ge_10_count | severe_ge_5_count | large_ge_2_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | no_filter | 2000 | 317 | 231 | 1.0 | 0.19912650746534144 | 0.04972949862767547 | 0.8568205664943855 | 1.4699638977367877 | 0.3885 | 0.652 | 0.731 | 11.212152574256098 | 9 | 35 | 214 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | exclude_abs_error_ge_5 | 1965 | 311 | 228 | 0.9825 | 0.08075731884448137 | 0.039441382566675474 | 0.7501660303244244 | 1.1240552792167193 | 0.39541984732824426 | 0.6636132315521629 | 0.7440203562340967 | 4.9998091571626535 | 0 | 0 | 179 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | exclude_top_1p0_percent_abs_error | 1980 | 313 | 228 | 0.99 | 0.11885709344555793 | 0.04598513601234354 | 0.7831945268082287 | 1.2048978412384173 | 0.3924242424242424 | 0.6585858585858586 | 0.7383838383838384 | 5.227995042759014 | 0 | 15 | 194 |

## Largest Outliers

| row_id | paper_id | sample_id | formula_raw | material_group_key | T_K | sigma_S_per_m | sigma_pred_S_per_m | abs_error_decades | error_direction | likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| step0_00000916 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 303.1588 | 7.56579e-06 | 1233124.3023102197 | 11.212152574256098 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000917 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 312.5066 | 7.67388e-06 | 687702.8622116582 | 10.952385828222255 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000918 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 313.6876 | 7.67388e-06 | 592266.9178256801 | 10.887502471337074 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000924 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 343.5231 | 9.4935e-06 | 580628.0476320564 | 10.786471656161954 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000923 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 342.3408 | 9.4935e-06 | 501984.9119602189 | 10.723264309126158 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000919 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 322.2502 | 7.67388e-06 | 306829.0692391565 | 10.601881498991325 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000920 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 323.4311 | 7.67388e-06 | 283913.3421698723 | 10.56817079850579 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000922 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 334.0653 | 8.47502e-06 | 276975.84122819646 | 10.514301158321803 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000921 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 332.8835 | 8.47502e-06 | 270874.993388987 | 10.504628181970778 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00009966 | 759 | 6738 | (TlBiS2)0.5PbS | broad::sulfide | 299.1262 | 0.000228701 | 14305.479282802928 | 7.796234349172459 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |

## Sample Concentration

| validation_sample_group_id | paper_id_examples | sample_id_examples | sample_key_examples | formula_raw_examples | material_name_raw_examples | material_group_key_values | row_count | config_row_count | mean_abs_error_decades | median_abs_error_decades | max_abs_error_decades | extreme_ge_10_row_count | severe_ge_5_row_count | large_ge_2_row_count | factor10_or_more_row_count | fraction_factor10_or_more | T_min_K | T_max_K | sigma_exp_min_S_per_m | sigma_exp_max_S_per_m | sigma0_row_median_S_per_m | dominant_error_direction | dominant_likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33875::41898 | 33875 | 41898 | 10.1002_adfm.201900615__sample_41898 | CH3NH3PbI3 | CH3NH3PbI3 | broad::other_formula_system | 9 | 9 | 10.750084275210359 | 10.723264309126158 | 11.212152574256098 | 9 | 9 | 9 | 9 | 1.0 | 303.1588 | 343.5231 | 7.56579e-06 | 9.4935e-06 | 4.4552659377000014e-07 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 759::6738 | 759 | 6738 | 10.1007_s10582-005-0076-0__sample_6738 | (TlBiS2)0.5PbS | (TlBiS2)0.5PbS | broad::sulfide | 2 | 2 | 7.366034938302517 | 7.366034938302517 | 7.796234349172459 | 0 | 2 | 2 | 2 | 1.0 | 188.4141 | 299.1262 | 0.000228701 | 0.0002998742 | 0.000128849320966284 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1333::23917 | 1333 | 23917 | 10.1002_qua.22282__sample_23917 | (LiF)0.01(Fe2O3)0.99 | (LiF)0.01(Fe2O3)0.99 | broad::oxide | 1 | 1 | 7.220860441518514 | 7.220860441518514 | 7.220860441518514 | 0 | 1 | 1 | 1 | 1.0 | 391.9735 | 391.9735 | 0.03255632243781742 | 0.03255632243781742 | 0.00031435376033822556 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 759::6743 | 759 | 6743 | 10.1007_s10582-005-0076-0__sample_6743 | PbS | PbS | broad::sulfide | 2 | 2 | 6.156897547955841 | 6.156897547955841 | 6.342139891275035 | 0 | 2 | 2 | 2 | 1.0 | 122.4355 | 188.5743 | 0.0007961165 | 0.001203883 | 0.0003930117647204231 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 27838::34475 | 27838 | 34475 | 10.1007_s10853-020-04750-z__sample_34475 | C | C | broad::other_formula_system | 8 | 8 | 5.2759742184073275 | 5.244148055498648 | 5.461989604558317 | 0 | 8 | 8 | 8 | 1.0 | 302.9901 | 373.1558 | 7.829862 | 10.25682 | 0.18794022067422844 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 27838::34473 | 27838 | 34473 | 10.1007_s10853-020-04750-z__sample_34473 | C | C | broad::other_formula_system | 8 | 8 | 5.04922626808458 | 5.014511383962843 | 5.23395855444989 | 0 | 4 | 8 | 8 | 1.0 | 303.4798 | 373.2174 | 4.693105 | 5.783654 | 0.3086766361441485 | over_predicted | other_or_needs_manual_check |
| 11045::26360 | 11045 | 26360 | 10.1007_s11664-009-0666-x__sample_26360 | LaRh0.9Mg0.1O3 | LaRh0.9Mg0.1O3 | broad::oxide | 11 | 11 | 2.5140112094403033 | 2.2144796307247168 | 5.194131169277659 | 0 | 1 | 6 | 11 | 1.0 | 9.347218 | 68.94773 | 0.009800923639043743 | 36.61154310698003 | 2.0176259808756623 | over_predicted | other_or_needs_manual_check |
| 27838::34471 | 27838 | 34471 | 10.1007_s10853-020-04750-z__sample_34471 | C | C | broad::other_formula_system | 8 | 8 | 5.068653152696115 | 5.055787126432292 | 5.1591875666913785 | 0 | 8 | 8 | 8 | 1.0 | 303.1523 | 373.2556 | 3.485956 | 4.209354 | 0.2881780189641492 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1300::23896 | 1300 | 23896 | 10.1002_er.3052__sample_23896 | Fe2O3 | Fe2O3 | broad::oxide | 1 | 1 | 4.297755021718072 | 4.297755021718072 | 4.297755021718072 | 0 | 0 | 1 | 1 | 1.0 | 973.15 | 973.15 | 1.079967 | 1.079967 | 0.5229535563212886 | over_predicted | other_or_needs_manual_check |
| 27838::34478 | 27838 | 34478 | 10.1007_s10853-020-04750-z__sample_34478 | PbS/PEDOT:PSS+0.2wt% SWCNT | PbS/PEDOT:PSS+0.2wt% SWCNT | broad::oxide | 8 | 8 | 3.8106994208622433 | 3.768197652113315 | 3.9165693501966414 | 0 | 0 | 8 | 8 | 1.0 | 303.3121 | 372.9901 | 4.45005 | 4.8855 | 0.46425184474055675 | over_predicted | other_or_needs_manual_check |

## Paper Concentration

| validation_paper_group_id | paper_id_examples | doi_examples | row_count | sample_count | material_group_key_values | mean_abs_error_decades | median_abs_error_decades | max_abs_error_decades | extreme_ge_10_row_count | severe_ge_5_row_count | large_ge_2_row_count | factor10_or_more_row_count | fraction_factor10_or_more | T_min_K | T_max_K | dominant_error_direction | dominant_likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 33875 | 33875 | 10.1002/adfm.201900615 | 9 | 1 | broad::other_formula_system | 10.750084275210359 | 10.723264309126158 | 11.212152574256098 | 9 | 9 | 9 | 9 | 1.0 | 303.1588 | 343.5231 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 759 | 759 | 10.1007/s10582-005-0076-0 | 4 | 2 | broad::sulfide | 6.761466243129179 | 6.638987709353805 | 7.796234349172459 | 0 | 4 | 4 | 4 | 1.0 | 122.4355 | 299.1262 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1333 | 1333 | 10.1002/qua.22282 | 1 | 1 | broad::oxide | 7.220860441518514 | 7.220860441518514 | 7.220860441518514 | 0 | 1 | 1 | 1 | 1.0 | 391.9735 | 391.9735 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 27838 | 27838 | 10.1007/s10853-020-04750-z | 40 | 5 | broad::other_formula_system | broad::oxide | 4.540775402212245 | 5.001704150863478 | 5.461989604558317 | 0 | 20 | 40 | 40 | 1.0 | 302.9645 | 373.2556 | over_predicted | other_or_needs_manual_check |
| 11045 | 11045 | 10.1007/s11664-009-0666-x | 11 | 1 | broad::oxide | 2.5140112094403033 | 2.2144796307247168 | 5.194131169277659 | 0 | 1 | 6 | 11 | 1.0 | 9.347218 | 68.94773 | over_predicted | other_or_needs_manual_check |
| 1300 | 1300 | 10.1002/er.3052 | 1 | 1 | broad::oxide | 4.297755021718072 | 4.297755021718072 | 4.297755021718072 | 0 | 0 | 1 | 1 | 1.0 | 973.15 | 973.15 | over_predicted | other_or_needs_manual_check |
| 10663 | 10663 | 10.1007/s11664-010-1117-4 | 21 | 2 | broad::CoSb_skutterudite_like | 1.446089046431583 | 1.7288547230547306 | 3.914672822427282 | 0 | 0 | 7 | 13 | 0.6190476190476191 | 4.170709 | 286.7668 | over_predicted | other_or_needs_manual_check |
| 3009 | 3009 | 10.1007/s10854-011-0509-4 | 4 | 1 | broad::BiTe_like | 3.6934338583796436 | 3.7424084691634762 | 3.8761799792020963 | 0 | 0 | 4 | 4 | 1.0 | 296.9853 | 423.2999 | over_predicted | other_or_needs_manual_check |
| 3020 | 3020 | 10.1007/s10854-013-1430-9 | 1 | 1 | broad::oxide | 3.8059100969349307 | 3.8059100969349307 | 3.8059100969349307 | 0 | 0 | 1 | 1 | 1.0 | 622.6572 | 622.6572 | over_predicted | other_or_needs_manual_check |
| 11273 | 11273 | 10.1007/s10909-011-0390-9 | 6 | 1 | broad::oxide | 3.005747590172799 | 3.1858463195469175 | 3.6454437941279125 | 0 | 0 | 5 | 6 | 1.0 | 48.59438 | 340.1606 | under_predicted | other_or_needs_manual_check |

## Error Contribution Summary

| item | value | comment |
| --- | --- | --- |
| top1_sample_fraction_of_total_abs_error | 0.07335483635126713 | Top 1 sample(s) contribution to total absolute error. |
| top1_sample_fraction_of_total_squared_error | 0.0452284276568551 | Top 1 sample(s) contribution to total squared error. |
| top5_samples_fraction_of_total_abs_error | 0.2756847147791518 | Top 5 sample(s) contribution to total absolute error. |
| top5_samples_fraction_of_total_squared_error | 0.4267864062923362 | Top 5 sample(s) contribution to total squared error. |
| top10_samples_fraction_of_total_abs_error | 0.3972996773908971 | Top 10 sample(s) contribution to total absolute error. |
| top10_samples_fraction_of_total_squared_error | 0.6056566208872435 | Top 10 sample(s) contribution to total squared error. |
| top1_paper_fraction_of_total_abs_error | 0.10599127938281111 | Top 1 paper(s) contribution to total absolute error. |
| top1_paper_fraction_of_total_squared_error | 0.195895191451028 | Top 1 paper(s) contribution to total squared error. |
| top5_papers_fraction_of_total_abs_error | 0.380432742205734 | Top 5 paper(s) contribution to total absolute error. |
| top5_papers_fraction_of_total_squared_error | 0.6265677332372048 | Top 5 paper(s) contribution to total squared error. |
| top10_papers_fraction_of_total_abs_error | 0.5092305810497094 | Top 10 paper(s) contribution to total absolute error. |
| top10_papers_fraction_of_total_squared_error | 0.7106058813682709 | Top 10 paper(s) contribution to total squared error. |

## Readiness Summary

| criterion | status | value | threshold_or_reason | comment |
| --- | --- | --- | --- | --- |
| coverage_is_high | pass | 0.997895622895623 | >= 0.95 | Broad material_family default coverage. |
| material_family_differs_from_global | pass | 1.0 | > 0.1 | Checks whether broad grouping changes predictions. |
| mae_improved_vs_original | pass | -0.1283360528809405 | < -0.05 | Broad minus original MAE. |
| factor2_improved_vs_original | pass | 0.1003501742547003 | > 0.02 | Broad minus original factor2. |
| robust_mae_remains_improved_after_excluding_extreme_outliers | pass | 0.7501660303244244 | exclude_abs_error_ge_5 broad MAE < original no_filter MAE | Robustness after removing severe outliers. |
| not_dominated_by_single_sample_abs_error | pass | 0.07335483635126713 | < 0.20 | Top sample share of absolute error. |
| not_dominated_by_single_paper_abs_error | pass | 0.10599127938281111 | < 0.30 | Top paper share of absolute error. |
| extreme_outliers_exist | caution | 9 | caution if > 0 | Extreme outliers are expected to be audited, not treated as fatal. |
| manual_review_needed | caution | 11.212152574256098 | caution if extreme count > 0 or max_abs > 5 | Manual review of shortlist is required before final reporting. |
| recommended_next_action | caution | Use broad_family as a main candidate only with explicit outlier audit; review shortlist and consider reporting robust metrics alongside no-filter metrics. | manual decision | Next decision point. |

## Warnings

- optional step0 input not found: [WindowsPath('data/processed/step0_te_analysis_base.parquet'), WindowsPath('data/processed/step0_te_analysis_base.csv')]

## Sanity Checks

- prediction_valid_all_ok: True
- sigma_positive_finite: True
- sigma_pred_positive_finite: True
- sigma_pred_over_exp_consistent: True
- log_error_consistent: True
- sigma0_ratio_equals_prediction_error: True
- default_4_configs_exist: True
- broad_material_family_default_nonempty: True
- report_created: True
- did_not_read_step4_full_data_reference_curve: True
- did_not_read_raw_data: True
- did_not_compute_new_sigma_pred: True
- outlier_rows_topN_created: True
- outlier_summary_by_row_id_created: True
- outlier_summary_by_sample_created: True
- outlier_summary_by_paper_created: True
- top_outlier_sample_context_rows_created: True
- robust_metrics_by_filter_created: True
- robust_metrics_by_config_created: True
- original_vs_broad_robust_metrics_comparison_created: True
- error_contribution_concentration_created: True
- error_contribution_summary_created: True
- manual_review_shortlist_created: True
- readiness_summary_created: True

## Notes

- This Step6D audits existing Step6B/Step6C outputs only.
- No new sigma predictions are computed.
- Starrydata2 raw data and Step4 full-data reference curves are not read.
- Extreme-outlier exclusion is a sensitivity analysis, not a final data deletion decision.

## Next Actions

- Manually inspect the shortlist paper/sample rows.
- Decide whether to report broad_family no-filter and robust metrics together.
- Compare formula_system_collapsed if a second repaired grouping is needed.
- Select final tables and figures for reporting.

- elapsed_seconds: 10.80
