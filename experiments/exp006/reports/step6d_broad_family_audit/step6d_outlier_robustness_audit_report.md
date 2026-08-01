# Step6D Outlier Robustness Audit Report

## Inputs

- broad_prediction_rows: 604440
- audit_prediction_rows_after_optional_limit: 604440
- broad_material_family_default_rows: 18968
- broad_metrics_rows: 64
- broad_default_summary_rows: 8
- default_comparison_rows: 8
- original_prediction_rows: 604856

## Outputs

- outlier_rows_topN: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_rows_topN.csv
- outlier_summary_by_row_id: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_row_id.csv
- outlier_summary_by_sample: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_sample.csv
- outlier_summary_by_paper: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_outlier_summary_by_paper.csv
- top_outlier_sample_context_rows: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_top_outlier_sample_context_rows.csv
- robust_metrics_by_filter: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_robust_metrics_by_filter.csv
- robust_metrics_by_config: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_robust_metrics_by_config.csv
- original_vs_broad_robust_metrics_comparison: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_original_vs_broad_robust_metrics_comparison.csv
- error_contribution_concentration: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_error_contribution_concentration.csv
- error_contribution_summary: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_error_contribution_summary.csv
- manual_review_shortlist: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_manual_review_shortlist.csv
- readiness_summary: experiments\exp006\data\processed\step6d_broad_family_audit\step6d_broad_family_main_result_readiness_summary.csv

## Broad Material Family Default Metrics

| default_label | config_id | filter_label | n_rows | n_samples | n_papers | retained_row_fraction | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | max_abs_log10_error | extreme_ge_10_count | severe_ge_5_count | large_ge_2_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | no_filter | 18968 | 3189 | 2200 | 1.0 | 0.24535917400915397 | 0.0030867094458137314 | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.6865773935048503 | 0.7837937579080557 | 14.570733133464506 | 13 | 246 | 1371 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | exclude_abs_error_ge_5 | 18722 | 3170 | 2187 | 0.9870307886967524 | 0.1722325897191872 | -0.0033668853830050114 | 0.6507679626856471 | 1.0179067308983507 | 0.43227219314175835 | 0.6955987608161521 | 0.7940925114838159 | 4.9998091571626535 | 0 | 0 | 1125 |
| broad_material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | exclude_top_1p0_percent_abs_error | 18778 | 3175 | 2190 | 0.9899831294812316 | 0.1871935698252737 | -0.0018681436301120895 | 0.6643018482722893 | 1.0551694182674758 | 0.4309830652891682 | 0.6935243369900947 | 0.7917243582916178 | 5.418080719090004 | 0 | 56 | 1181 |

## Largest Outliers

| row_id | paper_id | sample_id | formula_raw | material_group_key | T_K | sigma_S_per_m | sigma_pred_S_per_m | abs_error_decades | error_direction | likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| step0_00130231 | 10260 | 5395 | In0.5Co4Sb12 | broad::CoSb_skutterudite_like | 699.1597 | 2.28237e+19 | 61327.168945525395 | 14.570733133464506 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| step0_00130230 | 10260 | 5395 | In0.5Co4Sb12 | broad::CoSb_skutterudite_like | 549.2997 | 2.088395e+19 | 92931.96980516672 | 14.351647502116625 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| step0_00130229 | 10260 | 5395 | In0.5Co4Sb12 | broad::CoSb_skutterudite_like | 501.6807 | 2.020321e+19 | 100760.70018259765 | 14.302129201352974 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| step0_00130228 | 10260 | 5395 | In0.5Co4Sb12 | broad::CoSb_skutterudite_like | 398.0392 | 1.92442e+19 | 114987.55847699205 | 14.22364900903764 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| step0_00000916 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 303.1588 | 7.56579e-06 | 1233124.3023102197 | 11.212152574256098 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000917 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 312.5066 | 7.67388e-06 | 687702.8622116582 | 10.952385828222255 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000918 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 313.6876 | 7.67388e-06 | 592266.9178256801 | 10.887502471337074 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000924 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 343.5231 | 9.4935e-06 | 580628.0476320564 | 10.786471656161954 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000923 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 342.3408 | 9.4935e-06 | 501984.9119602189 | 10.723264309126158 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| step0_00000919 | 33875 | 41898 | CH3NH3PbI3 | broad::other_formula_system | 322.2502 | 7.67388e-06 | 306829.0692391565 | 10.601881498991325 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |

## Sample Concentration

| validation_sample_group_id | paper_id_examples | sample_id_examples | sample_key_examples | formula_raw_examples | material_name_raw_examples | material_group_key_values | row_count | config_row_count | mean_abs_error_decades | median_abs_error_decades | max_abs_error_decades | extreme_ge_10_row_count | severe_ge_5_row_count | large_ge_2_row_count | factor10_or_more_row_count | fraction_factor10_or_more | T_min_K | T_max_K | sigma_exp_min_S_per_m | sigma_exp_max_S_per_m | sigma0_row_median_S_per_m | dominant_error_direction | dominant_likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10260::5395 | 10260 | 5395 | 10.1063_1.3596811__sample_5395 | In0.5Co4Sb12 | In0.5Co4Sb12 | broad::CoSb_skutterudite_like | 4 | 4 | 14.362039711492937 | 14.3268883517348 | 14.570733133464506 | 4 | 4 | 4 | 4 | 1.0 | 398.0392 | 699.1597 | 1.92442e+19 | 2.28237e+19 | 1.204645818588692e+19 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| 33875::41898 | 33875 | 41898 | 10.1002_adfm.201900615__sample_41898 | CH3NH3PbI3 | CH3NH3PbI3 | broad::other_formula_system | 9 | 9 | 10.750084275210359 | 10.723264309126158 | 11.212152574256098 | 9 | 9 | 9 | 9 | 1.0 | 303.1588 | 343.5231 | 7.56579e-06 | 9.4935e-06 | 4.4552659377000014e-07 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 10316::7968 | 10316 | 7968 | 10.1063_1.4820564__sample_7968 | CoS2 | CoS2 | broad::sulfide | 88 | 88 | 6.149091337759467 | 5.878200173610361 | 8.412448832124268 | 0 | 82 | 88 | 88 | 1.0 | 4.445221 | 316.7911 | 0.0002654687 | 0.015258 | 0.0006765275326827043 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 9442::24514 | 9442 | 24514 | 10.1134_1.1541047__sample_24514 | La2CuO4.0011 | La2CuO4.0011 | broad::oxide | 3 | 3 | 7.567857023469247 | 7.607003208282069 | 7.909334860713662 | 0 | 3 | 3 | 3 | 1.0 | 208.6589 | 250.0221 | 8.071898009954264e-05 | 0.00048711863482079883 | 3.713076473561057e-05 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1496::81442 | 1496 | 81442 | 10.1016_j.jallcom.2004.04.095__sample_81442 | CoNb0.5Ti0.5Sn | CoNb0.5Ti0.5Sn | broad::other_formula_system | 9 | 9 | 7.089943645379784 | 7.033760578776491 | 7.803100735126177 | 0 | 9 | 9 | 9 | 1.0 | 368.9 | 747.1 | 0.11456065986940085 | 0.16305233980107614 | 0.003424092010253274 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 759::6738 | 759 | 6738 | 10.1007_s10582-005-0076-0__sample_6738 | (TlBiS2)0.5PbS | (TlBiS2)0.5PbS | broad::sulfide | 2 | 2 | 7.366034938302517 | 7.366034938302517 | 7.796234349172459 | 0 | 2 | 2 | 2 | 1.0 | 188.4141 | 299.1262 | 0.000228701 | 0.0002998742 | 0.000128849320966284 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1333::23917 | 1333 | 23917 | 10.1002_qua.22282__sample_23917 | (LiF)0.01(Fe2O3)0.99 | (LiF)0.01(Fe2O3)0.99 | broad::oxide | 1 | 1 | 7.220860441518514 | 7.220860441518514 | 7.220860441518514 | 0 | 1 | 1 | 1 | 1.0 | 391.9735 | 391.9735 | 0.03255632243781742 | 0.03255632243781742 | 0.00031435376033822556 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 10054::26272 | 10054 | 26272 | 10.1063_1.1728294__sample_26272 | Ba2FeMoO6 | Ba2FeMoO6 | broad::oxide | 4 | 4 | 6.470842358928086 | 6.342850927780541 | 6.992711392015239 | 0 | 4 | 4 | 4 | 1.0 | 95.46273 | 283.7163 | 0.02349687550297999 | 0.0642375865681776 | 0.000718532795086234 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1173::25151 | 1173 | 25151 | 10.1021_ic902072v__sample_25151 | LaCoO3 | LaCoO3 | broad::oxide | 4 | 4 | 5.747529542020097 | 6.078946456487343 | 6.947022845515172 | 0 | 3 | 4 | 4 | 1.0 | 46.41979 | 127.2385 | 0.0011401712902132713 | 0.42655637625002346 | 0.0010811867620345843 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 568::19213 | 568 | 19213 | 10.1021_cm050412c__sample_19213 | TlTiPS5 | TlTiPS5 | broad::sulfide | 5 | 5 | 6.75576447456775 | 6.796459411582546 | 6.894727259596571 | 0 | 5 | 5 | 5 | 1.0 | 250.0149 | 289.7921 | 0.004653392090164125 | 0.006433433265996732 | 0.002542514386980586 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |

## Paper Concentration

| validation_paper_group_id | paper_id_examples | doi_examples | row_count | sample_count | material_group_key_values | mean_abs_error_decades | median_abs_error_decades | max_abs_error_decades | extreme_ge_10_row_count | severe_ge_5_row_count | large_ge_2_row_count | factor10_or_more_row_count | fraction_factor10_or_more | T_min_K | T_max_K | dominant_error_direction | dominant_likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10260 | 10260 | 10.1063/1.3596811 | 4 | 1 | broad::CoSb_skutterudite_like | 14.362039711492937 | 14.3268883517348 | 14.570733133464506 | 4 | 4 | 4 | 4 | 1.0 | 398.0392 | 699.1597 | under_predicted | sigma0_ref_much_smaller_than_row_sigma0 |
| 33875 | 33875 | 10.1002/adfm.201900615 | 9 | 1 | broad::other_formula_system | 10.750084275210359 | 10.723264309126158 | 11.212152574256098 | 9 | 9 | 9 | 9 | 1.0 | 303.1588 | 343.5231 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 10316 | 10316 | 10.1063/1.4820564 | 88 | 1 | broad::sulfide | 6.149091337759467 | 5.878200173610361 | 8.412448832124268 | 0 | 82 | 88 | 88 | 1.0 | 4.445221 | 316.7911 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 9442 | 9442 | 10.1134/1.1541047 | 13 | 2 | broad::oxide | 2.7400568946643125 | 1.2905449264354425 | 7.909334860713662 | 0 | 3 | 3 | 13 | 1.0 | 187.791 | 368.9775 | over_predicted | other_or_needs_manual_check |
| 1496 | 1496 | 10.1016/j.jallcom.2004.04.095 | 13 | 2 | broad::other_formula_system | 6.687364167984073 | 6.7545076958807 | 7.803100735126177 | 0 | 13 | 13 | 13 | 1.0 | 335.0 | 814.8 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 759 | 759 | 10.1007/s10582-005-0076-0 | 4 | 2 | broad::sulfide | 6.761466243129179 | 6.638987709353805 | 7.796234349172459 | 0 | 4 | 4 | 4 | 1.0 | 122.4355 | 299.1262 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1333 | 1333 | 10.1002/qua.22282 | 1 | 1 | broad::oxide | 7.220860441518514 | 7.220860441518514 | 7.220860441518514 | 0 | 1 | 1 | 1 | 1.0 | 391.9735 | 391.9735 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 10054 | 10054 | 10.1063/1.1728294 | 4 | 1 | broad::oxide | 6.470842358928086 | 6.342850927780541 | 6.992711392015239 | 0 | 4 | 4 | 4 | 1.0 | 95.46273 | 283.7163 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |
| 1173 | 1173 | 10.1021/ic902072v | 11 | 2 | broad::oxide | 2.5779118429491064 | 0.8980629376269669 | 6.947022845515172 | 0 | 3 | 4 | 4 | 0.36363636363636365 | 29.63017 | 264.3884 | under_predicted | other_or_needs_manual_check |
| 568 | 568 | 10.1021/cm050412c | 5 | 1 | broad::sulfide | 6.75576447456775 | 6.796459411582546 | 6.894727259596571 | 0 | 5 | 5 | 5 | 1.0 | 250.0149 | 289.7921 | over_predicted | sigma0_ref_much_larger_than_row_sigma0 |

## Error Contribution Summary

| item | value | comment |
| --- | --- | --- |
| top1_sample_fraction_of_total_abs_error | 0.039417228095073586 | Top 1 sample(s) contribution to total absolute error. |
| top1_sample_fraction_of_total_squared_error | 0.1138836458118289 | Top 1 sample(s) contribution to total squared error. |
| top5_samples_fraction_of_total_abs_error | 0.07544944130997888 | Top 5 sample(s) contribution to total absolute error. |
| top5_samples_fraction_of_total_squared_error | 0.17589475190017018 | Top 5 sample(s) contribution to total squared error. |
| top10_samples_fraction_of_total_abs_error | 0.11170914664666833 | Top 10 sample(s) contribution to total absolute error. |
| top10_samples_fraction_of_total_squared_error | 0.2669259272929802 | Top 10 sample(s) contribution to total squared error. |
| top1_paper_fraction_of_total_abs_error | 0.039417228095073586 | Top 1 paper(s) contribution to total absolute error. |
| top1_paper_fraction_of_total_squared_error | 0.1138836458118289 | Top 1 paper(s) contribution to total squared error. |
| top5_papers_fraction_of_total_abs_error | 0.09483626735676644 | Top 5 paper(s) contribution to total absolute error. |
| top5_papers_fraction_of_total_squared_error | 0.21122510558572874 | Top 5 paper(s) contribution to total squared error. |
| top10_papers_fraction_of_total_abs_error | 0.1426502448425827 | Top 10 paper(s) contribution to total absolute error. |
| top10_papers_fraction_of_total_squared_error | 0.2949226237828728 | Top 10 paper(s) contribution to total squared error. |

## Readiness Summary

| criterion | status | value | threshold_or_reason | comment |
| --- | --- | --- | --- | --- |
| coverage_is_high | pass | 0.997895622895623 | >= 0.95 | Broad material_family default coverage. |
| material_family_differs_from_global | pass | 1.0 | > 0.1 | Checks whether broad grouping changes predictions. |
| mae_improved_vs_original | pass | -0.1283360528809405 | < -0.05 | Broad minus original MAE. |
| factor2_improved_vs_original | pass | 0.1003501742547003 | > 0.02 | Broad minus original factor2. |
| robust_mae_remains_improved_after_excluding_extreme_outliers | pass | 0.6507679626856471 | exclude_abs_error_ge_5 broad MAE < original no_filter MAE | Robustness after removing severe outliers. |
| not_dominated_by_single_sample_abs_error | pass | 0.039417228095073586 | < 0.20 | Top sample share of absolute error. |
| not_dominated_by_single_paper_abs_error | pass | 0.039417228095073586 | < 0.30 | Top paper share of absolute error. |
| extreme_outliers_exist | caution | 13 | caution if > 0 | Extreme outliers are expected to be audited, not treated as fatal. |
| manual_review_needed | caution | 14.570733133464506 | caution if extreme count > 0 or max_abs > 5 | Manual review of shortlist is required before final reporting. |
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

- elapsed_seconds: 50.33
