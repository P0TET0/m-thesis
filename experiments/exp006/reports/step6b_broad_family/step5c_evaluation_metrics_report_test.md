# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid_test.csv
- input_rows: 6400
- evaluated_rows: 6400
- dropped_rows: 0
- config_count: 32
- metric_weighting: ['row_equal', 'sample_equal']
- min_eval_rows: 30
- min_eval_samples: 5
- metrics_by_config rows: 64
- metrics_by_carrier_type rows: 128
- metrics_by_material_family rows: 728
- metrics_by_temperature_bin rows: 512
- metrics_by_eta_bin rows: 352
- metrics_by_reliability_level rows: 128
- elapsed_seconds: 18.08

## Parquet Status

- step5c_metrics_by_config_test.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 47 | 34 | 14 | 10 | 0.5528062807471611 | 0.0148308342332969 | 0.8178684621517095 | 2.3196580411837555 | 2.25847786247094 | -0.5788412875313835 | -0.21170351317035158 | 0.4015992037264114 | 1.2880612783883691 | 11.074686531963357 | 0.525 | 0.475 | 0.155 | 0.525 | 0.675 | 0.84 | 0.895 | 1.9386677657200253 | 6.574586780262435 | 76277.50118503075 | 68492.59626697673 | 2.6236894252893483 | 94.810225 | 10.6618 | 913.9761 | 161.0 | 71.0 | True | high | 929 | 928 | 1 | 0.9989235737351992 | 23 | 22 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 51 | 38 | 12 | 10 | 0.4152970389195913 | -0.061001943124084995 | 0.823180408230256 | 2.333612572587181 | 2.3021239937969584 | -0.688159820665814 | -0.2303000878485661 | 0.20310774454709052 | 1.910492934377067 | 11.06538413240066 | 0.445 | 0.555 | 0.18 | 0.595 | 0.71 | 0.85 | 0.885 | 1.68207734971379 | 6.655495716061042 | 71885.4569635675 | 69057.64800699629 | 2.477959439809393 | 98.54305 | 10.6618 | 913.9761 | 18.0 | 8.0 | True | high | 929 | 876 | 53 | 0.9429494079655544 | 173 | 130 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 48 | 15 | 9 | 7 | 0.039912052109060174 | 0.03776294917106645 | 0.33912576910421777 | 0.460027599328604 | 0.45944298834368247 | -0.6536554410682639 | -0.27407527301802703 | 0.2376090079506366 | 0.825454481735033 | 1.5741617365549485 | 0.545 | 0.455 | 0.135 | 0.575 | 0.73 | 0.87 | 0.96 | 1.7467342105364791 | 2.1833621090058 | 47689.8216305049 | 49047.584842762866 | 1.7690003747879834 | 120.05675 | 80.7021 | 704.172 | 185.0 | 59.0 | True | high | 1205 | 1205 | 0 | 1.0 | 24 | 22 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 56 | 17 | 10 | 5 | 0.05733462049164591 | -0.02760447578529085 | 0.29216384486954367 | 0.42660258557562747 | 0.42379300355503813 | -0.41672032090015115 | -0.17401596477590536 | 0.24554119084045678 | 0.8220265630823047 | 1.652940701124915 | 0.465 | 0.535 | 0.155 | 0.665 | 0.83 | 0.905 | 0.95 | 1.5961926667813324 | 1.959583819896809 | 56780.24139804156 | 59167.92904509296 | 1.8483560577110927 | 117.34565 | 123.3824 | 592.6479 | 21.0 | 7.0 | True | high | 1205 | 1062 | 143 | 0.8813278008298755 | 175 | 127 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 47 | 47 | 34 | 14 | 11 | 0.2740169518502932 | 0.0316605759704105 | 0.5996360234703954 | 1.6308906492932136 | 1.625087248108501 | -0.6036648777499037 | -0.30023578573567905 | 0.4089056663685988 | 1.059855277850421 | 10.700918022112369 | 0.5319148936170213 | 0.46808510638297873 | 0.10638297872340426 | 0.425531914893617 | 0.6595744680851063 | 0.851063829787234 | 0.9361702127659575 | 2.1274699452511325 | 3.9777366191075108 | 74445.1971681047 | 58408.88295666479 | 2.1964971133124562 | 106.9822 | 37.22425 | 637.607 | 161.0 | 66.0 | True | high | 929 | 928 | 1 | 0.9989235737351992 | 23 | 22 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 51 | 51 | 38 | 12 | 10 | 0.2267751766270272 | -0.015803477793012798 | 0.611387783296886 | 1.6178065060672546 | 1.6177726442693068 | -1.0438743304125144 | -0.21640696187033048 | 0.2770269089807824 | 1.0621202108438617 | 10.649213737053897 | 0.45098039215686275 | 0.5490196078431373 | 0.23529411764705882 | 0.5490196078431373 | 0.7058823529411765 | 0.7843137254901961 | 0.8823529411764706 | 1.8552622245915757 | 4.0868413918224915 | 89098.05151471142 | 82729.06354764166 | 2.4588713740521353 | 100.27359 | 37.22425 | 637.607 | 18.0 | 8.0 | True | high | 929 | 876 | 53 | 0.9429494079655544 | 173 | 130 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 48 | 48 | 15 | 9 | 8 | 0.09647099444462785 | 0.0543506430437065 | 0.38629255663838963 | 0.5147931392794927 | 0.5110243265027105 | -0.5889496341658306 | -0.2756989208695388 | 0.2837028519440611 | 1.1341019245198403 | 1.5729905155360078 | 0.5416666666666666 | 0.4583333333333333 | 0.10416666666666667 | 0.5208333333333334 | 0.6875 | 0.8541666666666666 | 0.9166666666666666 | 1.935952800819882 | 2.4338429825763686 | 40849.349778946766 | 47058.78709822052 | 1.7507808819400688 | 120.842175 | 80.7021 | 562.1384 | 160.0 | 63.75 | True | high | 1205 | 1205 | 0 | 1.0 | 24 | 22 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 56 | 56 | 17 | 10 | 5 | 0.10374911118359043 | -0.01040146306961545 | 0.3082467330933377 | 0.4701968807116813 | 0.4627583072156106 | -0.3843161385262485 | -0.1282819925197881 | 0.2550666210746143 | 0.9290106002750984 | 1.6219030151239835 | 0.48214285714285715 | 0.5178571428571429 | 0.14285714285714285 | 0.6785714285714286 | 0.8035714285714286 | 0.8571428571428571 | 0.9285714285714286 | 1.4597021179197838 | 2.033511969662971 | 71262.23160957663 | 89216.97293416798 | 1.9137054044883375 | 115.28540000000001 | 288.6527 | 560.6567 | 24.0 | 8.0 | True | high | 1205 | 1062 | 143 | 0.8813278008298755 | 175 | 127 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.823180408230256 | 2.333612572587181 | -0.061001943124084995 | 0.595 | 0.85 | 0.885 | 0.9429494079655544 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.611387783296886 | 1.6178065060672546 | -0.015803477793012798 | 0.5490196078431373 | 0.7843137254901961 | 0.8823529411764706 | 0.9429494079655544 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8178684621517095 | 2.3196580411837555 | 0.0148308342332969 | 0.525 | 0.84 | 0.895 | 0.9989235737351992 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.5996360234703954 | 1.6308906492932136 | 0.0316605759704105 | 0.425531914893617 | 0.851063829787234 | 0.9361702127659575 | 0.9989235737351992 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.29216384486954367 | 0.42660258557562747 | -0.02760447578529085 | 0.665 | 0.905 | 0.95 | 0.8813278008298755 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.3082467330933377 | 0.4701968807116813 | -0.01040146306961545 | 0.6785714285714286 | 0.8571428571428571 | 0.9285714285714286 | 0.8813278008298755 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.33912576910421777 | 0.460027599328604 | 0.03776294917106645 | 0.575 | 0.87 | 0.96 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.38629255663838963 | 0.5147931392794927 | 0.0543506430437065 | 0.5208333333333334 | 0.8541666666666666 | 0.9166666666666666 | 1.0 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.33912576910421777 | 0.460027599328604 | 0.575 | 0.96 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.33912576910421777 | 0.460027599328604 | 0.575 | 0.96 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.3397820190247073 | 0.4590112099897444 | 0.565 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3397820190247073 | 0.4590112099897444 | 0.565 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | 0.35116670602384564 | 0.4665327276776724 | 0.565 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.35116670602384564 | 0.4665327276776724 | 0.565 | 0.965 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.3550916685648331 | 0.49676983775469286 | 0.56 | 0.935 | 0.9988571428571428 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3552960371583827 | 0.4957287493664981 | 0.555 | 0.935 | 0.9988571428571428 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | 0.3555877704436257 | 0.47499295564280086 | 0.555 | 0.945 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.3555877704436257 | 0.47499295564280086 | 0.555 | 0.945 | 1.0 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.33912576910421777 | 0.460027599328604 | 0.575 | 0.96 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.33912576910421777 | 0.460027599328604 | 0.575 | 0.96 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.3397820190247073 | 0.4590112099897444 | 0.565 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3397820190247073 | 0.4590112099897444 | 0.565 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | 0.35116670602384564 | 0.4665327276776724 | 0.565 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.35116670602384564 | 0.4665327276776724 | 0.565 | 0.965 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.3550916685648331 | 0.49676983775469286 | 0.56 | 0.935 | 0.9988571428571428 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3552960371583827 | 0.4957287493664981 | 0.555 | 0.935 | 0.9988571428571428 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | 0.3555877704436257 | 0.47499295564280086 | 0.555 | 0.945 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.3555877704436257 | 0.47499295564280086 | 0.555 | 0.945 | 1.0 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.3306522485096308, 'sample_holdout': 0.5729303246041435}
- group_scheme median mae_log10: {'global': 0.4010186891647377, 'material_family': 0.3367680559640735}
- curve_method median mae_log10: {'row_median': 0.4415675553312495, 'sample_median': 0.37079429689838617}
- reference_source_subset median mae_log10: {'all_valid': 0.3928982418104461, 'conservative_valid': 0.38629255663838963}
- p/n median mae_log10: {'n': 0.39660462274537833, 'p': 0.39534742920813326}
- eta bin median mae_log10: {'[1, 2)': 0.2777752417168828, '[10, 20)': 1.3443076665324005, '[2, 5)': 0.3260524352951968, '[20, 50)': 1.559069395236669, '[5, 10)': 0.5290702802090512, '[50, inf)': 1.9733671941238087}
- temperature bin median mae_log10: {'-50_50K': 2.0819119168528366, '150_250K': 0.7018776546199161, '250_350K': 0.45381772489038735, '350_450K': 0.28046116302871776, '450_550K': 0.3008270943147101, '50_150K': 0.8239010974797718, '550_650K': 0.2703645833536035, '650_750K': 0.28985051568236425, '750_850K': 0.15458500661758323, '850_950K': 0.14182699080683717}
- reliability_level median mae_log10: {'high': 0.37347272680377497, 'low': 0.5227982299639913, 'medium': 0.3661789177049125}
- largest abs_log10 error: 11.074686531963357

## Sanity Check

- prediction_status_ok: True
- sigma_exp_positive: True
- sigma_pred_positive: True
- F0_positive: True
- log10_error_finite: True
- abs_error_consistent: True
- squared_error_consistent: True
- ratio_consistent: True
- log10_ratio_consistent: True
- factor_accuracy_range: True
- mae_nonnegative: True
- rmse_nonnegative: True
- max_abs_ge_mae: True
- n_rows_positive: True
- is_reliable_eval_group_rule: True
- eval_group_reliability_rule: True
- default_comparison_complete: True
- ranking_nonempty: True
- largest_error_rows_limit: True
- coverage_fraction_range: True

## Notes

- WARNING: none
- Main metric is log10(sigma_pred / sigma_exp).
- Sigma spans orders of magnitude, so log error is used instead of ordinary absolute error.
- Step5B train-only sigma0_ref is used; Step4 full-data curves are not used for independent validation.
- Test-side sigma0_S_per_m is not used to create predictions.
- Step5D should create predicted-vs-experimental, error distribution, config comparison, eta, temperature, and material-family plots.
