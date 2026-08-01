# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step5b_test_predictions_valid.parquet
- input_rows: 604856
- evaluated_rows: 604856
- dropped_rows: 0
- config_count: 32
- metric_weighting: ['row_equal', 'sample_equal']
- min_eval_rows: 30
- min_eval_samples: 5
- metrics_by_config rows: 64
- metrics_by_carrier_type rows: 128
- metrics_by_material_family rows: 64
- metrics_by_temperature_bin rows: 992
- metrics_by_eta_bin rows: 384
- metrics_by_reliability_level rows: 176
- elapsed_seconds: 956.33

## Parquet Status

- step5c_metrics_by_config.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 19000 | 3191 | 2202 | 1 | 16 | 0.2896011572267853 | -0.012827817216535947 | 0.852081750001124 | 1.4064629875967487 | 1.3763607844037564 | -1.2348653212912115 | -0.4058738464241891 | 0.6837310369065903 | 2.8812138076537566 | 14.89226223168901 | 0.4917894736842105 | 0.5082105263157894 | 0.05410526315789474 | 0.3263157894736842 | 0.48473684210526313 | 0.6264736842105263 | 0.7508421052631579 | 3.1349640886579215 | 7.113474024787634 | 74006.81042773258 | 65384.1854778585 | 3.363524323546182 | 78.798495 | 0.1536145 | 1513.773 | 1870.0 | 862.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 19000 | 3191 | 2202 | 1 | 16 | 0.2896011572267853 | -0.012827817216535947 | 0.852081750001124 | 1.4064629875967487 | 1.3763607844037564 | -1.2348653212912115 | -0.4058738464241891 | 0.6837310369065903 | 2.8812138076537566 | 14.89226223168901 | 0.4917894736842105 | 0.5082105263157894 | 0.05410526315789474 | 0.3263157894736842 | 0.48473684210526313 | 0.6264736842105263 | 0.7508421052631579 | 3.1349640886579215 | 7.113474024787634 | 74006.81042773258 | 65384.1854778585 | 3.363524323546182 | 78.798495 | 0.1536145 | 1513.773 | 1870.0 | 862.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 20154 | 3179 | 866 | 1 | 16 | 0.18289070602324636 | -0.06354951823101276 | 0.8696361454419639 | 1.4699765395399786 | 1.4585909563695973 | -1.5567770477888336 | -0.4925921874862322 | 0.5620780677176904 | 2.5332863338567053 | 10.223962643057764 | 0.46635903542721047 | 0.5336409645727895 | 0.05730872283417684 | 0.3128907412920512 | 0.46735139426416594 | 0.6159571300982435 | 0.7470973504019053 | 3.2773590506806523 | 7.406894271698278 | 81817.305 | 64204.30829085391 | 3.7016634394884456 | 72.86373 | 0.01 | 1470.757 | 1377.0 | 575.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 20154 | 3179 | 866 | 1 | 16 | 0.18289070602324636 | -0.06354951823101276 | 0.8696361454419639 | 1.4699765395399786 | 1.4585909563695973 | -1.5567770477888336 | -0.4925921874862322 | 0.5620780677176904 | 2.5332863338567053 | 10.223962643057764 | 0.46635903542721047 | 0.5336409645727895 | 0.05730872283417684 | 0.3128907412920512 | 0.46735139426416594 | 0.6159571300982435 | 0.7470973504019053 | 3.2773590506806523 | 7.406894271698278 | 81817.305 | 64204.30829085391 | 3.7016634394884456 | 72.86373 | 0.01 | 1470.757 | 1377.0 | 575.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3191 | 3191 | 2202 | 1 | 25 | 0.297589776359606 | 0.0012095470172210562 | 0.7372337311582936 | 1.2343448639232923 | 1.1981224587962072 | -0.8289403198590264 | -0.3500925437644197 | 0.6489961055633299 | 2.528492842149041 | 14.62486251838327 | 0.5001566906925728 | 0.49984330930742715 | 0.05390159824506424 | 0.350987151363209 | 0.5455969915387026 | 0.6950799122532122 | 0.7972422438107176 | 2.711840885523711 | 5.460516593184133 | 67535.89 | 60939.9653069517 | 2.5985643357794 | 96.071305 | 1.530612 | 1434.467 | 2221.0 | 992.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3191 | 3191 | 2202 | 1 | 25 | 0.297589776359606 | 0.0012095470172210562 | 0.7372337311582936 | 1.2343448639232923 | 1.1981224587962072 | -0.8289403198590264 | -0.3500925437644197 | 0.6489961055633299 | 2.528492842149041 | 14.62486251838327 | 0.5001566906925728 | 0.49984330930742715 | 0.05390159824506424 | 0.350987151363209 | 0.5455969915387026 | 0.6950799122532122 | 0.7972422438107176 | 2.711840885523711 | 5.460516593184133 | 67535.89 | 60939.9653069517 | 2.5985643357794 | 96.071305 | 1.530612 | 1434.467 | 2221.0 | 992.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3179 | 3179 | 866 | 1 | 25 | 0.20754101505468883 | -0.031061931256368816 | 0.733850844545466 | 1.2177864499192326 | 1.2001598468516517 | -1.1226524307461208 | -0.41113979960871827 | 0.5758527755649367 | 2.2483578695606288 | 9.969421632913956 | 0.4819125511167034 | 0.5180874488832966 | 0.056621579112928595 | 0.34256055363321797 | 0.5174583202264863 | 0.6860648002516515 | 0.8052846807172067 | 2.866148668703084 | 5.418147758831544 | 73665.84 | 59171.976010732076 | 2.682889807604163 | 93.946605 | 0.32348 | 1431.07 | 2234.5 | 862.5 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3179 | 3179 | 866 | 1 | 25 | 0.20754101505468883 | -0.031061931256368816 | 0.733850844545466 | 1.2177864499192326 | 1.2001598468516517 | -1.1226524307461208 | -0.41113979960871827 | 0.5758527755649367 | 2.2483578695606288 | 9.969421632913956 | 0.4819125511167034 | 0.5180874488832966 | 0.056621579112928595 | 0.34256055363321797 | 0.5174583202264863 | 0.6860648002516515 | 0.8052846807172067 | 2.866148668703084 | 5.418147758831544 | 73665.84 | 59171.976010732076 | 2.682889807604163 | 93.946605 | 0.32348 | 1431.07 | 2234.5 | 862.5 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.012827817216535947 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.0012095470172210562 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.012827817216535947 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.0012095470172210562 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.06354951823101276 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.031061931256368816 | 0.34256055363321797 | 0.6860648002516515 | 0.8052846807172067 | 1.0 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.06354951823101276 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.031061931256368816 | 0.34256055363321797 | 0.6860648002516515 | 0.8052846807172067 | 1.0 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.7609681213353524 | 1.223927923089769 | 0.34075661047527767 | 0.7774708237018662 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.7609681213353524 | 1.223927923089769 | 0.34075661047527767 | 0.7774708237018662 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.761704635320337 | 1.2207029048825329 | 0.3336528161470373 | 0.7793313412640244 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.761704635320337 | 1.2207029048825329 | 0.3336528161470373 | 0.7793313412640244 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.7719300394286521 | 1.2283642759894442 | 0.3401928172746237 | 0.7700851327732987 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.7719300394286521 | 1.2283642759894442 | 0.3401928172746237 | 0.7700851327732987 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.7731719542472103 | 1.2245607530275178 | 0.33077747082370185 | 0.7703106500535604 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.7731719542472103 | 1.2245607530275178 | 0.33077747082370185 | 0.7703106500535604 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.8323381124904484 | 1.4395958363051942 | 0.3175357982474888 | 0.7637315665740543 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.8323381124904484 | 1.4395958363051942 | 0.3175357982474888 | 0.7637315665740543 | 1.0 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.7609681213353524 | 1.223927923089769 | 0.34075661047527767 | 0.7774708237018662 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.7609681213353524 | 1.223927923089769 | 0.34075661047527767 | 0.7774708237018662 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.7719300394286521 | 1.2283642759894442 | 0.3401928172746237 | 0.7700851327732987 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.7719300394286521 | 1.2283642759894442 | 0.3401928172746237 | 0.7700851327732987 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.761704635320337 | 1.2207029048825329 | 0.3336528161470373 | 0.7793313412640244 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.761704635320337 | 1.2207029048825329 | 0.3336528161470373 | 0.7793313412640244 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.7731719542472103 | 1.2245607530275178 | 0.33077747082370185 | 0.7703106500535604 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.7731719542472103 | 1.2245607530275178 | 0.33077747082370185 | 0.7703106500535604 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.852081750001124 | 1.4064629875967487 | 0.3263157894736842 | 0.7508421052631579 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.852081750001124 | 1.4064629875967487 | 0.3263157894736842 | 0.7508421052631579 | 0.9995791245791246 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.7838454822944709, 'sample_holdout': 0.7492782437572293}
- group_scheme median mae_log10: {'global': 0.7492782437572293, 'material_family': 0.7492782437572293}
- curve_method median mae_log10: {'row_median': 0.7487323268338312, 'sample_median': 0.7547592028038792}
- reference_source_subset median mae_log10: {'all_valid': 0.7496465007497216, 'conservative_valid': 0.749100926246823}
- p/n median mae_log10: {'n': 0.7442448357214205, 'p': 0.7957378523769271}
- eta bin median mae_log10: {'[1, 2)': 0.5779087221153717, '[10, 20)': 1.0783789176988776, '[2, 5)': 0.6805439791628815, '[20, 50)': 1.19691007130419, '[5, 10)': 0.8717613228540417, '[50, inf)': 1.5537974650976074}
- temperature bin median mae_log10: {'-50_50K': 1.3859255384698899, '1050_1150K': 0.7057805992327393, '1150_1250K': 0.5686896286420315, '1250_1350K': 0.5348744132398098, '1350_1450K': 1.065479411522126, '1450_1550K': 0.719318195409248, '150_250K': 0.9699541278533828, '250_350K': 0.7603986171026642, '350_450K': 0.5973164063826097, '450_550K': 0.5173069162789474, '50_150K': 1.0153557591235929, '550_650K': 0.5241652242861505, '650_750K': 0.536284259227973, '750_850K': 0.594031347521367, '850_950K': 0.5678900920671868, '950_1050K': 0.6287546315333101}
- reliability_level median mae_log10: {'high': 0.7493579302429179, 'low': 1.0046393854848794, 'medium': 0.6116411441512877}
- largest abs_log10 error: 14.924287322210992

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
- metrics_by_config_has_32_configs: True
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
