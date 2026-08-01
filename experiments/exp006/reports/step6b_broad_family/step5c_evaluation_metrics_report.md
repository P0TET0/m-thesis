# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step6b_broad_family\step5b_test_predictions_valid.parquet
- input_rows: 604440
- evaluated_rows: 604440
- dropped_rows: 0
- config_count: 32
- metric_weighting: ['row_equal', 'sample_equal']
- min_eval_rows: 30
- min_eval_samples: 5
- metrics_by_config rows: 64
- metrics_by_carrier_type rows: 128
- metrics_by_material_family rows: 960
- metrics_by_temperature_bin rows: 928
- metrics_by_eta_bin rows: 384
- metrics_by_reliability_level rows: 184
- elapsed_seconds: 773.47

## Parquet Status

- step5c_metrics_by_config.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 19000 | 3191 | 2202 | 15 | 16 | 0.2896011572267853 | -0.012827817216535947 | 0.852081750001124 | 1.4064629875967487 | 1.3763607844037564 | -1.2348653212912115 | -0.4058738464241891 | 0.6837310369065903 | 2.8812138076537566 | 14.89226223168901 | 0.4917894736842105 | 0.5082105263157894 | 0.05410526315789474 | 0.3263157894736842 | 0.48473684210526313 | 0.6264736842105263 | 0.7508421052631579 | 3.1349640886579215 | 7.113474024787634 | 74006.81042773258 | 65384.1854778585 | 3.363524323546182 | 78.798495 | 0.1536145 | 1513.773 | 1870.0 | 862.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 18968 | 3189 | 2200 | 15 | 14 | 0.24535917400915397 | 0.0030867094458137314 | 0.7237456971201834 | 1.2508679717579398 | 1.2266005376138873 | -1.14624959223138 | -0.3152017140204299 | 0.5096063555950064 | 2.4686078644348384 | 14.570733133464506 | 0.5028996204133277 | 0.4971003795866723 | 0.0882539013074652 | 0.42666596372838467 | 0.5674293547026571 | 0.6865773935048503 | 0.7837937579080557 | 2.4326871506297865 | 5.293533881188107 | 74193.88187082935 | 75028.29655938873 | 3.363443872380753 | 78.80000000000001 | 0.1536145 | 1284.759 | 311.0 | 142.0 | True | high | 19008 | 18968 | 40 | 0.997895622895623 | 316 | 272 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 20154 | 3179 | 866 | 15 | 16 | 0.18289070602324636 | -0.06354951823101276 | 0.8696361454419639 | 1.4699765395399786 | 1.4585909563695973 | -1.5567770477888336 | -0.4925921874862322 | 0.5620780677176904 | 2.5332863338567053 | 10.223962643057764 | 0.46635903542721047 | 0.5336409645727895 | 0.05730872283417684 | 0.3128907412920512 | 0.46735139426416594 | 0.6159571300982435 | 0.7470973504019053 | 3.2773590506806523 | 7.406894271698278 | 81817.305 | 64204.30829085391 | 3.7016634394884456 | 72.86373 | 0.01 | 1470.757 | 1377.0 | 575.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 20120 | 3175 | 865 | 15 | 13 | 0.16593521082868654 | -0.023308142122029404 | 0.788113945871367 | 1.4026385467371438 | 1.3928233261706502 | -1.4696824485059348 | -0.4003722471170745 | 0.4501258687665042 | 2.362854995835188 | 10.269977794372917 | 0.4813121272365805 | 0.5186878727634194 | 0.07335984095427435 | 0.38901590457256463 | 0.5409542743538768 | 0.6720178926441351 | 0.7736083499005965 | 2.6397109527799745 | 6.139230590688097 | 81839.1 | 76271.54874752717 | 3.7021514966049685 | 72.85569999999998 | 0.01 | 1245.244 | 325.0 | 131.0 | True | high | 20154 | 20120 | 34 | 0.9983129899771758 | 312 | 271 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3191 | 3191 | 2202 | 15 | 25 | 0.297589776359606 | 0.0012095470172210562 | 0.7372337311582936 | 1.2343448639232923 | 1.1981224587962072 | -0.8289403198590264 | -0.3500925437644197 | 0.6489961055633299 | 2.528492842149041 | 14.62486251838327 | 0.5001566906925728 | 0.49984330930742715 | 0.05390159824506424 | 0.350987151363209 | 0.5455969915387026 | 0.6950799122532122 | 0.7972422438107176 | 2.711840885523711 | 5.460516593184133 | 67535.89 | 60939.9653069517 | 2.5985643357794 | 96.071305 | 1.530612 | 1434.467 | 2221.0 | 992.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3189 | 3189 | 2200 | 15 | 23 | 0.2515415432463362 | 0.011340591210378358 | 0.6264331342694931 | 1.1209937015591127 | 1.0925786277072922 | -0.8131373209248701 | -0.252533873384342 | 0.4321025043029916 | 2.3177097882382447 | 14.3268883517348 | 0.5117591721542804 | 0.48824082784571965 | 0.09783631232361242 | 0.4816556914393227 | 0.6390718093446222 | 0.7375352775164629 | 0.8256506741925368 | 2.0743038145967754 | 4.230903640023825 | 67583.3217878492 | 71643.38934181484 | 2.5958205783945183 | 96.138055 | 1.530612 | 1147.432 | 307.5 | 133.0 | True | high | 19008 | 18968 | 40 | 0.997895622895623 | 316 | 272 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 3179 | 3179 | 866 | 15 | 25 | 0.20754101505468883 | -0.031061931256368816 | 0.733850844545466 | 1.2177864499192326 | 1.2001598468516517 | -1.1226524307461208 | -0.41113979960871827 | 0.5758527755649367 | 2.2483578695606288 | 9.969421632913956 | 0.4819125511167034 | 0.5180874488832966 | 0.056621579112928595 | 0.34256055363321797 | 0.5174583202264863 | 0.6860648002516515 | 0.8052846807172067 | 2.866148668703084 | 5.418147758831544 | 73665.84 | 59171.976010732076 | 2.682889807604163 | 93.946605 | 0.32348 | 1431.07 | 2234.5 | 862.5 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 3175 | 3175 | 865 | 15 | 23 | 0.16271804760999542 | -0.018263520284621717 | 0.6383798691230225 | 1.1306923247085083 | 1.119098932754846 | -1.0790904144446305 | -0.31531435164944943 | 0.3809815543239552 | 2.0632338840530484 | 10.01543678422911 | 0.4825196850393701 | 0.51748031496063 | 0.08692913385826771 | 0.4566929133858268 | 0.6192125984251968 | 0.7376377952755906 | 0.8299212598425196 | 2.179882726001953 | 4.348904482634019 | 73915.1 | 69817.38754889343 | 2.67788079854852 | 93.96012999999999 | 0.32348 | 1245.244 | 296.0 | 119.5 | True | high | 20154 | 20120 | 34 | 0.9983129899771758 | 312 | 271 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.7237456971201834 | 1.2508679717579398 | 0.0030867094458137314 | 0.42666596372838467 | 0.6865773935048503 | 0.7837937579080557 | 0.997895622895623 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6264331342694931 | 1.1209937015591127 | 0.011340591210378358 | 0.4816556914393227 | 0.7375352775164629 | 0.8256506741925368 | 0.997895622895623 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.012827817216535947 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.0012095470172210562 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.788113945871367 | 1.4026385467371438 | -0.023308142122029404 | 0.38901590457256463 | 0.6720178926441351 | 0.7736083499005965 | 0.9983129899771758 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6383798691230225 | 1.1306923247085083 | -0.018263520284621717 | 0.4566929133858268 | 0.7376377952755906 | 0.8299212598425196 | 0.9983129899771758 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.06354951823101276 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.031061931256368816 | 0.34256055363321797 | 0.6860648002516515 | 0.8052846807172067 | 1.0 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.641952431956595 | 1.0809822610551238 | 0.438431151241535 | 0.8123024830699774 | 0.9990415515588882 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.6421234247007328 | 1.07439666337015 | 0.4337471783295711 | 0.8137133182844244 | 0.9990415515588882 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.6528053641535432 | 1.0927128362465575 | 0.44351015801354404 | 0.805530474040632 | 0.9990415515588882 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.6528417385473161 | 1.0864039236742844 | 0.4406884875846501 | 0.8071106094808126 | 0.9990415515588882 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | 0.7064101063340971 | 1.2159171493637655 | 0.41975959510754957 | 0.793019822859553 | 0.997895622895623 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | 0.7146408688606549 | 1.253078237969729 | 0.42413538591311684 | 0.7921762969211303 | 0.997895622895623 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.7217842706784077 | 1.2392220881729283 | 0.42418810628426823 | 0.7856389708983551 | 0.997895622895623 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.7837937579080557 | 0.997895622895623 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.7502727178843275 | 1.3668436646895337 | 0.4025473616611367 | 0.78930750294338 | 0.9983970933960248 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.7524992602197149 | 1.3637767724574037 | 0.39454399572078097 | 0.7894089328697512 | 0.9988779653772174 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.6528053641535432 | 1.0927128362465575 | 0.44351015801354404 | 0.805530474040632 | 0.9990415515588882 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.6528417385473161 | 1.0864039236742844 | 0.4406884875846501 | 0.8071106094808126 | 0.9990415515588882 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.641952431956595 | 1.0809822610551238 | 0.438431151241535 | 0.8123024830699774 | 0.9990415515588882 |
| sample_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | 0.6421234247007328 | 1.07439666337015 | 0.4337471783295711 | 0.8137133182844244 | 0.9990415515588882 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.7237456971201834 | 1.2508679717579398 | 0.42666596372838467 | 0.7837937579080557 | 0.997895622895623 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.7217842706784077 | 1.2392220881729283 | 0.42418810628426823 | 0.7856389708983551 | 0.997895622895623 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | 0.7146408688606549 | 1.253078237969729 | 0.42413538591311684 | 0.7921762969211303 | 0.997895622895623 |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | 0.7064101063340971 | 1.2159171493637655 | 0.41975959510754957 | 0.793019822859553 | 0.997895622895623 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | 0.7502727178843275 | 1.3668436646895337 | 0.4025473616611367 | 0.78930750294338 | 0.9983970933960248 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.7620509748335719 | 1.3824137248727064 | 0.3997645295943487 | 0.7863641228727389 | 0.9983970933960248 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.7428127849914106, 'sample_holdout': 0.7033402259882697}
- group_scheme median mae_log10: {'global': 0.7492782437572293, 'material_family': 0.6420379283286639}
- curve_method median mae_log10: {'row_median': 0.7231425476577201, 'sample_median': 0.7287982708328247}
- reference_source_subset median mae_log10: {'all_valid': 0.7268797832763618, 'conservative_valid': 0.7276949617874844}
- p/n median mae_log10: {'n': 0.7136652480052352, 'p': 0.7589617401887636}
- eta bin median mae_log10: {'[1, 2)': 0.53124022455042, '[10, 20)': 1.0212734117160207, '[2, 5)': 0.6478841414670183, '[20, 50)': 1.1412651499658215, '[5, 10)': 0.8503287800578636, '[50, inf)': 1.532519165491704}
- temperature bin median mae_log10: {'-50_50K': 1.2906294663128228, '1050_1150K': 0.692536216975786, '1150_1250K': 0.5177970551698405, '1250_1350K': 0.6773422180901643, '1350_1450K': 1.065479411522126, '1450_1550K': 0.719318195409248, '150_250K': 0.9069282478116518, '250_350K': 0.7134024829185478, '350_450K': 0.5421961426703594, '450_550K': 0.47364398214376213, '50_150K': 0.9942311547641208, '550_650K': 0.4816475845175427, '650_750K': 0.4689902572008321, '750_850K': 0.5391836851215642, '850_950K': 0.5108830140522997, '950_1050K': 0.6180860871344048}
- reliability_level median mae_log10: {'high': 0.7262734923561984, 'low': 0.44290462663862173, 'medium': 0.6252551233186301}
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
