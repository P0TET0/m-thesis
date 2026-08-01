# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step5b_test_predictions_valid.parquet
- input_rows: 6400
- evaluated_rows: 6400
- dropped_rows: 0
- config_count: 32
- metric_weighting: ['row_equal', 'sample_equal']
- min_eval_rows: 30
- min_eval_samples: 5
- metrics_by_config rows: 64
- metrics_by_carrier_type rows: 128
- metrics_by_material_family rows: 64
- metrics_by_temperature_bin rows: 544
- metrics_by_eta_bin rows: 352
- metrics_by_reliability_level rows: 64
- elapsed_seconds: 19.71

## Parquet Status

- step5c_metrics_by_config_test.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 47 | 34 | 1 | 10 | 0.5649084558537046 | 0.0012140057453916833 | 0.8282715832167409 | 2.3193671838137906 | 2.255165498803194 | -0.570785254633439 | -0.22305438399633815 | 0.45887824850884557 | 1.297441485403862 | 11.189345236835823 | 0.5 | 0.5 | 0.135 | 0.48 | 0.665 | 0.835 | 0.895 | 2.084729300801302 | 6.733976294520393 | 76277.50118503075 | 69884.07724393213 | 2.623689425289349 | 94.810225 | 10.6618 | 913.9761 | 2572.0 | 1123.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 47 | 34 | 1 | 10 | 0.5649084558537046 | 0.0012140057453916833 | 0.8282715832167409 | 2.3193671838137906 | 2.255165498803194 | -0.570785254633439 | -0.22305438399633815 | 0.45887824850884557 | 1.297441485403862 | 11.189345236835823 | 0.5 | 0.5 | 0.135 | 0.48 | 0.665 | 0.835 | 0.895 | 2.084729300801302 | 6.733976294520393 | 76277.50118503075 | 69884.07724393213 | 2.623689425289349 | 94.810225 | 10.6618 | 913.9761 | 2572.0 | 1123.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 48 | 15 | 1 | 7 | 0.06601365185125399 | 0.026322675980983583 | 0.3401719656045584 | 0.45487997083537224 | 0.45119382194491187 | -0.5607187924436206 | -0.24318140939251293 | 0.2772629005839424 | 0.8405748148873212 | 1.73540118494079 | 0.515 | 0.485 | 0.11 | 0.56 | 0.75 | 0.895 | 0.97 | 1.8662863739187714 | 2.1886280734689763 | 47689.8216305049 | 50423.17580996275 | 1.7690003747879834 | 120.05675 | 80.7021 | 704.172 | 3168.0 | 1158.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 48 | 15 | 1 | 7 | 0.06601365185125399 | 0.026322675980983583 | 0.3401719656045584 | 0.45487997083537224 | 0.45119382194491187 | -0.5607187924436206 | -0.24318140939251293 | 0.2772629005839424 | 0.8405748148873212 | 1.73540118494079 | 0.515 | 0.485 | 0.11 | 0.56 | 0.75 | 0.895 | 0.97 | 1.8662863739187714 | 2.1886280734689763 | 47689.8216305049 | 50423.17580996275 | 1.7690003747879834 | 120.05675 | 80.7021 | 704.172 | 3168.0 | 1158.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 47 | 47 | 34 | 1 | 11 | 0.28388857616079816 | 0.020390250618212192 | 0.6114612798232293 | 1.6301186408178525 | 1.6225625250561304 | -0.6076139234897492 | -0.31711802131292494 | 0.38537599861305627 | 1.0478218559092045 | 10.675499723656374 | 0.5319148936170213 | 0.46808510638297873 | 0.06382978723404255 | 0.425531914893617 | 0.6382978723404256 | 0.8297872340425532 | 0.9361702127659575 | 2.167803902046245 | 4.087533074714999 | 74445.1971681047 | 66454.92543064094 | 2.1964971133124562 | 106.9822 | 37.22425 | 637.607 | 2572.0 | 1123.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 47 | 47 | 34 | 1 | 11 | 0.28388857616079816 | 0.020390250618212192 | 0.6114612798232293 | 1.6301186408178525 | 1.6225625250561304 | -0.6076139234897492 | -0.31711802131292494 | 0.38537599861305627 | 1.0478218559092045 | 10.675499723656374 | 0.5319148936170213 | 0.46808510638297873 | 0.06382978723404255 | 0.425531914893617 | 0.6382978723404256 | 0.8297872340425532 | 0.9361702127659575 | 2.167803902046245 | 4.087533074714999 | 74445.1971681047 | 66454.92543064094 | 2.1964971133124562 | 106.9822 | 37.22425 | 637.607 | 2572.0 | 1123.0 | True | high | 19008 | 19000 | 8 | 0.9995791245791246 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 48 | 48 | 15 | 1 | 8 | 0.09437529384298993 | 0.01880691832056626 | 0.386261121781071 | 0.49792240874592103 | 0.4940704014371104 | -0.5151197660245205 | -0.28872434196973024 | 0.3176060385313497 | 1.030834882682193 | 1.7108761107391957 | 0.5 | 0.5 | 0.020833333333333332 | 0.4583333333333333 | 0.7291666666666666 | 0.875 | 0.9375 | 2.0361016138862 | 2.4336668239068797 | 40849.349778946766 | 43744.49674162835 | 1.7507808819400688 | 120.84217500000001 | 80.7021 | 562.1384 | 2576.0 | 997.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 48 | 48 | 15 | 1 | 8 | 0.09437529384298993 | 0.01880691832056626 | 0.386261121781071 | 0.49792240874592103 | 0.4940704014371104 | -0.5151197660245205 | -0.28872434196973024 | 0.3176060385313497 | 1.030834882682193 | 1.7108761107391957 | 0.5 | 0.5 | 0.020833333333333332 | 0.4583333333333333 | 0.7291666666666666 | 0.875 | 0.9375 | 2.0361016138862 | 2.4336668239068797 | 40849.349778946766 | 43744.49674162835 | 1.7507808819400688 | 120.84217500000001 | 80.7021 | 562.1384 | 2576.0 | 997.0 | True | high | 20154 | 20154 | 0 | 1.0 | 33 | 31 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.8282715832167409 | 2.3193671838137906 | 0.0012140057453916833 | 0.48 | 0.835 | 0.895 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6114612798232293 | 1.6301186408178525 | 0.020390250618212192 | 0.425531914893617 | 0.8297872340425532 | 0.9361702127659575 | 0.9995791245791246 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8282715832167409 | 2.3193671838137906 | 0.0012140057453916833 | 0.48 | 0.835 | 0.895 | 0.9995791245791246 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.6114612798232293 | 1.6301186408178525 | 0.020390250618212192 | 0.425531914893617 | 0.8297872340425532 | 0.9361702127659575 | 0.9995791245791246 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.3401719656045584 | 0.45487997083537224 | 0.026322675980983583 | 0.56 | 0.895 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.386261121781071 | 0.49792240874592103 | 0.01880691832056626 | 0.4583333333333333 | 0.875 | 0.9375 | 1.0 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.3401719656045584 | 0.45487997083537224 | 0.026322675980983583 | 0.56 | 0.895 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.386261121781071 | 0.49792240874592103 | 0.01880691832056626 | 0.4583333333333333 | 0.875 | 0.9375 | 1.0 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | 0.34810031186324425 | 0.46151725580336533 | 0.54 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | 0.34810031186324425 | 0.46151725580336533 | 0.54 | 0.97 | 1.0 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | 0.33918903033942416 | 0.45116709546876693 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | 0.3401719656045584 | 0.45487997083537224 | 0.56 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.34932043635109833 | 0.45993301741206793 | 0.55 | 0.97 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | 0.34932043635109833 | 0.45993301741206793 | 0.55 | 0.97 | 1.0 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.3677907790660847, 'sample_holdout': 0.519099790683588}
- group_scheme median mae_log10: {'global': 0.38671054330871524, 'material_family': 0.38671054330871524}
- curve_method median mae_log10: {'row_median': 0.38758120055721845, 'sample_median': 0.3862638980301608}
- reference_source_subset median mae_log10: {'all_valid': 0.38713733152775387, 'conservative_valid': 0.38670776705962545}
- p/n median mae_log10: {'n': 0.4146002877602005, 'p': 0.37318154359540145}
- eta bin median mae_log10: {'[1, 2)': 0.3094544036338469, '[10, 20)': 1.728665296636311, '[2, 5)': 0.3847026945493743, '[20, 50)': 1.7533722684669775, '[5, 10)': 0.569002857588865, '[50, inf)': 0.6670519387852087}
- temperature bin median mae_log10: {'-50_50K': 0.33629699264593976, '150_250K': 0.5641037571943557, '250_350K': 0.47110020761413895, '350_450K': 0.33619254948117616, '450_550K': 0.3413141866969728, '50_150K': 0.6491588322862475, '550_650K': 0.3747424955994675, '650_750K': 0.3915863254633869, '750_850K': 0.5247567440604681, '850_950K': 0.22952241980612764}
- reliability_level median mae_log10: {'high': 0.38671054330871524}
- largest abs_log10 error: 11.224423635905708

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
