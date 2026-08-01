# Step5C Evaluation Metrics Report

## Summary

- input_file: experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_test_predictions_valid_test.csv
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
- metrics_by_temperature_bin rows: 1528
- metrics_by_eta_bin rows: 336
- metrics_by_reliability_level rows: 160
- elapsed_seconds: 29.84

## Parquet Status

- step5c_metrics_by_config_test.parquet: saved

## Default Comparison

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | metric_weighting | n_rows | n_samples | n_papers | n_material_families | n_T_bins | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | std_log10_error | q05_log10_error | q25_log10_error | q75_log10_error | q95_log10_error | max_abs_log10_error | overprediction_fraction | underprediction_fraction | near_exact_fraction | factor_2_accuracy | factor_3_accuracy | factor_5_accuracy | factor_10_accuracy | median_abs_factor_error | mean_abs_factor_error_equiv | sigma_exp_median_S_per_m | sigma_pred_median_S_per_m | eta_median | S_abs_median_uV_per_K | T_min_K | T_max_K | train_sample_count_median | train_paper_count_median | is_reliable_eval_group | eval_group_reliability | test_rows | prediction_ok_rows | prediction_unavailable_rows | coverage_fraction | reference_bins_total | reference_bins_reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 47 | 34 | 14 | 31 | 0.5192837960091681 | -0.028164573706851402 | 0.8504383009182984 | 2.335262876881425 | 2.282508767719502 | -0.6147146760412616 | -0.265640199929524 | 0.3675576429351691 | 1.4537149049690576 | 11.102595848486564 | 0.46 | 0.54 | 0.115 | 0.49 | 0.665 | 0.815 | 0.88 | 2.0391172109323965 | 7.086606217374111 | 78197.49200852399 | 66490.78411445825 | 2.6236894252893483 | 94.810225 | 10.6618 | 890.3596 | 73.0 | 36.0 | True | high | 929 | 924 | 5 | 0.9946178686759956 | 81 | 77 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 56 | 42 | 12 | 26 | 0.33239879439788966 | -0.05743633981043795 | 0.862404009168269 | 2.410442933589117 | 2.393405130387113 | -0.9138797219618242 | -0.31822979487204583 | 0.08992649325129895 | 2.337149320888808 | 11.139285636309875 | 0.405 | 0.595 | 0.2 | 0.57 | 0.76 | 0.855 | 0.89 | 1.674190792449965 | 7.284571479778244 | 87727.1656236919 | 65136.80859503427 | 2.2644752103752577 | 104.41155 | 10.6618 | 868.3698 | 10.0 | 6.0 | True | high | 929 | 794 | 135 | 0.8546824542518837 | 520 | 297 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | row_equal | 200 | 48 | 15 | 9 | 23 | 0.042457778194235594 | 0.0173994396943514 | 0.35234593854593965 | 0.47527866690095416 | 0.4745663434691992 | -0.6921299779010462 | -0.24550770773523772 | 0.258427064516079 | 0.9033388254499775 | 1.6672357895196723 | 0.515 | 0.485 | 0.08 | 0.6 | 0.72 | 0.86 | 0.965 | 1.8093883118775977 | 2.2508468105901893 | 47689.8216305049 | 50895.71069694141 | 1.7690003747879834 | 120.05675 | 80.7021 | 704.172 | 74.0 | 35.0 | True | high | 1205 | 1205 | 0 | 1.0 | 82 | 79 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | row_equal | 200 | 59 | 18 | 10 | 14 | 0.1314112807239049 | 0.0134310844226124 | 0.39231717140819156 | 0.5971306213729577 | 0.5839529600726807 | -0.6425353104443774 | -0.15873668629055548 | 0.2876343482784865 | 1.486711840433874 | 2.2689398996244448 | 0.515 | 0.485 | 0.155 | 0.61 | 0.74 | 0.835 | 0.88 | 1.7245042206440546 | 2.467840978474599 | 54743.480979246044 | 73938.78156732989 | 1.897765580922678 | 115.69945 | 123.3824 | 581.9643 | 13.0 | 5.0 | True | high | 1205 | 752 | 453 | 0.6240663900414938 | 513 | 280 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 47 | 47 | 34 | 14 | 24 | 0.24052903620728 | 0.00146569332859595 | 0.5992661365312646 | 1.6363821807935401 | 1.6361071503719051 | -0.6055760296198602 | -0.3454031370607102 | 0.289346287069736 | 1.0561963041278104 | 10.735700448572455 | 0.5106382978723404 | 0.48936170212765956 | 0.0851063829787234 | 0.44680851063829785 | 0.6595744680851063 | 0.851063829787234 | 0.9361702127659575 | 2.27333205437903 | 3.97435023842321 | 74445.1971681047 | 60231.83202363165 | 2.1964971133124562 | 106.9822 | 37.22425 | 637.607 | 71.5 | 36.0 | True | high | 929 | 924 | 5 | 0.9946178686759956 | 81 | 77 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 56 | 56 | 42 | 12 | 22 | 0.05563833938241791 | -0.03710465211677245 | 0.7163658841691845 | 1.7356177681877571 | 1.7504249411169217 | -1.4722784310680135 | -0.32933559859588224 | 0.057150068827611374 | 1.4688516071944158 | 10.948960968235674 | 0.4642857142857143 | 0.5357142857142857 | 0.21428571428571427 | 0.5357142857142857 | 0.6964285714285714 | 0.7857142857142857 | 0.8571428571428571 | 1.7149197657782889 | 5.204342670572443 | 95997.88470255399 | 74300.39819229537 | 2.3790700874845636 | 101.31162499999999 | 10.6618 | 625.8185 | 10.0 | 6.0 | True | high | 929 | 794 | 135 | 0.8546824542518837 | 520 | 297 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | sample_equal | 48 | 48 | 15 | 9 | 20 | 0.10172132488894432 | 0.04538866433980612 | 0.4044505250493618 | 0.5304348435369758 | 0.5260989910581877 | -0.5606314975055363 | -0.22959092302521045 | 0.2892469153801725 | 1.126154913323542 | 1.650020026652277 | 0.5208333333333334 | 0.4791666666666667 | 0.020833333333333332 | 0.5208333333333334 | 0.6666666666666666 | 0.8125 | 0.9166666666666666 | 1.9631893993616483 | 2.53775986717152 | 40849.349778946766 | 48845.62817664709 | 1.7507808819400688 | 120.842175 | 80.7021 | 562.1384 | 73.0 | 36.25 | True | high | 1205 | 1205 | 0 | 1.0 | 82 | 79 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | sample_equal | 59 | 59 | 18 | 10 | 18 | 0.11858413652345813 | -0.0051518795800391 | 0.39133229800177266 | 0.6381912233290327 | 0.6324599560153767 | -0.7063328748317178 | -0.12836155734850196 | 0.25783665004708023 | 1.4055027067131027 | 2.2689398996244448 | 0.4915254237288136 | 0.5084745762711864 | 0.1694915254237288 | 0.6779661016949152 | 0.7457627118644068 | 0.8305084745762712 | 0.847457627118644 | 1.5059507051496006 | 2.462250861094061 | 67547.21 | 88222.96918148913 | 1.9062095307535845 | 115.4207 | 278.02 | 560.6567 | 13.0 | 5.0 | True | high | 1205 | 752 | 453 | 0.6240663900414938 | 513 | 280 |

## Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.862404009168269 | 2.410442933589117 | -0.05743633981043795 | 0.57 | 0.855 | 0.89 | 0.8546824542518837 |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.7163658841691845 | 1.7356177681877571 | -0.03710465211677245 | 0.5357142857142857 | 0.7857142857142857 | 0.8571428571428571 | 0.8546824542518837 |

## Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8504383009182984 | 2.335262876881425 | -0.028164573706851402 | 0.49 | 0.815 | 0.88 | 0.9946178686759956 |
| sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.5992661365312646 | 1.6363821807935401 | 0.00146569332859595 | 0.44680851063829785 | 0.851063829787234 | 0.9361702127659575 | 0.9946178686759956 |

## Paper Material Family Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.39231717140819156 | 0.5971306213729577 | 0.0134310844226124 | 0.61 | 0.835 | 0.88 | 0.6240663900414938 |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.39133229800177266 | 0.6381912233290327 | -0.0051518795800391 | 0.6779661016949152 | 0.8305084745762712 | 0.847457627118644 | 0.6240663900414938 |

## Paper Global Default

| config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.35234593854593965 | 0.47527866690095416 | 0.0173994396943514 | 0.6 | 0.86 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.4044505250493618 | 0.5304348435369758 | 0.04538866433980612 | 0.5208333333333334 | 0.8125 | 0.9166666666666666 | 1.0 |

## Best Configs By MAE

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.35234593854593965 | 0.47527866690095416 | 0.6 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.35234593854593965 | 0.47527866690095416 | 0.6 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.3546852055338449 | 0.4770248120796769 | 0.59 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3546852055338449 | 0.4770248120796769 | 0.59 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | 0.3592159469901607 | 0.5357604539106338 | 0.58 | 0.96 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.3592159469901607 | 0.5357604539106338 | 0.58 | 0.96 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | 0.3596511879698251 | 0.536324649656993 | 0.575 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.3596511879698251 | 0.536324649656993 | 0.575 | 0.965 | 1.0 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3818660057053949 | 0.5269349491486712 | 0.515 | 0.92 | 0.9942857142857144 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.38647535152683204 | 0.5375410148965049 | 0.52 | 0.92 | 0.9942857142857144 |

## Best Configs By Factor 2 Accuracy

| config_id | mae_log10 | rmse_log10 | factor_2_accuracy | factor_10_accuracy | coverage_fraction |
| --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | 0.35234593854593965 | 0.47527866690095416 | 0.6 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.35234593854593965 | 0.47527866690095416 | 0.6 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | 0.3546852055338449 | 0.4770248120796769 | 0.59 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3546852055338449 | 0.4770248120796769 | 0.59 | 0.965 | 1.0 |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | 0.3592159469901607 | 0.5357604539106338 | 0.58 | 0.96 | 1.0 |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | 0.3592159469901607 | 0.5357604539106338 | 0.58 | 0.96 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | 0.3596511879698251 | 0.536324649656993 | 0.575 | 0.965 | 1.0 |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | 0.3596511879698251 | 0.536324649656993 | 0.575 | 0.965 | 1.0 |
| sample_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | 0.38647535152683204 | 0.5375410148965049 | 0.52 | 0.92 | 0.9942857142857144 |
| sample_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | 0.3818660057053949 | 0.5269349491486712 | 0.515 | 0.92 | 0.9942857142857144 |

## Comparison Notes

- split_scheme median mae_log10: {'paper_holdout': 0.3983838482287767, 'sample_holdout': 0.5827267307593487}
- group_scheme median mae_log10: {'global': 0.4259092636983577, 'material_family': 0.4261716801390153}
- curve_method median mae_log10: {'row_median': 0.4588842342536883, 'sample_median': 0.4033342392302339}
- reference_source_subset median mae_log10: {'all_valid': 0.42973073662960054, 'conservative_valid': 0.42434220715115223}
- p/n median mae_log10: {'n': 0.4314848191952009, 'p': 0.43906602725936017}
- eta bin median mae_log10: {'[1, 2)': 0.30236131035209746, '[10, 20)': 1.3209127356030868, '[2, 5)': 0.3926776973327497, '[20, 50)': 1.6434086755925086, '[5, 10)': 0.5746260473792351, '[50, inf)': 0.7303706546247463}
- temperature bin median mae_log10: {'-12.5_12.5K': 2.419871798587294, '112.5_137.5K': 1.1916091652773162, '137.5_162.5K': 0.3802809540319674, '162.5_187.5K': 0.22340737677800898, '212.5_237.5K': 3.003105750430767, '237.5_262.5K': 3.082460561597928, '262.5_287.5K': 1.3620519079740219, '287.5_312.5K': 0.5034774611982828, '312.5_337.5K': 0.43101081299518484, '337.5_362.5K': 0.3924561094287907, '362.5_387.5K': 0.32798414923531916, '37.5_62.5K': 0.49117504111452215, '387.5_412.5K': 0.2700610225304575, '412.5_437.5K': 0.27511396986403064, '437.5_462.5K': 0.24264313277601218, '462.5_487.5K': 0.3795888254942593, '487.5_512.5K': 0.18184960812622616, '512.5_537.5K': 0.28844367798768117, '537.5_562.5K': 0.2502490246433607, '562.5_587.5K': 0.2925056097414719, '587.5_612.5K': 0.2621460130409164, '612.5_637.5K': 0.35799094763483297, '62.5_87.5K': 0.5357921097180021, '637.5_662.5K': 0.1911763039416824, '662.5_687.5K': 0.40099521536643856, '687.5_712.5K': 0.3529276479563916, '712.5_737.5K': 0.33134106155613957, '737.5_762.5K': 0.17573832813266937, '762.5_787.5K': 0.22317031983651997, '787.5_812.5K': 0.5161896441725179, '862.5_887.5K': 0.0315742592521655, '87.5_112.5K': 0.7203017570350045, '887.5_912.5K': 0.1591309660023869}
- reliability_level median mae_log10: {'high': 0.4135365825622818, 'low': 0.4291029597593781, 'medium': 0.4520427364249283}
- largest abs_log10 error: 11.139285636309875

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
