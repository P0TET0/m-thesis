# Phase 6 oxide_v1 re-prediction

## Fixed comparison design

- Changed factor: broad::oxide material keys were replaced by the frozen composition-only oxide_v1 keys.
- Fixed factors: sample holdout assignments, 100 K bins, conservative-valid training references, all-valid test target, sample-median reference aggregation, and 3 rows / 3 samples / 1 paper bin thresholds.
- Cohort policy: review and exclude labels were retained as separate predictive groups so the classification change is the only experimental intervention.
- Prediction mechanism: train-only median sigma0 reference curves; no test-row sigma0 value was used to make a prediction.

## Main result

- Phase 6 coverage: 18,890/19,008 rows (99.379%).
- Paired common-row MAE: 0.721707 -> 0.701151 log10.
- Paired common-row RMSE: 1.247269 -> 1.202613 log10.
- Paired common-row factor-2 accuracy: 42.684% -> 42.525%.
- Paired common-row factor-10 accuracy: 78.454% -> 79.031%.

These paired metrics isolate the prediction change from differences in coverage. Available-as-is metrics remain necessary for deployment-like interpretation.

## Inventory

| metric | value | unit | definition |
| --- | --- | --- | --- |
| source_rows | 97086 | rows | Frozen Step6A broad-family validation rows |
| source_samples | 16013 | samples | All train and test samples |
| phase0_material_groups | 16 | groups | Original broad-family keys |
| phase6_material_groups | 29 | groups | Non-oxide keys unchanged; broad::oxide replaced by oxide_v1 |
| oxide_rows_reclassified | 17715 | rows | Train and test broad::oxide rows |
| oxide_samples_reclassified | 3072 | samples | Train and test broad::oxide samples |
| oxide_v1_nonempty_groups | 14 | groups | Composition-only oxide_v1 groups |
| new_reference_bins | 548 | bins | All phase6 material/carrier/100 K bins |
| test_rows | 19008 | rows | Frozen sample-holdout test cohort |
| baseline_prediction_ok_rows | 18968 | rows | broad_family_100K |
| phase6_prediction_ok_rows | 18890 | rows | oxide_v1_100K |
| paired_prediction_rows | 18890 | rows | Predicted by both variants |
| paired_oxide_rows | 3359 | rows | Oxide rows predicted by both variants |
| paired_oxide_rows_with_changed_prediction | 3210 | rows | Absolute log10 prediction delta > 1e-12 |
| taxonomy_version | oxide_v1 | text | Frozen Phase 4 taxonomy |
| taxonomy_rules_sha256 | 185239a6fa68e0403c1984b9e6338dd5eec4842b116cbab6201d33e2fc706999 | sha256 | Hash of ordered oxide_v1 rules |

## Available-as-is overall metrics

| prediction_variant | comparison_basis | metric_weighting | n_rows | n_source_rows | n_samples | n_papers | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | max_abs_log10_error | overprediction_fraction | underprediction_fraction | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | test_rows | prediction_ok_rows | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_family_100K | available_as_is | row_equal | 18968 | 18968 | 3189 | 2200 | 0.24535917400915397 | 0.0030867094458137 | 0.7237456971201833 | 1.2508679717579398 | 14.570733133464506 | 0.5028996204133277 | 0.4971003795866723 | 0.42666596372838467 | 0.6865773935048503 | 0.7837937579080557 | 19008 | 18968 | 0.997895622895623 |
| broad_family_100K | available_as_is | sample_equal | 3189 | 18968 | 3189 | 2200 | 0.25154154324633626 | 0.0113405912103783 | 0.6264331342694931 | 1.1209937015591127 | 14.326888351734798 | 0.5117591721542804 | 0.48824082784571965 | 0.4816556914393227 | 0.7375352775164629 | 0.8256506741925368 | 19008 | 18968 | 0.997895622895623 |
| oxide_v1_100K | available_as_is | row_equal | 18890 | 18890 | 3187 | 2198 | 0.19621750586067482 | -0.007961581779382824 | 0.7011509945747983 | 1.2026127371920246 | 14.570733133464506 | 0.4926945473795659 | 0.5073054526204341 | 0.42525145579671786 | 0.6956061408152462 | 0.7903123345685548 | 19008 | 18890 | 0.9937920875420876 |
| oxide_v1_100K | available_as_is | sample_equal | 3187 | 18890 | 3187 | 2198 | 0.21975745403723093 | 0.005792338581024351 | 0.6036011903779671 | 1.0770501889679707 | 14.3268883517348 | 0.5051772827110135 | 0.4948227172889865 | 0.4854094759962347 | 0.7502353310323188 | 0.8346407279573267 | 19008 | 18890 | 0.9937920875420876 |

## Paired common-row metrics

| prediction_variant | comparison_basis | metric_weighting | n_rows | n_source_rows | n_samples | n_papers | mean_log10_error | median_log10_error | mae_log10 | rmse_log10 | max_abs_log10_error | overprediction_fraction | underprediction_fraction | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | test_rows | prediction_ok_rows | coverage_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad_family_100K | paired_common_rows | row_equal | 18890 | 18890 | 3187 | 2198 | 0.24230755766984777 | 0.0019417561336276 | 0.7217065025367313 | 1.2472691283758959 | 14.570733133464506 | 0.5016410799364743 | 0.49835892006352567 | 0.42683959767072527 | 0.6875066172578084 | 0.7845420857596612 | 19008 | 18890 | 0.9937920875420876 |
| broad_family_100K | paired_common_rows | sample_equal | 3187 | 18890 | 3187 | 2198 | 0.24884435210948944 | 0.0107223901973991 | 0.624015986311522 | 1.115755236406715 | 14.326888351734798 | 0.5111390021964229 | 0.48886099780357706 | 0.4822717288986508 | 0.7379981173517415 | 0.8271101349231252 | 19008 | 18890 | 0.9937920875420876 |
| oxide_v1_100K | paired_common_rows | row_equal | 18890 | 18890 | 3187 | 2198 | 0.19621750586067482 | -0.007961581779382824 | 0.7011509945747983 | 1.2026127371920246 | 14.570733133464506 | 0.4926945473795659 | 0.5073054526204341 | 0.42525145579671786 | 0.6956061408152462 | 0.7903123345685548 | 19008 | 18890 | 0.9937920875420876 |
| oxide_v1_100K | paired_common_rows | sample_equal | 3187 | 18890 | 3187 | 2198 | 0.21975745403723093 | 0.005792338581024351 | 0.6036011903779671 | 1.0770501889679707 | 14.3268883517348 | 0.5051772827110135 | 0.4948227172889865 | 0.4854094759962347 | 0.7502353310323188 | 0.8346407279573267 | 19008 | 18890 | 0.9937920875420876 |

## Metric deltas

| comparison_basis | metric_weighting | metric_name | baseline_value | phase6_value | delta_phase6_minus_baseline | lower_or_higher_is_better |
| --- | --- | --- | --- | --- | --- | --- |
| available_as_is | row_equal | mean_log10_error | 0.24535917400915397 | 0.19621750586067482 | -0.049141668148479145 | context |
| available_as_is | row_equal | median_log10_error | 0.0030867094458137 | -0.007961581779382824 | -0.011048291225196525 | context |
| available_as_is | row_equal | mae_log10 | 0.7237456971201833 | 0.7011509945747983 | -0.02259470254538498 | lower |
| available_as_is | row_equal | rmse_log10 | 1.2508679717579398 | 1.2026127371920246 | -0.048255234565915295 | lower |
| available_as_is | row_equal | factor_2_accuracy | 0.42666596372838467 | 0.42525145579671786 | -0.0014145079316668085 | higher |
| available_as_is | row_equal | factor_5_accuracy | 0.6865773935048503 | 0.6956061408152462 | 0.009028747310395935 | higher |
| available_as_is | row_equal | factor_10_accuracy | 0.7837937579080557 | 0.7903123345685548 | 0.0065185766604990425 | higher |
| available_as_is | row_equal | coverage_fraction | 0.997895622895623 | 0.9937920875420876 | -0.004103535353535359 | higher |
| available_as_is | sample_equal | mean_log10_error | 0.25154154324633626 | 0.21975745403723093 | -0.03178408920910533 | context |
| available_as_is | sample_equal | median_log10_error | 0.0113405912103783 | 0.005792338581024351 | -0.005548252629353949 | context |
| available_as_is | sample_equal | mae_log10 | 0.6264331342694931 | 0.6036011903779671 | -0.022831943891525985 | lower |
| available_as_is | sample_equal | rmse_log10 | 1.1209937015591127 | 1.0770501889679707 | -0.04394351259114204 | lower |
| available_as_is | sample_equal | factor_2_accuracy | 0.4816556914393227 | 0.4854094759962347 | 0.0037537845569119987 | higher |
| available_as_is | sample_equal | factor_5_accuracy | 0.7375352775164629 | 0.7502353310323188 | 0.012700053515855902 | higher |
| available_as_is | sample_equal | factor_10_accuracy | 0.8256506741925368 | 0.8346407279573267 | 0.008990053764789896 | higher |
| available_as_is | sample_equal | coverage_fraction | 0.997895622895623 | 0.9937920875420876 | -0.004103535353535359 | higher |
| paired_common_rows | row_equal | mean_log10_error | 0.24230755766984777 | 0.19621750586067482 | -0.046090051809172944 | context |
| paired_common_rows | row_equal | median_log10_error | 0.0019417561336276 | -0.007961581779382824 | -0.009903337913010424 | context |
| paired_common_rows | row_equal | mae_log10 | 0.7217065025367313 | 0.7011509945747983 | -0.02055550796193295 | lower |
| paired_common_rows | row_equal | rmse_log10 | 1.2472691283758959 | 1.2026127371920246 | -0.044656391183871325 | lower |
| paired_common_rows | row_equal | factor_2_accuracy | 0.42683959767072527 | 0.42525145579671786 | -0.0015881418740074094 | higher |
| paired_common_rows | row_equal | factor_5_accuracy | 0.6875066172578084 | 0.6956061408152462 | 0.008099523557437815 | higher |
| paired_common_rows | row_equal | factor_10_accuracy | 0.7845420857596612 | 0.7903123345685548 | 0.005770248808893541 | higher |
| paired_common_rows | row_equal | coverage_fraction | 0.9937920875420876 | 0.9937920875420876 | 0.0 | higher |
| paired_common_rows | sample_equal | mean_log10_error | 0.24884435210948944 | 0.21975745403723093 | -0.029086898072258505 | context |
| paired_common_rows | sample_equal | median_log10_error | 0.0107223901973991 | 0.005792338581024351 | -0.004930051616374749 | context |
| paired_common_rows | sample_equal | mae_log10 | 0.624015986311522 | 0.6036011903779671 | -0.020414795933554886 | lower |
| paired_common_rows | sample_equal | rmse_log10 | 1.115755236406715 | 1.0770501889679707 | -0.038705047438744256 | lower |
| paired_common_rows | sample_equal | factor_2_accuracy | 0.4822717288986508 | 0.4854094759962347 | 0.0031377470975839095 | higher |
| paired_common_rows | sample_equal | factor_5_accuracy | 0.7379981173517415 | 0.7502353310323188 | 0.012237213680577286 | higher |
| paired_common_rows | sample_equal | factor_10_accuracy | 0.8271101349231252 | 0.8346407279573267 | 0.007530593034201449 | higher |
| paired_common_rows | sample_equal | coverage_fraction | 0.9937920875420876 | 0.9937920875420876 | 0.0 | higher |

## Oxide-group coverage

| oxide_v1_group_key | oxide_v1_display_name_en | oxide_v1_display_name_ja | test_rows | n_samples | n_papers | default_eligible_rows | review_rows | excluded_rows | baseline_prediction_ok_rows | phase6_prediction_ok_rows | paired_prediction_rows | baseline_coverage_fraction | phase6_coverage_fraction | coverage_delta_phase6_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oxide_v1::Co_containing_oxide | Co-containing oxide | Co含有酸化物系 | 1248 | 226 | 162 | 1215 | 33 | 0 | 1248 | 1242 | 1242 | 1.0 | 0.9951923076923077 | -0.004807692307692308 |
| oxide_v1::Mn_containing_oxide | Mn-containing oxide | Mn含有酸化物系 | 517 | 92 | 65 | 489 | 28 | 0 | 517 | 517 | 517 | 1.0 | 1.0 | 0.0 |
| oxide_v1::Zn_containing_oxide | Zn-containing oxide | Zn含有酸化物系 | 284 | 56 | 40 | 272 | 12 | 0 | 284 | 280 | 280 | 1.0 | 0.9859154929577465 | -0.014084507042253521 |
| oxide_v1::Ti_containing_oxide | Ti-containing oxide | Ti含有酸化物系 | 225 | 64 | 55 | 216 | 9 | 0 | 225 | 219 | 219 | 1.0 | 0.9733333333333334 | -0.02666666666666667 |
| oxide_v1::Cu_containing_oxide | Cu-containing oxide | Cu含有酸化物系 | 175 | 24 | 15 | 173 | 2 | 0 | 175 | 172 | 172 | 1.0 | 0.9828571428571429 | -0.017142857142857144 |
| oxide_v1::Mo_W_V_Nb_containing_oxide | Mo/W/V/Nb-containing oxide | Mo・W・V・Nb含有酸化物系 | 152 | 29 | 20 | 142 | 10 | 0 | 152 | 147 | 147 | 1.0 | 0.9671052631578947 | -0.03289473684210526 |
| oxide_v1::excluded_organic_polymer_or_carbon_composite | Excluded organic/polymer or carbon composite | 除外：有機・高分子・炭素複合系 | 142 | 21 | 15 | 0 | 0 | 142 | 142 | 134 | 134 | 1.0 | 0.9436619718309859 | -0.056338028169014086 |
| oxide_v1::other_inorganic_O_containing | Other inorganic O-containing | その他の無機O含有系 | 140 | 17 | 13 | 140 | 0 | 0 | 140 | 136 | 136 | 1.0 | 0.9714285714285714 | -0.02857142857142857 |
| oxide_v1::mixed_anion_O_S_Se_Te | Mixed-anion O + S/Se/Te | 混合アニオン：O＋S・Se・Te | 140 | 36 | 28 | 127 | 13 | 0 | 140 | 140 | 140 | 1.0 | 1.0 | 0.0 |
| oxide_v1::In_Sn_Cd_containing_oxide | In/Sn/Cd-containing oxide | In・Sn・Cd含有酸化物系 | 128 | 19 | 15 | 115 | 13 | 0 | 128 | 128 | 128 | 1.0 | 1.0 | 0.0 |
| oxide_v1::Co_Mn_mixed_oxide | Co-Mn mixed oxide | Co-Mn混合酸化物系 | 104 | 9 | 7 | 59 | 45 | 0 | 104 | 79 | 79 | 1.0 | 0.7596153846153846 | -0.2403846153846154 |
| oxide_v1::mixed_anion_O_N_or_halogen | Mixed-anion O + N/halogen | 混合アニオン：O＋N・ハロゲン | 98 | 8 | 6 | 0 | 98 | 0 | 98 | 95 | 95 | 1.0 | 0.9693877551020408 | -0.030612244897959183 |
| oxide_v1::mixed_anion_O_As_P | Mixed-anion O + As/P | 混合アニオン：O＋As・P | 64 | 6 | 4 | 64 | 0 | 0 | 64 | 51 | 51 | 1.0 | 0.796875 | -0.203125 |
| oxide_v1::Fe_Ni_Cr_containing_oxide | Fe/Ni/Cr-containing oxide | Fe・Ni・Cr含有酸化物系 | 20 | 3 | 3 | 20 | 0 | 0 | 20 | 19 | 19 | 1.0 | 0.95 | -0.05 |

## Validation

| check | status | detail |
| --- | --- | --- |
| source_row_count_frozen | PASS | rows=97086 |
| test_row_count_frozen | PASS | rows=19008 |
| oxide_row_count_all_splits | PASS | rows=17715 |
| all_oxide_rows_assigned | PASS | missing=0 |
| nonoxide_keys_unchanged | PASS | rows=79371 |
| broad_oxide_key_removed | PASS | remaining=0 |
| taxonomy_version_frozen | PASS | 185239a6fa68e0403c1984b9e6338dd5eec4842b116cbab6201d33e2fc706999 |
| heldout_assignments_match_phase4 | PASS | phase6=3437; phase4=3437 |
| prediction_rows_complete | PASS | rows=19008 |
| reference_uses_train_only_keys | PASS | bins=548 |
| prediction_status_allowed | PASS | {'ok': 18890, 'unreliable_reference_bin': 70, 'missing_reference_bin': 48} |
| new_predictions_positive | PASS | ok_rows=18890 |
| prediction_formula_recomputed | PASS | ok_rows=18890 |
| sample_holdout_no_leakage | PASS | train versus test validation_sample_group_id |
| nonoxide_predictions_unchanged | PASS | paired_rows=15531; max_abs_delta=1.157e-15 |
| baseline_metrics_reproduce_phase0 | PASS | mae_diff=0.000e+00 |
| overall_metrics_complete | PASS | rows=4 |

## Interpretation limit

The experiment measures the effect of the oxide_v1 grouping under the frozen Phase 0 reference-curve protocol. It does not test a new regression algorithm, new descriptors, 25 K bins, or manual crystal-structure labels.
