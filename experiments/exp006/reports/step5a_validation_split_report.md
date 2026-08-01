# Step5A Validation Split Report

## Summary

- input_file: experiments\exp006\data\processed\step4_sigma0_binned_input_rows.parquet
- input_rows: 97086
- validation rows used: 97086
- dropped rows: 0
- test_size: 0.2
- n_folds: 5
- seed: 20260618
- sample_holdout row counts: {'train': 78078, 'test': 19008}
- sample_holdout sample counts: {'test': 3191, 'train': 12822}
- paper_holdout row counts: {'train': 76932, 'test': 20154}
- paper_holdout paper counts: {'test': 866, 'train': 3496}
- sample_cv_fold row counts: {0: 20269, 1: 20227, 2: 18784, 3: 19084, 4: 18722}
- paper_cv_fold row counts: {0: 19406, 1: 19392, 2: 19464, 3: 19151, 4: 19673}
- default coverage sample_holdout/material_family/conservative_ref/all_test/sample_median: 0.9995791245791246
- default coverage sample_holdout/global/conservative_ref/all_test/sample_median: 0.9995791245791246
- default coverage paper_holdout/material_family/conservative_ref/all_test/sample_median: 1.0
- default coverage paper_holdout/global/conservative_ref/all_test/sample_median: 1.0
- uncovered default example rows: 8
- elapsed_seconds: 37.31

## Parquet Status

- step5a_validation_rows_with_splits.parquet: saved

## Split Summary

| split_scheme | split_label | row_count | sample_count | paper_count | p_row_count | n_row_count | conservative_row_count | material_family_count | T_min_K | T_max_K | sigma_median_S_per_m | log10_sigma_median_S_per_m | sigma0_median_S_per_m | log10_sigma0_median_S_per_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout | test | 19008 | 3191 | 2202 | 10171 | 8837 | 17737 | 1 | 0.1536145 | 1824.224 | 73873.8486050544 | 4.868490724982383 | 20075.751570714052 | 4.302671812775934 |
| sample_holdout | train | 78078 | 12822 | 4129 | 42522 | 35556 | 73397 | 1 | 0.01 | 1663.82 | 73783.20878382068 | 4.8679575384696046 | 19036.022840707796 | 4.279576217184488 |
| paper_holdout | train | 76932 | 12834 | 3496 | 41420 | 35512 | 72418 | 1 | 0.05407739 | 1824.224 | 71705.89000000001 | 4.855554830468989 | 18664.984588768006 | 4.27102763533456 |
| paper_holdout | test | 20154 | 3179 | 866 | 11273 | 8881 | 18716 | 1 | 0.01 | 1470.757 | 81817.305 | 4.9128451672037246 | 21631.512898625475 | 4.335086893330987 |
| sample_cv_fold | 0 | 20269 | 3218 | 2173 | 10724 | 9545 | 19016 | 1 | 0.01 | 1824.224 | 73133.06 | 4.864113745355216 | 17786.186824585686 | 4.250082849795709 |
| sample_cv_fold | 1 | 20227 | 3243 | 2168 | 11293 | 8934 | 19241 | 1 | 0.4370841 | 1663.82 | 76800.45 | 4.885363764718287 | 20293.498310549458 | 4.307356919680339 |
| sample_cv_fold | 2 | 18784 | 3180 | 2188 | 10637 | 8147 | 17497 | 1 | 1.193243 | 1500.862 | 63104.468507986705 | 4.800060112667277 | 17632.682949975744 | 4.246318398613342 |
| sample_cv_fold | 3 | 19084 | 3226 | 2167 | 10268 | 8816 | 17933 | 1 | 0.04092544 | 1470.206 | 81585.75238617737 | 4.911614322643501 | 20989.281994680736 | 4.321997580466553 |
| sample_cv_fold | 4 | 18722 | 3146 | 2118 | 9771 | 8951 | 17447 | 1 | 0.4678308 | 1399.057 | 74453.62940938585 | 4.871885851929269 | 19293.708420543604 | 4.2854157108871025 |
| paper_cv_fold | 0 | 19406 | 3152 | 862 | 10934 | 8472 | 18081 | 1 | 0.01 | 1824.224 | 72002.84566537714 | 4.857349660428898 | 17452.94793738625 | 4.24186878186325 |
| paper_cv_fold | 1 | 19392 | 3392 | 916 | 11778 | 7614 | 18420 | 1 | 0.8800869 | 1278.161 | 64826.39920045637 | 4.811751899055407 | 19375.84641190937 | 4.2872606811976475 |
| paper_cv_fold | 2 | 19464 | 3099 | 858 | 9477 | 9987 | 18256 | 1 | 0.126506 | 1336.666 | 85057.10650806816 | 4.929710603915407 | 22587.301262469056 | 4.353864343979785 |
| paper_cv_fold | 3 | 19151 | 3042 | 850 | 10700 | 8451 | 18105 | 1 | 0.3316769 | 1313.809 | 73909.3929969372 | 4.868699635519462 | 17677.632508668547 | 4.2474241013292815 |
| paper_cv_fold | 4 | 19673 | 3328 | 876 | 9804 | 9869 | 18272 | 1 | 0.04092544 | 1426.798 | 74158.79 | 4.870162635098008 | 19016.65137304035 | 4.279134044765148 |

## Coverage Preflight

| split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | min_rows_per_bin | min_samples_per_bin | min_papers_per_bin | train_rows | train_samples | train_papers | test_rows | test_samples | test_papers | train_reference_keys_total | train_reference_keys_reliable | test_rows_with_reference | test_rows_without_reference | coverage_fraction | p_test_rows | n_test_rows | p_test_rows_with_reference | n_test_rows_with_reference | material_family_count_in_test | T_bin_count_in_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout | all_valid | all_valid | global | row_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | all_valid | all_valid | global | sample_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | all_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | all_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | all_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | all_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | all_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | all_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 78078 | 12822 | 4129 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | conservative_valid | all_valid | global | row_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | conservative_valid | all_valid | global | sample_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | conservative_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | conservative_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 19008 | 3191 | 2202 | 33 | 31 | 19000 | 8 | 0.9995791245791246 | 10171 | 8837 | 10163 | 8837 | 1 | 19 |
| sample_holdout | conservative_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | conservative_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 73397 | 12340 | 4043 | 17737 | 3073 | 2142 | 33 | 31 | 17737 | 0 | 1.0 | 9474 | 8263 | 9474 | 8263 | 1 | 14 |
| paper_holdout | all_valid | all_valid | global | row_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 20154 | 3179 | 866 | 34 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | all_valid | all_valid | global | sample_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 20154 | 3179 | 866 | 34 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | all_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 20154 | 3179 | 866 | 34 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | all_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 20154 | 3179 | 866 | 34 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | all_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 18716 | 3049 | 850 | 34 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | all_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 18716 | 3049 | 850 | 34 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | all_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 18716 | 3049 | 850 | 34 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | all_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 76932 | 12834 | 3496 | 18716 | 3049 | 850 | 34 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | conservative_valid | all_valid | global | row_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 20154 | 3179 | 866 | 33 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | conservative_valid | all_valid | global | sample_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 20154 | 3179 | 866 | 33 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | conservative_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 20154 | 3179 | 866 | 33 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | conservative_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 20154 | 3179 | 866 | 33 | 31 | 20154 | 0 | 1.0 | 11273 | 8881 | 11273 | 8881 | 1 | 16 |
| paper_holdout | conservative_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 18716 | 3049 | 850 | 33 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | conservative_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 18716 | 3049 | 850 | 33 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 18716 | 3049 | 850 | 33 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |
| paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 72418 | 12364 | 3427 | 18716 | 3049 | 850 | 33 | 31 | 18716 | 0 | 1.0 | 10516 | 8200 | 10516 | 8200 | 1 | 16 |

## Sanity Check

- input_rows_equal_used_plus_dropped: True
- row_id_unique: True
- validation_sample_group_id_not_missing: True
- validation_paper_group_id_not_missing: True
- sample_holdout_split_allowed: True
- paper_holdout_split_allowed: True
- sample_group_no_holdout_leak: True
- paper_group_no_holdout_leak: True
- sample_cv_fold_range: True
- paper_cv_fold_range: True
- sample_group_single_cv_fold: True
- paper_group_single_cv_fold: True
- T_inside_bins: True
- carrier_type_p_or_n_only: True
- positive_finite_values: True
- coverage_fraction_range: True
- coverage_counts_consistent: True
- sample_holdout_train_test_nonzero: True
- paper_holdout_train_test_nonzero: True
- coverage_preflight_nonempty: True

## Warnings And Step5B Notes

- WARNING: none
- Step5B should build sigma0_ref(T) using train rows only.
- Step5B should apply sigma_pred = sigma0_ref(T) * F0_eta to test rows only.
- Evaluate errors with log10(sigma_pred / sigma_exp).
