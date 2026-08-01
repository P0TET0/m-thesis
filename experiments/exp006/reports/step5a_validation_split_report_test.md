# Step5A Validation Split Report

## Summary

- input_file: experiments\exp006\data\processed\step4_sigma0_binned_input_rows.parquet
- input_rows: 5000
- validation rows used: 5000
- dropped rows: 0
- test_size: 0.2
- n_folds: 5
- seed: 20260618
- sample_holdout row counts: {'train': 4071, 'test': 929}
- sample_holdout sample counts: {'test': 158, 'train': 697}
- paper_holdout row counts: {'train': 3795, 'test': 1205}
- paper_holdout paper counts: {'test': 43, 'train': 189}
- sample_cv_fold row counts: {0: 1187, 1: 979, 2: 922, 3: 954, 4: 958}
- paper_cv_fold row counts: {0: 863, 1: 701, 2: 1039, 3: 851, 4: 1546}
- default coverage sample_holdout/material_family/conservative_ref/all_test/sample_median: 0.9989235737351991
- default coverage sample_holdout/global/conservative_ref/all_test/sample_median: 0.9989235737351991
- default coverage paper_holdout/material_family/conservative_ref/all_test/sample_median: 1.0
- default coverage paper_holdout/global/conservative_ref/all_test/sample_median: 1.0
- uncovered default example rows: 1
- elapsed_seconds: 2.60

## Parquet Status

- step5a_validation_rows_with_splits_test.parquet: saved

## Split Summary

| split_scheme | split_label | row_count | sample_count | paper_count | p_row_count | n_row_count | conservative_row_count | material_family_count | T_min_K | T_max_K | sigma_median_S_per_m | log10_sigma_median_S_per_m | sigma0_median_S_per_m | log10_sigma0_median_S_per_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout | test | 929 | 158 | 118 | 565 | 364 | 875 | 1 | 0.1536145 | 1105.581 | 96811.41 | 4.985926545407734 | 29192.832313358726 | 4.46527623265126 |
| sample_holdout | train | 4071 | 697 | 211 | 2213 | 1858 | 3830 | 1 | 0.05407739 | 1233.419 | 77506.11 | 4.889335940373266 | 21105.489872255555 | 4.3243954368670385 |
| paper_holdout | train | 3795 | 688 | 189 | 2162 | 1633 | 3512 | 1 | 0.05407739 | 1233.419 | 73025.4 | 4.8634739445201784 | 20834.226518227762 | 4.318777381678296 |
| paper_holdout | test | 1205 | 167 | 43 | 616 | 589 | 1193 | 1 | 1.103748 | 973.15 | 126977.2 | 5.103725746128185 | 29315.576400892798 | 4.467098437673678 |
| sample_cv_fold | 0 | 1187 | 183 | 117 | 556 | 631 | 1031 | 1 | 0.05407739 | 973.15 | 113816.47821313594 | 5.05620514324503 | 33115.62344477978 | 4.520032935619625 |
| sample_cv_fold | 1 | 979 | 169 | 112 | 589 | 390 | 979 | 1 | 0.4370841 | 1105.581 | 69422.63 | 4.841501062415943 | 21471.88296100494 | 4.331870131241867 |
| sample_cv_fold | 2 | 922 | 167 | 106 | 533 | 389 | 873 | 1 | 1.268499 | 1233.419 | 68700.23950569404 | 4.836957356200354 | 25414.664841922735 | 4.405084260741184 |
| sample_cv_fold | 3 | 954 | 165 | 111 | 538 | 416 | 909 | 1 | 0.1536145 | 963.3638 | 103611.40665350884 | 5.015407412356561 | 23495.723111093783 | 4.370988812130594 |
| sample_cv_fold | 4 | 958 | 171 | 113 | 562 | 396 | 913 | 1 | 0.4678308 | 973.5053 | 57908.515 | 4.7627424263523785 | 13892.37671872869 | 4.142776496575652 |
| paper_cv_fold | 0 | 863 | 165 | 42 | 636 | 227 | 836 | 1 | 0.05407739 | 973.15 | 150609.8 | 5.177853231807911 | 25552.309830765218 | 4.407430164800038 |
| paper_cv_fold | 1 | 701 | 118 | 37 | 267 | 434 | 677 | 1 | 7.340624 | 1233.419 | 48027.13 | 4.68148663489205 | 16172.605616340596 | 4.208779996012504 |
| paper_cv_fold | 2 | 1039 | 162 | 55 | 480 | 559 | 842 | 1 | 0.126506 | 1105.581 | 52603.89268805892 | 4.721017883134557 | 9727.538532962872 | 3.9880029598170808 |
| paper_cv_fold | 3 | 851 | 151 | 46 | 577 | 274 | 814 | 1 | 0.4009652 | 825.6974 | 134092.5 | 5.127404487771786 | 36132.62667102629 | 4.557899533869931 |
| paper_cv_fold | 4 | 1546 | 259 | 52 | 818 | 728 | 1536 | 1 | 1.103748 | 972.4949 | 78799.14950347248 | 4.896519687225892 | 24797.122165968736 | 4.394400837132786 |

## Coverage Preflight

| split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | min_rows_per_bin | min_samples_per_bin | min_papers_per_bin | train_rows | train_samples | train_papers | test_rows | test_samples | test_papers | train_reference_keys_total | train_reference_keys_reliable | test_rows_with_reference | test_rows_without_reference | coverage_fraction | p_test_rows | n_test_rows | p_test_rows_with_reference | n_test_rows_with_reference | material_family_count_in_test | T_bin_count_in_test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout | all_valid | all_valid | global | row_median | 3 | 3 | 1 | 4071 | 697 | 211 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | all_valid | all_valid | global | sample_median | 3 | 3 | 1 | 4071 | 697 | 211 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | all_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 4071 | 697 | 211 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | all_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 4071 | 697 | 211 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | all_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 4071 | 697 | 211 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | all_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 4071 | 697 | 211 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | all_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 4071 | 697 | 211 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | all_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 4071 | 697 | 211 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | conservative_valid | all_valid | global | row_median | 3 | 3 | 1 | 3830 | 684 | 208 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | conservative_valid | all_valid | global | sample_median | 3 | 3 | 1 | 3830 | 684 | 208 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | conservative_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 3830 | 684 | 208 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | conservative_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 3830 | 684 | 208 | 929 | 158 | 118 | 23 | 22 | 928 | 1 | 0.9989235737351991 | 565 | 364 | 564 | 364 | 1 | 12 |
| sample_holdout | conservative_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 3830 | 684 | 208 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | conservative_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 3830 | 684 | 208 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | conservative_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 3830 | 684 | 208 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| sample_holdout | conservative_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 3830 | 684 | 208 | 875 | 152 | 114 | 23 | 22 | 874 | 1 | 0.9988571428571429 | 523 | 352 | 522 | 352 | 1 | 12 |
| paper_holdout | all_valid | all_valid | global | row_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | all_valid | all_valid | global | sample_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | all_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | all_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | all_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | all_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | all_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | all_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 3795 | 688 | 189 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | conservative_valid | all_valid | global | row_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | conservative_valid | all_valid | global | sample_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | conservative_valid | all_valid | material_family | row_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | conservative_valid | all_valid | material_family | sample_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1205 | 167 | 43 | 24 | 22 | 1205 | 0 | 1.0 | 616 | 589 | 616 | 589 | 1 | 11 |
| paper_holdout | conservative_valid | conservative_valid | global | row_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | conservative_valid | conservative_valid | global | sample_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |
| paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 3 | 3 | 1 | 3512 | 671 | 185 | 1193 | 165 | 43 | 24 | 22 | 1193 | 0 | 1.0 | 611 | 582 | 611 | 582 | 1 | 11 |

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

## Warnings And Step5B Notes

- WARNING: none
- Step5B should build sigma0_ref(T) using train rows only.
- Step5B should apply sigma_pred = sigma0_ref(T) * F0_eta to test rows only.
- Evaluate errors with log10(sigma_pred / sigma_exp).
