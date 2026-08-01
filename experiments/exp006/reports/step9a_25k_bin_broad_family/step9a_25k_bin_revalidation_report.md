# Step9A 25 K Temperature-Bin Broad-Family Revalidation Report

## Input and change from the 100 K version

- Input file: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step6a_validation_rows_with_splits_key_broad_family.parquet`
- Input rows: 97086
- The Step6A broad-family `material_group_key` and all existing holdout/CV split columns were retained.
- The only model-condition change is the temperature-bin width: the existing 100 K bins were rebuilt as 25 K bins.
- Existing `T_bin_*` values were preserved as `old_T_bin_*` before replacement.
- The 25 K validation rows retain every input column and add temperature-bin provenance columns.

## 25 K temperature-bin definition

- `bin_width_K = 25.0`
- `bin_start_K = 12.5`
- Bin index: `floor((T_K - bin_start_K) / bin_width_K)`.
- Interval convention: left-closed and right-open (`T_bin_left_K <= T_K < T_bin_right_K`).
- With the defaults, `[12.5, 37.5)` has center 25 K and centers advance in 25 K increments.
- Unique valid-row bin centers: 70
- Rows outside the target-row conditions: 0

## Step5B / Step5C execution

- Step5B small test: passed
- Step5C small test: passed
- Step5B full run: passed
- Step5C full run: passed
- Step5B rebuilt `sigma0_ref(T)` from train rows only and assigned the resulting 25 K references to test rows.
- Step4 full-data reference curves were not used.

## 25 K default metrics

| default_label | config_id | metric_weighting | n_rows | coverage_fraction | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | max_abs_log10_error | extreme_ge_10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 18883 | 0.9934238215488216 | 0.7184792716957925 | 1.2438614350158783 | 0.0017694685822311 | 0.4293809246412117 | 0.6899327437377535 | 0.7854154530530106 | 14.539199102855582 | 13.0 |

## 25 K versus 100 K

- Coverage: 100 K=0.997896, 25 K=0.993424, delta=-0.004472.
- Coverage decreased: True.
- MAE: 100 K=0.723746, 25 K=0.718479, delta=-0.005266.
- RMSE: 100 K=1.250868, 25 K=1.243861, delta=-0.007007.
- Factor-2 accuracy: 100 K=0.426666, 25 K=0.429381, delta=+0.002715.
- Factor-10 accuracy: 100 K=0.783794, 25 K=0.785415, delta=+0.001622.
- Both MAE and RMSE improved: True.
- Both factor accuracies improved: True.

| config_label | metric_weighting | metric_name | value_100k | value_25k | delta_25k_minus_100k | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | coverage_fraction | 0.997895622895623 | 0.9934238215488216 | -0.0044718013468013 | higher_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | mae_log10 | 0.7237456971201834 | 0.7184792716957925 | -0.0052664254243909 | lower_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | rmse_log10 | 1.2508679717579398 | 1.2438614350158783 | -0.0070065367420615 | lower_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_2_accuracy | 0.4266659637283846 | 0.4293809246412117 | 0.002714960912827 | higher_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | factor_10_accuracy | 0.7837937579080557 | 0.7854154530530106 | 0.0016216951449549 | higher_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | max_abs_log10_error | 14.570733133464506 | 14.539199102855582 | -0.031534030608924 | lower_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | extreme_ge_10_count | 13.0 | 13.0 | 0.0 | lower_is_better |
| sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | severe_ge_5_count | 246.0 | 228.0 | -18.0 | lower_is_better |

## Material-family changes

The tables below rank groups that are reliable in both versions by the change in row-equal MAE.

### Most improved

| material_group_key | material_family_raw | n_rows_100k | n_rows_25k | mae_log10_100k | mae_log10_25k | delta_mae_log10_25k_minus_100k | delta_rmse_log10_25k_minus_100k | delta_factor_2_accuracy_25k_minus_100k | delta_factor_10_accuracy_25k_minus_100k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad::GeTe_like | unknown | 183 | 154 | 0.8167188187119881 | 0.103484468719092 | -0.7132343499928961 | -1.8902047723704591 | 0.1171315023774041 | 0.1256830601092896 |
| broad::telluride | unknown | 404 | 401 | 0.4434256029360213 | 0.4136801236680967 | -0.0297454792679245 | 0.0571555739419801 | 0.1405644305078887 | -0.0031974519147676 |
| broad::Mg2SiSn_like | unknown | 267 | 264 | 0.6898075916119215 | 0.6762936690686433 | -0.0135139225432782 | -0.0168844253983817 | -0.0091504937010554 | 0.0347718760640108 |
| broad::other_formula_system | unknown | 6132 | 6124 | 0.7808995674301903 | 0.7742424968680016 | -0.0066570705621886 | -0.002071463019684 | 0.0020212839840085 | 0.0017864119780675 |
| broad::CoSb_skutterudite_like | unknown | 1328 | 1325 | 0.6515682992832744 | 0.6462907626484121 | -0.0052775366348623 | -0.0108903597231653 | 0.0107541486701522 | -0.0004313480336439 |
| broad::SbTe_like | unknown | 974 | 971 | 0.3914131759124441 | 0.3889039293435482 | -0.0025092465688958 | -0.010038537165414 | -0.0083668691858559 | 0.0007348633999961 |
| broad::BiTe_like | unknown | 1776 | 1775 | 0.3718177910986153 | 0.3693740162545834 | -0.0024437748440319 | -0.0072947323065463 | 0.0003356173074483 | 0.0061546758025632 |
| broad::SnTe_like | unknown | 370 | 370 | 0.1647405686543639 | 0.1623412678403444 | -0.0023993008140195 | -0.0234518461424805 | -0.0054054054054053 | 0.0054054054054054 |

### Most worsened

| material_group_key | material_family_raw | n_rows_100k | n_rows_25k | mae_log10_100k | mae_log10_25k | delta_mae_log10_25k_minus_100k | delta_rmse_log10_25k_minus_100k | delta_factor_2_accuracy_25k_minus_100k | delta_factor_10_accuracy_25k_minus_100k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| broad::sulfide | unknown | 892 | 885 | 1.4434137313346973 | 1.5111668562381977 | 0.0677531249035003 | 0.101031114720846 | -0.0205809328367662 | -0.0105609181424336 |
| unknown_material_group | unknown | 110 | 109 | 1.1569728361166862 | 1.2142629909595792 | 0.057290154842893 | 0.0381008704397711 | -0.0788156797331109 | 0.0324437030859049 |
| broad::BiSbTe_tetradymite_like | unknown | 643 | 636 | 0.5983637146953943 | 0.6071187888499353 | 0.008755074154541 | 0.0156462367367675 | 0.0113388499271301 | -0.0140164519694435 |
| broad::selenide | unknown | 1783 | 1781 | 0.502884231521916 | 0.5072753063093984 | 0.0043910747874824 | 0.0107074187277108 | 0.009026229695077 | -0.0012886066326712 |
| broad::PbTe_like | unknown | 472 | 467 | 0.268231940409633 | 0.2717347407731169 | 0.0035028003634838 | 0.0064422769447574 | -0.0025178746415998 | -0.0068549704206439 |
| broad::SiGe_like | unknown | 197 | 188 | 0.7339675896268115 | 0.7367163213480764 | 0.0027487317212648 | -0.0129747442304939 | -0.0494923857868019 | 0.0233826547143319 |
| broad::oxide | unknown | 3437 | 3433 | 1.0158711010290336 | 1.0168447567620778 | 0.0009736557330441 | -0.0044455379798054 | -0.0031015606877776 | -0.0006719935155041 |

## Sanity checks

- input_file_exists: True
- validation_rows_created: True
- validation_row_count_matches_input: True
- old_T_bin_columns_exist: True
- T_bin_centers_follow_requested_width: True
- T_bin_centers_are_25K_multiples: True
- temperature_inside_assigned_bin: True
- temperature_bin_version_25K: True
- material_group_key_not_missing: True
- holdout_splits_preserved: True
- step5b_outputs_created: True
- step5c_outputs_created: True
- step5b_step5c_checks_passed: True
- default_metrics_summary_created: True
- comparison_created: True
- report_created: True
- existing_100K_directory_unchanged: True
- raw_data_not_read: True

## Important notes

- Finer temperature bins can express local temperature dependence more directly, but fewer observations per bin can make the reference coefficient unstable.
- The existing 100 K outputs were not overwritten.
- Starrydata2 raw data was not read.
- This is not a new model; it is a revalidation in which only the temperature-bin width was changed.
- No figures were created in Step9A.

## Recommended next steps

- If the 25 K version improves the main metrics without a material coverage loss, create 25 K scatter and material-family plots in Step9B.
- If coverage or RMSE worsens, test an intermediate 50 K width.
- Compare 100 K, 50 K, and 25 K on the same splits and reporting definitions.
- elapsed_seconds: 1707.62
