# Step6B Broad Family Revalidation Report

## Summary

- input_variant: broad_family
- input_file: experiments\exp006\data\processed\step6a_validation_rows_with_splits_key_broad_family.parquet
- input_rows: 97086
- material_group_key unique count: 16
- material_group_key top groups: {'broad::other_formula_system': 31682, 'broad::oxide': 17715, 'broad::selenide': 8815, 'broad::BiTe_like': 8807, 'broad::CoSb_skutterudite_like': 6278, 'broad::SbTe_like': 4993, 'broad::sulfide': 4544, 'broad::BiSbTe_tetradymite_like': 3283, 'broad::PbTe_like': 2350, 'broad::telluride': 2147, 'broad::SnTe_like': 2069, 'broad::SiGe_like': 1443, 'broad::Mg2SiSn_like': 1388, 'broad::GeTe_like': 1113, 'unknown_material_group': 458, 'broad::half_heusler': 1}
- Step5B small test: passed
- Step5B full run: passed
- Step5C small test: passed
- Step5C full run: passed

## Prediction Diff Summary

| comparison_label | joined_row_count | max_abs_delta_log10_sigma_pred | median_abs_delta_log10_sigma_pred | mean_abs_delta_log10_sigma_pred | max_abs_delta_log10_sigma0_ref | median_abs_delta_log10_sigma0_ref | exact_equal_prediction_count | approximately_equal_prediction_count | different_prediction_count | different_prediction_fraction | unique_material_group_key_for_prediction_count_material_family | unique_material_group_key_for_prediction_count_global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sample_holdout_material_family_vs_global | 18968 | 2.09268661740968 | 0.21284970501651035 | 0.3170542256146483 | 2.0926866174096785 | 0.21284970501651035 | 0 | 0 | 18968 | 1.0 | 15 | 1 |
| paper_holdout_material_family_vs_global | 20120 | 1.6585846846781784 | 0.2036824369034771 | 0.29795350112013486 | 1.6585846846781775 | 0.2036824369034771 | 0 | 0 | 20120 | 1.0 | 15 | 1 |

## Reference Group Diagnostics Preview

| config_id | split_scheme | reference_source_subset | eval_target_subset | group_scheme | curve_method | material_group_count | reference_bin_count | reliable_reference_bin_count | material_group_examples | T_bin_count | carrier_type_values |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| paper_holdout__ref_all_valid__eval_all_valid__global__row_median | paper_holdout | all_valid | all_valid | global | row_median | 1 | 34 | 31 | ALL | 19 | n | p |
| paper_holdout__ref_all_valid__eval_all_valid__global__sample_median | paper_holdout | all_valid | all_valid | global | sample_median | 1 | 34 | 31 | ALL | 19 | n | p |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__row_median | paper_holdout | all_valid | all_valid | material_family | row_median | 16 | 321 | 275 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 19 | n | p |
| paper_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | paper_holdout | all_valid | all_valid | material_family | sample_median | 16 | 321 | 275 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 19 | n | p |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__row_median | paper_holdout | all_valid | conservative_valid | global | row_median | 1 | 34 | 31 | ALL | 19 | n | p |
| paper_holdout__ref_all_valid__eval_conservative_valid__global__sample_median | paper_holdout | all_valid | conservative_valid | global | sample_median | 1 | 34 | 31 | ALL | 19 | n | p |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__row_median | paper_holdout | all_valid | conservative_valid | material_family | row_median | 16 | 321 | 275 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 19 | n | p |
| paper_holdout__ref_all_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | all_valid | conservative_valid | material_family | sample_median | 16 | 321 | 275 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 19 | n | p |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__row_median | paper_holdout | conservative_valid | all_valid | global | row_median | 1 | 33 | 31 | ALL | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | paper_holdout | conservative_valid | all_valid | global | sample_median | 1 | 33 | 31 | ALL | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__row_median | paper_holdout | conservative_valid | all_valid | material_family | row_median | 16 | 312 | 271 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | paper_holdout | conservative_valid | all_valid | material_family | sample_median | 16 | 312 | 271 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__row_median | paper_holdout | conservative_valid | conservative_valid | global | row_median | 1 | 33 | 31 | ALL | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__global__sample_median | paper_holdout | conservative_valid | conservative_valid | global | sample_median | 1 | 33 | 31 | ALL | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__row_median | paper_holdout | conservative_valid | conservative_valid | material_family | row_median | 16 | 312 | 271 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |
| paper_holdout__ref_conservative_valid__eval_conservative_valid__material_family__sample_median | paper_holdout | conservative_valid | conservative_valid | material_family | sample_median | 16 | 312 | 271 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |
| sample_holdout__ref_all_valid__eval_all_valid__global__row_median | sample_holdout | all_valid | all_valid | global | row_median | 1 | 33 | 31 | ALL | 18 | n | p |
| sample_holdout__ref_all_valid__eval_all_valid__global__sample_median | sample_holdout | all_valid | all_valid | global | sample_median | 1 | 33 | 31 | ALL | 18 | n | p |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__row_median | sample_holdout | all_valid | all_valid | material_family | row_median | 16 | 321 | 273 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |
| sample_holdout__ref_all_valid__eval_all_valid__material_family__sample_median | sample_holdout | all_valid | all_valid | material_family | sample_median | 16 | 321 | 273 | broad::BiSbTe_tetradymite_like | broad::BiTe_like | broad::CoSb_skutterudite_like | broad::GeTe_like | broad::Mg2SiSn_like | broad::PbTe_like | broad::SbTe_like | broad::SiGe_like | broad::SnTe_like | broad::half_heusler | 18 | n | p |

## Broad Family Default Metrics

| default_label | config_id | metric_weighting | mae_log10 | rmse_log10 | median_log10_error | factor_2_accuracy | factor_5_accuracy | factor_10_accuracy | coverage_fraction | n_rows | n_samples | n_papers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global_default | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.852081750001124 | 1.4064629875967487 | -0.0128278172165359 | 0.3263157894736842 | 0.6264736842105263 | 0.7508421052631579 | 0.9995791245791246 | 19000 | 3191 | 2202 |
| material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.7237456971201834 | 1.2508679717579398 | 0.0030867094458137 | 0.4266659637283846 | 0.6865773935048503 | 0.7837937579080557 | 0.997895622895623 | 18968 | 3189 | 2200 |
| paper_global_default | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | row_equal | 0.8696361454419639 | 1.4699765395399786 | -0.0635495182310127 | 0.3128907412920512 | 0.6159571300982435 | 0.7470973504019053 | 1.0 | 20154 | 3179 | 866 |
| paper_material_family_default | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | row_equal | 0.788113945871367 | 1.4026385467371438 | -0.0233081421220294 | 0.3890159045725646 | 0.6720178926441351 | 0.7736083499005965 | 0.9983129899771758 | 20120 | 3175 | 865 |
| global_default | sample_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.7372337311582936 | 1.2343448639232923 | 0.001209547017221 | 0.350987151363209 | 0.6950799122532122 | 0.7972422438107176 | 0.9995791245791246 | 3191 | 3191 | 2202 |
| material_family_default | sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6264331342694931 | 1.1209937015591127 | 0.0113405912103783 | 0.4816556914393227 | 0.7375352775164629 | 0.8256506741925368 | 0.997895622895623 | 3189 | 3189 | 2200 |
| paper_global_default | paper_holdout__ref_conservative_valid__eval_all_valid__global__sample_median | sample_equal | 0.733850844545466 | 1.2177864499192326 | -0.0310619312563688 | 0.3425605536332179 | 0.6860648002516515 | 0.8052846807172067 | 1.0 | 3179 | 3179 | 866 |
| paper_material_family_default | paper_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median | sample_equal | 0.6383798691230225 | 1.1306923247085083 | -0.0182635202846217 | 0.4566929133858268 | 0.7376377952755906 | 0.8299212598425196 | 0.9983129899771758 | 3175 | 3175 | 865 |

## Original vs Broad Family Default Metrics

| default_label | metric_weighting | metric_name | original_value | broad_family_value | delta_broad_minus_original | relative_change_if_applicable | interpretation_hint |
| --- | --- | --- | --- | --- | --- | --- | --- |
| material_family_default | row_equal | mae_log10 | 0.852081750001124 | 0.7237456971201834 | -0.12833605288094052 | -0.15061471845955066 | lower_is_better |
| material_family_default | row_equal | rmse_log10 | 1.4064629875967487 | 1.2508679717579398 | -0.15559501583880886 | -0.11062858902862219 | lower_is_better |
| material_family_default | row_equal | median_log10_error | -0.0128278172165359 | 0.0030867094458137 | 0.0159145266623496 | -1.2406262416831704 | count_or_context |
| material_family_default | row_equal | factor_2_accuracy | 0.3263157894736842 | 0.4266659637283846 | 0.10035017425470039 | 0.307524727554727 | higher_is_better |
| material_family_default | row_equal | factor_5_accuracy | 0.6264736842105263 | 0.6865773935048503 | 0.06010370929432396 | 0.09593971911216964 | higher_is_better |
| material_family_default | row_equal | factor_10_accuracy | 0.7508421052631579 | 0.7837937579080557 | 0.032951652644897855 | 0.043886261057974156 | higher_is_better |
| material_family_default | row_equal | coverage_fraction | 0.9995791245791246 | 0.997895622895623 | -0.0016835016835016203 | -0.0016842105263157262 | higher_is_better |
| material_family_default | row_equal | n_rows | 19000.0 | 18968.0 | -32.0 | -0.0016842105263157896 | count_or_context |
| material_family_default | row_equal | n_samples | 3191.0 | 3189.0 | -2.0 | -0.0006267627702914447 | count_or_context |
| material_family_default | row_equal | n_papers | 2202.0 | 2200.0 | -2.0 | -0.0009082652134423251 | count_or_context |
| material_family_default | sample_equal | mae_log10 | 0.7372337311582936 | 0.6264331342694931 | -0.1108005968888005 | -0.15029235940509372 | lower_is_better |
| material_family_default | sample_equal | rmse_log10 | 1.2343448639232923 | 1.1209937015591127 | -0.11335116236417964 | -0.09183103172957649 | lower_is_better |
| material_family_default | sample_equal | median_log10_error | 0.001209547017221 | 0.0113405912103783 | 0.0101310441931573 | 8.375899447409596 | count_or_context |
| material_family_default | sample_equal | factor_2_accuracy | 0.350987151363209 | 0.4816556914393227 | 0.13066854007611367 | 0.3722886708775703 | higher_is_better |
| material_family_default | sample_equal | factor_5_accuracy | 0.6950799122532122 | 0.7375352775164629 | 0.042455365263250666 | 0.06107983343328804 | higher_is_better |
| material_family_default | sample_equal | factor_10_accuracy | 0.7972422438107176 | 0.8256506741925368 | 0.028408430381819172 | 0.03563337317153498 | higher_is_better |
| material_family_default | sample_equal | coverage_fraction | 0.9995791245791246 | 0.997895622895623 | -0.0016835016835016203 | -0.0016842105263157262 | higher_is_better |
| material_family_default | sample_equal | n_rows | 3191.0 | 3189.0 | -2.0 | -0.0006267627702914447 | count_or_context |
| material_family_default | sample_equal | n_samples | 3191.0 | 3189.0 | -2.0 | -0.0006267627702914447 | count_or_context |
| material_family_default | sample_equal | n_papers | 2202.0 | 2200.0 | -2.0 | -0.0009082652134423251 | count_or_context |
| global_default | row_equal | mae_log10 | 0.852081750001124 | 0.852081750001124 | 0.0 | 0.0 | lower_is_better |
| global_default | row_equal | rmse_log10 | 1.4064629875967487 | 1.4064629875967487 | 0.0 | 0.0 | lower_is_better |
| global_default | row_equal | median_log10_error | -0.0128278172165359 | -0.0128278172165359 | 0.0 | -0.0 | count_or_context |
| global_default | row_equal | factor_2_accuracy | 0.3263157894736842 | 0.3263157894736842 | 0.0 | 0.0 | higher_is_better |
| global_default | row_equal | factor_5_accuracy | 0.6264736842105263 | 0.6264736842105263 | 0.0 | 0.0 | higher_is_better |
| global_default | row_equal | factor_10_accuracy | 0.7508421052631579 | 0.7508421052631579 | 0.0 | 0.0 | higher_is_better |
| global_default | row_equal | coverage_fraction | 0.9995791245791246 | 0.9995791245791246 | 0.0 | 0.0 | higher_is_better |
| global_default | row_equal | n_rows | 19000.0 | 19000.0 | 0.0 | 0.0 | count_or_context |
| global_default | row_equal | n_samples | 3191.0 | 3191.0 | 0.0 | 0.0 | count_or_context |
| global_default | row_equal | n_papers | 2202.0 | 2202.0 | 0.0 | 0.0 | count_or_context |
| global_default | sample_equal | mae_log10 | 0.7372337311582936 | 0.7372337311582936 | 0.0 | 0.0 | lower_is_better |
| global_default | sample_equal | rmse_log10 | 1.2343448639232923 | 1.2343448639232923 | 0.0 | 0.0 | lower_is_better |
| global_default | sample_equal | median_log10_error | 0.001209547017221 | 0.001209547017221 | 0.0 | 0.0 | count_or_context |
| global_default | sample_equal | factor_2_accuracy | 0.350987151363209 | 0.350987151363209 | 0.0 | 0.0 | higher_is_better |
| global_default | sample_equal | factor_5_accuracy | 0.6950799122532122 | 0.6950799122532122 | 0.0 | 0.0 | higher_is_better |
| global_default | sample_equal | factor_10_accuracy | 0.7972422438107176 | 0.7972422438107176 | 0.0 | 0.0 | higher_is_better |
| global_default | sample_equal | coverage_fraction | 0.9995791245791246 | 0.9995791245791246 | 0.0 | 0.0 | higher_is_better |
| global_default | sample_equal | n_rows | 3191.0 | 3191.0 | 0.0 | 0.0 | count_or_context |
| global_default | sample_equal | n_samples | 3191.0 | 3191.0 | 0.0 | 0.0 | count_or_context |
| global_default | sample_equal | n_papers | 2202.0 | 2202.0 | 0.0 | 0.0 | count_or_context |
| paper_material_family_default | row_equal | mae_log10 | 0.8696361454419639 | 0.788113945871367 | -0.08152219957059692 | -0.09374288315622616 | lower_is_better |
| paper_material_family_default | row_equal | rmse_log10 | 1.4699765395399786 | 1.4026385467371438 | -0.06733799280283481 | -0.04580888945609151 | lower_is_better |
| paper_material_family_default | row_equal | median_log10_error | -0.0635495182310127 | -0.0233081421220294 | 0.04024137610898329 | -0.633228657417975 | count_or_context |
| paper_material_family_default | row_equal | factor_2_accuracy | 0.3128907412920512 | 0.3890159045725646 | 0.07612516328051339 | 0.24329631156921452 | higher_is_better |
| paper_material_family_default | row_equal | factor_5_accuracy | 0.6159571300982435 | 0.6720178926441351 | 0.05606076254589165 | 0.09101406543820689 | higher_is_better |
| paper_material_family_default | row_equal | factor_10_accuracy | 0.7470973504019053 | 0.7736083499005965 | 0.02651099949869118 | 0.035485334654753405 | higher_is_better |
| paper_material_family_default | row_equal | coverage_fraction | 1.0 | 0.9983129899771758 | -0.0016870100228242313 | -0.0016870100228242313 | higher_is_better |
| paper_material_family_default | row_equal | n_rows | 20154.0 | 20120.0 | -34.0 | -0.0016870100228242532 | count_or_context |
| paper_material_family_default | row_equal | n_samples | 3179.0 | 3175.0 | -4.0 | -0.0012582573136206354 | count_or_context |
| paper_material_family_default | row_equal | n_papers | 866.0 | 865.0 | -1.0 | -0.0011547344110854503 | count_or_context |
| paper_material_family_default | sample_equal | mae_log10 | 0.733850844545466 | 0.6383798691230225 | -0.09547097542244354 | -0.13009588546781128 | lower_is_better |
| paper_material_family_default | sample_equal | rmse_log10 | 1.2177864499192326 | 1.1306923247085083 | -0.08709412521072424 | -0.07151838913670011 | lower_is_better |
| paper_material_family_default | sample_equal | median_log10_error | -0.0310619312563688 | -0.0182635202846217 | 0.012798410971747099 | -0.41202882287375386 | count_or_context |
| paper_material_family_default | sample_equal | factor_2_accuracy | 0.3425605536332179 | 0.4566929133858268 | 0.11413235975260888 | 0.33317426230812097 | higher_is_better |
| paper_material_family_default | sample_equal | factor_5_accuracy | 0.6860648002516515 | 0.7376377952755906 | 0.05157299502393908 | 0.07517219219674569 | higher_is_better |
| paper_material_family_default | sample_equal | factor_10_accuracy | 0.8052846807172067 | 0.8299212598425196 | 0.02463657912531292 | 0.030593626968503814 | higher_is_better |
| paper_material_family_default | sample_equal | coverage_fraction | 1.0 | 0.9983129899771758 | -0.0016870100228242313 | -0.0016870100228242313 | higher_is_better |
| paper_material_family_default | sample_equal | n_rows | 3179.0 | 3175.0 | -4.0 | -0.0012582573136206354 | count_or_context |
| paper_material_family_default | sample_equal | n_samples | 3179.0 | 3175.0 | -4.0 | -0.0012582573136206354 | count_or_context |
| paper_material_family_default | sample_equal | n_papers | 866.0 | 865.0 | -1.0 | -0.0011547344110854503 | count_or_context |
| paper_global_default | row_equal | mae_log10 | 0.8696361454419639 | 0.8696361454419639 | 0.0 | 0.0 | lower_is_better |
| paper_global_default | row_equal | rmse_log10 | 1.4699765395399786 | 1.4699765395399786 | 0.0 | 0.0 | lower_is_better |
| paper_global_default | row_equal | median_log10_error | -0.0635495182310127 | -0.0635495182310127 | 0.0 | -0.0 | count_or_context |
| paper_global_default | row_equal | factor_2_accuracy | 0.3128907412920512 | 0.3128907412920512 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | row_equal | factor_5_accuracy | 0.6159571300982435 | 0.6159571300982435 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | row_equal | factor_10_accuracy | 0.7470973504019053 | 0.7470973504019053 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | row_equal | coverage_fraction | 1.0 | 1.0 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | row_equal | n_rows | 20154.0 | 20154.0 | 0.0 | 0.0 | count_or_context |
| paper_global_default | row_equal | n_samples | 3179.0 | 3179.0 | 0.0 | 0.0 | count_or_context |
| paper_global_default | row_equal | n_papers | 866.0 | 866.0 | 0.0 | 0.0 | count_or_context |
| paper_global_default | sample_equal | mae_log10 | 0.733850844545466 | 0.733850844545466 | 0.0 | 0.0 | lower_is_better |
| paper_global_default | sample_equal | rmse_log10 | 1.2177864499192326 | 1.2177864499192326 | 0.0 | 0.0 | lower_is_better |
| paper_global_default | sample_equal | median_log10_error | -0.0310619312563688 | -0.0310619312563688 | 0.0 | -0.0 | count_or_context |
| paper_global_default | sample_equal | factor_2_accuracy | 0.3425605536332179 | 0.3425605536332179 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | sample_equal | factor_5_accuracy | 0.6860648002516515 | 0.6860648002516515 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | sample_equal | factor_10_accuracy | 0.8052846807172067 | 0.8052846807172067 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | sample_equal | coverage_fraction | 1.0 | 1.0 | 0.0 | 0.0 | higher_is_better |
| paper_global_default | sample_equal | n_rows | 3179.0 | 3179.0 | 0.0 | 0.0 | count_or_context |
| paper_global_default | sample_equal | n_samples | 3179.0 | 3179.0 | 0.0 | 0.0 | count_or_context |
| paper_global_default | sample_equal | n_papers | 866.0 | 866.0 | 0.0 | 0.0 | count_or_context |

## Revalidation Summary

| item | value | comment |
| --- | --- | --- |
| input_variant | broad_family | Step6A broad_family material_group_key |
| input_rows | 97086 | Rows in Step6A broad_family input |
| material_group_key_unique_count | 16 | Broad-family group count |
| material_group_key_unknown_fraction | 0.00471746698803123 | Unknown broad-family fraction |
| step5b_prediction_ok_rows | 604440 | All-config ok prediction rows |
| step5b_prediction_unavailable_rows | 480 | All-config unavailable rows |
| step5b_default_coverage_fraction | 0.997895622895623 | Step5B coverage |
| step5b_global_default_coverage_fraction | 0.9995791245791246 | Step5B coverage |
| step5c_default_mae_log10 | 0.7237456971201834 | Step5C row_equal default metric |
| step5c_default_factor_2_accuracy | 0.4266659637283846 | Step5C row_equal default metric |
| step5c_default_factor_10_accuracy | 0.7837937579080557 | Step5C row_equal default metric |
| step5c_global_default_mae_log10 | 0.852081750001124 | Step5C row_equal default metric |
| material_family_vs_global_predictions_identical | False | Sample holdout default |
| material_family_vs_global_different_prediction_fraction | 1.0 | Sample holdout default |
| paper_material_family_vs_global_predictions_identical | False | Paper holdout default |
| paper_material_family_vs_global_different_prediction_fraction | 1.0 | Paper holdout default |
| original_default_mae_log10 | 0.852081750001124 | Original Step5C default MAE |
| broad_family_default_mae_log10 | 0.7237456971201834 | Step6B broad-family default MAE |
| default_mae_delta_broad_minus_original | -0.12833605288094052 | Positive means broad_family is worse by MAE |
| recommended_next_action | Step6C visualization | Next decision based on material vs global difference |

## Sanity Check

- output_dir_is_step6b_specific: True
- input_unique_groups_gt_1: True
- input_material_group_key_not_missing: True
- step5b_full_outputs_exist: True
- step5c_full_outputs_exist: True
- step5b_valid_nonzero: True
- step5c_metrics_nonzero: True
- step5c_default_comparison_8_rows: True
- step5b_config_count_32: True
- step5c_config_count_32: True
- diff_summary_created: True
- reference_diag_created: True
- original_comparison_created: True
- default_summary_created: True
- summary_created: True
- report_created: True
- did_not_read_step4_full_data_reference_curve: True
- did_not_read_raw_data: True

## Notes

- Step4 full-data reference curves were not read.
- Starrydata2 raw data was not read.
- Existing Step5B/Step5C outputs were not overwritten; outputs are under step6b_broad_family.
- If material_family and global still match, inspect Step5B join keys and material_group_key_for_prediction.
- If they differ, Step6C should visualize the broad_family revalidation.
- elapsed_seconds: 12.41
