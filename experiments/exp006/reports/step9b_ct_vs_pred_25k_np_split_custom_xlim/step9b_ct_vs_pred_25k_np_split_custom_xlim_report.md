# Step9B custom x-axis figures

## Summary

- These are newly created figures; the original Step9B figures were not overwritten.
- Prediction rows: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_ct_vs_pred_25k_np_split\step9b_prediction_rows_used.csv`
- p/n-unsplit old C(T): `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_ct_vs_pred_25k_np_split\step9b_old_ct_curves_no_pn.csv`
- Only the horizontal axis range was changed.
- Point data, old C(T) data, y-axis scaling, colors, and legends follow the original Step9B figures.
- Combinations not explicitly assigned a custom range retain matplotlib automatic x-axis limits.
- Original-figure files unchanged: True
- PNG files: 14
- PDF files: 14
- elapsed_seconds: 11.92

## Figure index and x-axis ranges

| material_group_key | carrier_type | xlim_mode | requested_x_min_K | requested_x_max_K | applied_x_min_K | applied_x_max_K | figure_path_png |
| --- | --- | --- | --- | --- | --- | --- | --- |
| broad::SnTe_like | p | custom_fixed | 0.0 | 1000.0 | 0.0 | 1000.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SnTe_like_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::SnTe_like | n | unchanged_auto |  |  | -52.5 | 1102.5 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SnTe_like_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::PbTe_like | p | custom_fixed | 100.0 | 1000.0 | 100.0 | 1000.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_PbTe_like_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::PbTe_like | n | custom_fixed | 0.0 | 1000.0 | 0.0 | 1000.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_PbTe_like_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::BiTe_like | p | unchanged_auto |  |  | -61.25 | 1286.25 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_BiTe_like_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::BiTe_like | n | custom_fixed | 0.0 | 800.0 | 0.0 | 800.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_BiTe_like_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::SbTe_like | p | custom_fixed | 0.0 | 900.0 | 0.0 | 900.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SbTe_like_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::SbTe_like | n | custom_fixed | 0.0 | 1000.0 | 0.0 | 1000.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SbTe_like_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::SiGe_like | p | unchanged_auto |  |  | -65.0 | 1365.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SiGe_like_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::SiGe_like | n | custom_fixed | 50.0 | 900.0 | 50.0 | 900.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_SiGe_like_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::oxide | p | unchanged_auto |  |  | -63.75 | 1338.75 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_oxide_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::oxide | n | unchanged_auto |  |  | -64.23795 | 1348.99695 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_oxide_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::sulfide | p | custom_fixed | 0.0 | 1200.0 | 0.0 | 1200.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_sulfide_p_sigma_pred_vs_oldCT_25k_custom_xlim.png |
| broad::sulfide | n | custom_fixed | 0.0 | 1300.0 | 0.0 | 1300.0 | C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_ct_vs_pred_25k_np_split_custom_xlim\broad_sulfide_n_sigma_pred_vs_oldCT_25k_custom_xlim.png |

## Notes

- No new sigma_pred values were calculated.
- Existing Step9B CSVs were read without modification.
- Existing Step9B PNG/PDF files were not modified.
- Measured sigma and sigma0_ref are not plotted.
