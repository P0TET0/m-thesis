# Step9B: 25 K sigma0_ref(T) vs Step31 SS2026 C(T) overlay

## Inputs and detected Step31 schema

- exp006 input: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9a_25k_bin_broad_family\step5b_train_reference_curve_bins.parquet`
- exp005 Step31 C(T) CSV: `C:\Users\miots\m-thesis\m-thesis\data\output\starrydata2_step31_reclassified_ct_curves\ct_curves_material_system_np_split_step31.csv`
- adopted C(T) column: `C_T_sigma_median_S_per_m_step31`
- adopted temperature column: `temp_bin_center_K`
- adopted material column: `material_system_reclass_step31`
- adopted carrier column: `n_or_p_reclass_step31`

## Scope and results

- target material groups (3): broad::BiSbTe_tetradymite_like, broad::PbTe_like, broad::SnTe_like
- target carrier_type: p, n
- material-group x carrier combinations with both curves: 6
- overlay figures: 6
- log-ratio figures: 6
- combinations skipped because one side was missing: 0
- overall median log10(sigma0_ref_25k / C_T_step31): -0.137259

### Median ratio by material group

| material_group_key | median log10 ratio |
|---|---:|
| broad::BiSbTe_tetradymite_like | -0.290187 |
| broad::PbTe_like | 0.25139 |
| broad::SnTe_like | -0.561713 |

### Missing-side warnings

None.

## Figure index

| figure_id | group | carrier | type | PNG | PDF |
|---|---|---|---|---|---|
| step9b_001 | broad::BiSbTe_tetradymite_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_002 | broad::BiSbTe_tetradymite_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_p_p_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |
| step9b_003 | broad::BiSbTe_tetradymite_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_004 | broad::BiSbTe_tetradymite_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_BiSbTe_tetradymite_like_n_n_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |
| step9b_005 | broad::PbTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_006 | broad::PbTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |
| step9b_007 | broad::PbTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_008 | broad::PbTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_PbTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |
| step9b_009 | broad::SnTe_like | p | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_010 | broad::SnTe_like | p | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_p_p_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |
| step9b_011 | broad::SnTe_like | n | overlay | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_25k_sigma0ref_vs_step31_ct_overlay_test.pdf` |
| step9b_012 | broad::SnTe_like | n | log_ratio_nearest | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_log_ratio_sigma0ref25k_over_step31ct_test.png` | `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\figures\step9b_25k_vs_step31_ct_overlay\broad_SnTe_like_n_n_log_ratio_sigma0ref25k_over_step31ct_test.pdf` |

## Unmatched material labels

None.

## CSV outputs

- parse: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_step31_ct_parse_summary_test.csv`
- sigma: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_25k_sigma0ref_curves_test.csv`
- ct: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_step31_ct_curves_normalized_test.csv`
- comparison: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_curve_comparison_table_test.csv`
- summary: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_summary_by_group_carrier_test.csv`
- figures: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_figure_index_test.csv`
- mapping: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_material_mapping_test.csv`
- unmatched: `C:\Users\miots\m-thesis\m-thesis\experiments\exp006\data\processed\step9b_25k_vs_step31_ct_overlay\step9b_unmatched_material_labels_test.csv`

## Interpretation and safeguards

- The exp006 curve is the 25 K temperature-bin broad_family sigma0_ref(T) from Step9A.
- The exp005 curve is SS2026 C(T) from Step31, not the old Step12 prefactor C(T).
- No new sigma_pred was calculated.
- Starrydata2 raw data was not read.
- Existing Step0-Step9A and Step31 outputs were not overwritten.
- Input file metadata remained unchanged during this run: True.
