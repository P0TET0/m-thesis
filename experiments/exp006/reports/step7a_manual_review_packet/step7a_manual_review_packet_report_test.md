# Step7A Manual Review Packet Report

## Outputs

| file_role | file_path | description | intended_user_action |
| --- | --- | --- | --- |
| master | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_manual_review_master_test.csv | Unified row/sample/paper review case list. | Use as supporting review material. |
| row_cases | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_row_review_cases_test.csv | Row-level outlier cases. | Use as supporting review material. |
| sample_cases | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_sample_review_cases_test.csv | Sample-level concentration cases. | Use as supporting review material. |
| paper_cases | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_paper_review_cases_test.csv | Paper-level concentration cases. | Use as supporting review material. |
| decision_template | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_review_decisions_template_test.csv | Human-editable review decisions for Step7B. | Fill reviewer columns before Step7B. |
| source_trace | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_source_traceability_table_test.csv | Source traceability fields and scores. | Use to locate source files and curves. |
| sample_context | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_sample_context_for_review_test.csv | Rows around top outlier samples. | Use as supporting review material. |
| casebook | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_manual_review_casebook_test.md | Readable markdown summary of top review cases. | Read first for triage. |
| packet_index | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_review_packet_index_test.csv |  | Use as supporting review material. |
| readiness_update | experiments\exp006\data\processed\step7a_manual_review_packet\step7a_readiness_after_review_packet_summary_test.csv | Readiness state after packet creation. | Use as supporting review material. |
| report | experiments\exp006\reports\step7a_manual_review_packet\step7a_manual_review_packet_report_test.md | Step7A generation report. | Use as supporting review material. |

## Case Counts

- total review cases: 110
- row_case: 50
- sample_case: 30
- paper_case: 30
- extreme_ge_10_decades cases: 17
- severe_ge_5_decades cases: 37
- large_ge_2_decades cases: 0
- source_traceability_score distribution: {5: 50}
- cases with source metadata gaps: 60

## Top Review Cases

| review_case_id | review_case_type | review_priority | row_id | paper_id | sample_id | material_group_key | abs_error_decades | error_severity | likely_error_origin_hint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ROW_0001 | row_case | 1 | step0_00130231 | 10260 | 5395 | broad::CoSb_skutterudite_like | 14.570733133464506 | extreme_ge_10_decades | sigma0_ref_much_smaller_than_row_sigma0 |
| ROW_0002 | row_case | 2 | step0_00130230 | 10260 | 5395 | broad::CoSb_skutterudite_like | 14.351647502116624 | extreme_ge_10_decades | sigma0_ref_much_smaller_than_row_sigma0 |
| ROW_0003 | row_case | 3 | step0_00130229 | 10260 | 5395 | broad::CoSb_skutterudite_like | 14.302129201352974 | extreme_ge_10_decades | sigma0_ref_much_smaller_than_row_sigma0 |
| ROW_0004 | row_case | 4 | step0_00130228 | 10260 | 5395 | broad::CoSb_skutterudite_like | 14.22364900903764 | extreme_ge_10_decades | sigma0_ref_much_smaller_than_row_sigma0 |
| ROW_0005 | row_case | 5 | step0_00000916 | 33875 | 41898 | broad::other_formula_system | 11.212152574256098 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0006 | row_case | 6 | step0_00000917 | 33875 | 41898 | broad::other_formula_system | 10.952385828222257 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0007 | row_case | 7 | step0_00000918 | 33875 | 41898 | broad::other_formula_system | 10.887502471337074 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0008 | row_case | 8 | step0_00000924 | 33875 | 41898 | broad::other_formula_system | 10.786471656161954 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0009 | row_case | 9 | step0_00000923 | 33875 | 41898 | broad::other_formula_system | 10.723264309126158 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0010 | row_case | 10 | step0_00000919 | 33875 | 41898 | broad::other_formula_system | 10.601881498991323 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0011 | row_case | 11 | step0_00000920 | 33875 | 41898 | broad::other_formula_system | 10.56817079850579 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0012 | row_case | 12 | step0_00000922 | 33875 | 41898 | broad::other_formula_system | 10.514301158321803 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0013 | row_case | 13 | step0_00000921 | 33875 | 41898 | broad::other_formula_system | 10.504628181970778 | extreme_ge_10_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0014 | row_case | 14 | step0_00134097 | 10316 | 7968 | broad::sulfide | 8.412448832124268 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0015 | row_case | 15 | step0_00134098 | 10316 | 7968 | broad::sulfide | 7.98884535118387 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0016 | row_case | 16 | step0_00159174 | 9442 | 24514 | broad::oxide | 7.909334860713662 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0017 | row_case | 17 | step0_00045515 | 1496 | 81442 | broad::other_formula_system | 7.803100735126177 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0018 | row_case | 18 | step0_00009966 | 759 | 6738 | broad::sulfide | 7.796234349172459 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0019 | row_case | 19 | step0_00134099 | 10316 | 7968 | broad::sulfide | 7.669055714887127 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |
| ROW_0020 | row_case | 20 | step0_00159175 | 9442 | 24514 | broad::oxide | 7.607003208282069 | severe_ge_5_decades | sigma0_ref_much_larger_than_row_sigma0 |

## Decision Template Use

- Fill `review_status` with keep, keep_but_note, suspect, exclude_from_primary, exclude_from_all, or unresolved.
- Fill `review_reason_code` with the closest source/unit/temperature/pairing reason.
- Use `apply_to_scope` to distinguish row-only, sample-level, paper-level, or source-curve-level decisions.
- Step7B should read the completed decision template and create primary/sensitivity analysis flags.

## Readiness Update

| item | status | value | comment | next_action |
| --- | --- | --- | --- | --- |
| coverage_is_high | pass | 0.997895622895623 | Broad material_family default coverage. | record as supporting evidence |
| material_family_differs_from_global | pass | 1.0 | Checks whether broad grouping changes predictions. | record as supporting evidence |
| mae_improved_vs_original | pass | -0.1283360528809405 | Broad minus original MAE. | record as supporting evidence |
| factor2_improved_vs_original | pass | 0.1003501742547003 | Broad minus original factor2. | record as supporting evidence |
| robust_mae_remains_improved_after_excluding_extreme_outliers | pass | 0.6507679626856471 | Robustness after removing severe outliers. | record as supporting evidence |
| not_dominated_by_single_sample_abs_error | pass | 0.039417228095073586 | Top sample share of absolute error. | record as supporting evidence |
| not_dominated_by_single_paper_abs_error | pass | 0.039417228095073586 | Top paper share of absolute error. | record as supporting evidence |
| extreme_outliers_exist | caution | 13 | Extreme outliers are expected to be audited, not treated as fatal. | use Step7A decision template for human review |
| manual_review_needed | caution | 14.570733133464506 | Manual review of shortlist is required before final reporting. | use Step7A decision template for human review |
| recommended_next_action | caution | Use broad_family as a main candidate only with explicit outlier audit; review shortlist and consider reporting robust metrics alongside no-filter metrics. | Next decision point. | use Step7A decision template for human review |
| manual_review_packet_created | pending_review | 110 | Manual review cases are prepared but not adjudicated. | fill step7a_review_decisions_template.csv, then run Step7B |

## Warnings

- optional step0 not found: data\processed\step0_te_analysis_base.parquet

## Sanity Checks

- required_outputs_exist: True
- manual_review_master_created: True
- review_case_id_unique: True
- review_priority_not_missing: True
- review_status_initial_pending: True
- decision_template_created: True
- decision_template_human_columns_exist: True
- source_traceability_table_created: True
- sample_context_created: True
- casebook_created: True
- packet_index_created: True
- readiness_update_created: True
- report_created: True
- did_not_compute_new_sigma_pred: True
- did_not_read_step4_full_data_reference_curve: True
- did_not_read_raw_data: True
- did_not_auto_exclude_outliers: True

## Notes

- Step7A does not automatically exclude outliers.
- Step7A does not compute new sigma predictions.
- Step7A does not create figures.
- Step7A is a packet for human source verification, not a final research decision.

- elapsed_seconds: 0.97
