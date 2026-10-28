# exp007 Phase 0 baseline comparison

## Scope

This phase freezes the comparison basis for the AROB study. It reads existing exp006 outputs, recomputes the default metrics, and does not modify exp006.

Default config: `sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median`

## Row-equal metrics on each available cohort

| variant | n_rows | n_samples | coverage_fraction | mae_log10 | rmse_log10 | mean_log10_error | factor_2_accuracy | factor_10_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_old_default | 19000 | 3191 | 0.999579 | 0.852082 | 1.406463 | 0.289601 | 0.326316 | 0.750842 |
| broad_family_100K | 18968 | 3189 | 0.997896 | 0.723746 | 1.250868 | 0.245359 | 0.426666 | 0.783794 |
| broad_family_25K | 18883 | 3178 | 0.993424 | 0.718479 | 1.243861 | 0.241316 | 0.429381 | 0.785415 |

## Sample-equal metrics on each available cohort

| variant | n_rows | n_papers | coverage_fraction | mae_log10 | rmse_log10 | mean_log10_error | factor_2_accuracy | factor_10_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_old_default | 3191 | 2202 | 0.999579 | 0.737234 | 1.234345 | 0.297590 | 0.350987 | 0.797242 |
| broad_family_100K | 3189 | 2200 | 0.997896 | 0.626433 | 1.120994 | 0.251542 | 0.481656 | 0.825651 |
| broad_family_25K | 3178 | 2193 | 0.993424 | 0.626379 | 1.120759 | 0.249211 | 0.479547 | 0.828194 |

## Prediction coverage and grouping context

| variant | test_rows | prediction_ok_rows | missing_reference_bin | unreliable_reference_bin | material_prediction_groups | temperature_bins_all_statuses |
| --- | --- | --- | --- | --- | --- | --- |
| legacy_old_default | 19008 | 19000 | 2 | 6 | 1 | 19 |
| broad_family_100K | 19008 | 18968 | 27 | 13 | 15 | 19 |
| broad_family_25K | 19008 | 18883 | 41 | 84 | 15 | 63 |

## Common-row comparison

The intersection contains 18,883 rows with predictions from all three variants.

| variant | n_rows | n_samples | mae_log10 | rmse_log10 | mean_log10_error | factor_2_accuracy | factor_10_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| legacy_old_default | 18883 | 3178 | 0.844991 | 1.391834 | 0.280622 | 0.326643 | 0.752211 |
| broad_family_100K | 18883 | 3178 | 0.718717 | 1.236983 | 0.239157 | 0.426627 | 0.784409 |
| broad_family_25K | 18883 | 3178 | 0.718479 | 1.243861 | 0.241316 | 0.429381 | 0.785415 |

## Baseline decision

- Primary baseline: `broad_family_100K`.
- Sensitivity baseline: `broad_family_25K`.
- Legacy reproduction reference: `legacy_old_default`.

The 100 K broad-family variant is selected for the oxide-subclassification experiment because it changes the material taxonomy without simultaneously imposing the sparsity of 25 K temperature bins. The 25 K variant remains a required sensitivity analysis because its row-equal errors are slightly lower, while its coverage is lower. The old/default variant is not suitable as the primary baseline because its material-family prediction key is collapsed to a single unknown group.

## Validation

- Existing Step5C values reproduced: True
- Maximum absolute difference from Step5C: 1.11e-16
- exp006 source files unchanged during the run: True
