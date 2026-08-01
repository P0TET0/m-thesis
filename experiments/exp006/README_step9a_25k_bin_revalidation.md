# Step9A: 25 K temperature-bin broad-family revalidation

Step9A reuses the existing Step6A broad-family validation rows and the existing
Step5B/Step5C prediction and evaluation scripts. The difference from the
existing 100 K broad-family analysis is the temperature-bin width: Step9A
rebuilds the temperature bins from `T_K` at 25 K width with a default start of
12.5 K.

This is a bin-width sensitivity analysis, not a new model. The train side alone
is used to construct `sigma0_ref(T)`, and the reference is then assigned to the
test side. Step4 full-data reference curves and Starrydata2 raw data are not
read. Existing 100 K outputs are left unchanged.

The original `T_bin_*` columns are preserved as `old_T_bin_*`, after which the
active `T_bin_*` columns are rebuilt. The complete input row set is written to:

- `data/processed/step9a_25k_bin_broad_family/step9a_25k_validation_rows_with_splits.csv`
- `data/processed/step9a_25k_bin_broad_family/step9a_25k_validation_rows_with_splits.parquet`

All Step5B, Step5C, comparison, and summary outputs are kept in the same
Step9A-specific processed directory. Reports are kept in
`reports/step9a_25k_bin_broad_family/`. Step9A creates no figures.

## Run

```powershell
python experiments/exp006/run_step9a_25k_bin_revalidation.py `
    --input experiments/exp006/data/processed/step6a_validation_rows_with_splits_key_broad_family.parquet `
    --output experiments/exp006/data/processed/step9a_25k_bin_broad_family `
    --report-dir experiments/exp006/reports/step9a_25k_bin_broad_family `
    --bin-width-k 25 `
    --bin-start-k 12.5 `
    --min-rows-per-bin 3 `
    --min-samples-per-bin 3 `
    --min-papers-per-bin 1 `
    --min-eval-rows 30 `
    --min-eval-samples 5 `
    --max-rows 5000 `
    --max-rows-per-config 200
```

By default, the driver runs and checks a small Step5B/Step5C test before the
full run. `--skip-small-test` is available only for an intentional rerun that
does not need this gate. Small-test files use `_test` by default and cannot
overwrite the full outputs.

## Check

```powershell
python experiments/exp006/check_step9a_25k_bin_revalidation.py `
    --output experiments/exp006/data/processed/step9a_25k_bin_broad_family `
    --validation-rows experiments/exp006/data/processed/step9a_25k_bin_broad_family/step9a_25k_validation_rows_with_splits.csv `
    --metrics experiments/exp006/data/processed/step9a_25k_bin_broad_family/step5c_metrics_by_config.csv `
    --comparison experiments/exp006/data/processed/step9a_25k_bin_broad_family/step9a_25k_vs_100k_default_metrics_comparison.csv `
    --report experiments/exp006/reports/step9a_25k_bin_broad_family/step9a_25k_bin_revalidation_report.md
```
