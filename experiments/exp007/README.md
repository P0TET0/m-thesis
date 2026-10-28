# exp007 AROB study

exp007 is an isolated continuation of exp006 for the AROB conference study.
The immediate objectives are to diagnose electrical-conductivity prediction
errors and to test a finer, physically motivated classification of oxides.

exp006 is treated as a read-only source. New calculations and reports belong
under `experiments/exp007/`.

## Phase 0: freeze the comparison basis

Phase 0 compares three existing exp006 variants using the same default config:

```text
sample_holdout__ref_conservative_valid__eval_all_valid__material_family__sample_median
```

The variants are:

- `legacy_old_default`: poster-era result with the material key collapsed to one unknown group.
- `broad_family_100K`: 100 K broad-family result from Step6B.
- `broad_family_25K`: 25 K broad-family result from Step9A.

The frozen roles for the next phase are:

- Primary baseline: `broad_family_100K`.
- Temperature-bin sensitivity: `broad_family_25K`.
- Legacy reproduction reference: `legacy_old_default`.

The 100 K version is primary because the oxide-subclassification experiment
should first change only the material taxonomy. Combining a finer oxide
taxonomy with 25 K bins in the primary experiment would increase reference-bin
sparsity and make the cause of any change harder to identify.

## Run

```powershell
python experiments/exp007/build_phase0_baseline_comparison.py
python experiments/exp007/check_phase0_baseline_comparison.py
```

Both scripts use CSV inputs so the run does not depend on a Parquet engine.

## Phase 0 outputs

Data:

```text
experiments/exp007/data/processed/phase0_baseline_metrics.csv
experiments/exp007/data/processed/phase0_prediction_status_summary.csv
experiments/exp007/data/processed/phase0_reference_validation.csv
experiments/exp007/data/processed/phase0_common_row_metrics.csv
experiments/exp007/data/processed/phase0_common_row_predictions.csv
experiments/exp007/data/processed/phase0_metric_deltas_vs_100K.csv
experiments/exp007/data/processed/phase0_baseline_decision.csv
experiments/exp007/data/processed/phase0_input_manifest.csv
```

Reports:

```text
experiments/exp007/reports/phase0_baseline_comparison.md
experiments/exp007/reports/phase0_validation_log.txt
experiments/exp007/reports/phase0_check_log.txt
```

The build recomputes row-equal and sample-equal log10 metrics directly from
prediction rows. It validates those values against the existing Step5C
summaries and records SHA-256 hashes for all exp006 inputs.

## Phase 1: build the current-prediction residual data

Phase 1 uses the frozen `broad_family_100K` primary baseline and preserves all
19,008 held-out test rows. Residual values are calculated only when the
prediction status is `ok` and both observed and predicted conductivity are
finite and positive.

The signed residual is defined as:

```text
residual_log10_pred_over_observed = log10(sigma_predicted / sigma_observed)
```

A positive value means overprediction and a negative value means
underprediction. Each evaluable row is also assigned to a factor-2, factor-5,
factor-10, or beyond-factor-10 error band.

### Run

```powershell
python experiments/exp007/build_phase1_residual_dataset.py
node experiments/exp007/build_phase1_residual_workbook.mjs
python experiments/exp007/check_phase1_residual_dataset.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase1_sigma_residual_rows.csv
experiments/exp007/data/processed/phase1_sigma_sample_summary.csv
experiments/exp007/data/processed/phase1_sigma_material_group_summary.csv
experiments/exp007/data/processed/phase1_sigma_temperature_bin_summary.csv
experiments/exp007/data/processed/phase1_sigma_paper_summary.csv
experiments/exp007/data/processed/phase1_sigma_outlier_rows.csv
experiments/exp007/reports/phase1_residual_dataset.md
experiments/exp007/reports/phase1_sigma_residual_analysis.xlsx
```

The Excel workbook is a reader-facing view of the CSV outputs. The CSV files
remain the canonical machine-readable data for subsequent analysis.

## Phase 2: visualize the residual patterns

Phase 2 reads the Phase 1 residual table and creates four reproducible figure
sets without changing the model or taxonomy:

1. Global residual distribution, parity, residual-versus-observed behavior,
   and error-magnitude bands.
2. Broad material-group error, bias, and threshold accuracy.
3. Error and bias across the frozen 100 K temperature bins.
4. Broad-oxide residual comparisons and relationships with temperature and
   absolute Seebeck coefficient.

Each figure is written as a 300 dpi PNG and a one-page PDF under:

```text
experiments/exp007/figures/phase2/
```

The plotting script requires `matplotlib` in addition to `pandas` and `numpy`.

```powershell
python -m pip install matplotlib
python experiments/exp007/build_phase2_residual_visualizations.py
python experiments/exp007/check_phase2_residual_visualizations.py
```

Supporting summaries and validation files are written under
`experiments/exp007/data/processed/` and `experiments/exp007/reports/`.

## Phase 3: inventory the current broad-oxide cohort

Phase 3 audits the 3,437 held-out rows currently assigned to `broad::oxide`
without changing the model or the exp006 classification. The cohort contains
610 samples, 439 papers, and 506 normalized material labels. Every row entered
the current broad-oxide group because the Step6A parser found element `O` in
`formula_raw`.

The inventory adds a provisional composition-only split to expose group size,
residual behavior, and classification risks. It is not the final oxide
taxonomy. In particular, crystal-structure labels such as perovskite, layered
cobaltite, and delafossite require explicit rules or manual metadata review.

### Run

```powershell
python experiments/exp007/build_phase3_oxide_inventory.py
node experiments/exp007/build_phase3_oxide_inventory_workbook.mjs
python experiments/exp007/check_phase3_oxide_inventory.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase3_oxide_row_inventory.csv
experiments/exp007/data/processed/phase3_oxide_sample_inventory.csv
experiments/exp007/data/processed/phase3_oxide_label_inventory.csv
experiments/exp007/data/processed/phase3_oxide_review_queue.csv
experiments/exp007/data/processed/phase3_oxide_candidate_family_summary.csv
experiments/exp007/data/processed/phase3_oxide_element_summary.csv
experiments/exp007/data/processed/phase3_oxide_temperature_support.csv
experiments/exp007/reports/phase3_oxide_inventory.md
experiments/exp007/reports/phase3_oxide_inventory.xlsx
```

The review queue contains 184 samples. It includes 18 likely organic/polymer
O-containing false-positive candidates and 54 samples containing O together
with S, Se, or Te. The CSV files remain the canonical machine-readable outputs;
the workbook is the reader-facing audit view.

## Phase 4: define oxide taxonomy v1

Phase 4 freezes a reproducible, priority-ordered classification named
`oxide_v1` and applies it to all 3,437 Phase 3 broad-oxide rows. The definition
first separates organic/polymer/carbon-composite entries, then mixed-anion
systems, and finally ordinary O-containing compositions by selected element
markers. It assigns no crystal-structure labels from formula text.

The applied cohort contains 610 samples. Of these, 553 are eligible by default,
36 require review, and 21 are marked for exclusion from the inorganic-oxide v1
scope. These assignments define the comparison groups for later revalidation;
they are not post-retraining performance results.

### Run

```powershell
python experiments/exp007/build_phase4_oxide_taxonomy_v1.py
node experiments/exp007/build_phase4_oxide_v1_workbook.mjs
python experiments/exp007/check_phase4_oxide_taxonomy_v1.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase4_oxide_v1_classification_config.json
experiments/exp007/data/processed/phase4_oxide_v1_rules.csv
experiments/exp007/data/processed/phase4_oxide_v1_class_summary.csv
experiments/exp007/data/processed/phase4_oxide_v1_row_assignments.csv
experiments/exp007/data/processed/phase4_oxide_v1_sample_assignments.csv
experiments/exp007/data/processed/phase4_oxide_v1_label_assignments.csv
experiments/exp007/data/processed/phase4_oxide_v1_review_queue.csv
experiments/exp007/data/processed/phase4_oxide_v1_transition_from_phase3.csv
experiments/exp007/reports/phase4_oxide_v1_definition.md
experiments/exp007/reports/phase4_oxide_v1_definition.xlsx
```

The CSV and JSON files are canonical machine-readable outputs. The Excel file
is a reader-facing review book containing the ordered rules, assignments,
review queue, Phase 3 crosswalk, and validation evidence.

## Phase 5: reaggregate existing predictions by oxide_v1

Phase 5 joins the frozen Phase 1 `broad_family_100K` held-out conductivity
predictions to the Phase 4 `oxide_v1` assignments. It does not retrain or alter
the model. Metrics are recomputed for each new group using both row-equal and
sample-equal weighting. Sample-equal metrics use one median signed residual per
validation sample before calculating the group result.

The primary analysis contains the 553 default-eligible samples. Review and
excluded samples remain available as separate sensitivity scopes. Supporting
tables also split the primary cohort by carrier type and the existing 100 K
temperature bins.

### Run

```powershell
python experiments/exp007/build_phase5_oxide_v1_reaggregation.py
node experiments/exp007/build_phase5_oxide_v1_reaggregation_workbook.mjs
python experiments/exp007/check_phase5_oxide_v1_reaggregation.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase5_oxide_v1_overall_scope_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_primary_group_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_group_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_carrier_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_temperature_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_sample_metrics.csv
experiments/exp007/data/processed/phase5_oxide_v1_outlier_samples.csv
experiments/exp007/data/processed/phase5_oxide_v1_existing_prediction_rows.csv
experiments/exp007/reports/phase5_oxide_v1_reaggregation.md
experiments/exp007/reports/phase5_oxide_v1_reaggregation.xlsx
```

The CSV files are the canonical analysis tables. The Excel file is the
reader-facing view of the pooled, group, carrier, temperature, outlier, sample,
row, and validation results.

## Phase 6: re-predict after changing only the oxide classification

Phase 6 replaces `broad::oxide` with the frozen `oxide_v1` group keys before
rebuilding the train-only conductivity reference curves. All other Phase 0
conditions remain fixed: sample holdout assignments, 100 K temperature bins,
conservative-valid reference rows, all-valid evaluation rows, sample-median
aggregation, and the 3 rows / 3 samples / 1 paper reliability thresholds.

Review and exclude labels are retained as separate predictive groups in the
main controlled experiment. This avoids changing both the taxonomy and the
evaluation cohort at the same time. Default-eligible, review-only, and
excluded-only oxide cohorts are reported separately as sensitivity views.

Metrics are reported both on each variant's available predictions and on the
paired common rows predicted by both variants. The paired view isolates the
classification-induced prediction change from the effect of lost coverage.

### Run

```powershell
python experiments/exp007/build_phase6_oxide_v1_reprediction.py
node experiments/exp007/build_phase6_oxide_v1_reprediction_workbook.mjs
python experiments/exp007/check_phase6_oxide_v1_reprediction.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase6_oxide_v1_all_row_assignments.csv
experiments/exp007/data/processed/phase6_oxide_v1_train_reference_bins.csv
experiments/exp007/data/processed/phase6_oxide_v1_test_predictions.csv
experiments/exp007/data/processed/phase6_prediction_comparison_rows.csv
experiments/exp007/data/processed/phase6_overall_metrics.csv
experiments/exp007/data/processed/phase6_paired_metrics.csv
experiments/exp007/data/processed/phase6_metric_deltas.csv
experiments/exp007/data/processed/phase6_oxide_scope_metrics.csv
experiments/exp007/data/processed/phase6_oxide_group_metrics.csv
experiments/exp007/data/processed/phase6_oxide_group_coverage.csv
experiments/exp007/data/processed/phase6_sample_prediction_changes.csv
experiments/exp007/reports/phase6_oxide_v1_reprediction.md
experiments/exp007/reports/phase6_oxide_v1_reprediction.xlsx
```

The CSV files are canonical. The workbook is a reader-facing comparison of
coverage, overall and paired metrics, oxide scopes and groups, sample changes,
row predictions, assignments, reference support, and validation evidence.

## Phase 7: evaluate before and after the classification change

Phase 7 evaluates the frozen Phase 6 predictions without retraining or changing
the taxonomy. The primary cohort is the default-eligible oxide rows predicted
by both the broad-oxide baseline and oxide_v1. Paired row-equal metrics measure
the direct prediction change, while sample-equal metrics prevent samples with
many temperature rows from dominating the conclusion.

Uncertainty is summarized with a deterministic 2,000-replicate paired
bootstrap that resamples held-out validation samples as clusters. Coverage is
evaluated separately on all available rows so missing or unreliable fine-group
reference bins are not hidden by paired-row filtering. Descriptive breakdowns
are provided by oxide_v1 group, carrier type, 100 K temperature bin, and
observed conductivity decade.

### Run

```powershell
python experiments/exp007/build_phase7_classification_evaluation.py
node experiments/exp007/build_phase7_classification_evaluation_workbook.mjs
python experiments/exp007/check_phase7_classification_evaluation.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase7_paired_metric_evaluation.csv
experiments/exp007/data/processed/phase7_bootstrap_intervals.csv
experiments/exp007/data/processed/phase7_oxide_group_evaluation.csv
experiments/exp007/data/processed/phase7_stratified_evaluation.csv
experiments/exp007/data/processed/phase7_primary_sample_changes.csv
experiments/exp007/data/processed/phase7_coverage_loss_rows.csv
experiments/exp007/data/processed/phase7_coverage_loss_summary.csv
experiments/exp007/figures/figure_phase7_primary_metrics_before_after.png
experiments/exp007/figures/figure_phase7_group_mae_coverage_tradeoff.png
experiments/exp007/figures/figure_phase7_sample_error_changes.png
experiments/exp007/reports/phase7_classification_evaluation.md
experiments/exp007/reports/phase7_classification_evaluation.xlsx
```

The CSV files are canonical. The workbook is the reader-facing view of the
decision summary, paired metrics, bootstrap intervals, oxide-group results,
stratified results, sample changes, coverage losses, and validation evidence.

## Phase 8: robustness checks

Phase 8 rebuilds both the broad-oxide baseline and oxide_v1 under identical
alternative settings. It varies the temperature-bin width (50, 100, 150, and
200 K), reference-support thresholds (1/1/1, frozen 3/3/1, and strict 5/5/2
rows/samples/papers), and one aggregation alternative (row median at the frozen
100 K support setting). The frozen sample holdout, source subsets, and oxide_v1
rules do not change.

The evaluation also tests row-, sample-, and paper-equal weighting, symmetric
removal of the most extreme paired rows, and omission of each oxide group in
turn. These are internal sensitivity checks on the same dataset and are not an
external-validation or DOI-held-out experiment.

### Run

```powershell
python experiments/exp007/build_phase8_robustness_check.py
node experiments/exp007/build_phase8_robustness_workbook.mjs
python experiments/exp007/check_phase8_robustness.py
```

The Node.js workbook step uses the bundled `@oai/artifact-tool` package.

### Main outputs

```text
experiments/exp007/data/processed/phase8_model_sensitivity_predictions.csv
experiments/exp007/data/processed/phase8_model_sensitivity_metrics.csv
experiments/exp007/data/processed/phase8_evaluation_sensitivity.csv
experiments/exp007/data/processed/phase8_leave_one_group_out.csv
experiments/exp007/data/processed/phase8_group_direction_consistency.csv
experiments/exp007/data/processed/phase8_decision_summary.csv
experiments/exp007/figures/figure_phase8_model_sensitivity.png
experiments/exp007/figures/figure_phase8_evaluation_sensitivity.png
experiments/exp007/figures/figure_phase8_leave_one_group_out.png
experiments/exp007/reports/phase8_robustness_check.md
experiments/exp007/reports/phase8_robustness_check.xlsx
```

The row-level sensitivity CSV is the canonical audit trail. The workbook is a
compact reader-facing summary of model sensitivity, evaluation sensitivity,
leave-one-group-out results, group consistency, and validation evidence.

## Phase 9: synthesize the research result

Phase 9 does not fit a new model. It consolidates the frozen Phase 3-8
evidence into a research narrative with traceable claims, quantitative
support, group-level exceptions, limitations, a bilingual abstract, and one
conference-ready summary figure. The prior thermoelectric-society poster is
used only to align the motivation and terminology; it is not modified.

The principal conclusion is that the formula-only oxide_v1 taxonomy lowers
paired held-out conductivity MAE and RMSE by about 14% relative to one broad
oxide group. The direction is stable across the predefined internal
sensitivity checks. The synthesis also preserves the coverage penalty, the
inconclusive factor-2 interval, the Mixed-anion O + As/P exception, and the
absence of DOI-held-out or external validation.

### Run

```powershell
python experiments/exp007/build_phase9_research_summary.py
python experiments/exp007/check_phase9_research_summary.py
```

### Main outputs

```text
experiments/exp007/data/processed/phase9_key_results.csv
experiments/exp007/data/processed/phase9_claim_evidence_matrix.csv
experiments/exp007/data/processed/phase9_group_result_summary.csv
experiments/exp007/data/processed/phase9_study_flow.csv
experiments/exp007/figures/figure_phase9_research_summary.png
experiments/exp007/reports/phase9_research_results_summary.md
experiments/exp007/reports/phase9_abstract_ja_en.md
```

The claim-evidence matrix is the recommended source for deciding which
statements can be used in a poster, abstract, paper, or oral presentation.

## Supplementary conductivity parity plots (2026-10-16)

The explanation figures and their builder now live in exp007. Copies from the
original conversation visualization folder remain there; use the exp007 copies
for research work. The date in the filenames identifies the original diagnostic,
not a new prediction run.

Run from the repository root:

```powershell
python experiments/exp007/explain_oxide_results_20261016.py
```

This script reads `data/processed/phase6_prediction_comparison_rows.csv`, checks
agreement with the saved Phase 7 MAE and exp006 agreement metrics, and draws the
existing predictions. It does not refit reference values or train a new model.
The script requires numpy, pandas, and matplotlib.

Outputs relative to exp007:

```text
figures/oxide_parity_before_after_20261016.png
figures/oxide_parity_by_group_20261016.png
figures/oxide_parity_by_group_pn_20261017.png
```

- **Before (grey)**: predictions using one broad oxide group, with reference
  conductivity prefactors still separated by carrier type and 100 K temperature bin.
- **After (teal)**: Phase 6 predictions after replacing the broad oxide key with
  the frozen composition-based oxide_v1 keys and rebuilding training reference
  values for those keys. Non-classification settings remain fixed.
- Both variants use the same 2,984 default-eligible, jointly predictable rows from
  552 held-out samples. Every panel compares the same samples at the same temperatures.
- The 12 panels display both variants using the new subgroup labels. The grey
  points are not predictions from 12 separately fitted baseline groups; they are
  the old broad-oxide predictions partitioned only for display.
- This is not a before/after comparison of model training versus testing, nor
  an exp005 ML versus exp006 FML comparison.

Agreement metrics are printed to the terminal; the script does not write an
additional metrics CSV. Canonical Phase 6-9 tables remain unchanged.

### Additional p/n-colored view (2026-10-17)

`figures/oxide_parity_by_group_pn_20261017.png` retains the same 12 panels,
2,984 paired rows, group ordering, and full-range axes as the original subgroup
figure. It uses the stored `carrier_type` field: orange for p-type and blue for
n-type. Crosses indicate Before (broad-oxide reference), while filled circles
indicate After (oxide_v1 reference). This color key differs from the original
grey/teal figure. Each panel lists its p/n row counts, including zero when one
type is absent. The displayed MAE and RMSE still combine both carrier types and
show Before -> After; they are not carrier-specific metrics. The builder checks
that both prediction variants draw all 2,984 rows without unrecognized types.
No predictions, classification rules, or train/test assignments are changed.
