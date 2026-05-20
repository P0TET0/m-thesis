# Step25 Paper Output Notes

## Purpose
Step25 organizes Step12-24 outputs into paper-ready tables, figure data, figures, text drafts, and caution statements.

## Files Generated
The output folder contains paper_table_01-11, paper_figure_data_01-06, markdown drafts, key claims/cautions, a report, notes, and an Excel workbook.

## Main Results to Report
Report tau_eff fitting, sigma validation, PF/ZT fitting workflow performance, ML log_tau_eff performance, fitting-vs-ML comparison, error analysis, and candidate material screening.

## How to Use the Tables
Use tables 01-08 for methods/results context, table 09 for candidate materials, and tables 10-11 for manual review queues.

## How to Use the Figures
Use figure data CSVs for reproducible plotting. PNG files are draft figures and can be restyled for the thesis.

## Candidate Material Interpretation
Candidate ranks are screening priorities, not final material recommendations.

## Important Caveats
tau_eff is relative scale, not physical seconds.
S and kappa were not predicted.
PF/ZT use predicted sigma and observed S/kappa.
direct fitting uses sigma observations and is expected to outperform ML.
rare-metal/toxicity labels are provisional.
downstream ML predictions are for screening, not unbiased evaluation.
Step25 does not perform new prediction, tau_eff refitting, PF/ZT recalculation, or ML retraining.

## Recommended Next Actions
Inspect paper_table_09_candidate_materials_step25.csv and paper_key_claims_and_cautions_step25.csv, then manually review high-priority candidate materials before final claims.
