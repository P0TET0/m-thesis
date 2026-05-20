# Step17 Manual Review Instructions

## Purpose
Step17 prepares a manual literature-review template for the important samples selected in Step16. It does not infer original-paper content automatically.

## Files to Open
Open `manual_annotation_template_step17.csv` or `manual_annotation_template_step17.xlsx` first. Use `DOI`, `paper_title`, `sample_id`, `composition`, and `doi_url` to locate the original paper.

## What to Check
Check paper evidence for additives, structure information, and n/p type. Record the result in the Step17 paper/manual columns and keep short evidence notes.

## What Not to Check
Do not create new predictions, refit tau_eff, recalculate PF/ZT, or infer paper contents without reading the source. Do not change existing auto-extracted columns directly.

## How to Fill the Template
Set `paper_checked_step17` to `yes` after checking a paper. Use `high`, `medium`, `low`, or `unknown` in confidence columns. Keep automatic Step9 and existing n/p columns unchanged; write paper-confirmed values only in Step17 manual columns.

## How to Re-run the Script After Filling
After editing, save the filled file as `manual_annotation_template_step17_filled.csv`. Re-run:

```bash
python prepare_step17_literature_review.py --manual_annotations data/output/starrydata2_step17_literature_review/manual_annotation_template_step17_filled.csv
```

## Notes for Sintering Check
Check sintering methods only for samples with `step17_check_sintering=yes`. You do not need to investigate sintering methods for all samples.

## Output Files
The main filled-output table is `step17_annotated_samples.csv`. The Step18 input candidate is `step17_tau_eff_ml_annotation_base.csv`. The `doi_url` column is placed near the end of CSV files so commas are less likely to disturb URL handling during manual editing.
