# Step32: exp005 ML sigma prediction by exp006 material system

Step32 applies the existing exp005 ML method to electrical conductivity and
exports the results separately for every exp006 `broad_family` material system.

The calculation is:

```text
sigma_ML(T) = prefactor_C(T) * tau_eff_ML
```

The material classification is not copied or independently reimplemented.
`predict_sigma_ml_by_material_system_step32.py` directly calls
`experiments/exp006/build_step6a_material_group_keys.py::build_candidates`.

Two prediction scopes are kept separate:

- DOI-group holdout: the best Step19 model on the DOI test split; use this for
  unbiased evaluation.
- all-sample screening: the final Step19 model trained on all recommended ML
  samples; use this for material-system curves, not for unbiased evaluation.

## Run

```powershell
python experiments/exp005/predict_sigma_ml_by_material_system_step32.py
```

To replace an existing validated output directory:

```powershell
python experiments/exp005/predict_sigma_ml_by_material_system_step32.py --replace-output
```

## Outputs

Outputs are written to
`data/output/starrydata2_step32_sigma_ml_by_material_system/`.

- `sigma_ml_screening_predictions_by_material_system_step32.csv`
- `sigma_ml_doi_holdout_predictions_by_material_system_step32.csv`
- `sigma_ml_material_system_curves_step32.csv`
- `sigma_ml_metrics_by_material_system_step32.csv`
- `material_system_index_step32.csv`
- `sample_material_system_classification_step32.csv`
- `by_material_system/*.csv`
- `figures/material_curves/*.png`
- `figures/evaluation_scatter/*.png`
- `figures/sigma_ml_material_system_curves_grid_step32.png`
- `figures/sigma_ml_doi_holdout_scatter_all_step32.png`
- `figures/sigma_ml_doi_holdout_scatter_grid_step32.png`
- `step32_sigma_ml_by_material_system_report.txt`

The material-system curves are 25 K-bin medians across samples. They represent
the distribution of predictions within a family and are not a prediction for
one particular composition.
