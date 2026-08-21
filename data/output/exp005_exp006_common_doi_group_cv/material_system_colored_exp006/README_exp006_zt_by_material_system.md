# exp006 common DOI-CV ZT by material system

- Evaluation: sample-maximum reported ZT versus exp006 out-of-fold predicted ZT.
- Samples: 2,881.
- Step31-classified samples: 2,621; unclassified: 260.
- Material-system labels shown: 16.
- Source prediction: common DOI-group 5-fold CV strict shared cohort.
- No ZT<1 cutoff is applied.
- Color and marker shape both encode material system.

## Interpretation caution

- Groups with small n have unstable correlations and high-ZT metrics; consult the metrics CSV before comparing systems.
- `Other formula systems`, `Unknown material group`, and `Unclassified` are retained rather than silently removed.
- These categories describe the evaluation samples. The current exp006 Step5A reference key itself is mostly `unknown_material_family`, so this plot is a post-hoc performance breakdown, not evidence that exp006 trained a separate model for every displayed material system.
