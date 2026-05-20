# Step9 literature review instructions

1. Open `manual_annotation_template_step9.csv`.
2. Open the paper from `doi_url` when available, or search by `paper_title`.
3. Check only additive/dopant/composite information, structure/morphology information, and explicit n/p-type evidence.
4. Do not check or fill sintering method in Step9.
5. Write dopants, additive elements, or composite components in `additive_manual_step9`.
6. Write nanostructure, thin film, bulk, grain size, CNT, graphene, layered structure, or related morphology in `structure_manual_step9`.
7. If the paper explicitly states the carrier type, write `n`, `p`, `mixed`, or `unknown` in `np_type_paper_manual_step9`.
8. Briefly write the paper evidence in `np_basis_paper_manual_step9`.
9. Set `paper_checked_step9` to `yes` after checking.
10. Save the completed file as `manual_annotation_template_step9_filled.csv`, then rerun this script with `--manual_annotations`.

Do not check sintering method at this stage. Check it later only for high-ZT samples, high-error samples, final-paper samples, or samples with the same composition but very different properties.
