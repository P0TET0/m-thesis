"""Read-only element-feature impact audit. Never fits or predicts a model.
Without --output-dir, print aggregate JSON only.
With --output-dir, create a NEW directory; refuse an existing directory.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse
import ast
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "metadata": "data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_metadata_step18.csv",
    "features": "data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_feature_matrix_step18.csv",
    "recommended": "data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_dataset_recommended_step18.csv",
    "splits": "data/output/starrydata2_step18_tau_eff_ml_dataset/tau_eff_ml_splits_step18.csv",
    "old_columns": "data/output/starrydata2_step19_tau_eff_ml_model/best_model_feature_columns_step19.json",
    "common_samples": "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/three_method_sample_summary.csv",
    "common_rows": "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/three_method_row_comparisons.csv",
    "common_metrics": "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/three_method_model_metrics.csv",
    "fig4": "data/output/df_ml_fml_material_parity_20261025/plot_source_ML.csv",
    "fig7_samples": "experiments/exp011_oxide_v1_df_ml_fml/data/ml_sample_audit_fine.csv",
    "fig7_plot": "experiments/exp011_oxide_v1_df_ml_fml/data/plot_source_fine.csv",
    "fig7_predictions": "experiments/exp011_oxide_v1_df_ml_fml/data/predictions_fine.csv",
    "fig7_metrics": "experiments/exp011_oxide_v1_df_ml_fml/data/metrics.csv",
    "fig7_feature_audit": "experiments/exp011_oxide_v1_df_ml_fml/data/feature_audit_fine.csv",
    "fig7_importance": "experiments/exp011_oxide_v1_df_ml_fml/data/feature_importance_fine.csv",
    "fig7_model": "experiments/exp011_oxide_v1_df_ml_fml/models/ml_fine.joblib",
    "fig7_manifest": "experiments/exp011_oxide_v1_df_ml_fml/source_manifest.json",
    "step18_code": "experiments/exp005/build_tau_eff_ml_dataset_step18.py",
    "step19_code": "experiments/exp005/train_tau_eff_ml_step19.py",
    "fig7_code": "experiments/exp011_oxide_v1_df_ml_fml/build_oxide_comparison.py",
    "fig2_png": "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/figures/figure_three_method_parity_sigma_common_2K_cohort.png",
    "fig6_png": "data/output/exp005_ml_df_exp006_common_cohort_zt_comparison/figures/figure_three_method_parity_reported_zt_2K.png",
    "fig4_png": "data/output/df_ml_fml_material_parity_20261025/sigma/ML/overview_full_range.png",
    "fig7_png": "experiments/exp011_oxide_v1_df_ml_fml/figures/sigma/all_oxides_full_range.png",
}
PERIODIC_SYMBOLS = set(("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi "
    "Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn "
    "Nh Fl Mc Lv Ts Og").split())
assert len(PERIODIC_SYMBOLS) == 118
TOKEN = re.compile(r"([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?|\.[0-9]+)?")
GROUP_CONSTANTS = {
    "has_chalcogen_step18": "CHALCOGEN",
    "has_pnictogen_step18": "PNICTOGEN",
    "has_transition_metal_step18": "TRANSITION_METAL",
    "has_alkali_or_alkaline_step18": "ALKALI_OR_ALKALINE",
    "has_heavy_element_step18": "HEAVY_ELEMENT",
    "has_toxic_attention_element_step18": "TOXIC_ATTENTION",
    "has_rare_metal_attention_element_step18": "RARE_METAL_ATTENTION",
}
TRACKED = ["C","Bi","Te","Se","Sb","Pb","Sn","Ge","Si","Mg","Co","Fe","Cu","Ag","In","Ga","Ca","Na","K","O","N"]
ELEMENT_FEATURES = ["element_count_step18"] + [f"contains_{e}_step18" for e in TRACKED] + list(GROUP_CONSTANTS)
COHORT_LABELS = {
    "old_candidates": "旧ML候補",
    "old_doi_train": "旧ML DOI学習",
    "old_doi_test": "旧ML DOI評価",
    "fig2_fig6_common": "図2・6共通",
    "fig4_published": "図4掲載",
    "fig7_ml_train": "図7 ML学習（全材料）",
    "fig7_published": "図7 ML掲載",
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def read(name, **kwargs):
    return pd.read_csv(ROOT / SOURCES[name], low_memory=False, **kwargs)

def truth(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])

def literal_assignments(path, names):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(node.value)
    assert set(out) == set(names)
    return out

def strict_formula(raw):
    """Conservative, fully consumed flat formula grammar; no partial regex match."""
    text = str(raw).strip()
    if text == "" or text.lower() in {"unknown", "undefined", "nan", "none", "null"}:
        return "review", "missing_or_unspecified", None, None
    if any(c in text for c in "()[]{}"):
        return "review", "grouped_formula_or_component_not_adjudicated", None, None
    if re.search(r"[/\\+@:,;%_·∙•]", text):
        return "review", "composite_separator_or_annotation", None, None
    if re.search(r"\s", text):
        return "review", "internal_whitespace_or_description", None, None
    if not re.fullmatch(r"[A-Za-z0-9.]+", text):
        return "review", "variable_charge_or_unsupported_character", None, None
    pos = 0
    amounts = {}
    while pos < len(text):
        match = TOKEN.match(text, pos)
        if match is None or match.group(1) not in PERIODIC_SYMBOLS:
            return "review", "unconsumed_or_invalid_element_token", None, None
        symbol = match.group(1)
        try:
            amount = Decimal(match.group(2)) if match.group(2) is not None else Decimal(1)
        except InvalidOperation:
            return "review", "invalid_numeric_coefficient", None, None
        amounts[symbol] = amounts.get(symbol, Decimal(0)) + amount
        pos = match.end()
    present = {e for e, amount in amounts.items() if amount > 0}
    zero = {e for e, amount in amounts.items() if amount == 0}
    if not present:
        return "review", "all_coefficients_zero", None, None
    return "clear", "fully_parsed_flat_formula", present, zero

def old_extract(raw, constants, regex):
    # Exact Step18 boundary behavior, including before.isalpha()/after.islower().
    if pd.isna(raw):
        return set()
    text = str(raw)
    found = set()
    for match in regex.finditer(text):
        start, end = match.span()
        before = text[start-1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        if before.isalpha() or after.islower():
            continue
        found.add(match.group(0))
    return found

def element_features(elements, constants):
    out = {"element_count_step18": len(elements)}
    out.update({f"contains_{e}_step18": int(e in elements) for e in TRACKED})
    out.update({feature: int(bool(elements & constants[name])) for feature, name in GROUP_CONSTANTS.items()})
    return out

def json_default(value):
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.bool_): return bool(value)
    if isinstance(value, set): return sorted(value)
    raise TypeError(type(value).__name__)

def main(output_dir=None):
    before = {name: digest(ROOT / rel) for name, rel in SOURCES.items()}
    constants = literal_assignments(ROOT / SOURCES["step18_code"], {"ELEMENT_SYMBOLS", *GROUP_CONSTANTS.values()})
    assert set(constants["ELEMENT_SYMBOLS"]) == PERIODIC_SYMBOLS
    old_regex = re.compile("|".join(sorted(constants["ELEMENT_SYMBOLS"], key=len, reverse=True)))
    for formula, expected in [("CoSb3", {"Co","Sb"}), ("SrTiO3", {"Sr","Ti","O"}),
                              ("Bi2Te3", {"Bi","Te"}), ("C60", {"C"}), ("Bi0Te", {"Te"})]:
        assert strict_formula(formula)[2] == expected
    for formula in ["(Bi0.25Sb0.75)2Te3", "C@PbTe", "Bi2Te3/Sb2Te3", "Bi2 Te3", "CoSb3-x", "polyaniline", "TiX2"]:
        assert strict_formula(formula)[0] == "review"
    assert old_extract("CoSb3", constants, old_regex) == {"Co"}
    metadata = read("metadata", keep_default_na=False, usecols=["sample_key","composition"])
    feature = read("features")
    assert not metadata.sample_key.duplicated().any()
    assert not feature.sample_key.duplicated().any()
    assert set(metadata.sample_key) == set(feature.sample_key)
    base = metadata.merge(feature, on="sample_key", how="left", validate="one_to_one")
    old_columns = json.loads((ROOT / SOURCES["old_columns"]).read_text(encoding="utf-8-sig"))
    assert len(old_columns) == 49 and set(ELEMENT_FEATURES) <= set(old_columns)
    candidate = set(read("recommended", usecols=["sample_key"]).sample_key)
    split = read("splits").set_index("sample_key")
    sample7 = read("fig7_samples")
    assert not sample7.sample_key.duplicated().any()
    train7 = truth(sample7.used_to_train_ML)
    fig4 = read("fig4")
    plot7 = read("fig7_plot")
    plotted7 = plot7[plot7.method.eq("ML") & plot7.broad_group.eq("broad::oxide")
                    & np.isfinite(plot7.sigma_obs) & plot7.sigma_obs.gt(0)
                    & np.isfinite(plot7.sigma_pred) & plot7.sigma_pred.gt(0)]
    common = read("common_samples")
    cohorts = {
        "old_candidates": candidate,
        "old_doi_train": candidate & set(split.index[split.split_doi_group_80_20_step18.eq("train")]),
        "old_doi_test": candidate & set(split.index[split.split_doi_group_80_20_step18.eq("test")]),
        "fig2_fig6_common": set(common.sample_key),
        "fig4_published": set(fig4.sample_key),
        "fig7_ml_train": set(sample7.loc[train7, "sample_key"]),
        "fig7_published": set(plotted7.sample_key),
    }
    assert [len(cohorts[k]) for k in list(cohorts)[:5]] == [15623,12422,3201,109,2943]
    assert len(cohorts["fig7_published"]) == 280 and len(plotted7) == 1547
    assert all(keys <= set(base.sample_key) for keys in cohorts.values())
    assert cohorts["old_doi_train"].isdisjoint(cohorts["old_doi_test"])
    assert cohorts["old_doi_train"] | cohorts["old_doi_test"] == candidate
    assert cohorts["fig2_fig6_common"] <= cohorts["old_doi_test"]
    assert cohorts["fig4_published"] <= cohorts["old_doi_test"]
    assert cohorts["fig7_ml_train"].isdisjoint(cohorts["fig7_published"])

    # Recompute ONLY the original model-input preparation; no model load/fit/predict.
    tree = ast.parse((ROOT / SOURCES["step19_code"]).read_text(encoding="utf-8-sig"))
    leak = literal_assignments(ROOT / SOURCES["step19_code"], {"LEAKAGE_PATTERNS"})
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"prepare_features","is_leakage_column"}]
    ns = {"pd":pd, "np":np, **leak}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<read-only original feature preparation>", "exec"), ns)
    samples = sample7.merge(feature, on="sample_key", how="left", validate="one_to_one")
    candidate_columns = [c for c in feature if c != "sample_key" and not c.startswith("material_system__")
                         and not ns["is_leakage_column"](c)]
    groups = sorted(samples.loc[train7, "material_group"].unique())
    for group in groups:
        col = f"material_category::{group}"
        samples[col] = samples.material_group.eq(group).astype(float)
        candidate_columns.append(col)
    _, recomputed_used, medians, _ = ns["prepare_features"](samples, candidate_columns, train7)
    saved_audit = read("fig7_feature_audit")
    saved_used = saved_audit.loc[truth(saved_audit.used_in_model_step19), "feature_name"].tolist()
    importance = read("fig7_importance")
    assert recomputed_used == saved_used == importance.feature.tolist()
    saved_medians = saved_audit.set_index("feature_name").imputation_value
    assert all(np.isclose(medians[c], saved_medians[c], rtol=0, atol=0) for c in saved_used)
    assert set(ELEMENT_FEATURES) <= set(saved_used)

    records = []
    mismatch_old = []
    changes_by_key = {}
    omission_changes_by_key = {}
    for row in base.to_dict("records"):
        key = row["sample_key"]
        old = old_extract(row["composition"], constants, old_regex)
        old_f = element_features(old, constants)
        if any(row[col] != val for col, val in old_f.items()):
            mismatch_old.append(key)
        status, reason, expected, zero = strict_formula(row["composition"])
        changes = {}
        omission_changes = {}
        missing = set()
        extra = set()
        if status == "clear":
            missing = expected - old
            extra = old - expected
            new_f = element_features(expected, constants)
            changes = {col: {"saved":int(row[col]), "formula":new_f[col]} for col in ELEMENT_FEATURES
                       if row[col] != new_f[col]}
            # Strict token grammar means any extra old element must have zero coefficient.
            assert extra <= zero
            # Isolate the reported omission mechanism; keep zero-coefficient
            # detections as-is in this secondary feature comparison.
            omission_f = element_features(old | missing, constants)
            omission_changes = {col: {"saved":int(row[col]), "with_missing_elements":omission_f[col]}
                                for col in ELEMENT_FEATURES if row[col] != omission_f[col]}
        changes_by_key[key] = changes
        omission_changes_by_key[key] = omission_changes
        rec = {"sample_key":key, "composition":row["composition"], "parse_status":status,
               "review_reason":reason, "old_step18_elements":";".join(sorted(old)),
               "formula_elements":";".join(sorted(expected)) if expected is not None else "",
               "zero_coefficient_elements":";".join(sorted(zero)) if zero is not None else "",
               "missing_elements":";".join(sorted(missing)), "extra_elements":";".join(sorted(extra)),
               "element_set_diff":bool(missing or extra), "has_omission":bool(missing),
               "has_extra":bool(extra), "affected_feature_count":len(changes),
               "affected_features":";".join(changes), "feature_changes_json":json.dumps(changes,ensure_ascii=False),
               "omission_affected_feature_count":len(omission_changes),
               "omission_affected_features":";".join(omission_changes)}
        rec.update({k:key in members for k,members in cohorts.items()})
        records.append(rec)
    assert not mismatch_old, mismatch_old[:10]
    assessments = pd.DataFrame(records)
    summaries, counts, reasons = [], [], []
    for name, members in cohorts.items():
        sub = assessments[assessments.sample_key.isin(members)]
        clear = sub.parse_status.eq("clear")
        affected = sub.affected_feature_count.gt(0)
        active_columns = saved_used if name.startswith("fig7") else old_columns
        active_diff_keys = {key for key in members if set(changes_by_key[key]) & set(active_columns)}
        summaries.append({"cohort":name,"label":COHORT_LABELS[name],"samples":len(sub),
            "clear_formula":int(clear.sum()),"review":int((~clear).sum()),
            "element_set_diff":int(sub.element_set_diff.sum()),"omission_samples":int(sub.has_omission.sum()),
            "extra_samples":int(sub.has_extra.sum()),
            "both_missing_and_extra":int((sub.has_omission & sub.has_extra).sum()),
            "affected_feature_samples":int(affected.sum()),"affected_active_input_samples":len(active_diff_keys),
            "different_elements_but_no_feature_diff":int((sub.element_set_diff & ~affected).sum()),
            "affected_feature_cells":int(sub.affected_feature_count.sum()),
            "affected_feature_columns":sum(any(col in changes_by_key[key] for key in members) for col in ELEMENT_FEATURES),
            "omission_affected_feature_columns":sum(any(col in omission_changes_by_key[key] for key in members) for col in ELEMENT_FEATURES),
            "diff_percent_all":100*sub.element_set_diff.sum()/len(sub),
            "diff_percent_clear":100*sub.element_set_diff.sum()/clear.sum() if clear.any() else None})
        for col in ELEMENT_FEATURES:
            entries = [changes_by_key[key][col] for key in members if col in changes_by_key[key]]
            counts.append({"cohort":name,"feature":col,"different_samples":len(entries),
                "omission_different_samples":sum(col in omission_changes_by_key[key] for key in members),
                "saved_less_than_formula":sum(x["saved"]<x["formula"] for x in entries),
                "saved_greater_than_formula":sum(x["saved"]>x["formula"] for x in entries),
                "retained_in_input":col in active_columns})
        for reason, number in sub.loc[~clear,"review_reason"].value_counts().items():
            reasons.append({"cohort":name,"review_reason":reason,"samples":int(number)})
    summary = pd.DataFrame(summaries)
    feature_counts = pd.DataFrame(counts)
    final_columns = saved_audit.copy()
    final_columns["step18_element_feature"] = final_columns.feature_name.isin(ELEMENT_FEATURES)
    for name in ["fig7_ml_train","fig7_published"]:
        mapping = feature_counts[feature_counts.cohort.eq(name)].set_index("feature").different_samples
        final_columns[name+"_different_samples"] = final_columns.feature_name.map(mapping).fillna(0).astype(int)
    changed_files = [name for name, old in before.items() if digest(ROOT / SOURCES[name]) != old]
    assert not changed_files
    results = {"rule":"strict_flat_formula_v1","comparison_unit":"unique sample_key",
        "step9_used_as_truth":False,"model_loaded":False,"model_fitted":False,"predictions_recomputed":False,
        "metadata_rows":len(base),"original_element_feature_mismatches":len(mismatch_old),
        "fig7_train_samples":int(train7.sum()),"fig7_training_feature_available":bool(truth(sample7.loc[train7,"features_available"]).all()),
        "fig7_published_points":len(plotted7),"fig7_old_base_feature_candidates":len([c for c in feature if c!="sample_key" and not c.startswith("material_system__")]),
        "fig7_material_category_candidates":len(groups),"fig7_total_candidates":len(candidate_columns),
        "fig7_final_columns":len(saved_used),"fig7_element_features_retained":len(set(ELEMENT_FEATURES)&set(saved_used)),
        "fig7_final_columns_match_recomputed_preprocessing":True,"fig7_final_columns_match_importance":True,
        "fig7_imputation_matches":True,"cohorts":summaries,"review_reasons":reasons,
        "input_hashes_before_and_after_equal":True,
        "input_manifest":[{"name":name,"path":SOURCES[name],"sha256":value} for name,value in before.items()]}
    if output_dir is not None:
        out = Path(output_dir).resolve()
        if out.exists():
            raise FileExistsError("Refusing to overwrite an existing output directory: "+str(out))
        out.mkdir(parents=True, exist_ok=False)
        for filename, table in [
            ("cohort_summary.csv",summary),("feature_difference_counts.csv",feature_counts),
            ("sample_assessment.csv",assessments),("review_reason_counts.csv",pd.DataFrame(reasons)),
            ("review_cases.csv",assessments[assessments.parse_status.eq("review")]),
            ("fig7_final_input_columns.csv",final_columns)]:
            with (out/filename).open("x",encoding="utf-8-sig",newline="") as handle:
                table.to_csv(handle,index=False)
        with (out/"audit_results.json").open("x",encoding="utf-8") as handle:
            json.dump(results,handle,ensure_ascii=False,indent=2,default=json_default)
    print(json.dumps(results,ensure_ascii=False,indent=2,default=json_default))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--output-dir",default=None,help="Must not already exist")
    main(args.parse_args().output_dir)
