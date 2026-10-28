"""Hypothetical cutoffs on limited formula subsets; no classification updates."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp011_composition_review.audit_composition_notation import MASTER, PLOTS, parse_flat

OUT = Path(__file__).resolve().parent
THRESHOLDS = ["0.01", "0.02", "0.05", "0.1", "0.2", "0.3", "0.5"]
EXAMPLES = ["Ti0.3Zr0.35Hf0.35NiSn", "Bi0.48Sb1.52Te3", "Bi0.5Sb1.5Te3", "Bi0.52Sb1.48Te3", "Na0.5CoO2", "Pb0.997Bi0.003Te"]


def summarize(items, capacity):
    fractions = [q for _, q in items]
    return {
        "samples": len(items),
        "unique_raw_formulas": len({r["composition"] for r, _ in items}),
        "exact_50_50_samples": sum(q == F(1, 2) for q in fractions),
        "normalized_pair_coefficient_sum": capacity,
        "minority_removal_by_coefficient_cutoff": {t: sum(capacity*q <= F(t) and q < F(1, 2) for q in fractions) for t in THRESHOLDS},
        "minority_removal_by_pair_fraction_cutoff": {t: sum(q <= F(t) and q < F(1, 2) for q in fractions) for t in THRESHOLDS},
        "sample_evidence": [dict(r, minority_pair_fraction=str(q)) for r, q in items],
    }


def main():
    data = pd.read_csv(MASTER, usecols=["sample_key", "sample_id", "DOI", "composition"], dtype=str).fillna("")
    assert data.sample_key.is_unique
    parsed = {s: parse_flat(s) for s in data.composition.unique()}
    sige, bst, ternary = [], [], []
    for row in data.to_dict("records"):
        raw = parsed[row["composition"]]
        if raw is None:
            continue
        a = {k: F(v) for k, v in raw.items()}
        if set(a) == {"Si", "Ge"}:
            sige.append((row, min(a.values()) / sum(a.values())))
        if set(a) == {"Bi", "Sb", "Te"}:
            pair = a["Bi"] + a["Sb"]
            if F("1.49") <= a["Te"] / pair <= F("1.51"):
                bst.append((row, min(a["Bi"], a["Sb"]) / pair))
        if {"Ti", "Zr", "Hf", "Ni", "Sn"}.issubset(a):
            total = sum(a[e] for e in ["Ti", "Zr", "Hf"])
            if max(a[e] for e in ["Ti", "Zr", "Hf"]) / total < F(1, 2):
                ternary.append(row)
    inputs, methods = [MASTER], {}
    for method in ["DF", "ML", "FML"]:
        path = PLOTS / f"plot_source_{method}.csv"
        inputs.append(path)
        methods[method] = set(pd.read_csv(path, usecols=["sample_key"], dtype=str).sample_key)
        assert methods[method] <= set(data.sample_key)
    examples = []
    for formula in EXAMPLES:
        rows = data[data.composition.str.strip().eq(formula)]
        assert not rows.empty
        examples.append({"composition": formula, "samples_in_local_master": len(rows), "current_plot_samples": {m: int(rows.sample_key.isin(keys).sum()) for m, keys in methods.items()}, "sample_evidence": rows.to_dict("records")})
    result = {
        "status": "diagnostic_only_no_new_classification_or_model",
        "agreed_rule": "Si0.5Ge0.5 remains a mixed host. Other tie rules and tolerances are undecided.",
        "scope": "Local master, not online Starrydata2. Sample counts, not temperature points. Only exact Si/Ge or Bi/Sb/Te element sets; formulas with other elements excluded from sensitivity tables.",
        "normalization": "Si+Ge=1; Bi+Sb=2 with 1.49<=Te/(Bi+Sb)<=1.51 as a diagnostic subset filter. Exact rational arithmetic. Only minority components tested, retaining exact 50:50 pairs in this hypothetical comparison. These are not adopted global rules or verified dopant labels.",
        "SiGe": summarize(sige, 1), "BiSbTe": summarize(bst, 2),
        "Ti_Zr_Hf_Ni_Sn_no_majority_candidates": {"samples": len(ternary), "unique_raw_formulas": len({r["composition"] for r in ternary}), "note": "Formula screen only; no assigned structure or phase purity.", "sample_evidence": ternary},
        "examples": examples,
        "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }
    (OUT / "threshold_tradeoffs.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name in ["SiGe", "BiSbTe"]:
        print(name, json.dumps({k:v for k,v in result[name].items() if k != "sample_evidence"}))
    print("No-majority Ti/Zr/Hf formula candidates:",len(ternary))


if __name__ == "__main__":
    main()
