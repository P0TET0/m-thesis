"""Audit exp011_composition_review notation and hypothetical removal; do not assign new classes."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp005.material_group_classifier_step31 import ELEMENT_SYMBOLS

OUT = Path(__file__).resolve().parent
MASTER = ROOT / "data/output/starrydata2_prepared_for_relaxation_time_csv/sample_master.csv"
CLASS = ROOT / "data/output/starrydata2_step32_sigma_ml_by_material_system/sample_material_system_classification_step32.csv"
PLOTS = ROOT / "data/output/df_ml_fml_material_parity_20261025"
TOKEN = re.compile(r"([A-Z][a-z]?)(\d+(?:\.\d*)?|\.\d+)?")
SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def parse_flat(value: str) -> dict[str, Decimal] | None:
    # Only contiguous element-number formulas. Parentheses, prose, mixtures,
    # variable coefficients, and unknown symbols stay unassessed.
    text = value.strip().translate(SUBSCRIPTS)
    tokens = list(TOKEN.finditer(text))
    if not tokens or "".join(m.group() for m in tokens) != text:
        return None
    amounts: Counter = Counter()
    for match in tokens:
        symbol, number = match.groups()
        if symbol not in ELEMENT_SYMBOLS:
            return None
        amounts[symbol] += Decimal(number) if number else Decimal(1)
    positive = {symbol: amount for symbol, amount in amounts.items() if amount > 0}
    return positive or None


def counts(frame: pd.DataFrame, parsed: dict) -> dict:
    formulas = [parsed[value] for value in frame.composition]
    valid = [amounts for amounts in formulas if amounts is not None]
    result = {
        "samples": len(frame),
        "unique_compositions": int(frame.composition.nunique()),
        "flat_numeric_samples": len(valid),
        "unassessed_samples": len(frame) - len(valid),
        "coefficient_sum_exactly_1": sum(sum(a.values()) == 1 for a in valid),
        "coefficient_sum_exactly_100": sum(sum(a.values()) == 100 for a in valid),
    }
    for name, threshold, normalized in [
        ("coefficient_le_0_5", Decimal("0.5"), False),
        ("atomic_fraction_le_0_5", Decimal("0.5"), True),
        ("atomic_fraction_le_0_005", Decimal("0.005"), True),
    ]:
        removed_counts = []
        empty = 0
        for amounts in valid:
            # Compare by multiplication to preserve exact boundary values.
            limit = threshold * (sum(amounts.values()) if normalized else 1)
            removed = sum(value <= limit for value in amounts.values())
            removed_counts.append(removed)
            empty += removed == len(amounts)
        result[name] = {
            "samples_with_removed_elements": sum(n > 0 for n in removed_counts),
            "samples_with_no_remaining_elements": empty,
        }
    return result


def main() -> None:
    master = pd.read_csv(MASTER, usecols=["sample_key", "sample_id", "DOI", "composition"], dtype=str).fillna("")
    old = pd.read_csv(CLASS, usecols=["sample_key", "formula_raw", "material_group_key_broad_family"], dtype=str).fillna("")
    assert master.sample_key.is_unique and old.sample_key.is_unique
    old_in_master = old.merge(master, on="sample_key", how="left", validate="one_to_one", indicator=True)
    assert old_in_master._merge.eq("both").all()
    assert old_in_master.formula_raw.str.strip().eq(old_in_master.composition.str.strip()).all()
    parsed = {value: parse_flat(value) for value in master.composition.unique()}
    scopes = {"local_sample_master": master, "saved_step32_classification": old_in_master}
    inputs = [MASTER, CLASS]
    method_maps = {}
    for method in ["DF", "ML", "FML"]:
        path = PLOTS / f"plot_source_{method}.csv"
        inputs.append(path)
        groups = pd.read_csv(path, usecols=["sample_key", "material_group"], dtype=str).drop_duplicates()
        assert groups.sample_key.is_unique
        merged = groups.merge(master, on="sample_key", how="left", validate="one_to_one", indicator=True)
        assert merged._merge.eq("both").all()
        scopes[f"current_plot_{method}"] = merged
        method_maps[method] = groups.set_index("sample_key").material_group.to_dict()

    examples = []
    for formula in ["Pb0.997Bi0.003Te", "Si0.8Ge0.2", "Si0.5Ge0.5", "Bi0.5Sb1.5Te3", "Na0.5CoO2", "Hf0.5Zr0.5NiSn0.99Sb0.01", "Ti0.5Hf0.5CoSb0.8Sn0.2"]:
        rows = master[master.composition.str.strip().eq(formula)].copy()
        assert not rows.empty, formula
        rows["in_current_plots"] = rows.sample_key.map(lambda key: sum(key in m for m in method_maps.values()))
        row = rows.sort_values(["in_current_plots", "sample_key"], ascending=[False, True]).iloc[0]
        amounts = parsed[row.composition]
        assert amounts is not None
        prior = old.loc[old.sample_key.eq(row.sample_key), "material_group_key_broad_family"]
        examples.append({
            "composition": row.composition,
            "sample_key": row.sample_key,
            "sample_id": row.sample_id,
            "doi": row.DOI,
            "number_of_samples_with_this_formula": len(rows),
            "saved_step32_group": prior.iloc[0] if len(prior) else None,
            "current_plot_groups": {method: m.get(row.sample_key) for method, m in method_maps.items()},
            "coefficient_le_0_5_removed": sorted(k for k,v in amounts.items() if v <= Decimal("0.5")),
            "coefficient_le_0_5_remaining": sorted(k for k,v in amounts.items() if v > Decimal("0.5")),
        })
    equivalent = defaultdict(list)
    for row in master.drop_duplicates("composition").itertuples():
        amounts = parsed[row.composition]
        if amounts is None:
            continue
        total = sum(amounts.values())
        signature = tuple((k, Fraction(v) / Fraction(total)) for k, v in sorted(amounts.items()))
        equivalent[signature].append({
            "composition": row.composition,
            "sample_id": row.sample_id,
            "doi": row.DOI,
            "remaining_elements": sorted(k for k, v in amounts.items() if v > Decimal("0.5")),
        })
    scale_sensitive = [rows for rows in equivalent.values() if len({tuple(r["remaining_elements"]) for r in rows}) > 1]
    scale_sensitive.sort(key=lambda rows: min(len(r["composition"]) for r in rows))
    special = master[master.composition.map(parsed).isna()]
    result = {
        "scope_note": "Local retained Starrydata2 snapshot; counts are sample keys, not temperature points or the current entire online database. Removal counts are hypothetical diagnostics, not verified dopants or classification errors.",
        "parser_note": "Strict flat formulas only; missing coefficients mean 1; unicode subscript digits translated; repeated elements summed; zero amounts excluded; unsupported notation unassessed. Decimal arithmetic preserves <= boundaries.",
        "counts_by_scope": {name: counts(frame, parsed) for name, frame in scopes.items()},
        "equivalent_ratio_groups_with_scale_dependent_removal": len(scale_sensitive),
        "scale_dependent_removal_examples": scale_sensitive[:10],
        "examples": examples,
        "unassessed_notation_examples": special.drop_duplicates("composition").head(20)[["sample_id", "composition", "DOI"]].to_dict("records"),
        "input_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
    }
    (OUT / "composition_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts_by_scope"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
