"""Extend the diagnostic to explicit formula subsets; never change saved labels."""
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
THRESHOLDS = ["0.05", "0.1", "0.2", "0.3", "0.5"]
# name, exact positive-element set, tested elements, tested coefficient sum,
# constraints (numerator elements, denominator elements, inclusive bounds).
# These formula screens do not assert crystal structure or a correct host label.
SPECS = [
    ("Si–Ge", "Si Ge", "Si Ge", 1, []),
    ("Bi–Sb–Te", "Bi Sb Te", "Bi Sb", 2, [("Te", "Bi Sb", "1.49", "1.51")]),
    ("Bi–Te–Se", "Bi Te Se", "Te Se", 3, [("Te Se", "Bi", "1.49", "1.51")]),
    ("Sb–Te–Se", "Sb Te Se", "Te Se", 3, [("Te Se", "Sb", "1.49", "1.51")]),
    ("Pb–Sn–Te", "Pb Sn Te", "Pb Sn", 1, [("Te", "Pb Sn", "0.99", "1.01")]),
    ("Pb–Ge–Te", "Pb Ge Te", "Pb Ge", 1, [("Te", "Pb Ge", "0.99", "1.01")]),
    ("Sn–Ge–Te", "Sn Ge Te", "Sn Ge", 1, [("Te", "Sn Ge", "0.99", "1.01")]),
    ("Pb–Te–Se", "Pb Te Se", "Te Se", 1, [("Pb", "Te Se", "0.99", "1.01")]),
    ("Pb–S–Se", "Pb S Se", "S Se", 1, [("Pb", "S Se", "0.99", "1.01")]),
    ("Sn–S–Se", "Sn S Se", "S Se", 1, [("Sn", "S Se", "0.99", "1.01")]),
    ("Mg–Si–Sn", "Mg Si Sn", "Si Sn", 1, [("Mg", "Si Sn", "1.99", "2.01")]),
    ("Co–Fe–Sb", "Co Fe Sb", "Co Fe", 4, [("Sb", "Co Fe", "2.99", "3.01")]),
    ("Hf–Zr–Ni–Sn", "Hf Zr Ni Sn", "Hf Zr", 1, [("Ni", "Hf Zr", "0.99", "1.01"), ("Sn", "Hf Zr", "0.99", "1.01")]),
    ("Ti–Zr–Hf–Ni–Sn", "Ti Zr Hf Ni Sn", "Ti Zr Hf", 1, [("Ni", "Ti Zr Hf", "0.99", "1.01"), ("Sn", "Ti Zr Hf", "0.99", "1.01")]),
    ("Ca–Sr–Ti–O", "Ca Sr Ti O", "Ca Sr", 1, [("Ti", "Ca Sr", "0.99", "1.01"), ("O", "Ti", "2.99", "3.01")]),
    ("La–Sr–Ti–O", "La Sr Ti O", "La Sr", 1, [("Ti", "La Sr", "0.99", "1.01"), ("O", "Ti", "2.99", "3.01")]),
    ("Al–Zn–O", "Al Zn O", "Al Zn", 1, [("O", "Al Zn", "0.99", "1.01")]),
]


def summary(rows, tested, capacity):
    values = [[F(r["normalized_coefficients"][e]) for e in tested.split()] for r in rows]
    protected = [len(v) == 2 and v[0] == v[1] for v in values]
    counts = {t: sum(not tie and any(x <= F(t) for x in v) for v, tie in zip(values, protected)) for t in THRESHOLDS}
    empty = {t: sum(not tie and all(x <= F(t) for x in v) for v, tie in zip(values, protected)) for t in THRESHOLDS}
    fractions = {t: sum(not tie and any(x / capacity <= F(t) for x in v) for v, tie in zip(values, protected)) for t in ["0.01", "0.02", "0.05"]}
    assert list(counts.values()) == sorted(counts.values())
    assert all(empty[t] <= counts[t] <= len(rows) - sum(protected) for t in THRESHOLDS)
    assert len({r["sample_key"] for r in rows}) == len(rows)
    return {"samples": len(rows), "unique_raw_formulas": len({r["composition"] for r in rows}), "protected_equal_pair_samples": sum(protected), "removed_any_tested_element": counts, "removed_all_tested_elements": empty, "removed_any_by_tested_group_fraction": fractions}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = pd.read_csv(MASTER, usecols=["sample_key", "sample_id", "DOI", "composition"], dtype=str).fillna("")
    assert data.sample_key.is_unique
    parsed = {s: parse_flat(s) for s in data.composition.unique()}
    methods, inputs = {}, [MASTER]
    for method in ["DF", "ML", "FML"]:
        path = PLOTS / f"plot_source_{method}.csv"
        inputs.append(path)
        methods[method] = set(pd.read_csv(path, usecols=["sample_key"], dtype=str).sample_key)
        assert methods[method] <= set(data.sample_key)
    groups = []
    seen = set()
    for name, elements, tested, capacity, constraints in SPECS:
        rows = []
        formula_matches = 0
        for row in data.to_dict("records"):
            raw = parsed[row["composition"]]
            if raw is None or set(raw) != set(elements.split()):
                continue
            formula_matches += 1
            amounts = {e: F(v) for e, v in raw.items()}
            def total(names):
                return sum(amounts[e] for e in names.split())
            if not all(F(lo) <= total(n) / total(d) <= F(hi) for n, d, lo, hi in constraints):
                continue
            multiplier = F(capacity) / total(tested)
            assert row["sample_key"] not in seen
            seen.add(row["sample_key"])
            rows.append(dict(row, normalized_coefficients={e: str(v * multiplier) for e, v in amounts.items()}))
        group = {"name": name, "exact_positive_elements": elements.split(), "tested_elements": tested.split(), "tested_coefficient_sum": capacity, "ratio_constraints": constraints, "exact_element_set_samples_before_ratio_filter": formula_matches, "local": summary(rows, tested, capacity), "current_plots": {m: summary([r for r in rows if r["sample_key"] in keys], tested, capacity) for m, keys in methods.items()}, "sample_evidence": rows}
        groups.append(group)
    previous_path = OUT / "threshold_tradeoffs.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    for i, key in [(0, "SiGe"), (1, "BiSbTe")]:
        actual, expected = groups[i]["local"], previous[key]
        assert actual["samples"] == expected["samples"]
        for t in THRESHOLDS:
            assert actual["removed_any_tested_element"][t] == expected["minority_removal_by_coefficient_cutoff"][t]
    result = {"status": "hypothetical_diagnostic_no_classification_changes", "scope": "Local sample master; exact listed element sets only, with declared ratio filters. Additional dopants, missing listed elements, unsupported formulas, and off-ratio samples are excluded. Temperature points are deduplicated by sample_key. These are composition subsets, not the full existing material classes.", "cutoff_rule": "Scale full formula to declared tested-element coefficient sum; test only declared elements. Remove <= cutoff, except exactly equal two-element tested groups retained for this diagnostic. This tie extension beyond SiGe is not adopted policy. Framework elements are not tested; removed-all means tested group only.", "groups": groups, "input_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, "previous_diagnostic_sha256": hashlib.sha256(previous_path.read_bytes()).hexdigest()}
    (OUT / "material_threshold_counts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 材料別の閾値試算", "", "分類・予測値は変更せず、手元の保存データから17種類の組成群を抽出して試算しました。表の件数は、比較対象の元素のうち1つ以上が除外される試料数です。正しいドープ判定数や材料分類の変更数ではありません。", "", "表示した元素だけを含む数値式に限定し、下記の組成比条件も課しています。追加のドーパントを含む試料はこの表には含みません。既存の材料分類に属する全試料を集計した表でもありません。", "", "2成分の比較対象が正確に同率なら、この試算では両方保持しました。Si–Ge以外への同率規則の正式採用は未確定です。3成分の混合にはこの例外を適用していません。", "", "| 組成群 | 比較対象の係数の合計 | 対象試料数 | 0.05以下 | 0.1以下 | 0.2以下 | 0.3以下 | 0.5以下 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for g in groups:
        s = g["local"]
        lines.append("| " + " | ".join([g["name"], "+".join(g["tested_elements"]) + "=" + str(g["tested_coefficient_sum"]), f'{s["samples"]:,}'] + [f'{s["removed_any_tested_element"][t]:,}' for t in THRESHOLDS]) + " |")
    lines += ["", "係数の基準は材料系で異なります。例えばCo+Fe=4とした0.5は、この2元素内の12.5%に相当します。列の数値を共通の原子百分率と解釈しないでください。JSONには比較対象の元素群内で1%、2%、5%以下の場合の件数も保存しています。", "", "Ti–Zr–Hf–Ni–Snの19試料では、0.3以下で少なくとも1元素、0.5以下でTi・Zr・Hfの3元素すべてが除外されます。Ni・Snまで消えるという意味ではありません。同率の2成分を保持する例外だけでは、この混合群を保護できません。", "", "## 抽出に用いた組成比条件", "", "元素名を連結した分子・分母は、それらの係数の合計です。下限・上限を含みます。幅は今回の診断条件であり、母材の確定規則ではありません。", "", "| 組成群 | 比の条件 | 元素集合一致 | 比の条件も満たす |", "| --- | --- | ---: | ---: |"]
    for g in groups:
        conditions = "; ".join(f"{lo} ≤ ({n.replace(' ', '+')})/({d.replace(' ', '+')}) ≤ {hi}" for n,d,lo,hi in g["ratio_constraints"]) or "なし"
        lines.append(f'| {g["name"]} | {conditions} | {g["exact_element_set_samples_before_ratio_filter"]:,} | {g["local"]["samples"]:,} |')
    lines += ["", "## 直近のDF・ML・FML図に含まれる試料", "", "各セルは「0.5以下で除外対象が生じる試料数／この組成群の対象試料数」です。全温度版の図の入力を試料単位に重複除去しています。手法間に重複があるため合算しません。0/0は該当試料なしを表します。", "", "| 組成群 | DF | ML | FML |", "| --- | ---: | ---: | ---: |"]
    for g in groups:
        cells = [f'{g["current_plots"][m]["removed_any_tested_element"]["0.5"]}/{g["current_plots"][m]["samples"]}' for m in ["DF", "ML", "FML"]]
        lines.append("| " + " | ".join([g["name"]] + cells) + " |")
    lines += ["", "## 再現", "", "`python experiments/exp011_composition_review/audit_material_thresholds.py` を実行します。試料ID・DOI・元の組成・正規化後の全係数・抽出条件・入力ハッシュは `material_threshold_counts.json` に保存します。係数は有理数で計算し、前回のSi–GeとBi–Sb–Teの試算に一致することを検証しています。", ""]
    (OUT / "material_threshold_counts_ja.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:27]))


if __name__ == "__main__":
    main()
