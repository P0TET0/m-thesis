"""Sensitivity audit only; never change material classes or source compositions."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.exp011_composition_review.audit_composition_notation import parse_flat

OUT = Path(__file__).resolve().parent
SOURCE = OUT / "doi_host_review/cache/sample_host_review.jsonl"
THRESHOLDS = ["0.001", "0.005", "0.01", "0.02", "0.05", "0.1", "0.5"]
NORMALIZED = {"no_elements_removed_sum1", "confirmed_dopants_removed_then_sum1"}


def elements(value: str) -> set[str]:
    return set(filter(None, value.split(";")))


def example(row: dict, fractions: dict, removed: set[str]) -> dict:
    return {
        key: row[key]
        for key in ["doi", "sample_id", "composition_original", "sample_name",
                    "host_material", "host_elements", "dopant_elements",
                    "evidence_ids", "data_quality_flags", "source_sample_record"]
    } | {
        "atomic_fractions_from_recorded_coefficients": {
            key: float(value) for key, value in fractions.items()
        },
        "removed_elements": sorted(removed),
        "retained_elements": sorted(set(fractions) - removed),
    }


def main() -> None:
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    assert len({(r["doi"], r["sample_id"]) for r in rows}) == len(rows)
    formulas = {}
    for formula in {r["composition_original"] for r in rows}:
        amounts = parse_flat(formula)
        if amounts:
            amounts = {e: Fraction(n) for e, n in amounts.items()}
            total = sum(amounts.values())
            formulas[formula] = {e: n / total for e, n in amounts.items()}
            assert sum(formulas[formula].values()) == 1
        else:
            formulas[formula] = None
    valid = [(r, formulas[r["composition_original"]]) for r in rows
             if formulas[r["composition_original"]] is not None]
    reviewed = [(r, f) for r, f in valid if r["normalization_status"] in NORMALIZED]
    assert len(reviewed) == sum(r["normalization_status"] in NORMALIZED for r in rows)
    for r, f in reviewed:
        assert set(f) == (elements(r["host_elements"]) | elements(r["excluded_elements"])) & set(f)
        assert not (elements(r["host_elements"]) & elements(r["excluded_elements"]))
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "unit": "unique normalized DOI + sample_id; not curve rows",
        "method": "x_i=n_i/sum(n_j); remove x_i <= threshold once, then renormalize retained elements",
        "scope": "Numeric flat recorded formulas only; fractions are not verified physical concentrations",
        "total_samples": len(rows),
        "parseable_samples": len(valid),
        "unsupported_samples": len(rows) - len(valid),
        "reviewed_role_comparison_samples": len(reviewed),
        "reviewed_role_comparison_dois": len({r["doi"] for r, _ in reviewed}),
        "reviewed_samples_with_recorded_dopants": sum(bool(elements(r["excluded_elements"])) for r, _ in reviewed),
        "reviewed_samples_with_known_scale_mismatch": sum("coefficient_scale_mismatch" in r["data_quality_flags"] for r, _ in reviewed),
        "thresholds": [],
    }
    for threshold in THRESHOLDS:
        cutoff = Fraction(threshold)
        stats = Counter()
        compare = Counter()
        examples = {"host_removed": [], "dopant_retained": []}
        for r, f in valid:
            removed = {e for e, x in f.items() if x <= cutoff}
            left = len(f) - len(removed)
            stats["samples_with_any_element_removed"] += bool(removed)
            stats["samples_with_no_elements_left"] += left == 0
            stats["samples_with_one_element_left"] += left == 1
            stats["samples_with_multiple_elements_left"] += left >= 2
            if cutoff == Fraction(1, 2):
                assert left <= 1
                # Only the specific SiGe 50:50 exception agreed with the user.
                if set(f) == {"Si", "Ge"} and f["Si"] == f["Ge"]:
                    stats["pure_sige_50_50_exception_samples"] += 1
        for r, f in reviewed:
            removed = {e for e, x in f.items() if x <= cutoff}
            host_removed = removed & elements(r["host_elements"])
            dopant_left = elements(r["excluded_elements"]) - removed
            compare["samples_with_confirmed_host_element_removed"] += bool(host_removed)
            compare["samples_with_confirmed_recorded_dopant_retained"] += bool(dopant_left)
            compare["samples_matching_reviewed_element_removal"] += removed == elements(r["excluded_elements"])
            compare["samples_with_either_role_mismatch"] += bool(host_removed or dopant_left)
            for kind, condition in [("host_removed", host_removed), ("dopant_retained", dopant_left)]:
                if condition and len(examples[kind]) < 12:
                    examples[kind].append(example(r, f, removed))
        assert sum(stats[k] for k in ["samples_with_no_elements_left", "samples_with_one_element_left", "samples_with_multiple_elements_left"]) == len(valid)
        assert compare["samples_matching_reviewed_element_removal"] + compare["samples_with_either_role_mismatch"] == len(reviewed)
        result["thresholds"].append({"threshold_fraction": threshold,
            "threshold_at_percent": float(cutoff * 100),
            "all_parseable_samples": dict(stats),
            "reviewed_roles_comparison": dict(compare), "examples": examples})
    chosen = [
        ("10.1021/acs.chemmater.6b02308", "9531"),
        ("10.3390/nano11041014", "89330"),
        ("10.1016/j.jallcom.2018.09.029", "41700"),
        ("10.1073/pnas.1617663113", "31557"),
    ]
    by_key = {(r["doi"], r["sample_id"]): (r, f) for r, f in valid}
    result["case_studies"] = [example(*by_key[key], set()) for key in chosen]
    evidence_path = OUT / "doi_host_review/cache/host_evidence.jsonl"
    evidence_ids = {eid for case in result["case_studies"] for eid in case["evidence_ids"].split(";")}
    result["case_study_evidence"] = [e for line in evidence_path.read_text(encoding="utf-8").splitlines()
        if (e := json.loads(line))["evidence_id"] in evidence_ids]
    (OUT / "atomic_fraction_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 原子分率による少量元素除去の試算", "",
        "目的：組成式の係数を原子分率へ変換し、一律の閾値でドーパントを除けるか検討する。既存の分類・母材判定CSV・学習結果は変更しない。", "",
        "## 計算方法と対象", "",
        "- 元素iの原子分率は x_i = n_i / Σn_j。原子分率0.5は50 at.%、0.005は0.5 at.%。",
        "- 入力はDOI調査で作成した全試料表。DOI＋sample_idの組を1試料と数え、同じ試料の測定曲線は重複計上しない。",
        f"- 全{len(rows):,}試料中、元素記号と数値だけで完全に読める{len(valid):,}試料を計算。括弧・変数・混合物など未対応の記法{len(rows)-len(valid):,}試料は計算対象外。",
        "- 記載係数をそのまま用いる計算上の分率であり、全試料の公称濃度・実測濃度を検証した結果ではない。係数の尺度が元素間で混在する記録も修正しない。",
        "- 元の分率が閾値以下の元素を一度だけ除く。残った元素を合計1にしても、除去済みの母材元素やサイト占有情報は回復しない。境界は有理数で厳密に比較。", "",
        "## 全試料の閾値比較", "",
        "|除去する原子濃度（以下）|1元素以上を除去|全元素を除去|1元素だけ残る|2元素以上残る|",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in result["thresholds"]:
        a = item["all_parseable_samples"]
        lines.append(f"|{item['threshold_at_percent']:g} at.%|{a['samples_with_any_element_removed']:,}|{a['samples_with_no_elements_left']:,}|{a['samples_with_one_element_left']:,}|{a['samples_with_multiple_elements_left']:,}|")
    last = result["thresholds"][-1]["all_parseable_samples"]
    tie = last.get("pure_sige_50_50_exception_samples", 0)
    lines += ["", f"50 at.%基準では、分率の和が1であるため2元素以上を残すことは数学的にできない。合意済みの『Si0.5Ge0.5は両方残す』に該当する二元SiGeは{tie:,}試料。この例外だけを適用しても、全元素除去は{last['samples_with_no_elements_left']-tie:,}試料残る。他の同率組成への例外拡張は本試算では行っていない。", "",
        "## 文献で整理済みの元素役割との照合", "",
        f"文献調査表のうち、記載元素の役割に基づく正規化結果を持つ{len(reviewed):,}試料（{result['reviewed_role_comparison_dois']} DOI）に限定。うち記載されたドーパントを除く対象は{result['reviewed_samples_with_recorded_dopants']}試料。未記載のドーパントは検出できず、この比較には数えない。", "",
        "選択された確認済み試料であり、全データの精度推定ではない。濃度の表記が未検証のものや既知の尺度混在4試料も含むため、『記録上の係数から得た閾値判定』と『文献上の元素役割』との不一致として読む。下の2種類の不一致は重複し得る。", "",
        "|除去する原子濃度（以下）|母材元素を誤除去した試料|記載ドーパントが残った試料|文献に基づく除去元素と一致|",
        "|---:|---:|---:|---:|",
    ]
    for item in result["thresholds"]:
        c = item["reviewed_roles_comparison"]
        lines.append(f"|{item['threshold_at_percent']:g} at.%|{c['samples_with_confirmed_host_element_removed']:,}|{c['samples_with_confirmed_recorded_dopant_retained']:,}|{c['samples_matching_reviewed_element_removal']:,} / {len(reviewed):,}|")
    lines += ["", "## 実データに残る懸念", "",
        "1. 少量でも母材の固溶成分である。Mg2Si0.28Ge0.02Sn0.7（sample_id 9531、DOI 10.1021/acs.chemmater.6b02308）のGeは約0.667 at.%で、1 at.%以下除去では消える。論文抄録と補足資料の説明ではSi/SnのGe置換系列であり、母材成分として扱った。",
        "2. 多元素母材では1元素当たりの分率が小さい。LaCr0.2Mn0.2Fe0.2Co0.2Ni0.2O3（sample_id 89330、DOI 10.3390/nano11041014）のBサイト5元素は各4 at.%で、5 at.%以下除去では全て消える。Methodsの組成式(1)–(3)では高エントロピー母材の構成元素。",
        "3. 記載尺度の混在は正規化しても直らない。Si80Ge20B0.005（sample_id 41700、DOI 10.1016/j.jallcom.2018.09.029）から計算したBは約0.005 at.%だが、試料名・論文の系列はB 0.5 at.%。全係数に共通の尺度がある場合だけ正規化が有効。",
        "4. 結晶サイトでの置換率と全原子中の割合は異なる。例としてA0.99D0.01BO3のDはAサイトの1%だが、全原子では0.2 at.%。酸素を含む全原子と、陽イオンだけ・特定サイトだけの割合を混同しない。",
        "5. ドーパントが組成式に記載されない試料、添加相・析出相、酸素欠損、重量%表記は、単純な閾値では判別できない。元素が母材と添加相の両方にある場合、元素丸ごとの削除は不適切。",
        "6. ドーパントが必ず低濃度とは限らない。Nb0.8Ti0.2FeSb（sample_id 31557、DOI 10.1073/pnas.1617663113）のTiは全原子の約6.67 at.%で、5 at.%以下の除去でも残る。Results and DiscussionではTi dopingと記述し、Figure 7で対象組成を示す。固溶とドーピングは排他的な概念ではなく、この試算では論文での役割の記述に従っている。", "",
        "## 判断", "",
        "原子分率は係数の表記尺度をそろえる用途に使う。ドーパントの確定は文献の母材・固溶系列・ドーピング記述と試料対応に基づける。閾値は要確認候補を絞る補助とし、未確認元素を自動削除しない。今回の試算だけから全材料に共通する適切な閾値は決められない。", "",
        "## 根拠の所在", "",
    ]
    for e in result["case_study_evidence"]:
        lines.append(f"- `{e['evidence_id']}`：[DOI {e['doi']}]({e['source_url']})。箇所：{e['source_location']}。{e['interpretation_ja']}")
    lines += ["", "再現：リポジトリ直下で `python experiments/exp011_composition_review/audit_atomic_fraction.py` を実行。詳細値・例示試料・入力SHA-256は `atomic_fraction_audit.json` に保存。", ""]
    (OUT / "atomic_fraction_review_ja.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"thresholds", "case_studies", "case_study_evidence"}}, ensure_ascii=False, indent=2))
    for item in result["thresholds"]:
        print(json.dumps({k: v for k, v in item.items() if k != "examples"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
