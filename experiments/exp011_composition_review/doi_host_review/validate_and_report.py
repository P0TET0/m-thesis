"""Independently validate exported CSVs and write the Japanese coverage report."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path

BASE = Path(__file__).resolve().parent


def as_csv(value):
    if isinstance(value, bool):
        return str(value).lower()
    return "" if value is None else str(value)


def main():
    summary = json.loads((BASE / "coverage_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((BASE / "cache/manifest.json").read_text(encoding="utf-8"))
    validation = {"tables": {}, "input_hashes_unchanged": {}}
    for name in ("sample_host_review", "paper_coverage", "host_evidence"):
        path = BASE / "outputs" / (name + ".csv")
        count = 0
        with path.open(encoding="utf-8-sig", newline="") as f, (BASE / "cache" / (name + ".jsonl")).open(encoding="utf-8") as original:
            reader = csv.DictReader(f)
            for row, line in zip_longest(reader, original):
                assert row is not None and line is not None, (name, "row_count_mismatch")
                expected = json.loads(line)
                assert list(row) == list(expected), (name, "headers")
                assert row == {k: as_csv(v) for k, v in expected.items()}, (name, count, "round_trip")
                count += 1
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        validation["tables"][name] = {"rows": count, "bytes": path.stat().st_size, "sha256": digest, "all_cells_match_jsonl": True}
    for name, metadata in manifest["source_files"].items():
        with Path(metadata["path"]).open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        assert digest == metadata["sha256"], (name, "source_changed")
        validation["input_hashes_unchanged"][name] = True
    with (BASE / "outputs/sample_host_review.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(r["judgment_status"] for r in rows)
    assert dict(counts) == summary["sample_status_counts"]
    assert len({(r["doi"], r["sample_id"]) for r in rows}) == summary["samples_total"]
    assert sum(int(r["curve_count"]) for r in rows) == summary["input_counts"]["curve_rows"]
    with (BASE / "outputs/host_evidence.csv").open(encoding="utf-8-sig", newline="") as f:
        ev = {r["evidence_id"]: r for r in csv.DictReader(f)}
    for r in rows:
        for eid in filter(None, r["evidence_ids"].split(";")):
            assert ev[eid]["doi"] == r["doi"]
        if r["judgment_status"] == "automatic_candidate":
            assert not r["host_material"] and not r["excluded_elements"] and not r["retained_atomic_fractions"]
        if r["retained_atomic_fractions"]:
            assert abs(sum(json.loads(r["retained_atomic_fractions"]).values()) - 1) < 1e-10
        if r["excluded_elements"]:
            assert set(r["excluded_elements"].split(";")) <= set(r["dopant_elements"].split(";"))
    validation.update({"unique_doi_sample_keys": True, "curve_count_reconciled": True, "all_evidence_links_valid": True, "candidates_not_marked_confirmed_or_removed": True, "normalizations_sum_to_one": True})
    (BASE / "validation_result.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    total = summary["doi_total"]
    n = summary["samples_total"]
    confirmed = counts["literature_host_and_roles"] + counts["literature_host_only"]
    unresolved = counts["unresolved"] + counts["unresolved_after_review"]
    def number(value):
        return f"{value:,}"
    def fraction(value, denominator):
        return f"{value:,} / {denominator:,}（{value / denominator:.2%}）"
    report = f"""# 母材判定と文献取得の集計

対象全DOIの公開書誌照合と、全試料を含むCSV出力を行いました。**全論文の個別精読・全試料の母材確定は未完了です。** 文献の該当箇所と照合して母材を確認できたのは {fraction(confirmed, n)}、自動候補は {number(counts['automatic_candidate'])} 試料、未確定は {number(unresolved)} 試料です。

## 母数

- 元の曲線行数：{number(summary['input_counts']['curve_rows'])}。
- 元DOI表記：{number(summary['input_counts']['doi_raw_distinct'])} 種類。大文字・小文字等の統一後：{number(total)} DOI。
- `sample_id` 単独：{number(summary['input_counts']['sample_id_distinct'])}。正しい試料単位 `doi + sample_id`：{number(n)}。
- samples CSVとの結合：{fraction(summary['input_counts']['sample_metadata_matches'], n)}。結合時の組成不一致は0。

大小文字が異なる `10.1143/JJAP.41.3787` と `10.1143/jjap.41.3787` を同一DOIとして扱いました。

## 何件の情報を取得できたか

| 項目 | 取得・確認数 / 対象数 |
|---|---:|
| Crossref照合要求成功 | {fraction(summary['crossref_successful_queries'], total)} |
| Crossref書誌レコード | {fraction(summary['crossref_records'], total)} |
| DataCiteで追加取得した書誌レコード | {number(summary['datacite_records'])} |
| 公開API書誌レコードの合計（重複除外） | {fraction(summary['online_records_union'], total)} |
| API外の公式ページで補完した書誌情報 | {number(summary['official_bibliography_supplement'])} |
| API・公式ページで書誌を確認した合計 | {fraction(summary['bibliography_confirmed_union'], total)} |
| Europe PMC照合要求成功 | {fraction(summary['europepmc_successful_queries'], total)} |
| Europe PMC収録レコード | {fraction(summary['europepmc_records'], total)} |
| 公開要旨を取得 | {fraction(summary['abstract_obtained_dois'], total)} |
| 公開本文XMLを取得 | {fraction(summary['open_fulltext_xml_dois'], total)} |
| 一次資料の該当箇所を個別確認した論文 | {fraction(summary['individually_reviewed_dois'], total)} |
| 母材を確認できた試料が1件以上ある論文 | {fraction(summary['dois_with_confirmed_sample_host'], total)} |
| 収録された全試料の母材を確認できた論文 | {fraction(summary['dois_with_all_samples_confirmed_host'], total)} |

要旨取得数はCrossref・DataCite・Europe PMCの和集合です。同一論文を重複加算しません。本文XML取得数はEurope PMCで公開アクセス可能とされた本文を実際に取得し、XML中のDOIと照合した数です。別経路で個別確認した著者原稿PDF等は、このXML件数には含めません。取得した本文には自動照合だけのものも含まれます。

APIの検索成功・収録・本文取得・試料判定は別の段階です。索引で0件だったことを、論文不存在や全文取得不可と同一視しません。API未収録DOIの解決先HTTP結果は `paper_coverage.csv` に記録しています。

## 試料ごとの判定

| 判定 | 試料数 / 全試料数 |
|---|---:|
| 母材と成分の役割を文献で確認 | {fraction(counts['literature_host_and_roles'], n)} |
| 母材のみ確認し、ドーパント等は保留 | {fraction(counts['literature_host_only'], n)} |
| 文献テキストからの自動候補（未確定） | {fraction(counts['automatic_candidate'], n)} |
| 個別確認後も未確定 | {fraction(counts['unresolved_after_review'], n)} |
| 個別確認が必要 | {fraction(counts['unresolved'], n)} |
| 合計 | {number(n)} |

確認済みドーパントを実際に除き、残りを総和1にした試料は {number(summary['samples_with_confirmed_dopant_removal'])} 件です。自動候補と未確定の行では元素を除去していません。原組成はすべて保存しています。

## 根拠がどこに書かれているか

`sample_host_review.csv` の `evidence_ids` を `host_evidence.csv` の `evidence_id` に対応させてください。使用した根拠は {number(summary['evidence_records'])} 件あり、`source_url` が原資料、`source_location` が頁・節・図または取得データ項目、`short_excerpt` が短い該当記載、`interpretation_ja` が判定理由です。抜粋を省略した根拠も位置とURLを残しています。

| 具体例 | 判定と根拠の場所 |
|---|---|
| Si0.8Ge0.2系、DOI 10.1063/1.348408 | SiとGeを母材として保持し、B/Pをドーパントとして区別。`sige_vining_1991` の原論文の該当節を参照 |
| Mg2(Si,Ge,Sn)、DOI 10.1021/acs.chemmater.6b02308 | 少量Geを合金成分として保持、Sbはドーパント。`other_07` の出版社要旨・補足資料説明 |
| (Ti,Zr,Hf)NiSn、DOI 10.3390/ma11040649 | Ti/Zr/Hfを母材構成元素として保持。`other_01` のSection 2.1、2.2と該当表。相分離の可能性も付記 |
| SrTiO3–La2/3TiO3、DOI 10.1021/acsami.7b14231 | 母材系列は確認するがLa一律除去は保留。`other_02` の著者原稿Introduction・Methods・Phase Analysis |

一部SiGe試料のB係数尺度、試料名と組成欄のTi/Fe係数逆転、Gaの部分固溶と第二相など、検出した問題は `data_quality_flags` と理由欄に保存しています。原データを自動修正していません。

## 検証と残る作業

CSV全セルを生成元JSONLと照合し、試料キー重複、件数の不一致、別DOIの根拠参照、自動候補での元素除去がないことを確認しました。元の3ファイルはSHA-256が処理前後で一致しています。詳細は `validation_result.json`。

母材が空の行は、原文の試料表・図・補足資料との対応や、固溶体・第二相・添加物の役割をさらに確認する必要があります。自動候補を確認済み母材として学習に使わないでください。既存DF・ML・FMLの分類・学習データ・評価結果にはまだ適用していません。
"""
    (BASE / "coverage_report_ja.md").write_text(report, encoding="utf-8")
    print(json.dumps({"rows": {k: v["rows"] for k, v in validation["tables"].items()}, "verified": True, "host_confirmed": confirmed, "automatic_candidate": counts["automatic_candidate"], "unresolved": unresolved}, ensure_ascii=True))


if __name__ == "__main__":
    main()
