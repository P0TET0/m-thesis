"""文献を直接確認した SiGe 試料だけの根拠台帳を作る。分類の一般規則ではない。"""
import csv
import json
from pathlib import Path

SOURCE = Path(r"C:\Users\miots\OneDrive - 山陽小野田市立山口東京理科大学 (1)\USBにいたヤツ\大学\1_研究\starrydata2")
OUT = Path(__file__).with_name("sige_evidence.json")
papers = {r["DOI"].lower(): r for r in csv.DictReader((SOURCE / "starrydata_papers.csv").open(encoding="utf-8-sig", newline=""))}
samples = {(r["DOI"].lower(), r["sample_id"]): r for r in csv.DictReader((SOURCE / "starrydata_samples.csv").open(encoding="utf-8-sig", newline=""))}
reviews = []

def evidence(eid, url, location, quote, summary, access):
    assert len(quote.split()) <= 25
    return {"evidence_id": eid, "url": url, "location": location, "short_quote": quote,
            "summary_ja": summary, "access_method": access, "accessed_date": "2026-10-25"}

def rule(ids, dopants, reason, refs, *, status="host_and_dopant_confirmed", flags=None, candidates=None, components=None):
    return {"sample_ids": [str(x) for x in ids],
            "composition_condition": "DOIとsample_idが一致し、listed_samplesの組成・試料名に一致するものだけ。論文内の他試料には転用しない。",
            "host_family": "SiGe", "host_elements": ["Si", "Ge"],
            "dopant_elements": dopants, "dopant_candidates": candidates or [],
            "additive_components": components or [], "status": status,
            "reason_ja": reason, "evidence_ids": refs, "data_quality_flags": flags or []}

def add(doi, level, ev, rules, note=""):
    ids = [s for r in rules for s in r["sample_ids"]]
    assert all((doi, s) in samples for s in ids)
    reviews.append({"doi": doi, "title": papers[doi]["title"].strip('"'), "review_level": level,
                    "evidence": ev, "sample_rules": rules, "note_ja": note,
                    "listed_samples": [{k: samples[(doi,s)][k] for k in ("sample_id", "sample_name", "composition", "composition_details")} for s in ids]})

add("10.1063/1.348408", "full_text_relevant_sections",
    [evidence("sige_vining_1991", "https://cvining.com/system/files/articles/vining/Vining-JAP-1991b.pdf",
              "p.4333: Abstract, II. EXPERIMENTAL DETAILS 第1段落; pp.4334–4335: Table I",
              "heavily doped with either boron or phosphorus",
              "SiとGeの原子比4:1の溶融体に、n型ではP、p型ではBを添加。表IではB/P濃度の例外を区別している。",
              "web.open PDF本文")],
    [rule([21167], ["B"], "原文のp型Si0.8Ge0.2とB添加記述が試料名・composition_detailsに一致する。Geは母材構成元素。", ["sige_vining_1991"], flags=["B係数とwt.%の関係は本判定では換算・修正しない"]),
     rule([21168], ["P"], "原文のn型Si0.8Ge0.2とP添加記述が試料名・composition_detailsに一致する。Geは母材構成元素。", ["sige_vining_1991"], flags=["P係数0.0055はcomposition_detailsの0.55wt.%と混同しない。原子比としての正確性は別途確認が必要"] )])

add("10.1002/adma.200600527", "full_text_indexed_relevant_pages",
    [evidence("sige_dresselhaus_2007", "https://wiki.physics.udel.edu/wiki_qttg/images/6/67/DRESSELHAUS%3Dnew_direction_for_low_dimensional_thermoelectric_materials.pdf",
              "p.1049: Fig.9および周辺段落; p.1050: Fig.10, Fig.11およびFig.12説明段落",
              "Si0.80Ge0.20B0.016 sample doped with 1.6 % boron",
              "SGMA05/SGMA04はSiGeナノ複合材料で、B添加を記載。P-JIMOは比較用のbulk/advanced SiGe合金。HTB AはFig.9にあるがB量の個別確認は不足。",
              "web.searchで原論文PDFのp.1049–1050索引本文を確認。直接PDF取得はcache miss")],
    [rule([26769, 26771], ["B"], "SGMA04/05という試料名とSi0.80Ge0.20B0.016が原文の図・説明に一致し、B dopingが明記される。", ["sige_dresselhaus_2007"], flags=["nanocomposite: SiGe系分類は確定するが単一均一相とみなさない"]),
     rule([26770], [], "HTB AはSiGeナノ複合材料の比較図に対応する。B1の個別根拠は確認できず、Bの除外は保留。", ["sige_dresselhaus_2007"], status="host_confirmed_dopant_unresolved", candidates=["B"], flags=["nanocomposite", "B1の対応箇所未確認"]),
     rule([26772], [], "P-JIMOは本文・Fig.10/11でbulk SiGe alloyと明記される。ドーパント種類・量は確認できない。", ["sige_dresselhaus_2007"], status="host_confirmed_dopant_unresolved", flags=["reference_sample", "SiGeは材料名の可能性がありSi:Ge=1:1と解釈しない", "ドーパントが未記載でも無添加とは断定しない"])],
    "レビュー論文である。対象の試料名まで照合しており、論文全体へSiGe分類を一律適用する根拠ではない。")

add("10.1016/j.matchemphys.2013.09.019", "publisher_abstract_and_section_snippets",
    [evidence("sige_li_2013", "https://www.sciencedirect.com/science/article/pii/S0254058413006998",
              "Abstract 第1文; Experimental details 第1段落; Microstructure and phase composition 第1段落",
              "For fabricating such n-type material, phosphorus (P) powder",
              "Si80Ge20P2の合成、Si/Ge粉へのP添加、GeがSi結晶に固溶していく過程を記載。SiO2/ZrO2はミリング媒体。",
              "web.searchで出版社ページの公開abstract/section snippetsを確認。直接openは取得失敗")],
    [rule([21169], ["P"], "該当Si80Ge20P2のn型材料を得るためPをSi/Geへ添加しており、SiGeを母材とする。", ["sige_li_2013"], flags=["ミリング媒体を母材・ドーパントとして組成に追加しない", "媒体由来汚染と偏析の定量は未判定"])])

add("10.1016/j.materresbull.2019.05.008", "publisher_abstract_and_section_snippets",
    [evidence("sige_murugasami_p_2019", "https://www.sciencedirect.com/science/article/pii/S0025540818335815",
              "Abstract 第3文; Materials and methods 第2段落; Conclusions 項目i",
              "SiGe alloy doped with phosphorus (P) (0.5–2.0 at.%)",
              "Si:Ge=80:20 at.%の母材に、Pを0.5–2.0 at.%添加した系列。公開のMaterials and methodsに母材比を明記。",
              "web.searchで出版社の公開abstract/Materials and methods/Conclusionsを確認")],
    [rule([41600,41601,41602,41603], ["P"], "試料名・組成のP0.5/1.0/1.5/2.0が原論文のP添加系列に一致する。母材はSiGeの両元素。", ["sige_murugasami_p_2019"])])

add("10.1016/j.jallcom.2018.09.029", "publisher_abstract_and_section_snippets",
    [evidence("sige_murugasami_b_2019", "https://www.sciencedirect.com/science/article/abs/pii/S0925838818332584",
              "Introduction 最終段落; Material selection 最終文; Conclusion 第1文",
              "The SiGe alloy doped with B (0.5–2.0 at.%)",
              "BをドープしたSiGe系列を作製し、p型RTG SiGeと比較。Material selectionでもBをdoping elementと記載。",
              "web.searchで出版社の公開Introduction/Material selection/Conclusionを確認")],
    [rule([41700,41701,41702,41703], ["B"], "原文のSiGe-B系列と試料名のB0.5–2.0at%が一致し、Bはドーパントと明記される。", ["sige_murugasami_b_2019"], flags=["coefficient_scale_mismatch: Si/Geが80/20なのにB係数は0.005–0.02。試料名と原論文は0.5–2.0at%でありraw組成係数の尺度が一致しない", "原本組成は修正せず品質問題として保持"]),
     rule([41699], [], "SiGe RTG比較材料の母材はSiGe。比較試料のB量やドーパント種類は公開抜粋で個別確認できない。", ["sige_murugasami_b_2019"], status="host_confirmed_dopant_unresolved", flags=["reference_sample_likely", "ドーパント未記載を無添加とは扱わない"])])

add("10.1021/nl3003045", "full_text_indexed_relevant_pages",
    [evidence("sige_yu_2012", "https://citeseerx.ist.psu.edu/document?doi=2e2bba0596359dcc606c19f0506a5016d3b9295b&repid=rep1&type=pdf",
              "p.2078: 新しいmodulation-doping設計の段落、試料作製段落、Table 1",
              "Si95Ge5 as the matrix",
              "Si95Ge5母相とP添加Si70Ge30P3粒子を組み合わせる。Table 1に粒子5/15/25/35/45%と同組成均一試料の双方がある。",
              "web.searchで原論文PDFのp.2078索引本文を確認。出版社abstractとも照合")],
    [rule([1348,1349,1350,1351,1352], ["P"], "5–45%NPsという試料名と組成が、(Si95Ge5)1-x(Si70Ge30P3)xの総組成に一致。Pはドーピング用、Si/Geは両相の構成元素。", ["sige_yu_2012"], flags=["nanocomposite: 実際の連続母相はSi95Ge5、粒子側はSi70Ge30", "総組成のP除去後比率はSiGe系全体の比率であり連続母相そのものの比率ではない"], components=["P-doped Si70Ge30 nanoparticles in Si95Ge5 matrix"])],
    "SiGeという材料系と、複合材中のmatrix組成を区別する必要がある例。")

add("10.1142/s1793604714500088", "author_uploaded_full_text_relevant_sections",
    [evidence("sige_lin_2014", "https://www.researchgate.net/publication/262263058_Microstructure_and_thermoelectric_properties_of_Ga-DOPED_SiGe_alloys_prepared_by_mechanical_alloying_and_induction_hot_pressing",
              "Abstract; Table 1; Fig.2説明; temperature-dependent propertiesを述べる段落",
              "only part of Ga atoms is incorporated as dopant in the SiGe solid solution phase",
              "著者Dexuan Huoの公開全文でGa添加とp型化を記載。Gaの全量が固溶ドーパントになるわけではなく、Ge/Gaリッチ相やWSi2を含む試料もある。",
              "web.searchで著者投稿全文・Table 1・Fig.2を確認。web.openは取得失敗")],
    [rule([27605,27606,27607,27608,27609], [], "試料名1#–5#とSi80Ge(20-x)Gax系列をTable 1で対応。材料系としてSiGeを保持する。Gaは一部のみ固溶ドーパントで、偏析相への分配があるため全Gaの除去と母材比率の正規化は保留。", ["sige_lin_2014"], status="host_confirmed_dopant_unresolved", candidates=["Ga"], components=["Ge/Ga-rich secondary phases（Gaの一部がSiGe母相外に分配。試料別相分率未確定）", "WSi2（試料3#に報告された第二相）"], flags=["Ga全量を電気的に活性なドーパント濃度と解釈しない", "multiphase: 試料2#/3#で組成不均一; 3#はWSi2も報告", "母材SiGe分類のみ確定。全Ga除去・原子比正規化は禁止"])])

add("10.1063/1.323271", "publisher_abstract",
    [evidence("sige_heller_1976", "https://pubs.aip.org/aip/jap/article/47/9/4113/502358/Electrical-transpoet-properties-of-SiGe",
              "Abstract 第1–2文 (HTMLは記事見出し直後; p.4113のabstractに相当)",
              "The alloys were doped with As, P, or As+P",
              "Si80Ge20合金にAs、P、As+Pをドープした系列を測定。個別濃度を与えるTable 1は今回未取得。",
              "web.clickで出版社本文冒頭のabstractを直接確認")],
    [rule([21110], ["As","P"], "As/Pを含む試料組成とAsP試料名がabstractの共ドープ系列に一致する。", ["sige_heller_1976"], flags=["Table 1による個別濃度照合は未実施"]),
     rule([21111,21112], ["P"], "Pを含むSi80Ge20とP試料名がabstractのPドープ系列に一致する。", ["sige_heller_1976"], flags=["Table 1による個別濃度照合は未実施"]),
     rule([21113,21114], ["As"], "Asを含むSi80Ge20とAs試料名がabstractのAsドープ系列に一致する。", ["sige_heller_1976"], flags=["Table 1による個別濃度照合は未実施"])])

add("10.1016/j.jnucmat.2019.151856", "publisher_abstract_and_section_snippets",
    [evidence("sige_li_neutron_2020", "https://www.sciencedirect.com/science/article/pii/S002231151930844X",
              "Abstract; Experimentalの粉末調合を述べる第1段落（公開section snippet）",
              "Si0.7895Ge0.198Ga0.0025P0.01 or Si0.7925Ge0.2B0.0075",
              "n/p型SiGeの照射実験。Si/Ge、BまたはGaPとPの粉末から、元CSVと同じ二つの組成を作製したことを確認。公開部分では添加元素の固溶・析出の詳細は確認できない。",
              "web.searchで出版社のabstractと組成を含む公開実験段落を確認。直接openは403")],
    [rule([31425,31426,31428,31429,31431,31433,31435,31437,31438,31455], [], "本文の調合組成とn-type SiGe試料名が一致するため母材系はSiGe。Ga/Pを導入することは確認したが、元素ごとのドーパント/副相の区分を公開部分で確定していない。", ["sige_li_neutron_2020"], status="host_confirmed_dopant_unresolved", candidates=["Ga","P"], components=["GaP and P introduced in powder preparation"], flags=["Ga/Pの個別役割・相状態未確認"]),
     rule([31440,31441,31442,31443,31445,31456], [], "本文の調合組成とp-type SiGe試料名が一致するため母材系はSiGe。Bが添加されることは確認したが、公開部分における明示的な役割確認を保留する。", ["sige_li_neutron_2020"], status="host_confirmed_dopant_unresolved", candidates=["B"], flags=["B添加の役割を文献本文で明示確認してから除外"])])

result = {"reviewer": "review_sige_papers", "reviewed_at": "2026-10-25", "scope_ja": "指定5件と追加4件のSiGe文献を試料単位に照合。全論文を調査済みとする台帳ではない。", "reviewed_papers": reviews}
result["counts"] = {"reviewed_dois":len(reviews), "mapped_samples":sum(len(r["listed_samples"]) for r in reviews)}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result["counts"]))
