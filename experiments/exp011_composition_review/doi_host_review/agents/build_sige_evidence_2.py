"""追加SiGe文献の試料限定根拠。一般的な係数閾値には使わない。"""
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

SOURCE = Path(r"C:\Users\miots\OneDrive - 山陽小野田市立山口東京理科大学 (1)\USBにいたヤツ\大学\1_研究\starrydata2")
OUT = Path(__file__).with_name("sige_evidence_2.json")
papers = {r["DOI"].lower(): r for r in csv.DictReader((SOURCE / "starrydata_papers.csv").open(encoding="utf-8-sig", newline=""))}
samples = {(r["DOI"].lower(), r["sample_id"]): r for r in csv.DictReader((SOURCE / "starrydata_samples.csv").open(encoding="utf-8-sig", newline=""))}
curve_samples = {(r["doi"], r["sample_id"]): r for r in json.load(gzip.open(OUT.parent.parent / "cache/samples.json.gz", "rt", encoding="utf-8"))}
reviews = []

def ev(eid, url, loc, quote, summary, access="web.openで出版社公開abstractを確認"):
    assert len(quote.split()) <= 25
    return {"evidence_id": eid, "url": url, "location": loc, "short_quote": quote,
            "summary_ja": summary, "access_method": access, "accessed_date": "2026-10-25"}

def rule(ids, dopants, reason, refs, *, status="host_and_dopant_confirmed", flags=None, candidates=None, components=None, host="SiGe", elements=None):
    return {"sample_ids": [str(x) for x in ids],
            "composition_condition": "DOIとsample_idが一致し、listed_samplesの組成・試料名に一致するものだけ。論文内の他試料には転用しない。",
            "host_family": host, "host_elements": elements or ["Si", "Ge"],
            "dopant_elements": dopants, "dopant_candidates": candidates or [],
            "additive_components": components or [], "status": status,
            "reason_ja": reason, "evidence_ids": refs, "data_quality_flags": flags or []}

def add(doi, level, evidence, rules, note=""):
    ids = [s for r in rules for s in r["sample_ids"]]
    assert len(ids) == len(set(ids))
    assert all((doi, s) in samples and (doi, s) in curve_samples for s in ids)
    reviews.append({"doi": doi, "title": papers[doi]["title"].strip('"'), "review_level": level,
                    "evidence": evidence, "sample_rules": rules, "note_ja": note,
                    "listed_samples": [{k: samples[(doi,s)][k] for k in ("sample_id", "sample_name", "composition", "composition_details")} for s in ids]})

add("10.1016/j.actamat.2013.10.062", "author_institution_report_plus_secondary_abstract_reproduction",
    [ev("sige_favier_cea_2013", "https://liten.cea.fr/cea-tech/liten/Documents/Medias/Rapports/CEA-Liten-RS-2013.pdf",
        "PDF p.7（冊子pp.12–13）: Development of efficient thermoelectrical materials... / Approach, Results, References [2]",
        "a tailored volume fraction of metal-silicide inclusions during sintering",
        "原著者らのCEA報告はSi92Ge08合金の母相と金属シリサイド介在物を区別し、当該Favier論文をActa Mater 2013 in pressとして引用する。P種類・量はこの報告では確認できない。",
        "web.openで著者所属機関の報告PDF本文を確認"),
     ev("sige_favier_abstract_reproduction", "https://www.researchgate.net/publication/274024286_Influence_of_in-situ_formed_MoSi2_inclusions_on_the_thermoelectrical_properties_of_N-type_silicon-germanium_alloy",
        "Abstract 第1文と第5文（原出版社・HAL本文は取得失敗。転載abstractのみ）",
        "Si92Ge08 (7000 at. ppm P doping)",
        "転載abstractはP添加SiGe母材、1.3vol.%Mo添加、および焼結反応でのMoSi2介在物生成を記す。ただし原出版社またはHALでのabstract照合未了のためP除去を確定しない。",
        "web.openでResearchGate転載abstractを確認。原出版社・HALへの複数経路は取得失敗")],
    [rule([21108], [], "著者機関の報告とDOI書誌題名によりSiGe母材を確認。Pは転載abstractでドープと記されるが一次abstractが取得できていないため保留。", ["sige_favier_cea_2013","sige_favier_abstract_reproduction"], status="host_confirmed_dopant_unresolved", candidates=["P"], flags=["Pの直接一次根拠未取得", "原文公称Si92Ge08とCSVのSi0.913Ge0.080を自動的に同一比率へ補正しない"]),
     rule([21109], [], "SiGeが母相で、Moは金属シリサイド第二相を作る添加材。Moを原子ドーパントとみなさない。Pの一次abstract確認と相別組成が未了なので除去正規化は保留。", ["sige_favier_cea_2013","sige_favier_abstract_reproduction"], status="host_confirmed_dopant_unresolved", candidates=["P"], components=["Mo添加由来のMoSi2介在物（著者報告のmetal-silicide、DOI題名のMoSi2 inclusions、転載abstractで対応）"], flags=["multiphase", "Mo係数0.02と1.3vol.%は同じ単位ではない", "MoSi2は母相のSiも消費するためMoだけの除去では実際の母相組成にならない", "Pの直接一次根拠未取得"])],
    "一次資料が支持する母材/介在物の区別と、転載abstractによるP情報を別レベルで記録。全文確認済みとは扱わない。")

add("10.1016/j.proeng.2011.12.442", "publisher_abstract",
    [ev("sige_hu_2012", "https://www.sciencedirect.com/science/article/pii/S1877705811062783", "Abstract 第1文と最終文",
        "Dopant P was used to increase the carrier concentration",
        "Si80Ge20のキャリア濃度をPドープで調整する研究。P0.06at.%試料を明示する。原論文の公開転載全文Table 1にも0.03/0.04/0.05/0.06/0.1at.%系列がある。",
        "web.searchで出版社abstractを確認。直接openは403; 原論文公開転載本文でTable 1も照合")],
    [rule([11783,11784,11785,11786,11787,11790], ["P"], "P添加量を変えたSi80Ge20の系列で、Pをキャリア濃度調整用dopantと明記。母材はSi/Ge双方を残す。", ["sige_hu_2012"]),
     rule([11791], [], "Si80Ge20という母材は対応するが、この試料のP有無を個別照合できない。", ["sige_hu_2012"], status="host_confirmed_dopant_unresolved", flags=["組成式にPなし。無添加とは断定しない", "reference_or_baseline_sample_role_unresolved"])])

add("10.1016/j.enconman.2015.01.082", "publisher_abstract",
    [ev("sige_wang_sb_2015", "https://www.sciencedirect.com/science/article/pii/S0196890415000953", "Abstract 第3–5文; Highlights 第2–3項",
        "the intended doping element, Sb",
        "Si80Ge(20-x)SbxのSbは意図的ドープ元素。ただしSi-rich母相、Ge-rich相、Si-Ge-Sb析出物の多相で、Sbは主としてGe-rich/三元相に分配される。",
        "web.searchで出版社abstract全文とHighlightsを確認")],
    [rule([1338,1339,1340,1341,1342,1343], [], "SiGe系という分類とSbのドープ意図は確認できるが、Sbの全部がSi-rich母相のドーパントではない。SiとGeも複数相へ分配されるため全Sb除去による母相比率の算出を保留。", ["sige_wang_sb_2015"], status="host_confirmed_dopant_unresolved", candidates=["Sb"], components=["Ge-rich secondary phase", "Si-Ge-Sb ternary precipitates"], flags=["multiphase", "Si-rich matrixと材料全体のSiGe組成を区別", "相別分配未定量のため正規化不可"])])

add("10.1063/1.359663", "publisher_abstract",
    [ev("sige_cook_1995", "https://pubs.aip.org/aip/jap/article/78/9/5474/2103/Si80Ge20-thermoelectric-alloys-prepared-with-GaP", "Abstract 第1文; Ga/P比、P過不足、最適GaP添加量を記す段落",
        "GaP and P were added to a Si 0.8 Ge 0.2 matrix",
        "Si0.8Ge0.2母材へGaPとPを添加し、Ga/P比・総ドープ量・P溶解度を調べた。GaP添加量と実際の固溶量は別である。")],
    [rule(range(21819,21828), [], "A90–A99の組成詳細はSi:Ge=80:20とGaP/P添加に一致する。母材SiGeは確定するが、GaP由来Ga/Pの相別・活性ドーパント配分を未確認のため両元素の一括除去は保留。", ["sige_cook_1995"], status="host_confirmed_dopant_unresolved", candidates=["Ga","P"], components=["GaP（添加原料。添加後の残存GaP/固溶Ga/P配分未確認）"], flags=["Ga/P添加係数はSi80Ge20と単純加算するだけでは最終固溶組成を与えない", "dopant_solubility_and_precipitation_depend_on_processing"])])

add("10.1063/1.1352686", "publisher_abstract",
    [ev("sige_yamashita_2001", "https://pubs.aip.org/aip/jap/article-pdf/89/11/6241/18978860/6241_1_online.pdf", "Abstract 第1–3文と最終2文（PDF URLは出版社abstract HTMLへリダイレクト）",
        "doped with 0.4 at.% P and/or 0.5–2.0 mol% GaP",
        "Si0.95Ge0.05へのP/GaPドープを扱い、Ga/PとGeの粒界偏析を明記。比較対象Si0.8Ge0.2+2mol.%GaPもabstractにある。HTMLで欠ける数式は同DOIの出版社登録Crossref abstractで確認。")],
    [rule([21962], ["P"], "Si0.95Ge0.05P0.004はabstractの0.4at.%Pのみ添加した比較系列と一致。SiGe合金を母材とする。", ["sige_yamashita_2001"]),
     rule([21960], [], "Si0.949Ge0.051P0.0041は本文未取得の比較データで、abstractの公称比と完全一致しない。SiGe母材だけ確定しP役割の試料対応は保留。", ["sige_yamashita_2001"], status="host_confirmed_dopant_unresolved", candidates=["P"], flags=["reference_sample_identity_unverified", "trailing_fullwidth_whitespace_in_raw_formula"]),
     rule([21961,21963,21964], [], "SiGeへのGaP/P添加という材料系は確認できる。Ga/PがGeとともに粒界偏析するため全Ga/Pを一律に除いて母材組成とする処理は保留。", ["sige_yamashita_2001"], status="host_confirmed_dopant_unresolved", candidates=["Ga","P"], components=["GaP添加由来のGa/P/Ge粒界偏析（各元素の相別分配未確認）"], flags=["multiphase_or_grain_boundary_segregation", "21963のP0.04Ga0.05P0.05はabstractの0.4at.%P、0.5–2mol.%GaPとの係数尺度に疑義", "21961/21964はSi0.8Ge0.2比較試料のためSi0.95Ge0.05へ置換しない"])])

add("10.1063/1.3027060", "author_uploaded_full_text_relevant_sections",
    [ev("sige_wang_p_2008", "https://www.researchgate.net/publication/231175385_Enhanced_Thermoelectric_Figure_of_Merit_in_Nanostructured_n-Type_Silicon_Germanium_Bulk_Alloy", "p.193121-1: n-doped nanopowdersの合成段落; p.193121-2: Fig.3導入; p.193121-3: Fig.3 caption",
        "2.0 mol %P was added to Si and Ge chunks with a Si/Ge ratio of 80/20",
        "n型SiGeをSi/Ge塊とP微粉の機械合金化で作製。代表試料はSi80Ge20P2で、Fig.3は作製試料とそのアニール後を区別する。",
        "web.openで共著者Giri Joshi投稿の原論文全文を確認")],
    [rule([21195,21196,21197,21198], ["P"], "nano-sample1–3およびsample1アニール後は原論文Fig.3のSi80Ge20P2系列に対応。Pはn型ドーパント、Si/Ge双方が合金母材。", ["sige_wang_p_2008"], flags=["添加時公称組成を保持。Pの固溶量を2mol.%と断定しない"])])

add("10.1063/1.332354", "publisher_abstract",
    [ev("sige_savvides_1983", "https://pubs.aip.org/aip/jap/article-pdf/54/5/2402/18401386/2402_1_online.pdf", "Abstract 第1–2文と最終文（出版社abstract HTMLへのリダイレクト）",
        "boron doped Si70Ge30 alloys",
        "BドープSi70Ge30の初期状態と長時間熱処理でのB析出・再固溶を研究する。Bの析出でキャリア濃度等が変化する。")],
    [rule([21239,21240], ["B"], "組成B0.5Si70Ge30と初期/熱処理後の試料名がBドープSi70Ge30の経時実験に一致。Bの役割は確認できるが析出相との配分は未定量。", ["sige_savvides_1983"], components=["B precipitates（熱履歴依存。析出量未定量）"], flags=["dopant_role_confirmed_but_active_fraction_not_constant", "析出を含むため単一母相の定量組成へ正規化しない"])])

add("10.1063/1.4726041", "author_preprint_full_text_relevant_sections",
    [ev("sige_stoib_2012", "https://arxiv.org/pdf/1201.4370", "著者原稿p.1: NP synthesis段落; p.2末–p.3: Fig.2と説明段落",
        "1 % per volume of phosphine was admixed to the silane source as dopant",
        "Si95Ge05/Si80Ge20は合金NP、Si-Ge60:40/91:09はSi/Ge NP複合体として区別。対象SiGe系列はphosphine源によるPドープで、Pの表面偏析/コア取り込みも論じる。",
        "web.openで著者arXiv原稿本文を確認。TUM機関書誌で掲載DOIとの一致を確認")],
    [rule(range(21199,21205), ["P"], "合金NPのSi95Ge05/Si80Ge20についてLS/CAS/CAS(ZEM-3)がFig.2と一致する。式にはPがないが原文ではPドープを明記。", ["sige_stoib_2012"], flags=["dopant_P_omitted_from_raw_formula", "05は5、09は9の表記。Ge0.5/0.9と読み替えない", "P原料ガス1vol.%は固体中1at.%ではない"]),
     rule(range(21205,21211), ["P"], "60:40/91:09はSi/Ge NP複合体で双方を材料分類に残す。Pドープは確認できるが均一SiGe単相とみなせない。", ["sige_stoib_2012"], components=["separately synthesized Si and Ge nanoparticles（Si-Ge NP composite）"], flags=["composite_not_homogeneous_alloy", "dopant_P_omitted_from_raw_formula", "P原料ガス1vol.%は固体中1at.%ではない"])])

add("10.1002/pssa.201228102", "publisher_deposited_abstract_crossref",
    [ev("sige_zamanipour_2012", "https://api.crossref.org/works/10.1002%2Fpssa.201228102", "message.abstract 第1–3文; 出版社登録abstract（cache/crossrefに原JSON保存済み）",
        "different nanostructured p-type SiGe alloys",
        "機械合金化・焼結で作るp型SiGe系列を対象とする。取得できたabstractにはB等のドーパント元素名がない。",
        "親エージェントが取得したCrossref原JSONの出版社提供abstractを直接読み、DOI照合。Wiley直接openは403")],
    [rule(range(21181,21186), [], "原文abstractでSiGe系合金は確認できる。p型というだけではBドープと断定できないので、係数からもドーパントを補わない。", ["sige_zamanipour_2012"], status="host_confirmed_dopant_unresolved", flags=["dopant_identity_unreported_in_accessible_abstract", "試料別配合表未取得"])])

add("10.1016/j.scriptamat.2014.10.001", "publisher_abstract",
    [ev("sige_usenko_2015", "https://www.sciencedirect.com/science/article/pii/S1359646214003868", "Abstract 第1–3文",
        "n-type nanostructured SiGe bulk sample",
        "機械合金化とSPSで合成したn型SiGe系列の研究。公開abstractにはドーパント元素が記されていない。",
        "web.searchで出版社公開abstractを確認")],
    [rule(range(21190,21195), [], "Si80Ge20というCSV組成が研究対象SiGeに一致するため母材系のみ確認。n型からPを推定せず、個別の添加元素は保留する。", ["sige_usenko_2015"], status="host_confirmed_dopant_unresolved", flags=["dopant_identity_unreported_in_accessible_abstract", "試料別配合表未取得"])])

add("10.1021/cg501776h", "publisher_abstract",
    [ev("sige_omprakash_2015", "https://pubs.acs.org/doi/10.1021/cg501776h", "Abstract 第1–3文（GSG1/GSG2とSi0.68Ge0.32を明記）",
        "Compositionally homogeneous Ga-doped Si0.68Ge0.32 bulk crystals",
        "GSG1とGSG2はGa濃度10^18/10^19 cm^-3のSi0.68Ge0.32結晶。成長方向に切り出した各領域を測定する。",
        "web.searchで出版社公開abstract全文を確認")],
    [rule(range(21212,21218), ["Ga"], "GSG1/GSG2という試料名とSi0.68Ge0.32がabstractに一致。Gaは元組成式に省略されるがドーパントとして明記される。", ["sige_omprakash_2015"], flags=["dopant_Ga_omitted_from_raw_formula", "cm^-3濃度を組成式係数へ推測変換しない"])])

add("10.1063/1.4801388", "publisher_abstract",
    [ev("sige_zamanipour_2013", "https://pubs.aip.org/aip/jap/article-pdf/doi/10.1063/1.4801388/13210736/143715_1_online.pdf", "Abstract 第1–4文（出版社abstract HTMLへのリダイレクト）",
        "Boron precipitation process and its effect on electronic properties",
        "Si0.8Ge0.2のナノ構造/大粒径試料で、加熱冷却時のB析出とキャリア濃度等の変化を比較する。")],
    [rule(range(21450,21454), ["B"], "母材比とナノ構造/溶製・加熱/冷却という条件が原文abstractに一致。Bが電子特性に影響する添加種であるが、析出/固溶への配分は熱履歴で変わる。", ["sige_zamanipour_2013"], components=["B precipitates（温度・熱履歴で変化）"], flags=["dopant_B_omitted_from_raw_formula", "析出を含むため単一母相の定量組成へ正規化しない"])])

add("10.1021/acsami.9b20090", "publisher_abstract",
    [ev("sige_sto_2020", "https://pubs.acs.org/doi/10.1021/acsami.9b20090", "Abstract: P-doped SiGe powder、La-Nb-doped SrTiO3 powder、SiGe matrixを記す文",
        "La–Nb–STO nanoparticles added to SiGe matrix",
        "PドープSiGe粉とLa/NbドープSrTiO3粉を混合熱間プレスする複合材。SiGeが母材でSrTiO3は第二相、La/Nbは第二相側のドーパント。",
        "web.searchで出版社abstract全文を確認")],
    [rule([35339], ["P"], "SrTiO3添加0wt.%のSi80Ge20P1で、abstractのPドープSiGe粉単体に対応する。", ["sige_sto_2020"]),
     rule([35340,35341,35343,35344,35345], ["P"], "SrTiO3添加量0.5–10wt.%の複合系列でSiGeが母材。Sr/Ti/Oを単なるドーパント元素として除去しない。", ["sige_sto_2020"], components=["La/Nb-doped SrTiO3 nanoparticles（第二相; raw formulaのSr/Ti/Oに対応）"], flags=["multiphase_composite", "第二相のLa/NbがCSV式には省略", "添加量wt.%と式係数は単位が異なる", "複合体全体の係数から実際の母相比率は再計算しない"]),
     rule([35338], [], "RTG比較試料はSi63.5Ge36.5P0.17で研究作製試料Si80Ge20P1と異なる。母材SiGeは確認できるが、Pの個別資料対応は未確認。", ["sige_sto_2020"], status="host_confirmed_dopant_unresolved", candidates=["P"], flags=["reference_sample", "本論文の作製条件を比較試料へ一律適用しない"])])

add("10.1039/c4nr04470c", "publisher_accepted_manuscript_relevant_sections",
    [ev("sige_yusufu_2014", "https://pubs.rsc.org/en/content/getauthorversionpdf/C4NR04470C", "Accepted Manuscript本文p.2: Results and Discussion冒頭、Table 1; p.3–4: Ge置換・P析出の説明",
        "P-doping supplied electrons to Si",
        "Si100P1/Si100P3/Si99Ge1P3/Si97Ge3P3を個別に記載。Pは電子供給ドーパント、GeはSi置換して格子・合金散乱を変える成分。P-rich析出物によるナノ構造も扱う。",
        "web.openでRSC公開accepted manuscript本文を確認")],
    [rule([29807,29808], ["P"], "表1のSi100P1/Si100P3はSi母材にPを添加する系列。Pドープと電子供給を原文で明記。P-rich析出の存在/分率を試料ごとに確定できていないため正規化は保留。", ["sige_yusufu_2014"], host="Si", elements=["Si"], components=["P-rich nanoprecipitates（論文のナノ構造機構。試料別析出量は未確定）"], flags=["原文はP添加量と固溶P濃度を区別", "母材名のみ確定。P-rich相へのSi分配未定量"]),
     rule([29809,29810], ["P"], "Ge1/Ge3は量が小さくてもSiを置換し格子定数・合金散乱を変える成分で、Pと役割が異なる。Si/Ge双方を母材構成に残す。", ["sige_yusufu_2014"], components=["P-rich nanoprecipitates（Si母相と整合/半整合に形成）"], flags=["Si-rich SiGe alloy; 著者の広義材料名はbulk silicon", "Geの係数の大小で削除しない", "析出相へのSi/P分配未定量のため正規化不可"])])

add("10.2109/jcersj.113.558", "publisher_full_text_relevant_sections",
    [ev("sige_tajima_2005", "https://www.jstage.jst.go.jp/article/jcersj/113/1320/113_1320_558/_pdf", "p.558: Abstract; p.560: 3.3 Thermoelectric properties, Fig.4–6; p.561: Table 1",
        "A phosphorus-doped Si0.8Ge0.2 thin film",
        "RFスパッタで作製したPドープSi0.8Ge0.2薄膜をアニール。3.3節/Fig.4–5は10/20/50hの3試料を明示し、Fig.6と本文はSIMSでPドーパント濃度を測定する。",
        "web.open/findでJ-STAGE原論文PDFの本文・図説明を確認（見出し等テキストの一部は文字化け）")],
    [rule([16478,16479,16480], ["P"], "Si0.8Ge0.2の10/20/50hアニール薄膜は同じPドープ薄膜系列。元CSVはPを省略するが原文はPドープを明記する。", ["sige_tajima_2005"], flags=["dopant_P_omitted_from_raw_formula", "ドープ量を式へ補完しない"])])

counts = Counter()
for p in reviews:
    counts["reviewed_dois"] += 1
    for r in p["sample_rules"]:
        counts["mapped_samples"] += len(r["sample_ids"])
        counts[r["status"]] += len(r["sample_ids"])
        counts["source_curve_rows"] += sum(curve_samples[(p["doi"], s)]["curve_count"] for s in r["sample_ids"])
payload = {"reviewer": "review_sige_papers", "reviewed_at": "2026-10-25",
           "scope_ja": "SiGe候補から追加した15 DOI。試料IDはDOIとの組で照合。文献確認レベルと未解決を区別し、複相材料の全添加元素除去・正規化は行わない。",
           "reviewed_papers": reviews, "counts": dict(counts)}
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(dict(counts), ensure_ascii=False))
