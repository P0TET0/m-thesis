"""Manually reviewed primary-source evidence. No source compositions are rewritten."""
import collections
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
samples = json.load(gzip.open(ROOT / 'cache/samples.json.gz', 'rt', encoding='utf-8'))
papers = json.load(gzip.open(ROOT / 'cache/papers.json.gz', 'rt', encoding='utf-8'))
output = {'reviewed_papers': [], 'method_notes_ja': [
    '追加調査15 DOI。DOIとsample_idの組を適用範囲とし、同じ論文にある別母材の比較試料に主題母材を流用しない。',
    '原文の本文節・要旨・図説明を実際に確認した範囲のみ記録。検索エンジン経由の一次本文抜粋はprimary_fulltext_sections_web_indexと明示する。',
    '組成係数だけではドーパントを決めない。論文のドープ、固溶体、非化学量論性、充填、相分離の記載を区別する。原組成の補正・係数の再正規化は行わない。',
    'unresolvedは単一母材への採用を保留。候補はcandidate_host_familyに別記する。記載組成と論文の名目組成が一致しない場合も、候補を確定扱いしない。',
    '件数はデータベースの(doi,sample_id)数であり、物理的に独立な作製試料数ではない。']}

def add(doi, level, url, location, quote, summary):
    idx = len(output['reviewed_papers']) + 1
    p = {'doi': doi, 'title': papers[doi][0]['title'].strip('"'), 'review_level': level,
         'evidence': [{'evidence_id': f'other2_{idx:02}', 'url': url, 'location': location,
                      'short_quote': quote, 'summary_ja': summary}], 'sample_rules': []}
    output['reviewed_papers'].append(p)
    groups = collections.defaultdict(list)
    for s in samples:
        if s['doi'] == doi:
            groups[';'.join(s['compositions'])].append(s['sample_id'])
    return p, groups

def rule(p, c, ids, host, elements, dopants, reason, flags=(), status='host_and_roles_confirmed', candidate=''):
    r = {'doi': p['doi'], 'sample_ids': ids, 'composition_condition': c,
         'host_family': host, 'host_elements': elements, 'dopant_elements': dopants,
         'additive_components': [], 'status': status, 'reason_ja': reason,
         'data_quality_flags': list(flags), 'evidence_ids': [p['evidence'][0]['evidence_id']]}
    if candidate:
        r['candidate_host_family'] = candidate
    p['sample_rules'].append(r)

def unresolved(p, c, ids, reason, flags=(), candidate=''):
    rule(p, c, ids, '', [], [], reason, flags, 'unresolved', candidate)

p, groups = add('10.1038/srep19775', 'primary_fulltext_sections',
    'https://www.nature.com/articles/srep19775', 'Methods: sample preparation / single-crystal growth',
    'TbFeO3, TmFeO3 and YFeO3 single crystals were grown',
    'MethodsでTbFeO3、TmFeO3、YFeO3の化学量論的原料と単結晶作製を明記。対象は3種のRFeO3で、Rはドーパントではない。')
for c, ids in groups.items():
    r = c.split('Fe')[0]
    rule(p, c, ids, c, [r, 'Fe', 'O'], [], 'Methodsに一致するRFeO3母材。方位・磁場条件の違いは元素のドープを意味しない。')

p, groups = add('10.1038/srep11010', 'primary_fulltext_sections',
    'https://www.nature.com/articles/srep11010', 'Methods: alloy ingot preparation and annealing',
    'as-cast ingot of Ni43Mn46Sn11 was prepared by arc-melting',
    'Ni43Mn46Sn11リボンの焼鈍時間を変更する実験。秤量時の追加Mnは溶解中の蒸発補償であり、独立したドーパントではない。')
for c, ids in groups.items():
    rule(p, c, ids, 'Ni-Mn-Sn alloy', ['Ni', 'Mn', 'Sn'], [], 'Ni/Mn/Snは論文に明記された三元合金の構成元素。', ['composition_in_atomic_percent', 'annealing_changes_structure_not_nominal_elements'])

p, groups = add('10.1038/ncomms12146', 'primary_fulltext_sections',
    'https://www.nature.com/articles/ncomms12146', 'Methods > Sample preparation',
    'High-quality FeSe single crystals used in the present study',
    'MethodsでFeSe単結晶の作製を記載。主な制御変数は圧力であり、別元素の置換系列ではない。')
for c, ids in groups.items():
    rule(p, c, ids, 'FeSe', ['Fe', 'Se'], [], 'MethodsのFeSe単結晶と組成が一致。圧力条件をドープと扱わない。')

p, groups = add('10.1038/am.2017.8', 'primary_fulltext_sections',
    'https://www.nature.com/articles/am20178', 'Results: paragraph after Table 1; Figures 3 and 6 captions',
    'extrinsic Bi doping at Ge sites',
    '本文でGe1-xBixTeのBiをGeサイトの外因性ドーパントと記載。図3/6にはSnTe:I、PbTe:Naの比較データもあり、全試料をGeTeに統一できない。')
for c, ids in groups.items():
    if 'Sn' in c or 'Pb' in c:
        unresolved(p, c, ids, 'SnTe/PbTeの比較試料。GeTeの母材判定を適用しない。元の比較論文と個別条件の再確認が必要。', ['comparison_material_excluded'])
    elif c in ('Bi0.008GeTe', 'Bi0.007GeTe', 'Bi0.07Ge0.94Te'):
        unresolved(p, c, ids, 'BiドープGeTeが候補だが、記載係数は本文のGe1-xBixTe系列と一致しない。欠落係数や誤記を自動修正しない。', ['recorded_coefficients_do_not_match_nominal_series'], 'GeTe')
    else:
        rule(p, c, ids, 'GeTe', ['Ge', 'Te'], ['Bi'] if 'Bi' in c else [], '本文がBiの置換サイトと母材GeTeを明示。GeTe:Bi表記ではドープ量のみ未指定。', ['dopant_amount_unspecified'] if ':' in c else [])

p, groups = add('10.1038/ncomms4640', 'primary_abstract_and_figure_captions',
    'https://www.nature.com/articles/ncomms4640', 'Abstract; Figures 1-4 captions',
    'Substituting antimony or bismuth for lead',
    '要旨にSb/BiのPbサイト置換を明記。図説明がPb1-xSbxSe/Pb1-xBixSe系列を示す。Sbはナノ析出にも関与し、全量固溶を意味しない。')
for c, ids in groups.items():
    if c == 'undefined':
        unresolved(p, c, ids, '組成未記載であり、Sb系列/Bi系列/比較試料の対応を判別できない。', ['missing_composition'])
    else:
        dp = ['Sb'] if 'Sb' in c else (['Bi'] if 'Bi' in c else [])
        rule(p, c, ids, 'PbSe', ['Pb', 'Se'], dp, '要旨と図説明でPbSeへのSb/Bi置換を確認。ドーパント役割の確認であり、析出分率の同定ではない。', ['abstract_and_figure_captions_only'] + (['sb_dopant_can_form_nanoprecipitates'] if dp == ['Sb'] else []))

p, groups = add('10.1038/ncomms5515', 'primary_fulltext_sections_web_index',
    'https://www.nature.com/articles/ncomms5515', 'Methods > Synthesis; Abstract (phase separation)',
    'A series of x mol% K-doped PbTe0.7S0.3',
    'MethodsがK濃度0.5、1、1.5、2、2.5、3 mol%系列を明示。母体はPbTe/PbSに富む相に分かれるPbTe0.7S0.3で、Sを微量ドーパントとして消去しない。')
for c, ids in groups.items():
    if c.startswith('K'):
        rule(p, c, ids, 'PbTe-PbS', ['Pb', 'Te', 'S'], ['K'], 'Kドープ量系列と母体組成がMethodsに一致。相分離する混合母材としてPb/Te/Sを保持。', ['phase_separated_mixed_host', 'host_family_not_single_phase'])
    else:
        unresolved(p, c, ids, 'Na/SrまたはNa/Mgを含むPbTe比較試料にはKドープPbTe0.7S0.3の判定を適用できない。', ['comparison_material_excluded'])

p, groups = add('10.1038/srep14641', 'primary_fulltext_sections',
    'https://www.nature.com/articles/srep14641', 'Introduction (Ga and Yb site roles); Methods (nominal series); Figure 4 caption',
    'Yb atoms as second fillers (dopants)',
    'Co-Sb骨格の空隙をYbが充填。Gaは空隙充填とSb置換を組み合わせる。MethodsのGa含有組成はSb12-z/3であり、データの一律Sb12表記と差がある。')
for c, ids in groups.items():
    if 'Ga' in c:
        unresolved(p, c, ids, 'CoSb3骨格とGa/Ybの役割は本文に明記されるが、データのSb12はMethodsのSb欠損係数と不一致。試料対応の再確認まで保留。', ['recorded_sb_coefficient_differs_from_paper', 'ga_has_filler_and_substitution_roles'], 'CoSb3')
    else:
        rule(p, c, ids, 'CoSb3', ['Co', 'Sb'], ['Yb'], 'Yb充填Co4Sb12系列を本文・図4で確認。空隙の充填元素Ybをドーパントとして分離する。', ['dopant_is_void_filler'])

p, groups = add('10.1038/s41598-017-12792-8', 'primary_fulltext_sections',
    'https://www.nature.com/articles/s41598-017-12792-8', 'Methods: single-crystal growth, XRD and EDX characterization',
    'Single crystals of LuSb were grown from Sn flux',
    'LuSb単結晶をSnフラックスから成長。XRDに不純物相がないこととEDXによる組成確認を記載。Snは成長溶媒で、LuSb組成中のドーパントとは記載されない。')
for c, ids in groups.items():
    rule(p, c, ids, 'LuSb', ['Lu', 'Sb'], [], 'Methodsの母材LuSbと組成が一致。成長用フラックスを組成へ追加しない。')

p, groups = add('10.3390/ma11101807', 'primary_abstract_and_fulltext_sections_web_index',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6213430/', 'Abstract; Figures 4 and 5 captions; nominal composition series',
    'Ca-site dual doping',
    'Ca1-2xPrxYbxMnO3 (x=0〜0.05)をCaサイトへのPr/Yb共ドープとして明記。')
for c, ids in groups.items():
    rule(p, c, ids, 'CaMnO3', ['Ca', 'Mn', 'O'], ['Pr', 'Yb'] if 'Pr' in c else [], '記載組成がPr/Yb共ドープ系列の名目組成に一致。CaMnO3を母材とする。')

p, groups = add('10.3390/ma15124126', 'primary_fulltext_sections_web_index',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC9229988/', 'Abstract; 2. Materials and Methods; 3.1. Crystal Structure',
    'Substitution of Zr4+ with Yb3+',
    'SrxZr0.95Yb0.05O3-δ系列。YbはZr置換を中心とする受容体ドーパント。x=0.94ではZrO2系の副相が報告される。酸素欠損とSr欠損を単純な元素除去で修復できない。')
for c, ids in groups.items():
    if c.startswith('Sr0.94'):
        unresolved(p, c, ids, 'SrZrO3系母材が候補だが、x=0.94ではZrO2およびYb安定化ZrO2副相があり、単一母材・単一置換サイトへの確定は保留。', ['secondary_phases_reported', 'oxygen_deficiency_not_encoded'], 'SrZrO3')
    else:
        rule(p, c, ids, 'SrZrO3', ['Sr', 'Zr', 'O'], ['Yb'], 'Ybを受容体ドーパントと記載。Sr欠損量や酸素欠損は母材の欠陥として別途保持する必要がある。', ['oxygen_deficiency_not_encoded', 'sr_deficiency_not_repaired'])

p, groups = add('10.3390/cryst10070610', 'primary_abstract_and_fulltext_sections_web_index',
    'https://www.mdpi.com/2073-4352/10/7/610', 'Abstract; Results and Discussion (Al concentration and ZnAl2O4)',
    'Al dopant at the concentration range 0.5–3 mol %',
    'ZnOナノ構造へのAlドープ0.5〜3 mol%系列。高濃度でのZnAl2O4形成が議論され、Al量と固溶量は同一とは限らない。')
for c, ids in groups.items():
    if c == 'Al0.03(ZnO)0.97':
        unresolved(p, c, ids, 'ZnO母材・Al添加の意図は明確だが、3 mol%でZnAl2O4の副相形成があり、全Alを除去した単一母材試料としての採用を保留。', ['secondary_phase_zinc_aluminate'], 'ZnO')
    else:
        rule(p, c, ids, 'ZnO', ['Zn', 'O'], ['Al'] if 'Al' in c else [], '要旨にZnOへのAlドープ濃度系列が明記され、原組成のAl量と一致。', ['mixture_style_composition_notation'] if 'Al' in c else [])

p, groups = add('10.3390/ma12132057', 'primary_fulltext_sections_web_index',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC6651819/', '2. Materials and Methods; 3.1. Phase Composition and Microstructure',
    'segregation of ZnTa2O6 at the grain boundaries',
    'Zn1-xTaxO (x=0,0.002,0.004,0.006,0.008)を作製。Taの溶解度が限られ、高濃度試料でZnTa2O6副相を検出する。')
for c, ids in groups.items():
    if 'Ta0.006' in c or 'Ta0.008' in c:
        unresolved(p, c, ids, 'ZnOが主母材候補だが、ZnTa2O6副相を伴うため全Taが単一母材のドーパントであると扱わない。', ['secondary_phase_zn_ta_oxide', 'dopant_solubility_limit'], 'ZnO')
    else:
        rule(p, c, ids, 'ZnO', ['Zn', 'O'], ['Ta'] if 'Ta' in c else [], 'MethodsのZn1-xTaxO系列と一致。Taをドーパントとして分離する。XRDで未検出の微量副相の不存在までは断定しない。', ['dopant_solubility_limited'] if 'Ta' in c else [])

p, groups = add('10.3390/ma13173865', 'primary_fulltext_sections_web_index',
    'https://pmc.ncbi.nlm.nih.gov/articles/PMC7503398/', '2.1. Crystal Growth and Characterization; 4. Conclusions',
    'prepared crystal was homogeneous and single phase',
    'CeAgAs2単結晶を作製し、EDXでCe約24.6、Ag約25、As約50.4 at.%を確認。ヨウ素は輸送剤で、母材構成元素や意図的ドーパントとは記載されない。')
for c, ids in groups.items():
    rule(p, c, ids, 'CeAgAs2', ['Ce', 'Ag', 'As'], [], 'MethodsのCeAgAs2単結晶と原組成が一致。全3元素を母材として保持。')

p, groups = add('10.3390/app8010052', 'primary_abstract_in_author_institution_repository',
    'https://badap.agh.edu.pl/publikacja/116137', 'Abstract, second paragraph (phase constitution and cooling rate)',
    'The other alloys must be regarded as multi-phase materials.',
    'Ag1-xSb1+xTe2+xは非化学量論組成系列。急冷x=0.19だけが単一β相で、他は多相と記載。組成式だけで冷却条件や相分率を同定できない。')
for c, ids in groups.items():
    unresolved(p, c, ids, 'Ag/Sb/Teは三元系の構成元素であり少量元素除去の対象ではないが、本文要旨では冷却条件依存の多相性がある。試料ごとの相構成対応を未確認のため単一母材判定を保留。', ['nonstoichiometry_is_not_extrinsic_doping', 'cooling_dependent_multiphase_material', 'phase_assignment_needs_sample_condition'], 'AgSbTe2-based Ag-Sb-Te alloy')

p, groups = add('10.3390/nano11041014', 'primary_fulltext_sections_web_index',
    'https://www.mdpi.com/2079-4991/11/4/1014/pdf?version=1618999648', '2. Materials and Methods, printed page 3, nominal compositions (1)-(3)',
    'B-site was fixed',
    'BサイトCr/Mn/Fe/Co/Ni各0.2を維持し、AサイトLa、Nd、La0.5Nd0.5を比較する高エントロピーペロブスカイト。各0.2は母材構成元素で、ドープではない。')
for c, ids in groups.items():
    a = ['La', 'Nd'] if c.startswith('La0.5Nd0.5') else (['La'] if c.startswith('La') else ['Nd'])
    fam = '(La,Nd)(Cr,Mn,Fe,Co,Ni)O3' if len(a) == 2 else a[0] + '(Cr,Mn,Fe,Co,Ni)O3'
    rule(p, c, ids, fam, a + ['Cr', 'Mn', 'Fe', 'Co', 'Ni', 'O'], [], 'Methodsの高エントロピー母材組成に一致。Bサイトの各0.2および混合Aサイトの各0.5はすべて残す。', ['high_entropy_host', 'small_coefficients_are_host_constituents'])

lookup = {(s['doi'], s['sample_id']): s for s in samples}
seen = set()
counts = collections.Counter()
for p in output['reviewed_papers']:
    for e in p['evidence']:
        assert len(e['short_quote'].split()) <= 25
    for r in p['sample_rules']:
        for sid in r['sample_ids']:
            k = (p['doi'], sid)
            assert k in lookup and k not in seen, k
            assert ';'.join(lookup[k]['compositions']) == r['composition_condition'], k
            seen.add(k)
            counts[r['status']] += 1
assert len(output['reviewed_papers']) == 15 and len(seen) == 490
output['review_summary'] = {'doi_count': 15, 'database_sample_count': len(seen), 'sample_status_counts': dict(counts)}
dest = ROOT / 'agents/other_evidence_2.json'
dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(output['review_summary'], ensure_ascii=False))
print(dest)
