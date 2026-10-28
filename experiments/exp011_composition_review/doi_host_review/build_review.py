"""Build evidence-linked review tables; do not alter training/classification data."""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE / "agents"))
from experiments.exp011_composition_review.audit_composition_notation import parse_flat
from retrieve_literature import read, dump, normalize_doi
from auto_host_candidates import candidates_from_text


def text_only(value):
    # Do not add spaces inside a chemical formula interrupted by sub/sup tags.
    value = re.sub(r"</?(?:[\w-]+:)?(?:sub|sup)[^>]*>", "", value or "")
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def local_title(records):
    value = records[0].get("title", "") if records else ""
    try:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, str) else value
    except ValueError:
        return value


def gather_service(name):
    results, checked, failures, cache_locations = {}, set(), {}, {}
    for path in sorted((BASE / "cache" / name).glob("*.json.gz")):
        obj = read(path)
        wanted = set(obj["requested_dois"])
        if not obj.get("ok"):
            failures.update({d: obj.get("error", "request_failed") for d in wanted})
            continue
        checked.update(wanted)
        items = obj["data"].get("message", {}).get("items", []) if name == "crossref" else obj["data"].get("resultList", {}).get("result", [])
        total = obj["data"].get("message", {}).get("total-results", 0) if name == "crossref" else obj["data"].get("hitCount", 0)
        assert total <= 1000, (name, path, "pagination_needed")
        for item in items:
            doi = normalize_doi(item.get("DOI", item.get("doi", "")))
            if doi not in wanted:
                continue
            if doi not in results or item.get("abstractText") or item.get("abstract"):
                results[doi] = item
                cache_locations[doi] = str(path.relative_to(BASE))
    return results, checked, failures, cache_locations


def main():
    manifest = read(BASE / "cache" / "manifest.json")
    samples = read(BASE / "cache" / "samples.json.gz")
    papers = read(BASE / "cache" / "papers.json.gz")
    crossref, cr_checked, cr_fail, cr_cache = gather_service("crossref")
    epmc, ep_checked, ep_fail, ep_cache = gather_service("europepmc")
    datacite, dc_cache, resolvers = {}, {}, {}
    for path in (BASE / "cache/datacite").glob("*.json.gz"):
        obj = read(path)
        doi = obj["requested_doi"]
        if obj.get("ok"):
            attrs = obj["data"]["data"]["attributes"]
            assert normalize_doi(attrs["doi"]) == doi
            datacite[doi] = {"DOI": doi, "title": [t["title"] for t in attrs.get("titles", [])], "abstract": " ".join(d["description"] for d in attrs.get("descriptions", []) if d.get("descriptionType") == "Abstract"), "URL": attrs.get("url", "")}
            dc_cache[doi] = str(path.relative_to(BASE))
        if obj.get("resolver"):
            resolvers[doi] = obj["resolver"]
    metadata = {**datacite, **crossref}
    metadata_cache = {**dc_cache, **cr_cache}
    all_dois = set(manifest["doi_spellings"])
    fulltexts = {}
    for path in (BASE / "cache" / "fulltext").glob("*.json"):
        obj = read(path)
        if obj.get("ok"):
            fulltexts[obj["doi"]] = obj

    manual = {}
    manual_papers = {}
    evidence = {}
    quote_budgets = defaultdict(int)
    all_keys = {(s["doi"], s["sample_id"]) for s in samples}
    source_conflicts = {}
    conflict_path = BASE / "agents" / "source_conflicts.json"
    if conflict_path.exists():
        for conflict in read(conflict_path).get("conflicts", []):
            source_conflicts[(normalize_doi(conflict["doi"]), str(conflict["sample_id"]))] = conflict
    for path in sorted((BASE / "agents").glob("*evidence*.json")):
        for p in read(path).get("reviewed_papers", []):
            doi = normalize_doi(p["doi"])
            assert doi in all_dois, (path, doi)
            manual_papers[doi] = p
            for e in p.get("evidence", []):
                eid = e["evidence_id"]
                assert eid not in evidence, ("duplicate_evidence_id", eid)
                # One short excerpt budget per DOI in delivered evidence table.
                words = e.get("short_quote", "").split()
                remaining = max(0, 25 - quote_budgets[doi])
                quote = " ".join(words[:remaining])
                quote_budgets[doi] += min(len(words), remaining)
                evidence[eid] = {"evidence_id": eid, "doi": doi, "source_type": "primary_literature_manual", "source_url": e.get("url", ""), "source_location": e.get("location", ""), "short_excerpt": quote, "interpretation_ja": e.get("summary_ja", ""), "verification_level": p.get("review_level", ""), "host_candidate": "", "dopant_candidates": "", "applicable_sample_count": 0, "limitations": "抜粋だけでなく該当箇所と試料対応を参照。本文全体を精読した意味ではない。", "local_cache": str(path.relative_to(BASE))}
            for rule in p.get("sample_rules", []):
                for sid in rule["sample_ids"]:
                    key = (doi, str(sid))
                    assert key in all_keys, ("manual_sample_missing", key)
                    assert key not in manual, ("overlapping_manual_rules", key)
                    manual[key] = {**rule, "review_level": p.get("review_level", ""), "agent_source": path.name}

    old_labels = {}
    old_path = ROOT / "data/output/starrydata2_step32_sigma_ml_by_material_system/sample_material_system_classification_step32.csv"
    with old_path.open(encoding="utf-8-sig", newline="") as f:
        old_by_key = {r["sample_key"]: r["material_group_key_broad_family"] for r in csv.DictReader(f)}
    master_path = ROOT / "data/output/starrydata2_prepared_for_relaxation_time_csv/sample_master.csv"
    with master_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["sample_key"] in old_by_key:
                old_labels[(normalize_doi(r["DOI"]), r["sample_id"])] = old_by_key[r["sample_key"]]

    candidates = {}
    online_titles = {}
    abstract_sources = {}
    for doi in sorted(all_dois):
        cr = metadata.get(doi, {})
        ep = epmc.get(doi, {})
        provider = "Crossref" if doi in crossref else "DataCite"
        title = text_only(" ".join(cr.get("title", [])) or ep.get("title", ""))
        online_titles[doi] = title
        abstract = text_only(cr.get("abstract") or ep.get("abstractText", ""))
        abstract_sources[doi] = provider + (" /abstract" if provider == "Crossref" else " /attributes/descriptions[Abstract]") if cr.get("abstract") else ("Europe PMC /abstractText" if ep.get("abstractText") else "")
        use_title = title or local_title(papers.get(doi, []))
        extracted = candidates_from_text(use_title, abstract)
        if doi in fulltexts:
            ft = fulltexts[doi]
            with gzip.open(BASE / "cache/fulltext" / (ft["pmcid"] + ".xml.gz"), "rb") as f:
                root = ET.parse(f).getroot()
            seen_relations = set()
            for para_index, para in enumerate(root.findall("./body//p"), 1):
                paragraph = text_only("".join(para.itertext()))
                if not re.search(r"doped|based|alloy|solid.solution", paragraph, re.I):
                    continue
                for c in candidates_from_text("", paragraph):
                    signature = (c["host_formula"], tuple(c["host_elements"]), tuple(c["dopant_elements"]), c["method"])
                    if signature in seen_relations:
                        continue
                    seen_relations.add(signature)
                    extracted.append({**c, "source": "fulltext", "fulltext_location": f"Europe PMC {ft['pmcid']} XML body paragraph {para_index}; id={para.get('id', '')}", "fulltext_cache": "cache/fulltext/" + ft["pmcid"] + ".xml.gz", "fulltext_url": "https://pmc.ncbi.nlm.nih.gov/articles/" + ft["pmcid"] + "/", "cautions": c["cautions"] + ["fulltext_context_not_individually_linked_to_sample"]})
        candidates[doi] = []
        for ix, c in enumerate(extracted):
            # Source text is a paper-wide candidate, never a sample-level verdict.
            if not c.get("host_elements"):
                continue
            eid = "auto_" + hashlib.sha256((doi + "|" + str(ix)).encode()).hexdigest()[:16]
            c = {**c, "evidence_id": eid}
            candidates[doi].append(c)
            is_abstract = c.get("source") == "abstract"
            source = abstract_sources[doi] if is_abstract else (provider + (" /title/0" if provider == "Crossref" else " /attributes/titles") if cr.get("title") else ("Europe PMC /title" if title else "starrydata_papers.csv title"))
            source_cache = (metadata_cache.get(doi, "") if cr.get("abstract") else ep_cache.get(doi, "")) if is_abstract else (metadata_cache.get(doi, "") if cr.get("title") else ep_cache.get(doi, "cache/papers.json.gz"))
            source_url = "https://doi.org/" + doi
            if c.get("source") == "fulltext":
                source = c["fulltext_location"]
                source_cache = c["fulltext_cache"]
                source_url = c["fulltext_url"]
            raw_quote = c.get("matched_text", "")
            if not isinstance(raw_quote, str):
                raw_quote = str(raw_quote)
            words = raw_quote.split()
            remaining = max(0, 25 - quote_budgets[doi])
            quote = " ".join(words[:remaining])
            quote_budgets[doi] += min(remaining, len(words))
            name = c.get("host_formula") or c.get("host_name") or "-".join(c["host_elements"])
            evidence[eid] = {"evidence_id": eid, "doi": doi, "source_type": "automatic_text_extraction", "source_url": source_url, "source_location": source + "; normalized-text character span=" + compact(c.get("span", [])), "short_excerpt": quote, "interpretation_ja": "母材候補として " + name + " を抽出。個別試料への適用は未確認。", "verification_level": "automatic_candidate_not_verified", "host_candidate": name, "dopant_candidates": ";".join(c.get("dopant_elements", [])), "applicable_sample_count": 0, "limitations": ";".join(c.get("cautions", [])), "local_cache": source_cache}

    by_doi = defaultdict(Counter)
    records = []
    all_status = Counter()
    confirmed_statuses = {"host_and_roles_confirmed", "host_and_dopant_confirmed"}
    removed_count = 0
    for s in samples:
        doi, sid = s["doi"], s["sample_id"]
        formula = s["compositions"][0]
        amounts = parse_flat(formula)
        elements = set(amounts or {})
        info = s.get("sample_info", {})
        family = {k: v for k, v in info.items() if "family" in k.lower().replace(" ", "") or "farmily" in k.lower()}
        base = {"doi": doi, "doi_original": ";".join(manifest["doi_spellings"][doi]), "sample_id": sid, "SID": ";".join(s["SIDs"]), "composition_original": formula, "sample_name": s.get("sample_name", ""), "curve_count": s["curve_count"], "host_material": "", "host_elements": "", "dopant_elements": "", "additive_components": "", "judgment_status": "unresolved", "host_candidate": "", "dopant_candidates": "", "excluded_elements": "", "retained_atomic_fractions": "", "normalization_status": "not_applied", "evidence_ids": "", "reason_ja": "個別試料の母材・ドーパントを確定する根拠が未確認。", "data_quality_flags": "", "review_level": "", "previous_material_group": old_labels.get((doi, sid), ""), "source_sample_record": ";".join(map(str, s.get("metadata_records", []))), "source_material_family": compact(family) if family else "", "composition_details": s.get("composition_details", ""), "source_figures": ";".join(s["figures"])}
        flags = []
        if not amounts:
            flags.append("non_flat_formula_not_parsed")
        r = manual.get((doi, sid))
        if r:
            status = r.get("status", "unresolved")
            is_confirmed = status.startswith("host_") and "confirmed" in status
            base.update({"host_material": r.get("host_family", "") if is_confirmed else "", "host_elements": ";".join(r.get("host_elements", [])) if is_confirmed else "", "dopant_elements": ";".join(r.get("dopant_elements", [])), "additive_components": compact(r.get("additive_components", [])) if r.get("additive_components") else "", "judgment_status": "literature_host_and_roles" if status in confirmed_statuses else ("literature_host_only" if is_confirmed else "unresolved_after_review"), "evidence_ids": ";".join(r.get("evidence_ids", [])), "reason_ja": r.get("reason_ja", ""), "review_level": r.get("review_level", ""), "dopant_candidates": ";".join(r.get("dopant_candidates", []))})
            flags.extend(r.get("data_quality_flags", []))
            # Remove only independently evidenced dopants, and only when every
            # other elemental role is accounted for. Never refill lattice sites.
            hosts, dopants = set(r.get("host_elements", [])), set(r.get("dopant_elements", []))
            if amounts and dopants - elements:
                flags.append("reported_dopant_not_in_raw_composition")
            if status in confirmed_statuses and amounts and not r.get("additive_components") and elements <= (hosts | dopants) and not (hosts & dopants):
                excluded = elements & dopants
                retained = {e: v for e, v in amounts.items() if e not in excluded}
                if retained and set(retained) <= hosts:
                    total = sum(Fraction(v) for v in retained.values())
                    base["retained_atomic_fractions"] = compact({e: float(Fraction(v) / total) for e, v in retained.items()})
                    base["excluded_elements"] = ";".join(sorted(excluded))
                    base["normalization_status"] = "confirmed_dopants_removed_then_sum1" if excluded else "no_elements_removed_sum1"
                    removed_count += bool(excluded)
            if base["normalization_status"] == "not_applied" and is_confirmed:
                flags.append("host_known_but_composition_removal_not_fully_supported")
        else:
            possible = []
            if amounts:
                for c in candidates.get(doi, []):
                    hs = set(c["host_elements"])
                    ds = set(c.get("dopant_elements", []))
                    # Equality is permitted. If other elements exist, their
                    # identity must be explicitly mentioned as dopants in text.
                    # No concentration threshold is used.
                    if hs <= elements and len(hs) >= 1 and elements - hs <= ds:
                        possible.append(c)
            if possible:
                title_possible = [c for c in possible if c.get("source") == "title"]
                if title_possible:
                    possible = title_possible
                names = sorted({c.get("host_formula") or c.get("host_name") or "-".join(c["host_elements"]) for c in possible})
                paper_dopants = set().union(*(set(c.get("dopant_elements", [])) for c in possible))
                base.update({"judgment_status": "automatic_candidate", "host_candidate": ";".join(names), "dopant_candidates": ";".join(sorted(paper_dopants & elements)), "evidence_ids": ";".join(c["evidence_id"] for c in possible), "reason_ja": "論文タイトル・要旨・取得本文の材料記載と試料の元素集合が整合する候補。比較試料・結晶相・ドーパントの役割は個別未確認。", "review_level": "automatic_candidate_not_verified"})
                if paper_dopants - elements:
                    flags.append("paper_mentions_dopant_absent_from_sample")
                for c in possible:
                    flags.extend(c.get("cautions", []))
                    fixed = parse_flat(c.get("host_formula") or "")
                    if fixed and set(fixed) == elements:
                        a_total, b_total = sum(amounts.values()), sum(fixed.values())
                        if any(amounts[e] * b_total != fixed[e] * a_total for e in elements):
                            flags.append("candidate_formula_ratio_differs_from_sample")
                if len({tuple(sorted(c["host_elements"])) for c in possible}) > 1:
                    flags.append("multiple_host_candidates")
        conflict = source_conflicts.get((doi, sid))
        if conflict:
            flags.append(conflict["data_quality_flag"])
            base["reason_ja"] += " " + conflict["note_ja"]
            base["judgment_status"] = "unresolved_after_review"
            base["host_material"] = base["host_elements"] = base["excluded_elements"] = base["retained_atomic_fractions"] = ""
            base["normalization_status"] = "not_applied_source_conflict"
        base["data_quality_flags"] = ";".join(dict.fromkeys(flags))
        for eid in filter(None, base["evidence_ids"].split(";")):
            assert eid in evidence, (doi, sid, eid)
            evidence[eid]["applicable_sample_count"] += 1
        records.append(base)
        by_doi[doi][base["judgment_status"]] += 1
        by_doi[doi]["curve_count"] += s["curve_count"]
        all_status[base["judgment_status"]] += 1

    paper_rows = []
    for doi in sorted(all_dois):
        counts = by_doi[doi]
        cr, ep = crossref.get(doi, {}), epmc.get(doi, {})
        local = papers.get(doi, [])
        abstract = text_only(metadata.get(doi, {}).get("abstract") or ep.get("abstractText", ""))
        reviewed = manual_papers.get(doi, {})
        paper_rows.append({"doi": doi, "doi_original": ";".join(manifest["doi_spellings"][doi]), "doi_url": "https://doi.org/" + doi, "sample_count": sum(v for k, v in counts.items() if k != "curve_count"), "curve_count": counts["curve_count"], "paper_title": online_titles[doi] or local_title(local), "local_paper_metadata_found": bool(local), "local_paper_records": ";".join(str(r["record_number"]) for r in local), "crossref_request_success": doi in cr_checked, "crossref_doi_found": bool(cr), "europepmc_request_success": doi in ep_checked, "europepmc_doi_found": bool(ep), "abstract_obtained": bool(abstract), "abstract_source": abstract_sources[doi], "open_fulltext_xml_obtained": doi in fulltexts, "fulltext_url": "https://pmc.ncbi.nlm.nih.gov/articles/" + fulltexts[doi]["pmcid"] + "/" if doi in fulltexts else "", "individual_review_level": reviewed.get("review_level", "not_individually_reviewed"), "samples_host_and_roles_confirmed": counts["literature_host_and_roles"], "samples_host_only_confirmed": counts["literature_host_only"], "samples_automatic_candidates": counts["automatic_candidate"], "samples_unresolved": counts["unresolved"] + counts["unresolved_after_review"], "crossref_failure": cr_fail.get(doi, ""), "europepmc_failure": ep_fail.get(doi, ""), "crossref_cache": cr_cache.get(doi, ""), "europepmc_cache": ep_cache.get(doi, ""), "note_ja": "公開要旨・本文の取得と個別試料の母材確認は別。索引で見つからない場合も論文不存在とは判定しない。"})

    for row in paper_rows:
        doi = row["doi"]
        resolver = resolvers.get(doi, {})
        individual = manual_papers.get(doi, {})
        if doi not in metadata and individual.get("bibliography_verified"):
            row["paper_title"] = individual["title"]
        row.update({"datacite_doi_found": doi in datacite, "datacite_cache": dc_cache.get(doi, ""), "doi_resolver_http_status": resolver.get("status", ""), "doi_resolver_url": resolver.get("url", ""), "doi_resolver_error": resolver.get("error", ""), "official_bibliography_verified_individually": bool(individual.get("bibliography_verified")), "official_bibliography_url": individual.get("bibliography_source_url", "")})
    used_evidence = [e for e in evidence.values() if e["applicable_sample_count"]]
    tables = {"sample_host_review": records, "paper_coverage": paper_rows, "host_evidence": used_evidence}
    for name, data in tables.items():
        target = BASE / "cache" / (name + ".jsonl")
        with target.open("w", encoding="utf-8", newline="\n") as f:
            for record in data:
                f.write(compact(record) + "\n")
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "input_counts": manifest["counts"], "doi_total": len(all_dois), "samples_total": len(samples), "crossref_successful_queries": len(cr_checked), "crossref_records": len(crossref), "europepmc_successful_queries": len(ep_checked), "europepmc_records": len(epmc), "online_records_union": len(set(crossref) | set(epmc)), "abstract_obtained_dois": sum(bool(p["abstract_obtained"]) for p in paper_rows), "open_fulltext_xml_dois": len(fulltexts), "individually_reviewed_dois": sum(p.get("review_level") != "primary_source_not_retrieved" for p in manual_papers.values()), "dois_with_confirmed_sample_host": sum(p["samples_host_and_roles_confirmed"] + p["samples_host_only_confirmed"] > 0 for p in paper_rows), "dois_with_all_samples_confirmed_host": sum(p["sample_count"] == p["samples_host_and_roles_confirmed"] + p["samples_host_only_confirmed"] for p in paper_rows), "sample_status_counts": dict(all_status), "samples_with_confirmed_dopant_removal": removed_count, "evidence_records": len(used_evidence), "all_dois_queried_crossref": cr_checked == all_dois, "all_dois_queried_europepmc": ep_checked == all_dois, "unqueried_or_failed_crossref": sorted(all_dois - cr_checked), "unqueried_or_failed_europepmc": sorted(all_dois - ep_checked)}
    official_bib = {d for d, p in manual_papers.items() if p.get("bibliography_verified")}
    summary.update({"datacite_records": len(datacite), "online_records_union": len(set(metadata) | set(epmc)), "doi_resolver_supplement_count": len(resolvers), "official_bibliography_supplement": len(official_bib - set(metadata) - set(epmc)), "bibliography_confirmed_union": len(set(metadata) | set(epmc) | official_bib)})
    dump(BASE / "coverage_summary.json", summary)
    assert len({(r["doi"], r["sample_id"]) for r in records}) == len(samples)
    assert sum(r["curve_count"] for r in records) == manifest["counts"]["curve_rows"]
    assert sum(all_status.values()) == len(samples)
    assert sum(p["sample_count"] for p in paper_rows) == len(samples)
    for r in records:
        if r["retained_atomic_fractions"]:
            assert abs(sum(json.loads(r["retained_atomic_fractions"]).values()) - 1) < 1e-10
            assert r["judgment_status"] == "literature_host_and_roles"
        if r["judgment_status"] == "automatic_candidate":
            assert not r["excluded_elements"] and not r["host_material"]
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("unqueried_or_failed")}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
