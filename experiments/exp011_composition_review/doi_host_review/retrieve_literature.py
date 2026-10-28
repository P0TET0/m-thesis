"""Read-only Starrydata joins and resumable public DOI metadata retrieval.

The source CSVs are never modified. Network success is not host verification.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\miots\OneDrive - 山陽小野田市立山口東京理科大学 (1)\USBにいたヤツ\大学\1_研究\starrydata2")
UA = "StarrydataHostReview/1.0 (public bibliographic research; no credentials)"


def normalize_doi(value):
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I).lower()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def read(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def inventory():
    samples = {}
    papers = defaultdict(list)
    doi_spellings = defaultdict(set)
    rows = 0
    files = {}
    for filename in ("starrydata_curves.csv", "starrydata_samples.csv", "starrydata_papers.csv"):
        path = SOURCE / filename
        files[filename] = {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    with (SOURCE / "starrydata_curves.csv").open(encoding="utf-8-sig", newline="") as f:
        for rownum, r in enumerate(csv.DictReader(f), 2):
            doi = normalize_doi(r["DOI"])
            doi_spellings[doi].add(r["DOI"])
            key = (doi, r["sample_id"])
            s = samples.setdefault(key, {"doi": doi, "sample_id": r["sample_id"], "compositions": set(), "curve_count": 0, "SIDs": set(), "figures": set(), "curve_first_record": rownum})
            s["compositions"].add(r["composition"])
            s["SIDs"].add(r["SID"])
            s["figures"].add(r["figure_id"] + ":" + r.get("figure_name", ""))
            s["curve_count"] += 1
            rows += 1
    sample_join_count = 0
    for s in samples.values():
        for field in ("compositions", "SIDs", "figures"):
            s[field] = sorted(s[field])
    with (SOURCE / "starrydata_samples.csv").open(encoding="utf-8-sig", newline="") as f:
        for rownum, r in enumerate(csv.DictReader(f), 2):
            key = (normalize_doi(r["DOI"]), r["sample_id"])
            if key not in samples:
                continue
            s = samples[key]
            s.setdefault("metadata_records", []).append(rownum)
            s["sample_name"] = r["sample_name"]
            s["composition_details"] = r["composition_details"]
            try:
                info = json.loads(r["sample_info"]) if r["sample_info"] else {}
                info = {k: v for k, v in info.items() if any(str(x).strip() for x in v.values())} if isinstance(info, dict) else {"raw": info}
            except (ValueError, AttributeError):
                info = {"unparsed": r["sample_info"]}
            s["sample_info"] = info
            s["metadata_composition"] = r["composition"]
            sample_join_count += 1
    with (SOURCE / "starrydata_papers.csv").open(encoding="utf-8-sig", newline="") as f:
        for rownum, r in enumerate(csv.DictReader(f), 2):
            doi = normalize_doi(r["DOI"])
            if doi in doi_spellings:
                papers[doi].append({"record_number": rownum, **{k: r[k] for k in ("SID", "DOI", "URL", "title", "publisher", "issued")}})
    counts = {"curve_rows": rows, "doi_raw_distinct": sum(len(v) for v in doi_spellings.values()), "doi_normalized_distinct": len(doi_spellings), "sample_doi_keys": len(samples), "sample_id_distinct": len({k[1] for k in samples}), "sample_metadata_matches": sample_join_count, "local_papers_doi_matches": len(papers), "composition_conflicts": sum(len(s["compositions"]) != 1 for s in samples.values())}
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "source_files": files, "counts": counts, "doi_spellings": {k: sorted(v) for k, v in sorted(doi_spellings.items())}}
    dump(BASE / "cache" / "manifest.json", manifest)
    dump(BASE / "cache" / "samples.json.gz", list(samples.values()))
    dump(BASE / "cache" / "papers.json.gz", papers)
    print(json.dumps(counts), flush=True)


def fetch_json(url, retry=3):
    error = ""
    for attempt in range(retry):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=60) as response:
                return {"ok": True, "status": response.status, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "url": url, "headers": {k: v for k, v in response.headers.items() if k.lower().startswith("x-rate") or k.lower().startswith("x-concurrency")}, "data": json.load(response)}
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            error = str(exc)
            if attempt + 1 < retry:
                time.sleep(3 * (attempt + 1))
    return {"ok": False, "url": url, "error": error, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}


def fetch_service(service):
    dois = list(read(BASE / "cache" / "manifest.json")["doi_spellings"])
    # Arrays return an OR across repeated doi filters. The public Crossref
    # array endpoint advertises 1 request/second and 1 concurrent request.
    size = 70 if service == "crossref" else 40
    blocks = [dois[i:i + size] for i in range(0, len(dois), size)]
    for ix, batch in enumerate(blocks):
        path = BASE / "cache" / service / f"batch_{ix:04d}.json.gz"
        if path.exists():
            cached = read(path)
            if cached.get("ok") and cached.get("requested_dois") == batch:
                continue
        if service == "crossref":
            params = {"filter": ",".join("doi:" + d for d in batch), "rows": 1000, "select": "DOI,title,abstract,URL,link,license,type"}
            url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
        else:
            params = {"query": " OR ".join('DOI:"' + d.replace('"', '') + '"' for d in batch), "format": "json", "resultType": "core", "pageSize": 1000}
            url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(params)
        started = time.monotonic()
        result = fetch_json(url)
        result["requested_dois"] = batch
        dump(path, result)
        if ix % 10 == 0 or ix == len(blocks) - 1 or not result["ok"]:
            print(service, ix + 1, "/", len(blocks), "ok", result["ok"], flush=True)
        time.sleep(max(0, 1.1 - (time.monotonic() - started)))


def fetch_fulltext():
    records = {}
    for path in (BASE / "cache" / "europepmc").glob("*.json.gz"):
        obj = read(path)
        if not obj["ok"]:
            continue
        wanted = set(obj["requested_dois"])
        for r in obj["data"].get("resultList", {}).get("result", []):
            doi = normalize_doi(r.get("doi", ""))
            if doi in wanted and r.get("pmcid") and r.get("isOpenAccess") == "Y":
                records[r["pmcid"]] = doi
    def one(item):
        pmcid, doi = item
        path = BASE / "cache" / "fulltext" / f"{pmcid}.xml.gz"
        meta = BASE / "cache" / "fulltext" / f"{pmcid}.json"
        if path.exists():
            return True
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=45) as r:
                data = r.read()
                if b"<article" not in data:
                    raise ValueError("response contains no article XML")
                root = ET.fromstring(data)
                article_dois = {normalize_doi("".join(node.itertext())) for node in root.findall(".//front/article-meta/article-id") if node.attrib.get("pub-id-type") == "doi"}
                if article_dois and doi not in article_dois:
                    raise ValueError("fulltext article DOI differs from requested DOI")
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wb") as f:
                f.write(data)
            dump(meta, {"doi": doi, "pmcid": pmcid, "url": url, "ok": True, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": hashlib.sha256(data).hexdigest()})
            return True
        except Exception as exc:
            dump(meta, {"doi": doi, "pmcid": pmcid, "url": url, "ok": False, "error": str(exc)})
            return False
    success = 0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for ix, future in enumerate(as_completed([pool.submit(one, item) for item in records.items()]), 1):
            success += future.result()
            if ix % 50 == 0 or ix == len(records):
                print("fulltext", ix, "/", len(records), "success", success, flush=True)
    dump(BASE / "cache" / "fulltext_summary.json", {"eligible_oa_pmcids": len(records), "downloaded": success})


def fetch_other_registries():
    dois = set(read(BASE / "cache" / "manifest.json")["doi_spellings"])
    found = set()
    for path in (BASE / "cache" / "crossref").glob("*.json.gz"):
        result = read(path)
        if result.get("ok"):
            found.update(normalize_doi(r["DOI"]) for r in result["data"]["message"]["items"])
    for doi in sorted(dois - found):
        key = hashlib.sha256(doi.encode()).hexdigest()[:16]
        path = BASE / "cache/datacite" / (key + ".json.gz")
        if path.exists():
            continue
        url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi, safe="")
        result = fetch_json(url, retry=1)
        result["requested_doi"] = doi
        if not result.get("ok"):
            resolve_url = "https://doi.org/" + urllib.parse.quote(doi, safe="/")
            try:
                with urllib.request.urlopen(urllib.request.Request(resolve_url, headers={"User-Agent": UA}, method="HEAD"), timeout=25) as r:
                    result["resolver"] = {"url": r.url, "status": r.status}
            except urllib.error.HTTPError as exc:
                result["resolver"] = {"url": exc.url, "status": exc.code, "error": str(exc)}
            except Exception as exc:
                result["resolver"] = {"url": resolve_url, "error": str(exc)}
        dump(path, result)
        print("datacite", doi, result["ok"], result.get("resolver", {}), flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inventory", "crossref", "europepmc", "fulltext", "registries"])
    command = parser.parse_args().command
    if command == "inventory":
        inventory()
    elif command == "fulltext":
        fetch_fulltext()
    elif command == "registries":
        fetch_other_registries()
    else:
        fetch_service(command)
