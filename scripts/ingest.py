"""Step 1: build the corpus from data/seed_papers.tsv.

For each seed: fetch arXiv metadata, verify the title, apply the scope rule, record the license,
download the PDF (local only, gitignored), look up OpenAlex citations. Every accept/reject decision
is logged, and a dated manifest (metadata only, no text) is written to data/manifests/.

Usage: .venv/bin/python scripts/ingest.py [--limit N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, PDF_DIR, db, sources  # noqa: E402

SCOPE_START_YEAR = 2022


SEED_FILES = ("seed_papers.tsv", "seed_papers_expansion.tsv")
REFS_CACHE = os.path.join(DATA, "raw", "openalex_refs.json")


def read_seeds():
    seeds, seen = [], set()
    for name in SEED_FILES:
        path = os.path.join(DATA, name)
        if not os.path.exists(path):
            continue
        for line in open(path):
            if line.strip() and not line.startswith("#"):
                arxiv_id, tag, keyword = [x.strip() for x in line.rstrip("\n").split("\t")[:3]]
                if arxiv_id not in seen:
                    seen.add(arxiv_id)
                    seeds.append((arxiv_id, tag, keyword))
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    con = db.connect()
    seeds = read_seeds()[: args.limit]
    refs_cache = json.load(open(REFS_CACHE)) if os.path.exists(REFS_CACHE) else {}
    existing = {r["paper_id"]: dict(r) for r in con.execute(
        "SELECT * FROM papers WHERE pdf_path IS NOT NULL AND license IS NOT NULL")}
    os.makedirs(PDF_DIR, exist_ok=True)

    print(f"Fetching arXiv metadata for {len(seeds)} seeds…")
    meta = {pid: {k: existing[pid][k] for k in ("paper_id", "version", "title", "abstract", "published", "updated",
                                                 "doi", "url", "year")} for pid in existing}
    for pid in meta:
        meta[pid]["authors"] = json.loads(existing[pid]["authors"] or "[]")
        meta[pid]["categories"] = json.loads(existing[pid]["categories"] or "[]")
    new_ids = [s[0] for s in seeds if s[0] not in existing]
    oai_license = {}
    try:
        for i in range(0, len(new_ids), 100):
            meta.update(sources.arxiv_metadata(new_ids[i:i + 100]))
    except Exception as e:                       # search API rate-limited: use OAI-PMH, one record per paper
        print(f"arXiv search API unavailable ({e}); fetching metadata from OAI-PMH instead")
        for n, pid in enumerate(new_ids):
            if pid in meta:
                continue
            try:
                rec = sources.arxiv_record_oai(pid)
            except Exception as e2:
                print(f"  OAI failed for {pid}: {e2}")
                continue
            if rec:
                oai_license[pid] = rec.pop("license")
                meta[pid] = rec
            if n % 20 == 0:
                print(f"  {n + 1}/{len(new_ids)} records")

    def decide(pid, decision, reason):
        db.upsert(con, "decisions", {"paper_id": pid, "decision": decision, "reason": reason, "decided_at": now})
        print(f"  {decision.upper():7} {pid}: {reason}")

    con.execute("DELETE FROM decisions")
    accepted = []
    for arxiv_id, tag, keyword in seeds:
        m = meta.get(arxiv_id)
        if not m:
            decide(arxiv_id, "reject", "not found on arXiv")
            continue
        norm = lambda t: " ".join(re.sub(r"[^a-z0-9]+", " ", t.lower()).split())
        if norm(keyword) not in norm(m["title"]):
            decide(arxiv_id, "reject", f"title mismatch: expected '{keyword}', arXiv has '{m['title']}'")
            continue
        if m["year"] < SCOPE_START_YEAR and tag != "foundation":
            decide(arxiv_id, "reject", f"published {m['year']}, before {SCOPE_START_YEAR} and not tagged foundation")
            continue

        if arxiv_id in existing:                                # already ingested: no arXiv requests
            if arxiv_id not in refs_cache:
                oa = sources.openalex_work(arxiv_id)
                refs_cache[arxiv_id] = [r.rsplit("/", 1)[-1] for r in (oa or {}).get("referenced_works", [])]
            accepted.append((arxiv_id, refs_cache[arxiv_id]))
            decide(arxiv_id, "accept", f"{tag}, {m['year']}, {existing[arxiv_id]['rights_status']} (already ingested)")
            continue
        lic = oai_license[arxiv_id] if arxiv_id in oai_license else sources.arxiv_license(arxiv_id)
        status = sources.rights_status(lic)
        row = dict(m, license=lic, rights_status=status, scope_tag=tag, retrieved_at=now)
        if status == "unknown_quarantine":
            row["parse_status"] = "skipped: rights unknown"
            db.upsert(con, "papers", row)
            decide(arxiv_id, "quarantine", f"license unclear ({lic})")
            continue

        pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}{m['version'] or ''}.pdf")
        if not os.path.exists(pdf_path):
            try:
                data = sources.arxiv_pdf(f"{arxiv_id}{m['version'] or ''}")
            except Exception as e:  # keep going; record the failure
                row["parse_status"] = f"failed: download ({e})"
                db.upsert(con, "papers", row)
                decide(arxiv_id, "reject", f"PDF download failed: {e}")
                continue
            with open(pdf_path, "wb") as f:
                f.write(data)
        row["pdf_path"] = os.path.relpath(pdf_path, DATA)
        row["pdf_sha256"] = hashlib.sha256(open(pdf_path, "rb").read()).hexdigest()

        try:
            oa = sources.openalex_work(arxiv_id)
        except Exception as e:
            oa = None
            print(f"    (OpenAlex lookup failed: {e})")
        row["openalex_id"] = oa["id"].rsplit("/", 1)[-1] if oa else None
        row["_refs"] = [r.rsplit("/", 1)[-1] for r in (oa or {}).get("referenced_works", [])]
        row.setdefault("parse_status", "pending")
        refs = row.pop("_refs")
        refs_cache[arxiv_id] = refs
        db.upsert(con, "papers", row)
        accepted.append((arxiv_id, refs))
        decide(arxiv_id, "accept", f"{tag}, {m['year']}, {status}")
        con.commit()

    with open(REFS_CACHE, "w") as f:
        json.dump(refs_cache, f)

    # Citation edges inside the corpus, from OpenAlex reference lists
    oa_to_paper = {r["openalex_id"]: r["paper_id"] for r in con.execute(
        "SELECT paper_id, openalex_id FROM papers WHERE openalex_id IS NOT NULL")}
    con.execute("DELETE FROM citation_edges WHERE edge_source = 'openalex'")
    n_edges = 0
    for pid, refs in accepted:
        for w in refs:
            cited = oa_to_paper.get(w)
            if cited and cited != pid:
                db.upsert(con, "citation_edges", {"citing_paper_id": pid, "cited_paper_id": cited,
                                                  "edge_source": "openalex", "discovered_at": now})
                n_edges += 1
    con.commit()

    # Dated manifest: metadata, license, hash and decision for every seed (no paper text)
    stamp = dt.date.today().strftime("%Y%m%d")
    out = os.path.join(DATA, "manifests", f"corpus_manifest_{stamp}.jsonl")
    decisions = {r["paper_id"]: dict(r) for r in con.execute("SELECT * FROM decisions")}
    with open(out, "w") as f:
        for arxiv_id, tag, _ in seeds:
            p = con.execute("SELECT * FROM papers WHERE paper_id = ?", (arxiv_id,)).fetchone()
            rec = {"paper_id": arxiv_id, "scope_tag": tag, **decisions.get(arxiv_id, {})}
            if p:
                rec.update({k: p[k] for k in ("version", "title", "year", "doi", "openalex_id", "url",
                                              "license", "rights_status", "pdf_sha256", "retrieved_at")})
            f.write(json.dumps(rec) + "\n")
    n_acc = sum(1 for d in decisions.values() if d["decision"] == "accept")
    print(f"\nAccepted {n_acc}/{len(seeds)} papers, {n_edges} OpenAlex citation edges inside the corpus.")
    print(f"Manifest: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
