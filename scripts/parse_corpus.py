"""Step 2: parse every accepted PDF into section-aware chunks and add reference-list citation edges.

Usage: .venv/bin/python scripts/parse_corpus.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, db, parse  # noqa: E402


def main():
    con = db.connect()
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    papers = con.execute("SELECT paper_id, title, pdf_path FROM papers WHERE pdf_path IS NOT NULL").fetchall()
    corpus_titles = {p["paper_id"]: p["title"] for p in papers}
    con.execute("DELETE FROM chunks")
    con.execute("DELETE FROM citation_edges WHERE edge_source = 'pdf_reference_match'")

    stats = []
    for p in papers:
        pid = p["paper_id"]
        try:
            blocks, n_pages = parse.extract_blocks(os.path.join(DATA, p["pdf_path"]))
            sections, refs = parse.split_sections(blocks)
            chunks = parse.chunk_sections(pid, sections)
            problems = []
            if len(sections) < 4:
                problems.append(f"only {len(sections)} sections detected")
            if not refs:
                problems.append("no reference list found")
            if len(chunks) < 10:
                problems.append(f"only {len(chunks)} chunks")
            status = "ok" if not problems else "ok (check: " + "; ".join(problems) + ")"
        except Exception as e:
            con.execute("UPDATE papers SET parse_status = ? WHERE paper_id = ?", (f"failed: {e}", pid))
            print(f"FAILED {pid}: {e}")
            continue

        for c in chunks:
            db.upsert(con, "chunks", c)
        for cited in parse.reference_matches(refs, corpus_titles, pid):
            db.upsert(con, "citation_edges", {"citing_paper_id": pid, "cited_paper_id": cited,
                                              "edge_source": "pdf_reference_match", "discovered_at": now})
        con.execute("UPDATE papers SET parse_status = ?, n_pages = ? WHERE paper_id = ?", (status, n_pages, pid))
        stats.append((pid, n_pages, len(sections), len(chunks), status))
        con.commit()

    for pid, n_pages, n_sec, n_chunks, status in stats:
        flag = "" if status == "ok" else "  <-- " + status
        print(f"{pid:12} {n_pages:3} pages {n_sec:3} sections {n_chunks:4} chunks{flag}")
    tok = [r[0] for r in con.execute("SELECT est_tokens FROM chunks")]
    edges = Counter(r[0] for r in con.execute("SELECT edge_source FROM citation_edges"))
    print(f"\n{len(stats)} papers, {len(tok)} chunks; est. tokens per chunk: "
          f"min {min(tok)}, median {sorted(tok)[len(tok)//2]}, max {max(tok)}")
    print("citation edges by source:", dict(edges))
    both = con.execute("""SELECT COUNT(*) FROM (SELECT citing_paper_id, cited_paper_id FROM citation_edges
                          GROUP BY 1, 2 HAVING COUNT(DISTINCT edge_source) = 2)""").fetchone()[0]
    print("edges found by both sources:", both)


if __name__ == "__main__":
    main()
