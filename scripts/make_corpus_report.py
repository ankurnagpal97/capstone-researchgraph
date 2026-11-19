"""Write reports/CORPUS.md: scope rule, composition, licenses, parse flags, citation graph, every rejection.

Usage: .venv/bin/python scripts/make_corpus_report.py
"""
import glob
import json
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, DB_PATH, ROOT  # noqa: E402


def main():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    papers = [dict(r) for r in c.execute("""SELECT p.*, (SELECT COUNT(*) FROM chunks ch WHERE ch.paper_id = p.paper_id) AS n_chunks
                                            FROM papers p WHERE p.pdf_path IS NOT NULL""")]
    decisions = [dict(r) for r in c.execute("SELECT * FROM decisions WHERE decision != 'accept' ORDER BY paper_id")]
    edges = c.execute("SELECT edge_source, COUNT(*) FROM citation_edges GROUP BY 1").fetchall()
    pairs = c.execute("SELECT COUNT(*) FROM (SELECT DISTINCT citing_paper_id, cited_paper_id FROM citation_edges)").fetchone()[0]
    both = c.execute("""SELECT COUNT(*) FROM (SELECT citing_paper_id, cited_paper_id FROM citation_edges
                        GROUP BY 1, 2 HAVING COUNT(DISTINCT edge_source) = 2)""").fetchone()[0]
    indeg = c.execute("""SELECT p.paper_id, p.title, p.year, COUNT(DISTINCT e.citing_paper_id) n FROM citation_edges e
                         JOIN papers p ON p.paper_id = e.cited_paper_id GROUP BY 1 ORDER BY n DESC LIMIT 10""").fetchall()
    excluded = []
    for line in open(os.path.join(DATA, "seed_papers_expansion.tsv")):
        if line.startswith("#EXCLUDED"):
            reason = line.split("(manual review: ", 1)[1].split(")", 1)[0]
            pid = line.split(") ", 1)[1].split("\t", 1)[0]
            title = line.rsplit("|", 1)[-1].strip()
            excluded.append((pid, title, reason))
    manifest = sorted(glob.glob(os.path.join(DATA, "manifests", "corpus_manifest_*.jsonl")))[-1]

    tags = Counter(p["scope_tag"] for p in papers)
    years = Counter(p["year"] for p in papers)
    lic = Counter(p["rights_status"] for p in papers)
    flagged = [p for p in papers if p["parse_status"] and p["parse_status"].startswith("ok (check")]
    n_chunks = sum(p["n_chunks"] for p in papers)
    n_pages = sum(p["n_pages"] or 0 for p in papers)

    out = ["# Corpus report", "",
           f"**{len(papers)} papers · {n_pages:,} pages · {n_chunks:,} chunks · {pairs:,} citation pairs** · "
           f"manifest `{os.path.relpath(manifest, ROOT)}`", "",
           "## Scope rule", "",
           "Included: papers on RAG architectures, dense/sparse/hybrid retrieval and reranking, multi-hop retrieval, "
           "graph-based RAG, and RAG evaluation and benchmarks, published 2022 or later. Older papers are included only "
           "when tagged *foundation*: hand-picked, or connected to at least 6 corpus papers. Excluded: medical, "
           "multimodal, code and tool-retrieval papers, and broad surveys that aren't specific to RAG. A paper also needs an arXiv "
           "version (for full text) and a title that matches the seed list's keyword exactly.", "",
           "## How the corpus was built", "",
           "1. **56 hand-picked seeds** (`data/seed_papers.tsv`).",
           "2. **Citation snowballing** (`scripts/expand_corpus.py`): for every seed, its references and up to 200 citing "
           "works via OpenAlex. Each candidate is scored by the number of corpus papers it connects to. The top 100 that pass "
           "the scope, year and arXiv filters are kept, and 7 of those were removed by manual review.",
           "3. **Ingestion** (`scripts/ingest.py`): arXiv metadata and license (OAI-PMH), title check, PDF download "
           "(3 s between requests), OpenAlex reference list. Every decision is logged in the manifest.", "",
           "## Composition", "",
           "| Scope tag | Papers |", "|---|---|"]
    out += [f"| {t} | {n} |" for t, n in tags.most_common()]
    out += ["", "| Year | Papers |", "|---|---|"]
    out += [f"| {y} | {years[y]} |" for y in sorted(years)]
    out += ["", "## Licenses", "", "| Rights status | Papers | Meaning |", "|---|---|---|",
            f"| open_license | {lic.get('open_license', 0)} | Creative Commons: reuse allowed with attribution |",
            f"| local_index_only | {lic.get('local_index_only', 0)} | arXiv's default licence: arXiv may distribute it; "
            "we index locally and never redistribute |",
            f"| unknown_quarantine | {lic.get('unknown_quarantine', 0)} | Not indexed |", "",
            "PDFs, extracted text and indexes are gitignored. The repository holds only metadata.", "",
            "## Parsing quality", "",
            f"{len(papers) - len(flagged)} papers parsed cleanly; {len(flagged)} were flagged for a structure check:", ""]
    out += [f"- `{p['paper_id']}` {p['title']}: {p['parse_status'][len('ok (check: '):-1]}" for p in flagged]
    out += ["", "Chunks: about 300 estimated tokens each, at most 1,500 characters (under the 512-token embedding limit), "
            "within one section, with page ranges kept. Known limitation: tables are flattened to text.", "",
            "## Citation graph", "", "| Edge source | Edges |", "|---|---|"]
    out += [f"| {s} | {n:,} |" for s, n in edges]
    out += ["", f"{pairs:,} distinct citing→cited pairs; {both:,} were found by both sources.", "",
            "Most-cited papers inside the corpus:", "", "| Paper | Year | Cited by (corpus papers) |", "|---|---|---|"]
    out += [f"| [{r['title']}](https://arxiv.org/abs/{r['paper_id']}) | {r['year']} | {r['n']} |" for r in indeg]
    out += ["", "## Rejected and excluded candidates", "", "| arXiv ID | Decision | Reason |", "|---|---|---|"]
    out += [f"| {d['paper_id']} | {d['decision']} | {d['reason']} |" for d in decisions]
    out += [f"| {pid} | excluded | Manual review: {reason} ({title}) |" for pid, title, reason in excluded]
    out.append("")
    path = os.path.join(ROOT, "reports", "CORPUS.md")
    open(path, "w").write("\n".join(out))
    print("Wrote", os.path.relpath(path, ROOT), f"({len(papers)} papers, {len(decisions)} rejected, {len(excluded)} excluded)")


if __name__ == "__main__":
    main()
