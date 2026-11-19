# AI ResearchGraph — IIT KGP Capstone, Track B (Qwen3-8B)

Evidence-grounded question answering over **146 research papers on retrieval-augmented generation (RAG) and LLM retrieval** (mostly 2022–2026).
Every claim in an answer cites the exact passage (paper, section, page) it comes from. Results that can't be compared are flagged, and the system says so when the indexed papers don't answer the question.
A citation graph between papers is available as an optional retrieval step. A citation is never treated as evidence or agreement.

Results: [reports/RESULTS.md](reports/RESULTS.md) · Full write-up: [reports/REPORT.md](reports/REPORT.md) · Demo: [reports/DEMO.md](reports/DEMO.md)

## Pipeline

```
seed_papers.tsv ──► expand_corpus.py ──► ingest.py ──► parse_corpus.py ──► build_index.py ──► app.py / ask.py
 (56 hand-picked)    (citation snowball    arXiv metadata,  PDF → sections →    bge-small vectors
                      via OpenAlex, +93)   license, PDF,    chunks + reference-  (cached) + BM25
                                           OpenAlex refs    list citation edges
                                                                                   evaluate.py · eval_qasper.py · make_report.py
```

| Step | What it does | Where |
|---|---|---|
| Corpus | Hand-picked seeds, plus one hop of citation snowballing through OpenAlex (a candidate's score is the number of corpus papers it connects to). Each title is checked against arXiv. The scope rule is 2022+ unless the paper is a highly connected foundation. Licenses are recorded. Each accept or reject decision is logged, and a dated manifest is written. | `data/seed_papers*.tsv`, `scripts/expand_corpus.py`, `scripts/ingest.py`, `data/manifests/` |
| Parsing | PyMuPDF text blocks. Section headings are detected (dates and table-of-contents lines are rejected). The reference list is kept separate. Unicode is normalized (ligatures), arXiv stamps are removed and words split by line-end hyphens are rejoined. | `researchgraph/parse.py` |
| Chunks | Paragraphs within one section, about 300 tokens, about 50 tokens of overlap, at most 1500 characters (under the embedding service's 512-token limit). Section and page range are kept. | `researchgraph/parse.py` |
| Citation edges | OpenAlex reference lists, plus title or arXiv-ID matches in each PDF's reference list. Each edge records its source. | `ingest.py`, `parse_corpus.py` |
| Indexes | Dense: bge-small-en-v1.5 from the course service (384 dimensions, cached by text hash). Sparse: in-memory BM25. | `researchgraph/index.py` |
| Retrieval | B0 BM25 · B1 dense · B2 hybrid (RRF, k = 60) · **B3 hybrid + cross-encoder rerank (default)** · B4 B3 + one-hop citation expansion · B5 B3 + personalized PageRank over the citation graph. At most 3 chunks per paper. | `researchgraph/retrieve.py`, `researchgraph/rerank.py` |
| Answers | One Qwen3-8B call: at most 7 passages / about 3,000 tokens of evidence → JSON claim ledger (≤ 1,024 tokens). The prompt checks the evidence first (`answer-v2`), and the code adds an abstention guard and a citation audit (claims citing passages that weren't provided are flagged). Every answer is logged to `query_runs`. | `researchgraph/answer.py` |
| API client | Retries 429/503 after 2, 4 and 8 seconds, sends 1 chat request at a time, stays under 60 per minute, checks `max_tokens` and batches embeddings 16 at a time. | `api_client.py`, `api_tests.ipynb` |

## Setup

1. Put the certificates in `secrets/` (gitignored): `vm06-ca.crt` for chat (groups 016–030) and `vm02-ca.crt` for embeddings.
2. `cp .env.example .env` and fill in your keys. **Never commit `.env`.** `OPENALEX_API_KEY` is optional: without it, OpenAlex requests share a small free daily budget.
3. Create the environment (works with the macOS system Python 3.9):
   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   python3 check_connection.py              # both endpoints answer
   ```

## Build, run, evaluate

```bash
# corpus (only needed to rebuild; ~15 min, arXiv requests are spaced 3 s apart)
.venv/bin/python scripts/expand_corpus.py --n 100      # optional: regenerate data/seed_papers_expansion.tsv
.venv/bin/python scripts/ingest.py                     # incremental: skips papers already ingested
.venv/bin/python scripts/parse_corpus.py
.venv/bin/python scripts/build_index.py                # embeddings cached by text hash

# use it
.venv/bin/streamlit run app.py --server.address localhost      # http://localhost:8501
.venv/bin/python scripts/ask.py "How does Self-RAG decide when to retrieve?"   # --mode bm25|dense|hybrid|rerank|graph1hop|graph_ppr

# evaluate
.venv/bin/python scripts/make_synthetic_qa.py --n 150  # synthetic dev/test questions (Qwen, ~8 min)
.venv/bin/python scripts/evaluate.py --answers         # B0–B5 on all sets + answers (~15 min)
.venv/bin/python scripts/tune_answers.py               # prompt comparison on the answer-tuning set only
.venv/bin/python scripts/eval_qasper.py --papers 60    # external benchmark (see QASPER below)
.venv/bin/python scripts/make_report.py                # → reports/RESULTS.md
.venv/bin/python -m pytest -q                          # offline unit tests
```

**QASPER** (CC BY 4.0) is not committed. Download it with:
`mkdir -p data/raw/qasper && curl -o data/raw/qasper/t.tgz https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz && tar -xzf data/raw/qasper/t.tgz -C data/raw/qasper`

## Evaluation sets

| Set | Size | Purpose | Gold labels |
|---|---|---|---|
| `questions_v0.jsonl` (curated) | 42: 14 single-paper, 9 comparison, 7 multi-hop, 3 disagreement, 3 metadata, 6 unanswerable | **Main test set** | Gold papers drafted from titles; **team review in `reports/gold_review_sheet.csv`** |
| `synthetic_dev.jsonl` | 105 | Tuning | Exact gold passage (question written from it) |
| `synthetic_test.jsonl` | 45 | Held out, reported once | Exact gold passage |
| `answer_dev.jsonl` | 20 (8 unanswerable + 12 synthetic dev) | Tuning the answer prompt only | Answerable or not |
| QASPER test (60 papers, 194 questions) | external | Independent check of within-paper evidence retrieval | Human annotations |

## Data and licensing

- PDFs, the SQLite database and the indexes stay on your machine (`data/raw/`, `data/*.db` and `data/index/` are gitignored). Only metadata, seeds, manifests and evaluation questions are committed.
- `rights_status` for each paper: `open_license` (Creative Commons), `local_index_only` (arXiv's default licence, which lets arXiv distribute the paper but gives us no right to redistribute it), or `unknown_quarantine` (not indexed).
- Only documented access routes are used: the arXiv API and OAI-PMH (with a fallback when the search API rate-limits), PDFs at 3 seconds apart, and the OpenAlex API. Semantic Scholar was rate-limited without a key, so it isn't used.

## Track B limits and how the design follows them

| Limit | Design response |
|---|---|
| 8192 tokens per request, input and output together | Evidence budget of about 3,000 tokens (at most 7 passages); compact JSON output |
| `max_tokens` of 1024 or less | One JSON answer per call, turned into text in code |
| 1 active request per group key | The client sends chat calls one at a time; evaluations run sequentially |
| Embeddings: 16 texts per request, under 512 tokens each | Batches of 16; a 1500-character chunk cap; if the service reports a text as too long, it is shortened in proportion to the reported token count |
| Qwen3 thinking mode | Disabled on the server (verified in `api_tests.ipynb`), so all output tokens go to the answer |

## Repository layout

```
api_client.py          shared Track B client (chat + embeddings)
api_tests.ipynb        API checklist tests (46/47 pass; thinking-mode warning)
app.py                 Streamlit UI (Ask · Citation neighbours · Corpus & provenance · Evaluation)
researchgraph/         package: db, sources, parse, index, retrieve, rerank, answer, llm
scripts/               expand_corpus, ingest, parse_corpus, build_index, ask, make_synthetic_qa,
                       evaluate, tune_answers, eval_qasper, make_report
data/seed_papers*.tsv  corpus seeds, hand-picked and snowballed (committed)
data/manifests/        dated corpus manifests (committed, metadata only)
data/eval/             evaluation question sets (committed)
reports/               RESULTS.md, REPORT.md, DEMO.md, review sheets, raw JSON results
tests/                 offline unit tests
```
