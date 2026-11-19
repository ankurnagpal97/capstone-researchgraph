# Corpus report

**146 papers · 3,050 pages · 11,148 chunks · 2,075 citation pairs** · manifest `data/manifests/corpus_manifest_20261114.jsonl`

## Scope rule

Included: papers on RAG architectures, dense/sparse/hybrid retrieval and reranking, multi-hop retrieval, graph-based RAG, and RAG evaluation and benchmarks, published 2022 or later. Older papers are included only when tagged *foundation*: hand-picked, or connected to at least 6 corpus papers. Excluded: medical, multimodal, code and tool-retrieval papers, and broad surveys that aren't specific to RAG. A paper also needs an arXiv version (for full text) and a title that matches the seed list's keyword exactly.

## How the corpus was built

1. **56 hand-picked seeds** (`data/seed_papers.tsv`).
2. **Citation snowballing** (`scripts/expand_corpus.py`): for every seed, its references and up to 200 citing works via OpenAlex. Each candidate is scored by the number of corpus papers it connects to. The top 100 that pass the scope, year and arXiv filters are kept, and 7 of those were removed by manual review.
3. **Ingestion** (`scripts/ingest.py`): arXiv metadata and license (OAI-PMH), title check, PDF download (3 s between requests), OpenAlex reference list. Every decision is logged in the manifest.

## Composition

| Scope tag | Papers |
|---|---|
| snowball | 66 |
| foundation | 24 |
| rag_method | 15 |
| survey | 15 |
| retrieval | 8 |
| graph_rag | 6 |
| multihop | 4 |
| evaluation | 4 |
| benchmark | 4 |

| Year | Papers |
|---|---|
| 2017 | 1 |
| 2018 | 2 |
| 2019 | 3 |
| 2020 | 12 |
| 2021 | 6 |
| 2022 | 10 |
| 2023 | 32 |
| 2024 | 50 |
| 2025 | 29 |
| 2026 | 1 |

## Licenses

| Rights status | Papers | Meaning |
|---|---|---|
| open_license | 69 | Creative Commons: reuse allowed with attribution |
| local_index_only | 77 | arXiv's default licence: arXiv may distribute it; we index locally and never redistribute |
| unknown_quarantine | 0 | Not indexed |

PDFs, extracted text and indexes are gitignored. The repository holds only metadata.

## Parsing quality

143 papers parsed cleanly; 3 were flagged for a structure check:

- `2309.01431` Benchmarking Large Language Models in Retrieval-Augmented Generation: no reference list found
- `2502.09891` ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation: no reference list found
- `2502.18139` LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers: no reference list found

Chunks: about 300 estimated tokens each, at most 1,500 characters (under the 512-token embedding limit), within one section, with page ranges kept. Known limitation: tables are flattened to text.

## Citation graph

| Edge source | Edges |
|---|---|
| openalex | 1,549 |
| pdf_reference_match | 1,929 |

2,075 distinct citing→cited pairs; 1,403 were found by both sources.

Most-cited papers inside the corpus:

| Paper | Year | Cited by (corpus papers) |
|---|---|---|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | 94 |
| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | 2020 | 81 |
| [HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering](https://arxiv.org/abs/1809.09600) | 2018 | 81 |
| [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) | 2020 | 60 |
| [Active Retrieval Augmented Generation](https://arxiv.org/abs/2305.06983) | 2023 | 55 |
| [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426) | 2021 | 55 |
| [REPLUG: Retrieval-Augmented Black-Box Language Models](https://arxiv.org/abs/2301.12652) | 2023 | 54 |
| [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299) | 2022 | 54 |
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | 2023 | 53 |
| [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) | 2023 | 50 |

## Rejected and excluded candidates

| arXiv ID | Decision | Reason |
|---|---|---|
| 2107.07566 | reject | published 2021, before 2022 and not tagged foundation |
| 2112.07899 | reject | published 2021, before 2022 and not tagged foundation |
| 2310.07554 | reject | title mismatch: expected 'Retrieve Anything', arXiv has 'A Multi-Task Embedder For Retrieval Augmented LLMs' |
| 2311.05232 | excluded | Manual review: general hallucination survey, not RAG-specific (A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions) |
| 2310.07521 | excluded | Manual review: general factuality survey, not RAG-specific (Survey on Factuality in Large Language Models) |
| 2410.14594 | excluded | Manual review: tool retrieval for agents (out of scope (Toolshed: Scale Tool-Equipped Agents with Advanced RAG-Tool Fusion and Tool Knowledge Bases) |
| 2502.04413 | excluded | Manual review: medical domain (out of scope (MedRAG: Enhancing Retrieval-augmented Generation with Knowledge Graph-Elicited Reasoning for Healthcare Copilot) |
| 2506.14035 | excluded | Manual review: multimodal document retrieval (out of scope (SimpleDoc: Multi‐Modal Document Understanding with Dual‐Cue Page Retrieval and Iterative Refinement) |
| 2502.07223 | excluded | Manual review: tool retrieval for agents (out of scope (Graph RAG-Tool Fusion) |
| 2303.12570 | excluded | Manual review: code completion (out of scope (RepoCoder: Repository-Level Code Completion Through Iterative Retrieval and Generation) |
