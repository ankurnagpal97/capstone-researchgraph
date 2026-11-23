# AI ResearchGraph — stage-by-stage evaluation

Run 2026-11-23 20:05 · LLM `Qwen/Qwen3-8B` · embeddings `bge-small-en-v1.5` · reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` · answer prompt `answer-v3` · follow-up prompt `followup-v1`. Means with 95% bootstrap CIs.

**Read with care.** Curated reference answers and gold passages (`data/eval/curated_references.jsonl`) were drafted by Claude from the gold papers and still need team review; the gold passages were picked from each gold paper with help from the cross-encoder, so passage-level numbers may slightly favour B3. Synthetic reference answers were written by Qwen3-8B from one passage, so they share its wording. ROUGE counts word overlap: a correct answer phrased differently scores low, so read it with token F1, semantic similarity and the grounding numbers, not alone.

## S1 · Parsing and chunking (whole corpus)

| Measure | Value |
|---|---|
| Papers indexed / with PDF | 146 / 146 |
| Chunks | 11148 (median 69 per paper) |
| Chunk length p50 / p95 / max (est. tokens) | 252 / 414 / 428 |
| Chunks within the 512-token embedding limit | 100.0% |
| Chunks with a detected section heading | 97.7% |

## curated (n = 42)

### S2–S3 · Retrieval and reranking (top 10)

| Stage | System | Paper Recall@10 | Passage Hit@10 | MRR | nDCG@10 | ROUGE-L recall of reference in top 10 |
|---|---|---|---|---|---|---|
| S2 | B0 BM25 | 0.81 [0.71, 0.91] | 0.58 [0.42, 0.75] | 0.37 [0.24, 0.51] | 0.32 [0.21, 0.43] | 0.57 [0.54, 0.61] |
| S2 | B1 Dense | 0.82 [0.72, 0.92] | 0.53 [0.36, 0.69] | 0.31 [0.19, 0.44] | 0.26 [0.16, 0.37] | 0.57 [0.53, 0.61] |
| S2 | B2 Hybrid | 0.86 [0.77, 0.94] | 0.64 [0.47, 0.78] | 0.43 [0.30, 0.56] | 0.34 [0.25, 0.45] | 0.58 [0.54, 0.62] |
| S3 | B3 Hybrid+rerank | 0.85 [0.76, 0.93] | 0.89 [0.78, 0.97] | 0.64 [0.52, 0.77] | 0.55 [0.44, 0.65] | 0.58 [0.55, 0.61] |
| S3 | B3 + per-entity (app default) | 0.85 [0.77, 0.93] | 0.89 [0.78, 0.97] | 0.69 [0.56, 0.81] | 0.57 [0.47, 0.67] | 0.59 [0.55, 0.62] |

MRR and nDCG use gold passages where the question has them, else gold papers.

### S4–S7 · Evidence pack, answer, grounding, follow-ups (B3 + per-entity, prompt v3)

| Stage | Measure | Value |
|---|---|---|
| S4 Evidence pack | Gold paper in pack (recall) | 0.84 [0.74, 0.92] |
| S4 Evidence pack | Gold passage in pack | 0.86 [0.75, 0.97] |
| S4 Evidence pack | Context precision (pack passages from gold papers) | 0.58 [0.50, 0.66] |
| S4 Evidence pack | ROUGE-L recall of reference in pack (upper bound for the answer) | 0.57 [0.54, 0.60] |
| S5 Answer | ROUGE-1 F1 | 0.37 [0.34, 0.39] |
| S5 Answer | ROUGE-2 F1 | 0.13 [0.11, 0.15] |
| S5 Answer | ROUGE-L F1 (full answer / short answer only) | 0.25 [0.23, 0.27] / 0.29 [0.26, 0.31] |
| S5 Answer | ROUGE-L recall | 0.39 [0.35, 0.43] |
| S5 Answer | Token F1 (SQuAD-style) | 0.30 [0.28, 0.33] |
| S5 Answer | Semantic similarity to reference (bge-small cosine) | 0.89 [0.88, 0.90] |
| S5 Answer | Mean answer length (words) | 112.94 [101.08, 124.94] |
| S5 Answer | Declined unanswerable / answered answerable | 4/6 / 35/36 |
| S6 Grounding | Citation IDs that exist in the evidence | 1.00 [1.00, 1.00] |
| S6 Grounding | Claim words found in its cited passages (ROUGE-L precision) | 0.80 [0.76, 0.84] |
| S6 Grounding | Qwen-judged claims supported | 0.929 |
| S7 Follow-ups | Suggestions generated | 126 |
| S7 Follow-ups | Novelty vs original question (1 − ROUGE-L) | 0.74 [0.71, 0.76] |
| S7 Follow-ups | Covered by the corpus (top rerank score ≥ -5.0) | 1.00 [1.00, 1.00] |

<details><summary>Example reference vs answer</summary>

**q01** (ROUGE-L F1 0.295)

- Reference: Self-RAG trains the LM to emit special reflection tokens. A retrieval token, predicted from the input and preceding generation, decides whether retrieving passages would help and calls the retriever on demand. Critique tokens then judge the relevance of each retrieved passage and whether the generated output is supported and useful, and are used to pick the best continuation.
- Answer: Self-RAG uses reflection tokens to decide when to retrieve and to critique its own generations. Specifically, it categorizes reflection tokens into retrieval and critique tokens. Retrieval tokens are used to determine if retrieving additional information would be helpful, while critique tokens are used to evaluate the quality of the generated output and select the best one. Self-RAG uses reflection tokens to indicate the need for retrieval and to evaluate the quality of its own generation. Reflection tokens are categorized into retrieval and critique tokens to indicate the need for retrieval a
- Follow-ups: How do retrieval tokens in Self-RAG differ from those in Auto-RAG? · What specific mechanisms are used in Self-RAG to evaluate the quality of generated output? · How does the performance of Self-RAG compare when using reflection tokens versus not using them?

**q02** (ROUGE-L F1 0.348)

- Reference: CRAG's lightweight retrieval evaluator scores how relevant the retrieved documents are and maps the score to one of three actions: Correct, Incorrect or Ambiguous. When the action is Incorrect, CRAG discards the retrieved documents and turns to large-scale web search for external knowledge instead; for Correct it refines the documents into knowledge strips, and Ambiguous combines both.
- Answer: When the retrieval evaluator in CRAG judges the retrieved documents to be incorrect, the retrieved documents are discarded, and web searches are resorted to as complementary knowledge sources for corrections. If the action Incorrect is triggered, the retrieved documents will be discarded. Instead, web searches are resorted to and regarded as complementary knowledge sources for corrections.
- Follow-ups: How does CRAG determine if the retrieved documents are incorrect? · How does CRAG's approach compare to Speculative RAG's error handling? · What evaluation metrics are used to assess CRAG's correction mechanism?

**q03** (ROUGE-L F1 0.298)

- Reference: RAPTOR splits text into chunks, embeds them with SBERT as leaf nodes, clusters similar chunks, and uses a language model to summarise each cluster. The summaries are re-embedded and the embed-cluster-summarise cycle repeats until further clustering is infeasible, giving a bottom-up multi-layer tree whose parent nodes summarise their children. At query time it retrieves from different levels of the tree.
- Answer: RAPTOR builds a tree of summaries over a document collection by recursively embedding, clustering, and summarizing text chunks. The process starts with text chunks as leaf nodes, which are clustered, summarized, and re-embedded iteratively until further clustering is infeasible, forming a multi-layered tree structure. RAPTOR builds a tree of summaries by recursively embedding, clustering, and summarizing text chunks. The chunks and their corresponding SBERT embeddings form the leaf nodes of the tree structure. RAPTOR clusters chunks of text, generates text summaries of those clusters, and then
- Follow-ups: How does RAPTOR handle the embedding and clustering of text chunks during the recursive process? · How does RAPTOR's approach compare to SiReRAG's method for organizing information in a tree structure? · What are the limitations of RAPTOR's tree-building process when dealing with very large document collections?

</details>

## synthetic_test (n = 45)

### S2–S3 · Retrieval and reranking (top 10)

| Stage | System | Paper Recall@10 | Passage Hit@10 | MRR | nDCG@10 | ROUGE-L recall of reference in top 10 |
|---|---|---|---|---|---|---|
| S2 | B0 BM25 | 0.98 [0.93, 1.00] | 0.80 [0.67, 0.91] | 0.67 [0.55, 0.79] | 0.70 [0.58, 0.82] | 0.80 [0.75, 0.85] |
| S2 | B1 Dense | 0.96 [0.89, 1.00] | 0.69 [0.56, 0.82] | 0.53 [0.41, 0.66] | 0.57 [0.45, 0.69] | 0.77 [0.71, 0.81] |
| S2 | B2 Hybrid | 1.00 [1.00, 1.00] | 0.84 [0.73, 0.93] | 0.65 [0.53, 0.76] | 0.70 [0.59, 0.80] | 0.81 [0.76, 0.85] |
| S3 | B3 Hybrid+rerank | 1.00 [1.00, 1.00] | 0.87 [0.76, 0.96] | 0.73 [0.61, 0.84] | 0.77 [0.66, 0.87] | 0.81 [0.77, 0.86] |
| S3 | B3 + per-entity (app default) | 1.00 [1.00, 1.00] | 0.87 [0.76, 0.96] | 0.69 [0.58, 0.80] | 0.74 [0.63, 0.84] | 0.82 [0.78, 0.87] |

MRR and nDCG use gold passages where the question has them, else gold papers.

### S4–S7 · Evidence pack, answer, grounding, follow-ups (B3 + per-entity, prompt v3)

| Stage | Measure | Value |
|---|---|---|
| S4 Evidence pack | Gold paper in pack (recall) | 1.00 [1.00, 1.00] |
| S4 Evidence pack | Gold passage in pack | 0.87 [0.76, 0.96] |
| S4 Evidence pack | Context precision (pack passages from gold papers) | 0.42 [0.37, 0.48] |
| S4 Evidence pack | ROUGE-L recall of reference in pack (upper bound for the answer) | 0.81 [0.76, 0.85] |
| S5 Answer | ROUGE-1 F1 | 0.32 [0.28, 0.36] |
| S5 Answer | ROUGE-2 F1 | 0.22 [0.18, 0.27] |
| S5 Answer | ROUGE-L F1 (full answer / short answer only) | 0.28 [0.24, 0.33] / 0.40 [0.34, 0.46] |
| S5 Answer | ROUGE-L recall | 0.71 [0.65, 0.77] |
| S5 Answer | Token F1 (SQuAD-style) | 0.30 [0.26, 0.35] |
| S5 Answer | Semantic similarity to reference (bge-small cosine) | 0.80 [0.77, 0.82] |
| S5 Answer | Mean answer length (words) | 105.71 [90.58, 120.36] |
| S5 Answer | Declined unanswerable / answered answerable | 0/0 / 44/45 |
| S6 Grounding | Citation IDs that exist in the evidence | 1.00 [1.00, 1.00] |
| S6 Grounding | Claim words found in its cited passages (ROUGE-L precision) | 0.77 [0.72, 0.82] |
| S6 Grounding | Qwen-judged claims supported | – |

<details><summary>Example reference vs answer</summary>

**s006** (ROUGE-L F1 0.122)

- Reference: RoPE and AliBI position encodings have shown potential to enable length extrapolation.
- Answer: Techniques used to enable length extrapolation in models handling long context inputs include RoPE (Su et al., 2021), AliBI (Press et al., 2021), position reorganization (Jin et al., 2024; An et al., 2024), position interpolation (Chen et al., 2023a; Peng et al., 2023; Liu et al., 2024), and landmark attention (Mohtashami & Jaggi, 2023). These methods allow models to handle longer context inputs by modifying positional encodings or attention mechanisms. RoPE and AliBI have been used to enable length extrapolation in models handling long context inputs. Position interpolation has been used to e

**s009** (ROUGE-L F1 0.286)

- Reference: Bi-encoder systems compute cosine distances between pre-encoded documents and online-encoded queries, while cross-encoder systems use full attention between query and document terms but require |D| encoding passes online, making them slower.
- Answer: Bi-encoder systems are more computationally efficient as they map documents offline and compute matching scores through fast cosine distance. Cross-encoder systems are less efficient as they require online computation of full attention between query and document terms, which involves multiple encoding passes. Bi-encoder systems map documents offline to a dense vector space and compute matching scores through fast cosine distance computation. Cross-encoder systems concatenate query and document as a single input sequence and compute full attention between query and document terms, which is comp

**s012** (ROUGE-L F1 0.111)

- Reference: MMOA-RAG introduces a multi-agent reinforcement learning framework with trainable components and shared rewards, whereas traditional RAG does not involve parameter updates during training.
- Answer: The main difference between the RAG framework described in MMOA-RAG and traditional RAG approaches lies in the temporal nature and evolution patterns of the content being handled. Traditional RAG systems deal with static, historically complete documents, while MMOA-RAG must handle dynamic content that evolves through real-time updates, ongoing conversations, and emerging social interactions. The main difference between the RAG framework described in MMOA-RAG and traditional RAG approaches is the temporal nature and evolution patterns of the content being handled.

</details>
