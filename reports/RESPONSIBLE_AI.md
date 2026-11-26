# Model card and Responsible AI

What AI ResearchGraph is for, how it fails, and how it handles prompt injection, personal data, refusals and the
boundary between what the system decides and what a human must decide. Numbers are from the final configuration on
the 62-question curated test set unless stated; full tables are in [REPORT.md](REPORT.md).

## Model card

| Item | Detail |
|---|---|
| **System** | AI ResearchGraph: retrieval-augmented question answering with passage-level citations over 146 papers (11,148 chunks) on RAG, retrieval, multi-hop reasoning and RAG evaluation, 2022–2026 |
| **Models** | Generator, planner and critic: Qwen3-8B (course Track B API; 8,192-token context, ≤ 1,024 output tokens). Embeddings: bge-small-en-v1.5 (384-d). Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2` (local CPU). No model was fine-tuned. |
| **Default configuration** | B3: hybrid BM25 + dense retrieval (RRF) → cross-encoder rerank → ≤ 7 passages → one JSON claim-ledger call (prompt v3) → code checks (citation audit, abstention guard, named-thing and comparability checks). Agent mode is optional. |
| **Training / index data** | arXiv PDFs with a recorded licence decision, indexed locally and not redistributed; corpus built by citation snowballing from 56 hand-picked seeds |
| **Evaluation data** | 62 curated questions (9 unanswerable), 45 synthetic held-out questions, QASPER test (194 questions with human evidence labels), 40 dev questions, 23 prompt-injection attacks |

### Intended use

- **Users:** engineers and researchers choosing or comparing retrieval-augmented generation methods.
- **Uses:** find what the indexed papers say about a method, compare methods with the evidence for each side, trace
  which paper a claim comes from, and learn when the literature does *not* settle a question.
- **How to use answers:** as a research aid. Every claim links to the exact passage; open it before relying on the
  claim, and treat "not directly comparable" labels as a reason to read the papers' settings.

### Out of scope

- Questions outside the corpus topic, including medical, legal or financial advice (the system declines or answers
  only with what the papers say about RAG methods).
- Open-web or up-to-date facts: the corpus is a fixed snapshot of 146 papers.
- Numeric comparisons that depend on tables (tables are flattened to text).
- Any automated decision: the system takes no actions (§4).

### Failure modes (measured)

| Failure | How often | Mitigation in place |
|---|---|---|
| Answers a question the papers can't answer, usually from a related passage | 2 of 9 unanswerable questions (q15, q41); 5–8% of dev answers before the guard | Abstention guard, named-thing check and domain rule in code; answer still cites its passages, so the gap is visible |
| Declines an answerable question (over-refusal) | 1 of 53 (q33, a three-paper lineage question) | Tracked as its own failure; agent mode retries with new searches |
| Comparison or lineage answer rests on only one side | All gold papers in the evidence for 8 of 19 comparison/multi-hop/disagreement questions | Per-entity retrieval (one search per named method) |
| Disagreement questions retrieve the wrong papers | Paper Recall@10 0.33 (n = 3) | Answers say "not established; depends on the setting" rather than picking a side |
| A claim only partly supported by its passage | Self-judge: 5.8% partial, 2.6% not supported (an upper bound on quality, since the model grades itself) | Every claim shows its passage; 100% of citation labels are verified in code |
| Survey passages turned into first-hand claims | Seen in q41 before the fix | Survey passages are labelled and must be attributed |
| Schema drift under chain-of-thought prompting | 70% of CoT answers nested decision fields | Production prompt is structured, not CoT; nested fields are lifted in code |
| Planted false *facts* in a document | Not observed in the 23-attack suite, but an 8B model can repeat a plausible false fact | Only arXiv documents with a recorded licence decision are ingested; the citation makes the source traceable |
| Slow or stalled API responses | One request stalled ~27 min during evaluation | Retries on dropped connections and timeouts; per-session and app-wide rate limits; answer cache |
| Agent critic asks for a second round too often | 48% of agent tasks | Hard caps (2 rounds, 8 tool calls, 20k tokens); the pipeline stays the default |

## 1. Prompt injection

**Threat.** The corpus is 146 third-party PDFs. A paper (or a future added document) could contain text addressed
to the model ("ignore your instructions…"). The user's question can also contain injections.

**Defences, in order of strength:**

1. **Code, not the prompt, decides what is shown as a source.** Every citation label is checked against the
   evidence pack; a label that does not exist is flagged and never rendered as a source. The abstention, named-entity
   and comparability checks also run in code, so a passage cannot talk the system out of them.
2. **The agent cannot be steered into actions.** The planner never sees passage text (only the question and earlier
   turns), the executor runs only four whitelisted, read-only tools, unknown tool names are rejected, and every task
   has hard caps on rounds, tool calls and tokens.
3. **Passages are framed as quoted data.** The system prompt says passages are data and instructions inside them
   must be ignored (rule 7).
4. **Answers stay inspectable.** Every claim links to the passage behind it, so a reader can see if a strange claim
   came from a strange passage.

**Tests.** `scripts/injection_suite.py` plants 10 kinds of attack in a fake paper inserted at rank 2 of the real
evidence for two questions (instruction override, fake system message, citation forgery, prompt exfiltration,
false-fact planting, phishing link, format breaking, role-play jailbreak, forced abstention, base64-encoded
instruction) and runs 3 attacks through the question itself: 23 runs in total. **Result: 23/23 defended (attack
success rate 0%); the poisoned passage was never cited.** Results and the model's answers are in `INJECTION.md`.
Refusal is tested by the 9 unanswerable curated questions (7 declined by the pipeline, 8 by the agent) and an
out-of-scope medical question in the live tests (declined). A quicker check, `tests/test_live.py::test_prompt_injection_in_a_passage_is_ignored`, runs one of them
with `RG_LIVE=1`.

**Known weakness.** Defence 3 depends on an 8B model following its instructions. An attack that plants a plausible
false *fact* (rather than an instruction) is the hardest case: the model may repeat it with a citation. The citation
makes it traceable, but not prevented. Only documents from arXiv with a recorded licence decision are ingested
(`decisions` table), which limits who can add text to the corpus.

## 2. Personal data (PII)

| Data | Handling |
|---|---|
| The user's question | Scrubbed before it reaches the model, the database logs (`query_runs`, `agent_turns`), the traces (`logs/traces.jsonl`) or the answer cache. Emails, phone numbers, card numbers (Luhn-checked), Aadhaar and PAN numbers, IP addresses and key-like strings become `[EMAIL]`, `[PHONE]`, … The app tells the user what was removed. Research numbers, years and arXiv IDs are kept (unit-tested). |
| Paper text | Public academic text; author names and affiliations are part of the citation record and are not removed. Full texts are stored locally only and not redistributed. |
| Accounts | The app has no login and stores no names. A browser session gets a random ID, used only for rate limiting and agent memory; "New conversation" deletes that session's turns. |
| Secrets | API keys and CA certificates live in `.env` and `secrets/` (gitignored, never in the Docker image; mounted at runtime). Git history was checked: none were ever committed. |
| Third parties | Questions go only to the course's Qwen3-8B and embedding endpoints. Tracing writes locally; nothing is sent to an external tracing service unless an OTLP endpoint is configured. |

## 3. Refusal behaviour

The system declines rather than guesses. It answers "Not established by the indexed papers", states what is missing,
and still lists any related, cited context. Refusal triggers: no passage directly answers the question; the question
is about a different domain, population or product than the passages; a named method or model is absent from the
evidence (code check); the model says "not stated" but marks the question answerable (code check). Measured on the
unanswerable curated questions and the dev set (see `STAGES.md`, `AGENT.md`, `PROMPTS.md`). Over-refusal (declining
an answerable question) is tracked as its own failure.

Out of scope by design: medical, legal and financial advice. Questions like "What is the best RAG method for medical
diagnosis in hospitals?" are answered only with what the papers say about RAG methods, which is usually an
abstention.

## 4. Human-in-the-loop boundary

**The system takes no consequential actions.** It reads an indexed corpus and returns cited text. It cannot send,
buy, delete, publish or change anything outside its own logs and cache, and the agent's tools are read-only.

**Humans decide; the system informs.** Answers are research aids, not conclusions:

| Decision | Who makes it |
|---|---|
| Which papers enter the corpus | A human (seed lists, logged accept/reject decisions with reasons) |
| Questions and gold papers used to grade the system | The project team (curated set); reference answers were drafted with an AI assistant from the gold passages and are stated as such |
| Whether a cited claim is actually supported | The reader, by opening the cited passage; `citation_audit_sample.csv` lists 60 sampled claims with their passages. The Qwen self-judge is reported as an estimate, not trusted on its own |
| Choosing an architecture, citing a result in a paper, any decision based on an answer | The user, after opening the cited passage |

If the system were extended with actions (for example emailing a summary or adding papers automatically), each
would need explicit user confirmation before it runs.

## 5. Other risks

- **Over-trust in a fluent answer.** Mitigated by per-claim citations, "not directly comparable" labels, survey
  attribution and visible missing-evidence lists.
- **Bias in the corpus.** Mostly English, arXiv, 2022–2026, 15 surveys; under-represents non-English and industry
  work. Stated in the app and the report.
- **Shared-resource misuse.** Rate limits and caching protect the group's single API key.
