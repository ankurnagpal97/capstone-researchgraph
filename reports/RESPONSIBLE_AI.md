# Responsible AI

How AI ResearchGraph handles prompt injection, personal data, refusals and the boundary between what the system
decides and what a human must decide.

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
instruction) and runs 3 attacks through the question itself: 23 runs in total. Results and the model's answers are in
`INJECTION.md`. A quicker check, `tests/test_live.py::test_prompt_injection_in_a_passage_is_ignored`, runs one of them
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
| Gold labels and reference answers used to grade the system | A human reviews every one (`reference_review_sheet.csv`, `gold_review_sheet.csv`) |
| Whether a cited claim is actually supported | Humans label a 60-claim sample (`citation_audit_sample.csv`); the Qwen self-judge is reported but not trusted |
| Choosing an architecture, citing a result in a paper, any decision based on an answer | The user, after opening the cited passage |

If the system were extended with actions (for example emailing a summary or adding papers automatically), each
would need explicit user confirmation before it runs.

## 5. Other risks

- **Over-trust in a fluent answer.** Mitigated by per-claim citations, "not directly comparable" labels, survey
  attribution and visible missing-evidence lists.
- **Bias in the corpus.** Mostly English, arXiv, 2022–2026, 15 surveys; under-represents non-English and industry
  work. Stated in the app and the report.
- **Shared-resource misuse.** Rate limits and caching protect the group's single API key.
