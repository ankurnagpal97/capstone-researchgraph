# Prompting strategies (dev sets, 40 questions, 2026-11-23 23:49)

Same B3 evidence and the same code checks for every strategy; only the prompt changes. Correct = the answer/abstain decision matches the label and, if answerable, a gold paper is cited. Mean [95% bootstrap CI].

| Metric | zero-shot | few-shot | CoT | decompose | CoT + repair |
|---|---|---|---|---|---|
| End-to-end correct | 0.82 [0.70, 0.93] | 0.85 [0.72, 0.95] | 0.28 [0.15, 0.42] | 0.88 [0.78, 0.97] | 0.82 [0.70, 0.93] |
| Answer/abstain decision right | 0.88 [0.78, 0.97] | 0.90 [0.80, 0.97] | 0.28 [0.15, 0.42] | 0.97 [0.93, 1.00] | 0.93 [0.82, 1.00] |
| Gold papers cited (answerable) | 0.73 [0.61, 0.84] | 0.75 [0.64, 0.86] | 0.61 [0.46, 0.75] | 0.75 [0.61, 0.88] | 0.61 [0.46, 0.75] |
| Gold papers in evidence | 0.80 [0.68, 0.91] | 0.80 [0.68, 0.91] | 0.80 [0.68, 0.91] | 0.82 [0.66, 0.95] | 0.80 [0.68, 0.91] |
| Claims per answer | 3.00 [2.50, 3.45] | 3.62 [3.12, 4.12] | 2.23 [1.77, 2.65] | 3.20 [2.60, 3.77] | 2.23 [1.77, 2.65] |
| Tokens per question | 2582.05 [2457.72, 2700.43] | 3116.05 [2873.40, 3465.32] | 2713.95 [2578.22, 2836.18] | 2976.93 [2882.40, 3074.75] | 2713.95 [2578.22, 2836.18] |
| Latency p50 (s) | 8.9 | 7.6 | 9.6 | 12.8 | 9.6 |

**How Qwen3-8B breaks** (share of questions)

| Metric | zero-shot | few-shot | CoT | decompose | CoT + repair |
|---|---|---|---|---|---|
| JSON valid but 'answerable' missing at top level (schema drift) | 0% | 0% | 70% | 0% | 0% |
| Invalid JSON on first try (repaired) | 0% | 2% | 0% | 0% | 0% |
| Invalid JSON twice (no answer) | 0% | 2% | 0% | 0% | 0% |
| Says 'not stated' but marks answerable (code fixed) | 8% | 0% | 0% | 8% | 0% |
| A named method absent from evidence (code flagged) | 5% | 5% | 0% | 5% | 0% |
| 'Directly comparable' across papers (code fixed) | 0% | 0% | 0% | 0% | 0% |
| Cites a passage label that does not exist | 0% | 0% | 0% | 0% | 0% |
| Answered an unanswerable question | 8% | 5% | 0% | 2% | 2% |
| Abstained on an answerable question | 5% | 2% | 2% | 0% | 5% |

**Correct by question type**

| Type | zero-shot | few-shot | CoT | decompose | CoT + repair |
|---|---|---|---|---|---|
| comparison | 15/16 | 15/16 | 3/16 | 16/16 | 13/16 |
| synthetic_definition | 1/1 | 1/1 | 0/1 | 1/1 | 1/1 |
| synthetic_finding | 4/5 | 4/5 | 1/5 | 4/5 | 4/5 |
| synthetic_limitation | 2/2 | 2/2 | 0/2 | 1/2 | 2/2 |
| synthetic_method | 2/2 | 2/2 | 0/2 | 1/2 | 1/2 |
| synthetic_number | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| unanswerable | 9/12 | 9/12 | 7/12 | 11/12 | 11/12 |

**CoT + repair** re-scores the same CoT answers after code lifts the fields Qwen nested inside "reasoning" back to the top level (no new calls).

n = 40, so differences under about 0.15 are not established. These are the dev sets used for earlier prompt tuning, so zero-shot (answer-v3, tuned on them) has a home advantage.
