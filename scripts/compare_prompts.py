"""Prompting strategies on the dev sets: zero-shot structured vs few-shot vs chain-of-thought vs decomposition.

Usage: .venv/bin/python scripts/compare_prompts.py [--reuse]

Dev sets only (answer_dev + comparison_dev, 40 questions), so the curated test set stays untouched.
Every strategy gets the same B3 retrieval (hybrid + rerank, no per-entity step, so the decomposition
strategy's own effect is visible) and the same code checks; only the prompt differs.
  zero-shot  answer-v3: rules + JSON schema                                 1 call
  few-shot   answer-v3 + two worked examples (one answer, one abstention)   1 call
  CoT        answer-v3 + a "reasoning" field written before the answer      1 call
  decompose  a planning call splits the question into <= 3 sub-questions;   2 calls
             each is retrieved separately; evidence merged; answer-v3 answers all
Also records how the small model breaks: invalid JSON, contradictions caught by the code checks, invalid
citations, answering the unanswerable, over-abstaining.
Output: reports/prompt_comparison.json, reports/PROMPTS.md (answers cached in reports/prompt_answers.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from researchgraph import DATA, ROOT, agent, answer  # noqa: E402
from researchgraph.llm import ApiError, client  # noqa: E402
from researchgraph.retrieve import Result, Retriever  # noqa: E402
from eval_stages import summ  # noqa: E402

SETS = ["answer_dev.jsonl", "comparison_dev.jsonl"]
CACHE = os.path.join(ROOT, "reports", "prompt_answers.jsonl")
OUT_JSON = os.path.join(ROOT, "reports", "prompt_comparison.json")
OUT_MD = os.path.join(ROOT, "reports", "PROMPTS.md")
STRATEGIES = {"zero-shot": "answer-v3", "few-shot": "answer-v3-fewshot", "CoT": "answer-v3-cot",
              "decompose": "answer-v3-decomp"}
DECOMPOSE = """Split a question about research papers on RAG / LLM retrieval into at most 3 simpler sub-questions that can each
be answered from one search over the papers (for a comparison: one per method being compared, plus the comparison
point). A simple question needs only 1 sub-question (itself).
Reply with ONLY JSON: {"subquestions": ["...", "..."]}"""


def decompose(question: str) -> tuple[list[str], int, bool]:
    try:
        r = client().chat([{"role": "system", "content": DECOMPOSE}, {"role": "user", "content": question}],
                          max_tokens=200, temperature=0.0)
        subs = answer._parse_json(r["choices"][0]["message"]["content"] or "").get("subquestions") or []
        subs = [str(x).strip() for x in subs if str(x).strip()][:3]
        return subs or [question], int((r.get("usage") or {}).get("total_tokens") or 0), True
    except (ApiError, ValueError, json.JSONDecodeError, AttributeError):
        return [question], 0, False


def run_one(R, q, strategy):
    version = STRATEGIES[strategy]
    extra_tokens, subs, ok = 0, None, True
    t0 = time.time()
    if strategy == "decompose":
        subs, extra_tokens, ok = decompose(q["question"])
        calls = [R.search(s, mode="rerank", k=6).hits for s in subs] + [R.search(q["question"], mode="rerank", k=10).hits]
        res = Result(q["question"], "rerank", agent.pool(calls, k=10))
    else:
        res = R.search(q["question"], mode="rerank", k=10)
    a = answer.ask(q["question"], res, log=False, prompt_version=version, subquestions=subs if subs and len(subs) > 1 else None)
    a["latency_s"] = round(time.time() - t0, 2)
    a["usage"]["total_tokens"] = int(a["usage"].get("total_tokens") or 0) + extra_tokens
    a["subquestions"], a["decompose_ok"] = subs, ok
    return a


def metrics(q, a):
    gp = set(q["gold_papers"])
    cited = {e["paper_id"] for e in a.get("evidence", []) if any(e["label"] in c.get("evidence", []) for c in a.get("claims", []))}
    in_pack = {e["paper_id"] for e in a.get("evidence", [])}
    dec = a.get("answerable")
    correct = (dec is True and (not gp or bool(gp & cited))) if q["answerable"] else dec is False
    return {"qid": q["qid"], "type": q["type"], "expected": q["answerable"], "answerable": dec, "correct": float(correct),
            "decision_correct": float(dec is q["answerable"]),
            "gold_cited": (len(gp & cited) / len(gp)) if (gp and q["answerable"]) else None,
            "gold_in_pack": (len(gp & in_pack) / len(gp)) if gp else None,
            "json_retry": float(a.get("json_retries", 0) > 0), "json_failed": float(bool(a.get("error"))),
            "contradiction_fixed": float(bool(a.get("abstain_guard"))),
            "missing_name_flag": float(bool(a.get("entity_guard"))),
            "comparability_fixed": float(any(c.get("comparability_guard") for c in a.get("comparisons", []) or [])),
            "invalid_citation": float(any(c.get("invalid_citations") for c in a.get("claims", []))),
            "answered_unanswerable": float(not q["answerable"] and dec is True),
            "over_abstained": float(q["answerable"] and dec is False),
            "n_claims": len(a.get("claims", [])), "tokens": a["usage"].get("total_tokens"), "latency_s": a.get("latency_s")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    qs = [json.loads(l) for s in SETS for l in open(os.path.join(DATA, "eval", s)) if l.strip()]
    cached = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            r = json.loads(l)
            cached[(r["strategy"], r["qid"])] = r
    R = Retriever()
    rows = {s: [] for s in STRATEGIES}
    for q in qs:
        for s in STRATEGIES:
            key = (s, q["qid"])
            if key not in cached:
                if args.reuse:
                    continue
                try:
                    a = run_one(R, q, s)
                except ApiError as e:
                    print(f"  {q['qid']} {s} API error {e.status}; skipped", flush=True)
                    continue
                cached[key] = {"strategy": s, "qid": q["qid"], "answer": {k: v for k, v in a.items() if k != "evidence"}
                               | {"evidence": [{k: e[k] for k in ("label", "paper_id", "chunk_id")} for e in a["evidence"]]}}
                with open(CACHE, "a") as f:
                    f.write(json.dumps(cached[key], default=str) + "\n")
                print(f"  {q['qid']} {s:9} answerable={a.get('answerable')} {a['latency_s']}s", flush=True)
            rows[s].append(metrics(q, cached[key]["answer"]))
    # CoT answers often nest answerable/evidence_check inside "reasoning" (schema drift). Score the same answers
    # again after a code repair that lifts the nested fields to the top level (no new LLM calls).
    rows["CoT + repair"] = []
    for q in qs:
        r = cached.get(("CoT", q["qid"]))
        if r:
            a = json.loads(json.dumps(r["answer"]))
            nested = a.get("reasoning")
            if isinstance(nested, dict):
                for k in ("answerable", "evidence_check", "short_answer"):
                    if a.get(k) is None and k in nested:
                        a[k] = nested[k]
            a = answer.abstain_guard(a)
            rows["CoT + repair"].append(metrics(q, a))
    for s_ in STRATEGIES:
        for r, q in zip(rows[s_], [q for q in qs if (s_, q["qid"]) in cached]):
            a = cached[(s_, q["qid"])]["answer"]
            r["schema_violation"] = float(a.get("answerable") is None and not a.get("error"))
    for r in rows["CoT + repair"]:
        r["schema_violation"] = 0.0
    keys = ["schema_violation", "correct", "decision_correct", "gold_cited", "gold_in_pack", "json_retry", "json_failed", "contradiction_fixed",
            "missing_name_flag", "comparability_fixed", "invalid_citation", "answered_unanswerable", "over_abstained",
            "n_claims", "tokens", "latency_s"]
    rep = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "n_questions": len(qs), "sets": SETS,
           "strategies": {s: {k: summ([r[k] for r in rs]) for k in keys} for s, rs in rows.items()},
           "by_type": {s: {t: f"{int(sum(r['correct'] for r in rs if r['type'] == t))}/{sum(r['type'] == t for r in rs)}"
                           for t in sorted({r['type'] for r in rs})} for s, rs in rows.items()},
           "per_question": rows}
    lat = {s: [r["latency_s"] for r in rs if r["latency_s"]] for s, rs in rows.items()}
    for s in rep["strategies"]:
        rep["strategies"][s]["latency_p50"] = round(float(np.median(lat[s])), 1) if lat[s] else None
    json.dump(rep, open(OUT_JSON, "w"), indent=1, default=str)
    write_md(rep)
    for s, m in rep["strategies"].items():  # noqa: B007
        print(f"{s:10} correct {m['correct']['mean']}  decision {m['decision_correct']['mean']}  gold cited {m['gold_cited']['mean']}")
    print("Saved", os.path.relpath(OUT_JSON, ROOT), "and", os.path.relpath(OUT_MD, ROOT))


def _f(m, pct=False):
    if not isinstance(m, dict):
        return "–" if m is None else str(m)
    if m.get("mean") is None:
        return "–"
    lo, hi = m["ci95"]
    if pct:
        return f"{m['mean'] * 100:.0f}%"
    return f"{m['mean']:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None and m["n"] > 1 else f"{m['mean']:.2f}"


def write_md(rep):
    S = rep["strategies"]
    names = list(S)
    head = "| Metric | " + " | ".join(names) + " |"
    sep = "|---" * (len(names) + 1) + "|"
    lines = [f"# Prompting strategies (dev sets, {rep['n_questions']} questions, {rep['run_at']})", "",
             "Same B3 evidence and the same code checks for every strategy; only the prompt changes. "
             "Correct = the answer/abstain decision matches the label and, if answerable, a gold paper is cited. "
             "Mean [95% bootstrap CI].", "", head, sep]
    for label, k in [("End-to-end correct", "correct"), ("Answer/abstain decision right", "decision_correct"),
                     ("Gold papers cited (answerable)", "gold_cited"), ("Gold papers in evidence", "gold_in_pack"),
                     ("Claims per answer", "n_claims"), ("Tokens per question", "tokens")]:
        lines.append(f"| {label} | " + " | ".join(_f(S[n][k]) for n in names) + " |")
    lines.append("| Latency p50 (s) | " + " | ".join(str(S[n]["latency_p50"]) for n in names) + " |")
    lines += ["", "**How Qwen3-8B breaks** (share of questions)", "", head, sep]
    for label, k in [("JSON valid but 'answerable' missing at top level (schema drift)", "schema_violation"),
                     ("Invalid JSON on first try (repaired)", "json_retry"), ("Invalid JSON twice (no answer)", "json_failed"),
                     ("Says 'not stated' but marks answerable (code fixed)", "contradiction_fixed"),
                     ("A named method absent from evidence (code flagged)", "missing_name_flag"),
                     ("'Directly comparable' across papers (code fixed)", "comparability_fixed"),
                     ("Cites a passage label that does not exist", "invalid_citation"),
                     ("Answered an unanswerable question", "answered_unanswerable"),
                     ("Abstained on an answerable question", "over_abstained")]:
        lines.append(f"| {label} | " + " | ".join(_f(S[n][k], pct=True) for n in names) + " |")
    lines += ["", "**Correct by question type**", "", "| Type | " + " | ".join(names) + " |", sep]
    for t in sorted({t for s in rep["by_type"].values() for t in s}):
        lines.append(f"| {t} | " + " | ".join(rep["by_type"][n].get(t, "–") for n in names) + " |")
    lines += ["", "**CoT + repair** re-scores the same CoT answers after code lifts the fields Qwen nested inside "
              "\"reasoning\" back to the top level (no new calls).", "",
              "n = 40, so differences under about 0.15 are not established. These are the dev sets used for "
              "earlier prompt tuning, so zero-shot (answer-v3, tuned on them) has a home advantage."]
    open(OUT_MD, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
