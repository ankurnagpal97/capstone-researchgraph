"""Agent vs pipeline on the curated set: accuracy, task completion, tool calls, cost, latency, failure taxonomy.

Usage:
  .venv/bin/python scripts/eval_agent.py            # runs the agent on new questions (cached in reports/agent_answers.jsonl)
  .venv/bin/python scripts/eval_agent.py --reuse    # no LLM calls; rescore cached answers

The pipeline (B3 + per-entity retrieval, one answer call) answers come from reports/stage_answers.jsonl
(scripts/eval_stages.py), so both systems are scored with the same code.

End-to-end correct = the answer/abstain decision matches the gold label AND, for answerable questions,
at least one gold paper is among the cited sources.

Failure taxonomy (first matching cause, for each question that is not end-to-end correct):
  tool-call error  the agent's tool call failed / planner or critic fell back / the generator returned no valid JSON
  retrieval miss   answerable, but no gold paper reached the evidence pack
  hallucination    answered an unanswerable question, or a claim cites an invalid label, or the judge marks a claim
                   not supported
  reasoning error  the gold paper was in the evidence, but the model abstained or cited other papers
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
from researchgraph import ROOT, agent  # noqa: E402
from researchgraph.llm import ApiError, client  # noqa: E402
from researchgraph.retrieve import Retriever  # noqa: E402
from eval_stages import CACHE as PIPE_CACHE, answer_rows, load_set, semantic_sims, summ  # noqa: E402
from evaluate import judge  # noqa: E402

CACHE = os.path.join(ROOT, "reports", "agent_answers.jsonl")
OUT_JSON = os.path.join(ROOT, "reports", "agent_results.json")
OUT_MD = os.path.join(ROOT, "reports", "AGENT.md")
CAUSES = ["tool-call error", "retrieval miss", "hallucination", "reasoning error"]


def outcome(q: dict, a: dict, labels: list[str], tool_trouble: bool) -> tuple[bool, str | None]:
    gp = set(q["gold_papers"])
    cited = {e["paper_id"] for e in a.get("evidence", []) if any(e["label"] in c.get("evidence", []) for c in a.get("claims", []))}
    in_pack = {e["paper_id"] for e in a.get("evidence", [])}
    decided = a.get("answerable")
    if q["answerable"]:
        correct = decided is True and (not gp or bool(gp & cited))
    else:
        correct = decided is False
    if correct:
        return True, None
    invalid = any(c.get("invalid_citations") for c in a.get("claims", []))
    if tool_trouble or a.get("error"):
        return False, "tool-call error"
    if q["answerable"] and gp and not (gp & in_pack):
        return False, "retrieval miss"
    if (not q["answerable"] and decided is True) or invalid or "not_supported" in labels:
        return False, "hallucination"
    return False, "reasoning error"


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if xs else None


def run(qs, cached, R, do_judge):
    recs = []
    for q in qs:
        if q["qid"] in cached:
            recs.append(cached[q["qid"]])
            continue
        try:
            a = agent.Agent(R, session_id=f"eval-{q['qid']}", log=False).run(q["question"])
        except ApiError as e:
            print(f"  {q['qid']} API error {e.status}; skipped")
            continue
        labels = judge(a) if (do_judge and a.get("claims")) else []
        rec = {"qid": q["qid"], "type": q["type"], "answer": a, "judge_labels": labels}
        with open(CACHE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        recs.append(rec)
        g = a["agent"]
        print(f"  {q['qid']} answerable={a.get('answerable')} tools={g['n_tool_calls']} llm={g['n_llm_calls']} "
              f"rounds={g['rounds']} stop={g['stop_reason']} {g['latency_s']}s", flush=True)
    return recs


def system_summary(qs, recs, R, is_agent):
    byq = {q["qid"]: q for q in qs}
    rows, outcomes = [], []
    for r in recs:
        q, a = byq[r["qid"]], r["answer"]
        row = answer_rows(q, r, R)
        g = a.get("agent") or {}
        trouble = bool(g.get("tool_errors") or g.get("fallbacks"))
        ok, cause = outcome(q, a, r.get("judge_labels", []), trouble)
        row.update(correct=ok, cause=cause, latency_s=a.get("latency_s"),
                   tokens=(a.get("usage") or {}).get("total_tokens"),
                   llm_calls=g.get("n_llm_calls", 1 + (1 if "invalid JSON" in str(a.get("error", "")) else 0)),
                   tool_calls=g.get("n_tool_calls", 1), rounds=g.get("rounds", 1), stop=g.get("stop_reason", "-"),
                   completed=g.get("completed", bool(not a.get("error") and a.get("answerable") is not None)))
        rows.append(row)
        outcomes.append(ok)
    semantic_sims(rows)
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    tok = [r["tokens"] for r in rows if r["tokens"]]
    labels = [l for r in recs for l in r.get("judge_labels", [])]
    unans = [r for r in rows if not r["expected_answerable"]]
    ans_ = [r for r in rows if r["expected_answerable"]]
    s = {"n": len(rows),
         "end_to_end_accuracy": summ([float(r["correct"]) for r in rows]),
         "task_completion_rate": summ([float(r["completed"]) for r in rows]),
         "avg_tool_calls": summ([r["tool_calls"] for r in rows]),
         "avg_llm_calls": summ([r["llm_calls"] for r in rows]),
         "tokens_per_query": summ(tok), "latency_p50_s": pct(lat, 50), "latency_p95_s": pct(lat, 95),
         "abstention": f"{sum(r['model_answerable'] is False for r in unans)}/{len(unans)}",
         "answered_when_answerable": f"{sum(r['model_answerable'] is True for r in ans_)}/{len(ans_)}",
         "judge_supported": round(labels.count("supported") / len(labels), 3) if labels else None,
         "failures": {c: sorted(r["qid"] for r in rows if r["cause"] == c) for c in CAUSES}}
    for k in ("pack_gold_paper_recall", "pack_gold_passage_hit", "rougeL_f", "token_f1", "semantic_sim",
              "citation_validity", "claim_rougeL_precision_vs_cited"):
        s[k] = summ([r.get(k) for r in rows])
    if is_agent:
        s["tool_errors"] = sum(r["answer"]["agent"]["tool_errors"] for r in recs)
        s["json_errors"] = sum(r["answer"]["agent"]["json_errors"] for r in recs)
        s["fallbacks"] = sum(len(r["answer"]["agent"]["fallbacks"]) for r in recs)
        s["retry_rate"] = round(float(np.mean([r["rounds"] > 1 for r in rows])), 3) if rows else None
        s["stop_reasons"] = {k: sum(r["stop"] == k for r in rows) for k in sorted({r["stop"] for r in rows})}
        s["tools_used"] = {}
        for r in recs:
            for c in r["answer"]["agent"]["tool_calls"]:
                if c["status"] != "skipped":
                    s["tools_used"][c["tool"]] = s["tools_used"].get(c["tool"], 0) + 1
    by_type = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(float(r["correct"]))
    s["accuracy_by_type"] = {t: f"{int(sum(v))}/{len(v)}" for t, v in sorted(by_type.items())}
    s["per_question"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                          if not k.startswith("_") and k not in ("judge_labels",)} for r in rows]
    return s


def _f(m):
    if isinstance(m, dict):
        if m.get("mean") is None:
            return "–"
        lo, hi = m["ci95"]
        return f"{m['mean']:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None and m["n"] > 1 else f"{m['mean']:.2f}"
    return "–" if m is None else str(m)


def write_md(rep):
    P, A = rep["pipeline"], rep["agent"]
    lines = [f"# Agent vs pipeline (curated set, {rep['run_at']})", "",
             f"Agent `{rep['agent_version']}`: planner → executor (tools) → generator → critic, at most "
             f"{rep['limits']['max_rounds']} rounds, {rep['limits']['max_tool_calls']} tool calls and "
             f"{rep['limits']['token_budget']:,} tokens per task. Pipeline: B3 hybrid + rerank + per-entity retrieval, "
             "one answer call. Both answer with Qwen3-8B (prompt answer-v3) and pass the same code checks. "
             "Mean [95% bootstrap CI].", "",
             "| Metric | Pipeline (B3) | Agent |", "|---|---|---|"]
    rows = [("End-to-end accuracy (decision right + a gold paper cited)", "end_to_end_accuracy"),
            ("Task completion rate", "task_completion_rate"), ("Tool calls per task", "avg_tool_calls"),
            ("LLM calls per task", "avg_llm_calls"), ("Tokens per query (cost)", "tokens_per_query"),
            ("Latency p50 (s)", "latency_p50_s"), ("Latency p95 (s)", "latency_p95_s"),
            ("Gold paper in evidence pack", "pack_gold_paper_recall"), ("Gold passage in evidence pack", "pack_gold_passage_hit"),
            ("ROUGE-L F1 vs reference", "rougeL_f"), ("Token F1 vs reference", "token_f1"),
            ("Semantic similarity vs reference", "semantic_sim"), ("Citation IDs valid", "citation_validity"),
            ("Claims supported (Qwen self-judge)", "judge_supported"),
            ("Abstained on unanswerable", "abstention"), ("Answered when answerable", "answered_when_answerable")]
    for label, k in rows:
        lines.append(f"| {label} | {_f(P.get(k))} | {_f(A.get(k))} |")
    lines += ["", "**Accuracy by question type**", "", "| Type | Pipeline | Agent |", "|---|---|---|"]
    for t in sorted(set(P["accuracy_by_type"]) | set(A["accuracy_by_type"])):
        lines.append(f"| {t} | {P['accuracy_by_type'].get(t, '–')} | {A['accuracy_by_type'].get(t, '–')} |")
    lines += ["", "**Failure taxonomy** (questions not end-to-end correct, first matching cause)", "",
              "| Cause | Pipeline | Agent |", "|---|---|---|"]
    for c in CAUSES:
        p, a = P["failures"][c], A["failures"][c]
        lines.append(f"| {c} | {len(p)} {', '.join(p)} | {len(a)} {', '.join(a)} |")
    lines += ["", f"**Agent behaviour:** retry rate {A['retry_rate']}, stop reasons {A['stop_reasons']}, "
              f"tools used {A['tools_used']}, tool errors {A['tool_errors']}, malformed JSON {A['json_errors']} "
              f"(all repaired or fell back), fallbacks {A['fallbacks']}.", "",
              "**Cost:** the course endpoint has no per-token price; cost is reported as tokens and LLM calls per query. "
              "Every LLM call holds the group's single request slot, so latency is also the cost to teammates.", "",
              "Curated reference answers were drafted by Claude from the gold passages; "
              f"n = {A['n']}, so differences under about 0.10 are not established."]
    open(OUT_MD, "w").write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    qs = load_set("curated")
    if args.limit:
        qs = qs[:args.limit]
    cached = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            if l.strip():
                r = json.loads(l)
                cached[r["qid"]] = r
    pipe = {}
    for l in open(PIPE_CACHE):
        r = json.loads(l)
        if r.get("set") == "curated":
            pipe[r["qid"]] = r
    R = Retriever()
    recs = [cached[q["qid"]] for q in qs if q["qid"] in cached] if args.reuse else run(qs, cached, R, not args.no_judge)
    done = {r["qid"] for r in recs}
    qs_done = [q for q in qs if q["qid"] in done and q["qid"] in pipe]
    rep = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "agent_version": agent.AGENT_VERSION, "llm": client().llm.model,
           "limits": {"max_rounds": agent.MAX_ROUNDS, "max_tool_calls": agent.MAX_TOOL_CALLS,
                      "token_budget": agent.TASK_TOKEN_BUDGET},
           "pipeline": system_summary(qs_done, [pipe[q["qid"]] for q in qs_done], R, False),
           "agent": system_summary(qs_done, [r for r in recs if r["qid"] in {q["qid"] for q in qs_done}], R, True)}
    json.dump(rep, open(OUT_JSON, "w"), indent=1, default=str)
    write_md(rep)
    P, A = rep["pipeline"], rep["agent"]
    print(f"\npipeline acc {_f(P['end_to_end_accuracy'])}  agent acc {_f(A['end_to_end_accuracy'])}  "
          f"agent completion {_f(A['task_completion_rate'])} tools {_f(A['avg_tool_calls'])}")
    print("Saved", os.path.relpath(OUT_JSON, ROOT), "and", os.path.relpath(OUT_MD, ROOT))


if __name__ == "__main__":
    main()
