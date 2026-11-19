"""Evaluate retrieval systems B0-B5 and end-to-end answers.

Question sets (data/eval/):
  questions_v0.jsonl     curated test set (hand-written; gold papers are DRAFTS until the team reviews them)
  synthetic_dev.jsonl    generated from known passages; used for tuning
  synthetic_test.jsonl   generated, held out; reported, never tuned on

Retrieval metrics, with 95% bootstrap CIs over questions:
  Paper Recall@5/@10, paper MRR (all sets); Passage Hit@10 and passage MRR (sets with gold_chunks).
Graph ablation: B4/B5 with graph="always" on multi-hop/comparison questions, to show what the
graph adds when it is forced on (by default it fires only for questions matching the MULTIHOP rule).

Answer metrics (--answers; system B4, one Qwen call per question + one judge call per answer):
  abstention on unanswerable, answer rate on answerable, JSON failures, citation-ID validity,
  claims without valid citation, gold-paper recall among cited passages, latency, tokens,
  and an AUTOMATED citation-support estimate (Qwen judges each claim against its cited passages;
  self-judged, so it is an estimate; the human labels in reports/citation_audit_sample.csv are the real measure).

Usage: .venv/bin/python scripts/evaluate.py [--sets curated,synthetic_dev,synthetic_test] [--answers]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, ROOT, answer, db  # noqa: E402
from researchgraph.llm import ApiError, client  # noqa: E402
from researchgraph.retrieve import SYSTEMS, Retriever  # noqa: E402

SETS = {"curated": "questions_v0.jsonl", "synthetic_dev": "synthetic_dev.jsonl", "synthetic_test": "synthetic_test.jsonl"}
JUDGE = """For each numbered claim, decide whether its cited passages support it.
"supported" = the passages state it; "partial" = they support part of it or something weaker; "not_supported" = they don't.
Judge only against the passages, not your own knowledge.

{items}

Reply with only JSON: {{"labels": ["supported", ...]}} with one label per claim, in order."""


def boot_ci(xs, n=2000, seed=0):
    if not xs:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [round(means[int(0.025 * n)], 3), round(means[int(0.975 * n)], 3)]


def retrieval_scores(hits, gold_papers, gold_chunks):
    out = {}
    papers_at = lambda k: {h.paper_id for h in hits[:k]}
    out["paper_r5"] = len(gold_papers & papers_at(5)) / len(gold_papers)
    out["paper_r10"] = len(gold_papers & papers_at(10)) / len(gold_papers)
    out["paper_mrr"] = next((1 / (i + 1) for i, h in enumerate(hits) if h.paper_id in gold_papers), 0.0)
    if gold_chunks:
        out["passage_hit10"] = float(any(h.chunk_id in gold_chunks for h in hits[:10]))
        out["passage_mrr"] = next((1 / (i + 1) for i, h in enumerate(hits) if h.chunk_id in gold_chunks), 0.0)
    return out


def summarise(rows: list[dict]) -> dict:
    keys = sorted({k for r in rows for k in r})
    return {k: {"mean": round(float(np.mean([r[k] for r in rows if k in r])), 3),
                "ci95": boot_ci([r[k] for r in rows if k in r]),
                "n": sum(1 for r in rows if k in r)} for k in keys}


def judge(ans) -> list[str]:
    ev = {e["label"]: e for e in ans["evidence"]}
    claims = [c for c in ans.get("claims", []) if c["evidence"]]
    if not claims:
        return []
    items = []
    for n, c in enumerate(claims, 1):
        passages = "\n".join(f"  [{l}] {ev[l]['text'][:900]}" for l in c["evidence"])
        items.append(f"Claim {n}: {c['text']}\nCited passages:\n{passages}")
    try:
        raw = client().ask(JUDGE.format(items="\n\n".join(items)), max_tokens=200, temperature=0.0)
        labels = answer._parse_json(raw).get("labels", [])
    except (ApiError, ValueError, json.JSONDecodeError):
        return []
    return [str(l) for l in labels[:len(claims)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="curated,synthetic_dev,synthetic_test")
    ap.add_argument("--answers", action="store_true")
    ap.add_argument("--answer-set", default="curated")
    ap.add_argument("--answer-mode", default="rerank")
    ap.add_argument("--answers-only", action="store_true", help="keep the saved retrieval results; redo answers only")
    args = ap.parse_args()

    corpus = {r[0] for r in db.connect().execute("SELECT DISTINCT paper_id FROM chunks")}
    R = Retriever()
    report = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "n_chunks": R.meta["n_chunks"], "n_papers": len(corpus),
              "systems": SYSTEMS, "retrieval": {}, "graph_ablation": {}, "per_question": {}}
    saved = os.path.join(ROOT, "reports", "eval_results.json")
    if args.answers_only and os.path.exists(saved):
        report = json.load(open(saved))
        report["answers_run_at"] = time.strftime("%Y-%m-%d %H:%M")
    questions = {}
    for name in args.sets.split(","):
        path = os.path.join(DATA, "eval", SETS[name])
        if not os.path.exists(path):
            print(f"skip {name}: {path} not found")
            continue
        qs = [json.loads(l) for l in open(path) if l.strip()]
        for q in qs:
            q["gold_in_corpus"] = [p for p in q["gold_papers"] if p in corpus]
        questions[name] = qs
        if args.answers_only:
            continue
        scored = [q for q in qs if q["gold_in_corpus"]]
        print(f"\n== {name}: {len(qs)} questions, {len(scored)} with gold papers in corpus")
        per_sys, per_q = {}, []
        for q in scored:
            gp, gc = set(q["gold_in_corpus"]), set(q.get("gold_chunks") or [])
            row = {"qid": q["qid"], "type": q["type"], "question": q["question"]}
            for m, label in SYSTEMS.items():
                res = R.search(q["question"], mode=m, k=10)
                s = retrieval_scores(res.hits, gp, gc)
                per_sys.setdefault(label, []).append(s)
                row[f"{label} paper R@10"] = round(s["paper_r10"], 2)
                if m == "graph1hop":
                    row["graph fired"] = res.graph_used
            per_q.append(row)
        report["retrieval"][name] = {label: summarise(rows) for label, rows in per_sys.items()}
        report["per_question"][name] = per_q
        for label, summ in report["retrieval"][name].items():
            line = f"  {label:20} paper R@10 {summ['paper_r10']['mean']:.3f} {summ['paper_r10']['ci95']}  MRR {summ['paper_mrr']['mean']:.3f}"
            if "passage_hit10" in summ:
                line += f"  passage Hit@10 {summ['passage_hit10']['mean']:.3f} {summ['passage_hit10']['ci95']}"
            print(line)

        # graph forced on, for multi-hop / comparison questions
        mh = [q for q in scored if q["type"] in ("multihop", "comparison", "disagreement")]
        if mh:
            abl = {}
            for m, label in (("rerank", "B3 no graph"), ("graph1hop", "B4 1-hop forced"), ("graph_ppr", "B5 PPR forced")):
                rows = [retrieval_scores(R.search(q["question"], mode=m, k=10, graph="always").hits,
                                         set(q["gold_in_corpus"]), set(q.get("gold_chunks") or [])) for q in mh]
                abl[label] = summarise(rows)
            report["graph_ablation"][name] = {"n": len(mh), "results": abl}
            print(f"  graph forced on {len(mh)} multi-hop/comparison questions:")
            for label, summ in abl.items():
                print(f"    {label:18} paper R@10 {summ['paper_r10']['mean']:.3f}  MRR {summ['paper_mrr']['mean']:.3f}")

    if args.answers and args.answer_set in questions:
        qs = questions[args.answer_set]
        print(f"\n== Answers: {args.answer_set}, system {args.answer_mode}, {len(qs)} questions")
        recs, audit_rows = [], []
        prompt_version = answer.PROMPT_VERSION
        for q in qs:
            res = R.search(q["question"], mode=args.answer_mode, k=10)
            a = answer.ask(q["question"], res)
            labels = judge(a) if a.get("claims") else []
            cited = {e["paper_id"] for e in a["evidence"] for c in a["claims"] if e["label"] in c["evidence"]}
            gold = set(q["gold_in_corpus"])
            rec = {"qid": q["qid"], "type": q["type"], "expected_answerable": q["answerable"],
                   "model_answerable": a.get("answerable"), "error": a.get("error"),
                   "citation_validity": a.get("citation_validity"), "n_claims": len(a.get("claims", [])),
                   "unsupported_citation": a.get("n_unsupported_claims"), "judge_labels": labels,
                   "gold_cited_recall": (len(gold & cited) / len(gold)) if gold else None,
                   "graph_used": a["graph_used"], "latency_s": a["latency_s"],
                   "total_tokens": a["usage"].get("total_tokens"), "short_answer": a.get("short_answer", "")}
            recs.append(rec)
            ev = {e["label"]: e for e in a["evidence"]}
            for c in a.get("claims", []):
                if c["evidence"]:
                    audit_rows.append({"qid": q["qid"], "question": q["question"], "claim": c["text"],
                                       "cited_chunk_ids": " ".join(c["chunk_ids"]),
                                       "cited_passages": "\n---\n".join(ev[l]["text"] for l in c["evidence"]),
                                       "auto_label": "", "human_label (supported/partial/not_supported)": ""})
            for row, l in zip([r for r in audit_rows if r["qid"] == q["qid"]], labels):
                row["auto_label"] = l
            print(f"  {q['qid']} expected={q['answerable']!s:5} model={a.get('answerable')!s:5} "
                  f"claims={rec['n_claims']} judge={labels} {a['latency_s']}s")
        unans = [r for r in recs if not r["expected_answerable"]]
        ans_ = [r for r in recs if r["expected_answerable"]]
        all_labels = [l for r in recs for l in r["judge_labels"]]
        lat = sorted(r["latency_s"] for r in recs)
        report["answers"] = {
            "set": args.answer_set, "system": SYSTEMS[args.answer_mode], "prompt": prompt_version, "n": len(recs),
            "abstention_on_unanswerable": f"{sum(r['model_answerable'] is False for r in unans)}/{len(unans)}",
            "answered_when_answerable": f"{sum(r['model_answerable'] is True for r in ans_)}/{len(ans_)}",
            "json_failures": sum(1 for r in recs if r["error"]),
            "citation_id_validity": round(float(np.mean([r["citation_validity"] for r in recs if r["citation_validity"] is not None] or [0])), 3),
            "claims_without_valid_citation": f"{sum(r['unsupported_citation'] or 0 for r in recs)}/{sum(r['n_claims'] for r in recs)}",
            "auto_support_supported": round(all_labels.count("supported") / len(all_labels), 3) if all_labels else None,
            "auto_support_partial": round(all_labels.count("partial") / len(all_labels), 3) if all_labels else None,
            "auto_support_not_supported": round(all_labels.count("not_supported") / len(all_labels), 3) if all_labels else None,
            "gold_paper_recall_in_citations": round(float(np.mean([r["gold_cited_recall"] for r in ans_ if r["gold_cited_recall"] is not None] or [0])), 3),
            "latency_p50_s": lat[len(lat) // 2], "latency_p95_s": lat[min(len(lat) - 1, int(0.95 * len(lat)))],
            "mean_total_tokens": int(np.mean([r["total_tokens"] or 0 for r in recs])),
            "note": "auto_support_* are Qwen3-8B judging its own answers: an estimate. Human labels in reports/citation_audit_sample.csv are the real measure.",
            "per_question": recs}
        print(json.dumps({k: v for k, v in report["answers"].items() if k != "per_question"}, indent=1))
        random.Random(1).shuffle(audit_rows)
        with open(os.path.join(ROOT, "reports", "citation_audit_sample.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
            w.writeheader()
            w.writerows(audit_rows[:60])
        print("Wrote reports/citation_audit_sample.csv for human labelling")

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "eval_results.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print("\nSaved reports/eval_results.json")


if __name__ == "__main__":
    main()
