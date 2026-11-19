"""Compare answer prompt versions on the answer-tuning dev set (never on the curated test set).

Usage: .venv/bin/python scripts/tune_answers.py [--mode rerank] [--versions answer-v1,answer-v2]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, ROOT, answer  # noqa: E402
from researchgraph.retrieve import Retriever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="rerank")
    ap.add_argument("--versions", default="answer-v2,answer-v3")
    ap.add_argument("--set", default="answer_dev.jsonl")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(DATA, "eval", a.set))]
    R = Retriever()
    results = {}
    for v in a.versions.split(","):
        rows = []
        for q in qs:
            search = R.search if v in ("answer-v1", "answer-v2") else R.search_decomposed   # v3 = prompt + per-entity retrieval
            ans = answer.ask(q["question"], search(q["question"], mode=a.mode, k=10), log=False, prompt_version=v)
            rows.append({"qid": q["qid"], "expected": q["answerable"], "model": ans.get("answerable"),
                         "guard": ans.get("abstain_guard", False), "entity_guard": ans.get("entity_guard"), "claims": len(ans.get("claims", [])),
                         "error": ans.get("error"),
                         "comparability_flags": [c.get("directly_comparable") for c in ans.get("comparisons", [])],
                         "guarded_comparisons": sum(1 for c in ans.get("comparisons", []) if c.get("comparability_guard")),
                         "survey_claims_attributed": sum(1 for c in ans.get("claims", []) if "according to" in c.get("text", "").lower()),
                         "short_answer": ans.get("short_answer", "")[:200]})
        un = [r for r in rows if not r["expected"]]
        an = [r for r in rows if r["expected"]]
        results[v] = {"declined_unanswerable": f"{sum(r['model'] is False for r in un)}/{len(un)}",
                      "answered_answerable": f"{sum(r['model'] is True for r in an)}/{len(an)}",
                      "guard_fired": sum(r["guard"] for r in rows), "json_errors": sum(bool(r["error"]) for r in rows),
                      "comparisons_marked_comparable": sum(sum(1 for f in r["comparability_flags"] if f) for r in rows),
                      "comparability_guard_fired": sum(r["guarded_comparisons"] for r in rows),
                      "rows": rows}
        print(v, {k: x for k, x in results[v].items() if k != "rows"})
    json.dump(results, open(os.path.join(ROOT, "reports", "answer_tuning.json"), "w"), indent=1)
    print("Saved reports/answer_tuning.json")
