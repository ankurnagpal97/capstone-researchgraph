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
    ap.add_argument("--versions", default="answer-v1,answer-v2")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(DATA, "eval", "answer_dev.jsonl"))]
    R = Retriever()
    results = {}
    for v in a.versions.split(","):
        rows = []
        for q in qs:
            ans = answer.ask(q["question"], R.search(q["question"], mode=a.mode, k=10), log=False, prompt_version=v)
            rows.append({"qid": q["qid"], "expected": q["answerable"], "model": ans.get("answerable"),
                         "guard": ans.get("abstain_guard", False), "claims": len(ans.get("claims", [])),
                         "error": ans.get("error")})
        un = [r for r in rows if not r["expected"]]
        an = [r for r in rows if r["expected"]]
        results[v] = {"declined_unanswerable": f"{sum(r['model'] is False for r in un)}/{len(un)}",
                      "answered_answerable": f"{sum(r['model'] is True for r in an)}/{len(an)}",
                      "guard_fired": sum(r["guard"] for r in rows), "json_errors": sum(bool(r["error"]) for r in rows),
                      "rows": rows}
        print(v, {k: x for k, x in results[v].items() if k != "rows"})
    json.dump(results, open(os.path.join(ROOT, "reports", "answer_tuning.json"), "w"), indent=1)
    print("Saved reports/answer_tuning.json")
