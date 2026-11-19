"""Does per-entity retrieval put every compared paper into the evidence pack? Retrieval only (no LLM calls).

Metric: gold-paper coverage of the evidence pack (the passages the model actually sees, ≤ 7), and the share
of questions where ALL gold papers are present. Run on the tuning set; use --set questions_v0.jsonl once for the test.

Usage: .venv/bin/python scripts/tune_decompose.py [--set comparison_dev.jsonl] [--types comparison,multihop]
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
    ap.add_argument("--set", default="comparison_dev.jsonl")
    ap.add_argument("--types", default="")
    ap.add_argument("--mode", default="rerank")
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(os.path.join(DATA, "eval", a.set)) if l.strip()]
    if a.types:
        qs = [q for q in qs if q["type"] in a.types.split(",")]
    qs = [q for q in qs if q["gold_papers"]]
    R = Retriever()
    rows, agg = [], {"plain": [], "decomposed": []}
    for q in qs:
        gold = set(q["gold_papers"])
        row = {"qid": q["qid"], "question": q["question"]}
        for name, fn in (("plain", R.search), ("decomposed", R.search_decomposed)):
            res = fn(q["question"], mode=a.mode, k=10)
            pack_papers = {p["hit"].paper_id for p in answer.build_pack(res)}
            cov = len(gold & pack_papers) / len(gold)
            agg[name].append(cov)
            row[name] = round(cov, 2)
            if name == "decomposed":
                row["entities"] = res.entities
        rows.append(row)
        print(f"{q['qid']:6} plain {row['plain']:.2f}  decomposed {row['decomposed']:.2f}  {row['entities']}")
    summary = {k: {"mean_coverage": round(sum(v) / len(v), 3), "all_gold_in_pack": f"{sum(x == 1.0 for x in v)}/{len(v)}"}
               for k, v in agg.items()}
    print(json.dumps(summary, indent=1))
    out = os.path.join(ROOT, "reports", f"decompose_{os.path.splitext(a.set)[0]}.json")
    json.dump({"set": a.set, "types": a.types, "mode": a.mode, "summary": summary, "rows": rows}, open(out, "w"), indent=1)
    print("Saved", os.path.relpath(out, ROOT))
