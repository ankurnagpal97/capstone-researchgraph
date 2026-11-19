"""Ask a question from the command line.

Usage: .venv/bin/python scripts/ask.py "Which methods decide when to retrieve?" [--mode hybrid_graph] [--search-only]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import answer  # noqa: E402
from researchgraph.retrieve import MODES, Retriever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="rerank", choices=MODES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    res = Retriever().search(a.question, mode=a.mode, k=a.k)
    if a.search_only:
        if res.graph_used:
            print("graph expansion added:", res.graph_papers)
        for h in res.hits:
            print(f"{h.score:.4f} {h.chunk_id:16} {'G' if h.via_graph else ' '} {h.title[:60]} | {h.section[:30]} | p.{h.page_start}")
        sys.exit()
    ans = answer.ask(a.question, res)
    print(json.dumps({k: v for k, v in ans.items() if k != "evidence"}, indent=1) if a.json else answer.render_markdown(ans))
    print(f"\n({ans['latency_s']}s, {ans['usage'].get('total_tokens')} tokens, mode {a.mode}"
          f"{', graph used' if ans['graph_used'] else ''})")
