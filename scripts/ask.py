"""Ask a question from the command line.

Usage: .venv/bin/python scripts/ask.py "Which methods decide when to retrieve?" [--mode hybrid_graph] [--search-only]
       .venv/bin/python scripts/ask.py "How does Self-RAG work?" --agent --session demo   # agent mode, remembers turns
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import agent, answer  # noqa: E402
from researchgraph.retrieve import MODES, Retriever  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="rerank", choices=MODES)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--search-only", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-followups", action="store_true", help="skip the follow-up question suggestions (one extra LLM call)")
    ap.add_argument("--agent", action="store_true", help="planner -> tools -> answer -> critic loop")
    ap.add_argument("--session", default=None, help="agent session ID; turns with the same ID share memory")
    a = ap.parse_args()

    if a.agent:
        ans = agent.Agent(Retriever(), session_id=a.session).run(a.question)
        if not a.no_followups:
            answer.suggest_followups(ans)
        g = ans["agent"]
        print(json.dumps({k: v for k, v in ans.items() if k != "evidence"}, indent=1, default=str) if a.json
              else answer.render_markdown(ans))
        print(f"\n[agent] standalone question: {g['standalone_question']}")
        for c in g["tool_calls"]:
            print(f"  round {c['round']} {c['status']:7} {c['tool']}({json.dumps(c['args'])[:90]}) {c.get('note') or c.get('error', '')}")
        for c in g["critiques"]:
            print(f"  critic round {c['round']}: {c['verdict']} {c['problems'][:2]} {c['new_queries']}")
        print(f"  {g['latency_s']}s · {g['n_llm_calls']} LLM calls · {g['n_tool_calls']} tool calls · {g['tokens']} tokens · "
              f"stop: {g['stop_reason']} · session {g['session_id']} · trace {g.get('trace_id')}")
        for f in ans.get("followups") or []:
            print("  →", f)
        sys.exit()

    res = Retriever().search_decomposed(a.question, mode=a.mode, k=a.k)
    if a.search_only:
        if res.graph_used:
            print("graph expansion added:", res.graph_papers)
        for h in res.hits:
            print(f"{h.score:.4f} {h.chunk_id:16} {'G' if h.via_graph else ' '} {h.title[:60]} | {h.section[:30]} | p.{h.page_start}")
        sys.exit()
    ans = answer.ask(a.question, res)
    if not a.no_followups:
        answer.suggest_followups(ans)
    print(json.dumps({k: v for k, v in ans.items() if k != "evidence"}, indent=1) if a.json else answer.render_markdown(ans))
    print(f"\n({ans['latency_s']}s, {ans['usage'].get('total_tokens')} tokens, mode {a.mode}"
          f"{', graph used' if ans['graph_used'] else ''})")
    if ans.get("followups") and not a.json:
        print("\nYou might ask next:")
        for f in ans["followups"]:
            print("  →", f)
