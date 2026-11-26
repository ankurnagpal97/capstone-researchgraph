"""Build the static trace dashboard (docs/traces/index.html) from logs/traces.jsonl.

The page is self-contained (no external scripts) and is published with GitHub Pages from docs/.
It shows: headline numbers, latency and token use per span type, end-to-end task latency, agent behaviour
(tools, critic verdicts, stop reasons) and a trace explorer with a span waterfall for every agent/pipeline task.

Only span names, timings, token counts and the attributes recorded by researchgraph/tracing.py are published.
Questions are already PII-scrubbed before they are traced (researchgraph/guard.py); session IDs are dropped.

Usage: .venv/bin/python scripts/make_trace_dashboard.py [--traces logs/traces.jsonl] [--out docs/traces/index.html]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP_ATTRS = {"session"}


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def category(name: str) -> str:
    if name.startswith("llm."):
        return "llm"
    if name.startswith(("retrieve.", "tool.")):
        return "retrieval"
    return "orchestration"


def build(spans: list[dict]) -> dict:
    by_trace = collections.defaultdict(list)
    for s in spans:
        s["attributes"] = {k: v for k, v in s["attributes"].items() if k not in DROP_ATTRS}
        by_trace[s["trace_id"]].append(s)

    # per span type
    types = collections.defaultdict(list)
    for s in spans:
        types[s["name"]].append(s)
    span_types = []
    for name, ss in sorted(types.items(), key=lambda kv: -len(kv[1])):
        d = [s["duration_ms"] for s in ss]
        tok = [s["attributes"]["tokens"] for s in ss if isinstance(s["attributes"].get("tokens"), (int, float))]
        span_types.append({"name": name, "category": category(name), "count": len(ss),
                           "p50": round(pct(d, .5), 1), "p95": round(pct(d, .95), 1),
                           "mean_tokens": round(statistics.mean(tok)) if tok else None,
                           "total_tokens": sum(tok) if tok else None,
                           "errors": sum(1 for s in ss if s["status"] == "ERROR")})

    # end-to-end tasks with full span trees
    tasks = []
    for tid, ss in by_trace.items():
        root = next((s for s in ss if s["name"] in ("agent.task", "pipeline.task")), None)
        if not root:
            continue
        a = root["attributes"]
        t0 = min(s["start"] for s in ss)
        llm_tok = sum(s["attributes"].get("tokens", 0) for s in ss if s["name"].startswith("llm."))
        tasks.append({
            "id": tid, "kind": root["name"].split(".")[0], "question": a.get("question", ""),
            "start": dt.datetime.fromtimestamp(root["start"]).strftime("%d %b %H:%M"),
            "ts": root["start"], "duration_ms": root["duration_ms"],
            "tokens": a.get("tokens", llm_tok), "llm_calls": a.get("llm_calls", sum(s["name"].startswith("llm.") for s in ss)),
            "tool_calls": a.get("tool_calls", sum(s["name"].startswith("tool.") for s in ss)),
            "stop_reason": a.get("stop_reason"),
            "spans": [{"id": s["span_id"], "parent": s["parent_id"], "name": s["name"], "cat": category(s["name"]),
                       "t": round((s["start"] - t0) * 1000, 1), "d": s["duration_ms"], "status": s["status"],
                       "attrs": s["attributes"]} for s in sorted(ss, key=lambda s: s["start"])],
        })
    tasks.sort(key=lambda t: t["ts"], reverse=True)

    agent = [t for t in tasks if t["kind"] == "agent"]
    agent_lat = [t["duration_ms"] / 1000 for t in agent]
    tools = collections.Counter(s["name"].split(".", 1)[1] for s in spans if s["name"].startswith("tool."))
    verdicts = collections.Counter(str(s["attributes"].get("verdict")) for s in spans
                                   if s["name"] == "agent.critic" and "verdict" in s["attributes"])
    stops = collections.Counter(str(t["stop_reason"]) for t in agent if t["stop_reason"])
    rounds = collections.Counter(max([s["attrs"].get("round", 1) for s in t["spans"] if s["name"] == "agent.execute"] or [1])
                                 for t in agent)  # rounds are numbered from 1
    llm = [s for s in spans if s["name"].startswith("llm.")]
    starts = [s["start"] for s in spans]
    return {
        "generated": dt.datetime.now().strftime("%d %b %Y %H:%M"),
        "window": [dt.datetime.fromtimestamp(min(starts)).strftime("%d %b %Y %H:%M"),
                   dt.datetime.fromtimestamp(max(starts)).strftime("%d %b %Y %H:%M")],
        "kpi": {"traces": len(by_trace), "spans": len(spans), "tasks": len(tasks), "agent_tasks": len(agent),
                "llm_calls": len(llm), "tokens": sum(s["attributes"].get("tokens", 0) for s in llm),
                "errors": sum(1 for s in spans if s["status"] == "ERROR"),
                "agent_p50": round(pct(agent_lat, .5), 1) if agent_lat else None,
                "agent_p95": round(pct(agent_lat, .95), 1) if agent_lat else None,
                "answer_p50": round(pct([s["duration_ms"] / 1000 for s in spans if s["name"] == "llm.answer"], .5), 1)},
        "span_types": span_types,
        "agent_latency_s": [round(x, 2) for x in agent_lat],
        "tools": tools.most_common(), "verdicts": verdicts.most_common(), "stops": stops.most_common(),
        "rounds": sorted(rounds.items()),
        "tasks": tasks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=os.path.join(ROOT, "logs", "traces.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "traces", "index.html"))
    args = ap.parse_args()
    spans = [json.loads(l) for l in open(args.traces) if l.strip()]
    data = build(spans)
    tpl = open(os.path.join(ROOT, "scripts", "trace_dashboard_template.html")).read()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    with open(args.out, "w") as f:
        f.write(tpl.replace("/*__DATA__*/null", payload))
    print(f"Wrote {os.path.relpath(args.out, ROOT)}: {data['kpi']['spans']} spans, {data['kpi']['tasks']} tasks, "
          f"{os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
