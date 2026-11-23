"""Agent mode: planner -> executor -> generator -> critic, with session memory and hard budgets.

    question ──► PLANNER (Qwen, JSON plan of tool calls; rewrites follow-ups into standalone questions)
                    │
                    ▼
              EXECUTOR (code) runs tools: search_papers · read_paper · citation_neighbours · paper_info
                    │   evidence pool (round-robin per tool call, so every sub-topic is represented)
                    ▼
              GENERATOR = answer.ask (one Qwen call, JSON claim ledger, citation audit, code guards)
                    │
                    ▼
              CRITIC = code checks (missing names, uncited claims, errors) + Qwen review
                    │ "retry" with new queries ──► EXECUTOR (next round, at most MAX_ROUNDS)
                    ▼ "accept" / budget or round limit reached
              final answer + per-task metrics (tool calls, LLM calls, tokens, latency, stop reason)

Failure handling: malformed planner/critic JSON gets one repair request, then a safe fallback (plan = one
search; critic = accept). A failing tool is recorded and skipped; the task continues with what it has.
Repeated identical tool calls are skipped (loop guard). Every LLM call is checked against a per-task token
budget before it is sent. Session turns persist in SQLite (agent_turns) so follow-up questions resolve.

Track B: one active chat request per key, so every task costs 3 (plan, answer, critique) to 5 calls.
"""
from __future__ import annotations

import json
import re
import time
import uuid

from . import answer, db, tracing
from .llm import ApiError, client
from .retrieve import MAX_PER_PAPER, Hit, Result

AGENT_VERSION = "agent-v1"
MAX_ROUNDS = 2                 # executor/generator rounds; the critic can ask for at most one more
MAX_CALLS_PER_ROUND = 4
MAX_TOOL_CALLS = 8             # whole task
TASK_TOKEN_BUDGET = 20000      # prompt + completion tokens over all LLM calls of one task
EST_TOKENS = {"plan": 1500, "answer": 6000, "critic": 2000}   # reserved before each call
HISTORY_TURNS = 3
POOL_K = 10
PER_CALL_FIRST = 2             # passages from each tool call guaranteed a place, before score order

TOOLS = ("search_papers", "read_paper", "citation_neighbours", "paper_info")

PLANNER_SYSTEM = """You plan how to research a question about papers on retrieval-augmented generation (RAG) and LLM retrieval.
You do not answer the question. You choose tool calls; their results become the only evidence.
Tools:
- search_papers {"query": "..."}: hybrid search + rerank over all indexed papers. Use one search per distinct sub-topic, and one per named method when comparing methods.
- read_paper {"paper": "<arXiv id, title or method name>", "query": "..."}: the best passages inside one specific paper.
- citation_neighbours {"paper": "...", "query": "..."}: indexed papers that cite or are cited by that paper, with their most relevant passages. Use for "builds on", "extends", "compares against" and lineage questions. Citing does not mean agreeing.
- paper_info {"paper": "..."}: title, authors, arXiv date and abstract. Use for who / when / which-paper questions.
If earlier turns are given, rewrite the question so it stands alone (replace "it", "that method", "they" with the names).
Use 1 to 4 tool calls; a question about one method usually needs only 1 or 2.
Reply with ONLY JSON: {"standalone_question": "...", "plan": [{"tool": "search_papers", "args": {"query": "..."}, "why": "..."}]}"""

CRITIC_SYSTEM = """You review a draft answer that was written only from retrieved paper passages. Decide whether ONE more round of searching would likely fix a specific gap.
Say "retry" only if part of the question (for example one of the methods being compared) has no supporting claim, or the answer says evidence is missing for something the papers could plausibly contain. Then give 1 or 2 new search queries aimed exactly at the gap, different from the queries already run.
Say "accept" if the answer covers the question, or if it says the papers do not establish something and more searching is unlikely to change that. Abstaining is a valid outcome; do not retry just to force an answer.
Reply with ONLY JSON: {"verdict": "accept", "problems": ["..."], "new_queries": ["..."]}"""


class ToolError(Exception):
    pass


class BudgetExceeded(Exception):
    pass


# -- session memory -----------------------------------------------------------------------------------
SESSION_SCHEMA = """CREATE TABLE IF NOT EXISTS agent_turns (
    session_id TEXT, turn INTEGER, timestamp TEXT, question TEXT, standalone_question TEXT,
    short_answer TEXT, answerable INTEGER, paper_ids TEXT, metrics TEXT,
    PRIMARY KEY (session_id, turn))"""


def _con():
    con = db.connect()
    con.execute(SESSION_SCHEMA)
    return con


def history(session_id: str, n: int = HISTORY_TURNS) -> list[dict]:
    rows = _con().execute("SELECT * FROM agent_turns WHERE session_id = ? ORDER BY turn DESC LIMIT ?",
                          (session_id, n)).fetchall()
    return [dict(r) for r in reversed(rows)]


def remember(session_id: str, ans: dict) -> None:
    con = _con()
    turn = con.execute("SELECT COALESCE(MAX(turn), 0) + 1 FROM agent_turns WHERE session_id = ?",
                       (session_id,)).fetchone()[0]
    a = ans["agent"]
    db.upsert(con, "agent_turns", {
        "session_id": session_id, "turn": turn, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "question": a["question"], "standalone_question": a["standalone_question"],
        "short_answer": ans.get("short_answer", ""), "answerable": {True: 1, False: 0}.get(ans.get("answerable")),
        "paper_ids": list(dict.fromkeys(e["paper_id"] for e in ans.get("evidence", []))),
        "metrics": {k: a[k] for k in ("n_llm_calls", "n_tool_calls", "tool_errors", "rounds", "tokens",
                                      "latency_s", "stop_reason", "completed")}})
    con.commit()


def forget(session_id: str) -> None:
    con = _con()
    con.execute("DELETE FROM agent_turns WHERE session_id = ?", (session_id,))
    con.commit()


# -- tools (pure code, no LLM) -----------------------------------------------------------------------
_PAPERS: dict | None = None
ARXIV_ID = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b")


def _papers() -> dict:
    global _PAPERS
    if _PAPERS is None:
        _PAPERS = {r["paper_id"]: dict(r) for r in db.connect().execute(
            "SELECT paper_id, title, year, authors, published, url, abstract, scope_tag FROM papers "
            "WHERE pdf_path IS NOT NULL")}
    return _PAPERS


def _words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2}


def resolve_paper(R, ref: str) -> str:
    """arXiv ID, (part of) a title, or a method name -> one indexed paper ID."""
    ref = str(ref or "").strip()
    if not ref:
        raise ToolError("no paper given")
    m = ARXIV_ID.search(ref)
    if m and m.group(1) in R.pidx:
        return m.group(1)
    owners = R.entity_owners(ref) if len(ref.split()) <= 3 else []
    if owners:
        return owners[0]
    want = _words(ref)
    best, score = None, 0.0
    for pid, p in _papers().items():
        if pid not in R.pidx:
            continue
        have = _words(p["title"] or "")
        s = len(want & have) / max(1, len(want))
        if s > score:
            best, score = pid, s
    if best is None or score < 0.6:
        raise ToolError(f"no indexed paper matches '{ref[:80]}'")
    return best


def _best_in_papers(R, query: str, papers: list[str], k: int, graph_papers=()) -> list[Hit]:
    rd = R.rank_dense(query)
    want = set(papers)
    idx = [int(i) for i in rd if R.paper_of[int(i)] in want][:12 * max(1, len(want))]
    return R._hits(R._rerank(query, idx), k, graph_papers=set(graph_papers))


def tool_search_papers(R, args: dict, question: str) -> tuple[list[Hit], str]:
    q = str(args.get("query") or "").strip()
    if not q:
        raise ToolError("search_papers needs a non-empty 'query'")
    hits = R.search_decomposed(q, mode="rerank", k=6).hits
    return hits, f"{len(hits)} passages from {len({h.paper_id for h in hits})} papers"


def tool_read_paper(R, args: dict, question: str) -> tuple[list[Hit], str]:
    pid = resolve_paper(R, args.get("paper"))
    hits = _best_in_papers(R, str(args.get("query") or question), [pid], 4)
    return hits, f"{len(hits)} passages from {pid}"


def tool_citation_neighbours(R, args: dict, question: str) -> tuple[list[Hit], str]:
    pid = resolve_paper(R, args.get("paper"))
    con = db.connect()
    cites = {r[0] for r in con.execute("SELECT DISTINCT cited_paper_id FROM citation_edges WHERE citing_paper_id = ?", (pid,))}
    cited_by = {r[0] for r in con.execute("SELECT DISTINCT citing_paper_id FROM citation_edges WHERE cited_paper_id = ?", (pid,))}
    nbrs = [p for p in sorted(cites | cited_by) if p in R.pidx]
    if not nbrs:
        return [], f"{pid} has no citation neighbours in the corpus"
    query = str(args.get("query") or question)
    hits = _best_in_papers(R, query, nbrs, 6, graph_papers=nbrs)
    return hits, f"{pid}: cites {len(cites & set(nbrs))}, cited by {len(cited_by & set(nbrs))} indexed papers"


def tool_paper_info(R, args: dict, question: str) -> tuple[list[Hit], str]:
    pid = resolve_paper(R, args.get("paper"))
    p = _papers()[pid]
    try:
        authors = ", ".join(json.loads(p["authors"] or "[]"))
    except (TypeError, json.JSONDecodeError):
        authors = str(p["authors"] or "")
    text = (f"Title: {p['title']}. Authors: {authors or 'unknown'}. First published on arXiv: "
            f"{(p['published'] or '')[:10] or 'unknown'}. Abstract: {p['abstract'] or ''}")
    hit = Hit(f"{pid}#meta", pid, p["title"], p["year"], "arXiv metadata", 1, 1, text, p["url"] or "", 10.0,
              False, p["scope_tag"] or "")
    return [hit], f"metadata for {pid}"


TOOL_FUNCS = {"search_papers": tool_search_papers, "read_paper": tool_read_paper,
              "citation_neighbours": tool_citation_neighbours, "paper_info": tool_paper_info}


def pool(calls: list[list[Hit]], k: int = POOL_K) -> list[Hit]:
    """Merge tool results: each call's best PER_CALL_FIRST passages first (round-robin), then the rest by
    score; no duplicates; at most MAX_PER_PAPER passages per paper."""
    out, seen, per_paper = [], set(), {}

    def add(h):
        if h.chunk_id in seen or per_paper.get(h.paper_id, 0) >= MAX_PER_PAPER:
            return
        seen.add(h.chunk_id)
        per_paper[h.paper_id] = per_paper.get(h.paper_id, 0) + 1
        out.append(h)

    for rank in range(PER_CALL_FIRST):
        for hits in calls:
            if rank < len(hits):
                add(hits[rank])
    for h in sorted((h for hits in calls for h in hits[PER_CALL_FIRST:]), key=lambda h: -h.score):
        add(h)
    return out[:k]


# -- the loop -----------------------------------------------------------------------------------------
class Agent:
    def __init__(self, retriever, session_id: str | None = None, max_rounds: int = MAX_ROUNDS,
                 token_budget: int = TASK_TOKEN_BUDGET, log: bool = True):
        self.R, self.session_id = retriever, session_id or uuid.uuid4().hex[:12]
        self.max_rounds, self.token_budget, self.log = max_rounds, token_budget, log

    # LLM helpers
    def _reserve(self, kind: str):
        if self.state["tokens"] + EST_TOKENS[kind] > self.token_budget:
            raise BudgetExceeded(kind)

    def _llm_json(self, kind: str, system: str, user: str, max_tokens: int) -> dict:
        """One JSON call with one repair attempt. Raises ApiError, ValueError (bad JSON twice), BudgetExceeded."""
        self._reserve(kind)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            with tracing.span(f"llm.{kind}", attempt=attempt) as sp:
                r = client().chat(messages, max_tokens=max_tokens, temperature=0.0)
                used = int((r.get("usage") or {}).get("total_tokens") or 0)
                sp.set("tokens", used)
            self.state["n_llm_calls"] += 1
            self.state["tokens"] += used
            raw = r["choices"][0]["message"]["content"] or ""
            try:
                out = answer._parse_json(raw)
                if not isinstance(out, dict):
                    raise ValueError("JSON is not an object")
                return out
            except (ValueError, json.JSONDecodeError) as e:
                self.state["json_errors"] += 1
                if attempt == 1:
                    raise ValueError(f"invalid JSON from {kind}: {e}")
                messages += [{"role": "assistant", "content": raw},
                             {"role": "user", "content": f"That was not valid JSON ({e}). Reply again with only the JSON object."}]

    # stages
    def plan(self, question: str) -> tuple[str, list[dict]]:
        hist = history(self.session_id) if self.session_id else []
        ctx = ""
        if hist:
            ctx = "Earlier turns in this conversation:\n" + "\n".join(
                f"Q{i + 1}: {h['standalone_question'] or h['question']}\nA{i + 1}: {(h['short_answer'] or '')[:300]}"
                for i, h in enumerate(hist)) + "\n\n"
        user = f"{ctx}Question: {question}"
        try:
            out = self._llm_json("plan", PLANNER_SYSTEM, user, 400)
            standalone = str(out.get("standalone_question") or question).strip() or question
            steps = out.get("plan") or []
            if not isinstance(steps, list):
                raise ValueError("plan is not a list")
            steps = [s for s in steps if isinstance(s, dict)]
            if not steps:
                raise ValueError("empty plan")
            return standalone, steps[:MAX_CALLS_PER_ROUND]
        except (ApiError, ValueError, BudgetExceeded) as e:
            self.state["fallbacks"].append(f"planner: {type(e).__name__}: {str(e)[:120]}")
            return question, [{"tool": "search_papers", "args": {"query": question}, "why": "fallback plan"}]

    def execute(self, steps: list[dict], question: str, rnd: int) -> None:
        for s in steps:
            if self.state["n_tool_calls"] >= MAX_TOOL_CALLS:
                self.state["fallbacks"].append("tool-call cap reached")
                break
            name, args = str(s.get("tool", "")), s.get("args") if isinstance(s.get("args"), dict) else {}
            key = name + json.dumps(args, sort_keys=True)
            rec = {"round": rnd, "tool": name, "args": args, "why": s.get("why", ""), "status": "ok"}
            if key in self.seen_calls:
                rec.update(status="skipped", note="repeated call")
                self.state["tool_calls"].append(rec)
                continue
            self.seen_calls.add(key)
            self.state["n_tool_calls"] += 1
            t0 = time.time()
            with tracing.span(f"tool.{name}", **{f"arg.{k}": str(v)[:200] for k, v in args.items()}) as sp:
                try:
                    if name not in TOOL_FUNCS:
                        raise ToolError(f"unknown tool '{name}'")
                    hits, note = TOOL_FUNCS[name](self.R, args, question)
                    self.calls.append(hits)
                    rec.update(n_hits=len(hits), papers=list(dict.fromkeys(h.paper_id for h in hits)), note=note)
                except (ToolError, ApiError, KeyError, TypeError, ValueError) as e:
                    rec.update(status="error", error=f"{type(e).__name__}: {str(e)[:200]}")
                    self.state["tool_errors"] += 1
                    sp.set("error", rec["error"])
            rec["seconds"] = round(time.time() - t0, 2)
            self.state["tool_calls"].append(rec)

    def generate(self, question: str) -> dict:
        hits = pool(self.calls)
        if not hits:   # every tool failed or found nothing: one plain search so the generator has something
            self.state["fallbacks"].append("empty evidence pool: plain search")
            hits = self.R.search_decomposed(question, mode="rerank", k=POOL_K).hits
        self._reserve("answer")
        with tracing.span("agent.generate", passages=len(hits)):
            ans = answer.ask(question, Result(question, "rerank", hits), log=self.log)
        self.state["n_llm_calls"] += 1 + (1 if "invalid JSON" in str(ans.get("error", "")) else 0)
        self.state["tokens"] += int(ans["usage"].get("total_tokens") or 0)
        return ans

    def code_checks(self, ans: dict) -> list[str]:
        problems = []
        if ans.get("error"):
            problems.append(f"generator error: {ans['error']}")
        if ans.get("entity_guard"):
            problems.append("no passage mentions: " + ", ".join(ans["entity_guard"]))
        if ans.get("n_unsupported_claims"):
            problems.append(f"{ans['n_unsupported_claims']} claim(s) without a valid citation")
        return problems

    def critique(self, question: str, ans: dict) -> dict:
        problems = self.code_checks(ans)
        missing = list(ans.get("entity_guard") or [])
        claims = "\n".join(f"- {c['text']}" for c in (ans.get("claims") or [])[:10]) or "(none)"
        queries = [c["args"].get("query", c["args"].get("paper", "")) for c in self.state["tool_calls"]]
        user = (f"Question: {question}\nAnswered from the papers: {ans.get('answerable')}\n"
                f"Short answer: {ans.get('short_answer', '')}\nClaims:\n{claims}\n"
                f"Missing evidence noted: {ans.get('missing_evidence') or 'none'}\n"
                f"Code checks: {'; '.join(problems) or 'all passed'}\nQueries already run: {queries}")
        try:
            out = self._llm_json("critic", CRITIC_SYSTEM, user, 250)
            verdict = "retry" if str(out.get("verdict", "")).lower().strip() == "retry" else "accept"
            new = [str(q).strip() for q in (out.get("new_queries") or []) if str(q).strip()][:2]
            crit = {"verdict": verdict, "problems": [str(p) for p in out.get("problems") or []][:4],
                    "new_queries": new, "source": "llm"}
        except (ApiError, ValueError) as e:
            self.state["fallbacks"].append(f"critic: {type(e).__name__}")
            crit = {"verdict": "accept", "problems": [], "new_queries": [], "source": "fallback"}
        if missing:        # code critic overrides: a named thing has no evidence -> search for it directly
            crit["verdict"] = "retry"
            crit["new_queries"] = list(dict.fromkeys([f"{m} {question}" for m in missing] + crit["new_queries"]))[:2]
            crit["source"] += "+code"
        crit["code_problems"] = problems
        if crit["verdict"] == "retry" and not crit["new_queries"]:
            crit["verdict"] = "accept"
            crit["problems"].append("retry requested without a query; accepted")
        return crit

    def run(self, question: str) -> dict:
        from .guard import scrub_pii
        question, pii = scrub_pii(question)
        t0 = time.time()
        self.state = {"n_llm_calls": 0, "n_tool_calls": 0, "tool_errors": 0, "json_errors": 0, "tokens": 0,
                      "tool_calls": [], "critiques": [], "fallbacks": []}
        self.calls, self.seen_calls = [], set()
        stop, ans, best = "accepted", None, None
        with tracing.span("agent.task", session=self.session_id, question=question[:200]) as task_span:
            with tracing.span("agent.plan"):
                standalone, steps = self.plan(question)
            first_plan = steps
            rnd = 0
            while True:
                rnd += 1
                with tracing.span("agent.execute", round=rnd):
                    self.execute(steps, standalone, rnd)
                try:
                    ans = self.generate(standalone)
                except BudgetExceeded:
                    stop = "token budget"
                    break
                if best is None or not ans.get("error"):
                    best = ans
                if rnd >= self.max_rounds:
                    stop = "max rounds" if rnd > 1 else "single round (critic off)"
                    break
                try:
                    with tracing.span("agent.critic", round=rnd) as sp:
                        crit = self.critique(standalone, ans)
                        sp.set("verdict", crit["verdict"])
                except BudgetExceeded:
                    stop = "token budget"
                    break
                crit["round"] = rnd
                self.state["critiques"].append(crit)
                if crit["verdict"] != "retry":
                    break
                steps = [{"tool": "search_papers", "args": {"query": q}, "why": "critic: " + "; ".join(crit["problems"])[:200]}
                         for q in crit["new_queries"]]
            ans = best
            if ans is None:     # budget too small for even one answer
                ans = {"answerable": None, "short_answer": "", "claims": [], "comparisons": [], "limitations": [],
                       "missing_evidence": [], "error": "token budget too small", "evidence": [], "usage": {},
                       "graph_used": False, "graph_papers": [], "entities": [], "citation_validity": None}
            self.state.update(rounds=rnd, stop_reason=stop, latency_s=round(time.time() - t0, 2),
                              completed=bool(not ans.get("error") and ans.get("answerable") is not None))
            task_span.set("tool_calls", self.state["n_tool_calls"])
            task_span.set("llm_calls", self.state["n_llm_calls"])
            task_span.set("tokens", self.state["tokens"])
            task_span.set("stop_reason", stop)
            trace_id = task_span.trace_id
        ans.update(mode="agent", question=question, pii_removed=pii, latency_s=self.state["latency_s"],
                   usage={"total_tokens": self.state["tokens"]})
        ans["agent"] = {"version": AGENT_VERSION, "session_id": self.session_id, "question": question,
                        "standalone_question": standalone, "plan": first_plan,
                        "final_code_problems": self.code_checks(ans), "trace_id": trace_id, **self.state}
        if self.session_id and self.log:
            remember(self.session_id, ans)
        return ans
