"""Generate a passage-grounded question set with Qwen3-8B (one call per sampled passage).

Each question is written from ONE chunk, so its gold passage is known exactly. Used for:
  - dev split  (tuning: chunk caps, RRF depth, reranker pool, graph settings)
  - test split (held out: reported once, never used for tuning)
The hand-written, team-reviewed questions in questions_v0.jsonl remain the main test set;
synthetic questions are easier (they echo their passage's wording) and are reported separately.

Sampling: content sections only (no front matter, references, acknowledgements, prompt listings),
chunks with >= 150 estimated tokens, at most PER_PAPER chunks per paper, stratified across papers.
Quality filter: drop questions that copy >= 8 consecutive words from the passage, that are too short,
or that refer to "this paper"/"the passage" (unanswerable without knowing which paper).

Usage: .venv/bin/python scripts/make_synthetic_qa.py [--n 150] [--seed 13]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import DATA, db  # noqa: E402
from researchgraph.answer import _parse_json  # noqa: E402
from researchgraph.llm import ApiError, client  # noqa: E402

PER_PAPER = 2
SKIP_SECTIONS = re.compile(r"front matter|acknowledg|reference|bibliograph|prompt|instruction|demonstration|"
                           r"example|case stud|checklist|author|contribution", re.I)
PROMPT = """You write evaluation questions for a search engine over research papers about retrieval-augmented generation.
Read the passage from the paper "{title}" ({year}), section "{section}".

Write ONE specific question that this passage answers, as a researcher would ask it.
- Name the method, model, dataset or benchmark explicitly (e.g. "In Self-RAG, ..." or "What does RAPTOR ..."), never "this paper" or "the authors".
- The question must be answerable from the passage alone, and should not be answerable from general knowledge.
- Do not copy long phrases from the passage; paraphrase.
- Prefer questions about methods, design choices, findings, numbers or limitations.

Passage:
{text}

Reply with only JSON: {{"question": "...", "answer": "one-sentence answer from the passage", "kind": "method|finding|number|definition|limitation"}}"""


def copies_passage(q: str, text: str, n: int = 8) -> bool:
    qw, tw = q.lower().split(), " ".join(text.lower().split())
    return any(" ".join(qw[i:i + n]) in tw for i in range(len(qw) - n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    con = db.connect()
    rows = con.execute("""SELECT c.chunk_id, c.paper_id, c.section, c.text, c.est_tokens, p.title, p.year
                          FROM chunks c JOIN papers p USING (paper_id) WHERE c.est_tokens >= 150""").fetchall()
    by_paper = {}
    for r in rows:
        if not SKIP_SECTIONS.search(r["section"]) and r["section"].lower() != "abstract":
            by_paper.setdefault(r["paper_id"], []).append(r)
    papers = sorted(by_paper)
    rng.shuffle(papers)
    picks = []
    for pid in papers:
        picks += rng.sample(by_paper[pid], min(PER_PAPER, len(by_paper[pid])))
    rng.shuffle(picks)
    picks = picks[: int(args.n * 1.25)]          # over-sample; the filter drops some

    out, dropped = [], {}
    for r in picks:
        if len(out) >= args.n:
            break
        prompt = PROMPT.format(title=r["title"], year=r["year"], section=r["section"], text=r["text"])
        try:
            raw = client().ask(prompt, max_tokens=200, temperature=0.3)
            q = _parse_json(raw)
        except (ApiError, ValueError, json.JSONDecodeError) as e:
            dropped["error"] = dropped.get("error", 0) + 1
            continue
        question = (q.get("question") or "").strip()
        reason = None
        if len(question.split()) < 6:
            reason = "too short"
        elif re.search(r"\b(this|the) (paper|passage|study|work|authors?)\b", question, re.I):
            reason = "refers to 'this paper'"
        elif copies_passage(question, r["text"]):
            reason = "copies passage"
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        out.append({"qid": f"s{len(out) + 1:03d}", "type": "synthetic_" + (q.get("kind") or "other"),
                    "question": question, "reference_answer": q.get("answer", ""),
                    "gold_papers": [r["paper_id"]], "gold_chunks": [r["chunk_id"]], "answerable": True,
                    "source": "qwen3-8b from gold chunk", "notes": "synthetic; spot-check before relying on it"})
        if len(out) % 20 == 0:
            print(f"  {len(out)} questions")

    rng.shuffle(out)
    n_test = int(len(out) * args.test_frac)
    splits = {"test": out[:n_test], "dev": out[n_test:]}
    for name, qs in splits.items():
        path = os.path.join(DATA, "eval", f"synthetic_{name}.jsonl")
        with open(path, "w") as f:
            for q in sorted(qs, key=lambda q: q["qid"]):
                f.write(json.dumps(q) + "\n")
        print(f"{name}: {len(qs)} questions -> {os.path.relpath(path)}")
    print("dropped:", dropped)


if __name__ == "__main__":
    main()
