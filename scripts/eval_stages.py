"""Stage-by-stage evaluation: how much of the right information survives each step of the pipeline.

  S1 Parsing & chunking     chunk length vs the 512-token embedding limit, section detection, chunks per paper
  S2 First-stage retrieval  B0 BM25 / B1 dense / B2 hybrid: paper Recall@10, passage Hit@10, MRR, nDCG@10,
                            and ROUGE-L recall of the reference answer against the top-10 passages (answer coverage)
  S3 Reranking              B3 cross-encoder (and B3 + per-entity retrieval, the app's default): same metrics
  S4 Evidence pack          the <= 7 passages Qwen actually sees: gold passage / gold paper in pack,
                            context precision (share of pack passages from gold papers), ROUGE-L recall of the reference
  S5 Answer generation      ROUGE-1/2/L F1 and token F1 against the reference answer, semantic similarity
                            (bge-small cosine), abstention on unanswerable questions
  S6 Grounding              citation-ID validity, ROUGE-L precision of each claim against its own cited passages
                            (share of the claim's words found in its evidence), Qwen-judged support (curated only)
  S7 Follow-up questions    3 suggestions per answer: novelty (ROUGE-L vs the original question) and whether the
                            corpus covers them (top cross-encoder score above the evidence threshold)

Sets: curated (42 hand-written questions; reference answers in data/eval/curated_references.jsonl are DRAFTS until
the team reviews them) and synthetic_test (45 held-out, reference answers written by Qwen from one known passage).
Answers are cached in reports/stage_answers.jsonl; --reuse recomputes metrics without new LLM calls.

Usage: .venv/bin/python scripts/eval_stages.py [--sets curated,synthetic_test] [--reuse] [--no-judge]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from researchgraph import DATA, ROOT, answer, db, parse, rerank  # noqa: E402
from researchgraph.llm import ApiError, client  # noqa: E402
from researchgraph.retrieve import Retriever  # noqa: E402
from evaluate import boot_ci, judge  # noqa: E402

SETS = {"curated": "questions_v0.jsonl", "synthetic_test": "synthetic_test.jsonl"}
CACHE = os.path.join(ROOT, "reports", "stage_answers.jsonl")
OUT_JSON = os.path.join(ROOT, "reports", "stage_results.json")
OUT_MD = os.path.join(ROOT, "reports", "STAGES.md")
EMBED_LIMIT = 512
SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
RETRIEVERS = [("bm25", "B0 BM25", "S2"), ("dense", "B1 Dense", "S2"), ("hybrid", "B2 Hybrid", "S2"),
              ("rerank", "B3 Hybrid+rerank", "S3"), ("rerank+entity", "B3 + per-entity (app default)", "S3")]


# ---------- metric helpers ----------
def rouge(ref: str, hyp: str) -> dict:
    s = SCORER.score(ref, hyp)
    return {k: {"p": v.precision, "r": v.recall, "f": v.fmeasure} for k, v in s.items()}


def _norm_tokens(t: str) -> list[str]:
    t = re.sub(r"\b(a|an|the)\b", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower()))
    return t.split()


def token_f1(ref: str, hyp: str) -> float:
    r, h = _norm_tokens(ref), _norm_tokens(hyp)
    common = sum((Counter(r) & Counter(h)).values())
    if not r or not h or not common:
        return 0.0
    p, rc = common / len(h), common / len(r)
    return 2 * p * rc / (p + rc)


def ndcg10(hits, gold_chunks, gold_papers) -> float:
    """Binary-gain nDCG@10: gold passage if the question has gold passages, else gold paper (first hit per paper)."""
    if gold_chunks:
        rel = [1.0 if h.chunk_id in gold_chunks else 0.0 for h in hits[:10]]
        n_rel = len(gold_chunks)
    else:
        seen, rel = set(), []
        for h in hits[:10]:
            rel.append(1.0 if h.paper_id in gold_papers and h.paper_id not in seen else 0.0)
            seen.add(h.paper_id)
        n_rel = len(gold_papers)
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(rel))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel, 10)))
    return dcg / idcg if idcg else 0.0


def answer_text(a: dict) -> str:
    return " ".join([a.get("short_answer", "")] + [c["text"] for c in a.get("claims", [])]).strip()


def summ(xs) -> dict:
    xs = [float(x) for x in xs if x is not None]
    return {"mean": round(float(np.mean(xs)), 3) if xs else None, "ci95": boot_ci(xs), "n": len(xs)}


def cosine(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ---------- stages ----------
def stage1() -> dict:
    con = db.connect()
    rows = con.execute("SELECT paper_id, section, text FROM chunks").fetchall()
    toks = np.array([parse.est_tokens(r[2]) for r in rows])
    no_section = sum(1 for r in rows if not r[1] or r[1].lower() in ("front matter", "unknown", ""))
    per_paper = Counter(r[0] for r in rows)
    papers = con.execute("SELECT COUNT(*) FROM papers WHERE pdf_path IS NOT NULL").fetchone()[0]
    return {"papers_with_pdf": papers, "papers_indexed": len(per_paper), "chunks": len(rows),
            "chunk_tokens_p50": int(np.percentile(toks, 50)), "chunk_tokens_p95": int(np.percentile(toks, 95)),
            "chunk_tokens_max": int(toks.max()), "share_under_embed_limit": round(float((toks <= EMBED_LIMIT).mean()), 4),
            "share_with_detected_section": round(1 - no_section / len(rows), 3),
            "chunks_per_paper_median": int(np.median(list(per_paper.values())))}


def load_set(name: str) -> list[dict]:
    qs = [json.loads(l) for l in open(os.path.join(DATA, "eval", SETS[name])) if l.strip()]
    if name == "curated":
        refs = {r["qid"]: r for r in map(json.loads, open(os.path.join(DATA, "eval", "curated_references.jsonl")))}
        for q in qs:
            r = refs.get(q["qid"], {})
            q["reference_answer"] = r.get("reference_answer") if q["answerable"] else None
            q["gold_chunks"] = r.get("gold_chunks", [])
    return qs


def retrieval_rows(R: Retriever, q: dict) -> dict:
    gp, gc, ref = set(q["gold_papers"]), set(q.get("gold_chunks") or []), q.get("reference_answer")
    out = {}
    for mode, label, _ in RETRIEVERS:
        res = R.search_decomposed(q["question"], mode="rerank", k=10) if mode == "rerank+entity" \
            else R.search(q["question"], mode=mode, k=10)
        hits = res.hits[:10]
        row = {"paper_r10": len(gp & {h.paper_id for h in hits}) / len(gp),
               "mrr": next((1 / (i + 1) for i, h in enumerate(hits)
                            if (h.chunk_id in gc if gc else h.paper_id in gp)), 0.0),
               "ndcg10": ndcg10(hits, gc, gp)}
        if gc:
            row["passage_hit10"] = float(any(h.chunk_id in gc for h in hits))
        if ref:
            row["rougeL_recall_top10"] = rouge(ref, " ".join(h.text for h in hits))["rougeL"]["r"]
        out[label] = row
    return out


def run_answers(R: Retriever, qs: list[dict], name: str, cached: dict, do_judge: bool, do_followups: bool) -> list[dict]:
    recs = []
    for q in qs:
        key = f"{name}:{q['qid']}"
        if key in cached:
            recs.append(cached[key])
            continue
        res = R.search_decomposed(q["question"], mode="rerank", k=10)
        try:
            a = answer.ask(q["question"], res, log=False)
            if do_followups:
                answer.suggest_followups(a)
        except ApiError as e:
            print(f"  {q['qid']} API error {e.status}; skipped")
            continue
        labels = judge(a) if (do_judge and a.get("claims")) else []
        rec = {"key": key, "qid": q["qid"], "set": name, "type": q["type"], "answer": a, "judge_labels": labels}
        recs.append(rec)
        with open(CACHE, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"  {q['qid']} answerable={a.get('answerable')} claims={len(a.get('claims', []))} "
              f"followups={len(a.get('followups', []))} {a['latency_s']}s")
    return recs


def answer_rows(q: dict, rec: dict, R: Retriever) -> dict:
    a = rec["answer"]
    ev = {e["label"]: e for e in a["evidence"]}
    gp, gc, ref = set(q["gold_papers"]), set(q.get("gold_chunks") or []), q.get("reference_answer")
    pack_text = " ".join(e["text"] for e in a["evidence"])
    row = {"qid": q["qid"], "type": q["type"], "expected_answerable": q["answerable"],
           "model_answerable": a.get("answerable")}
    # S4 evidence pack
    if gp:
        row["pack_gold_paper_recall"] = len(gp & {e["paper_id"] for e in a["evidence"]}) / len(gp)
        row["pack_context_precision"] = (sum(e["paper_id"] in gp for e in a["evidence"]) / len(a["evidence"])) if a["evidence"] else 0.0
    if gc:
        row["pack_gold_passage_hit"] = float(any(e["chunk_id"] in gc for e in a["evidence"]))
    if ref:
        row["pack_rougeL_recall"] = rouge(ref, pack_text)["rougeL"]["r"]
    # S5 generation (answerable questions with a reference only)
    if ref and q["answerable"]:
        hyp = answer_text(a)
        r = rouge(ref, hyp)
        row.update({"rouge1_f": r["rouge1"]["f"], "rouge2_f": r["rouge2"]["f"], "rougeL_f": r["rougeL"]["f"],
                    "rougeL_r": r["rougeL"]["r"], "token_f1": token_f1(ref, hyp),
                    "short_rougeL_f": rouge(ref, a.get("short_answer", ""))["rougeL"]["f"],
                    "answer_words": len(hyp.split()), "_ref": ref, "_hyp": hyp})
    # S6 grounding
    grounded = [rouge(" ".join(ev[l]["text"] for l in c["evidence"]), c["text"])["rougeL"]["p"]
                for c in a.get("claims", []) if c.get("evidence")]
    row["claim_rougeL_precision_vs_cited"] = float(np.mean(grounded)) if grounded else None
    row["citation_validity"] = a.get("citation_validity")
    row["n_claims"] = len(a.get("claims", []))
    row["judge_labels"] = rec.get("judge_labels", [])
    # S7 follow-ups
    fus = a.get("followups") or []
    row["n_followups"] = len(fus)
    if fus:
        row["followup_novelty"] = float(np.mean([1 - rouge(q["question"], f)["rougeL"]["f"] for f in fus]))
        covered = []
        for f in fus:
            hits = R.search(f, mode="rerank", k=1).hits
            covered.append(float(bool(hits) and hits[0].score >= answer.MIN_RERANK_SCORE))
        row["followup_covered"] = float(np.mean(covered))
        row["_followups"] = fus
    return row


def semantic_sims(rows: list[dict]) -> None:
    todo = [r for r in rows if "_ref" in r]
    if not todo:
        return
    try:
        vecs = client().embed([t[:1500] for r in todo for t in (r["_ref"], r["_hyp"])])
    except (ApiError, Exception) as e:  # embeddings are optional for this metric
        print("semantic similarity skipped:", e)
        return
    for i, r in enumerate(todo):
        r["semantic_sim"] = cosine(vecs[2 * i], vecs[2 * i + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="curated,synthetic_test")
    ap.add_argument("--reuse", action="store_true", help="only use cached answers; no new LLM calls")
    ap.add_argument("--no-judge", action="store_true")
    args = ap.parse_args()

    cached = {}
    if os.path.exists(CACHE):
        for l in open(CACHE):
            if l.strip():
                r = json.loads(l)
                cached[r["key"]] = r
    R = Retriever()
    report = {"run_at": time.strftime("%Y-%m-%d %H:%M"), "llm": client().llm.model, "reranker": rerank.MODEL,
              "embeddings": "bge-small-en-v1.5", "prompt": answer.PROMPT_VERSION, "followup_prompt": answer.FOLLOWUP_VERSION,
              "S1": stage1(), "sets": {}}
    print("S1", report["S1"])
    for name in args.sets.split(","):
        qs = load_set(name)
        print(f"\n== {name}: {len(qs)} questions")
        ret = {label: [] for _, label, _ in RETRIEVERS}
        for q in qs:
            if q["gold_papers"]:
                for label, row in retrieval_rows(R, q).items():
                    ret[label].append(row)
        if args.reuse:
            recs = [cached[f"{name}:{q['qid']}"] for q in qs if f"{name}:{q['qid']}" in cached]
        else:
            recs = run_answers(R, qs, name, cached, do_judge=(name == "curated" and not args.no_judge),
                               do_followups=(name == "curated"))
        byq = {q["qid"]: q for q in qs}
        rows = [answer_rows(byq[r["qid"]], r, R) for r in recs]
        semantic_sims(rows)
        keys = ["paper_r10", "passage_hit10", "mrr", "ndcg10", "rougeL_recall_top10"]
        s = {"n_questions": len(qs), "retrieval": {label: {k: summ([r.get(k) for r in rs]) for k in keys}
                                                    for label, rs in ret.items()}}
        akeys = ["pack_gold_paper_recall", "pack_gold_passage_hit", "pack_context_precision", "pack_rougeL_recall",
                 "rouge1_f", "rouge2_f", "rougeL_f", "rougeL_r", "short_rougeL_f", "token_f1", "semantic_sim",
                 "answer_words", "claim_rougeL_precision_vs_cited", "citation_validity", "followup_novelty", "followup_covered"]
        s["answers"] = {k: summ([r.get(k) for r in rows]) for k in akeys}
        unans = [r for r in rows if not r["expected_answerable"]]
        ans_ = [r for r in rows if r["expected_answerable"]]
        labels = [l for r in rows for l in r["judge_labels"]]
        s["answers"]["abstention"] = f"{sum(r['model_answerable'] is False for r in unans)}/{len(unans)}"
        s["answers"]["answered_when_answerable"] = f"{sum(r['model_answerable'] is True for r in ans_)}/{len(ans_)}"
        s["answers"]["judge_supported"] = round(labels.count("supported") / len(labels), 3) if labels else None
        s["answers"]["n_followups_total"] = sum(r["n_followups"] for r in rows)
        s["per_question"] = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                              if k not in ("_ref", "_hyp")} for r in rows]
        s["examples"] = [{"qid": r["qid"], "reference": r["_ref"], "answer": r["_hyp"][:600],
                          "rougeL_f": round(r["rougeL_f"], 3), "followups": r.get("_followups", [])}
                         for r in rows if "_ref" in r][:6]
        report["sets"][name] = s
        a = s["answers"]
        print(f"  ROUGE-L F1 {a['rougeL_f']['mean']}  token F1 {a['token_f1']['mean']}  "
              f"semantic {a['semantic_sim']['mean']}  abstention {a['abstention']}")
    json.dump(report, open(OUT_JSON, "w"), indent=1, default=str)
    write_md(report)
    print("\nSaved", os.path.relpath(OUT_JSON, ROOT), "and", os.path.relpath(OUT_MD, ROOT))


def _f(m):
    if not m or m.get("mean") is None:
        return "–"
    lo, hi = m["ci95"]
    return f"{m['mean']:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None and m["n"] > 1 else f"{m['mean']:.2f}"


def write_md(rep: dict) -> None:
    s1 = rep["S1"]
    out = ["# AI ResearchGraph — stage-by-stage evaluation", "",
           f"Run {rep['run_at']} · LLM `{rep['llm']}` · embeddings `{rep['embeddings']}` · reranker `{rep['reranker']}` · "
           f"answer prompt `{rep['prompt']}` · follow-up prompt `{rep['followup_prompt']}`. Means with 95% bootstrap CIs.", "",
           "**Read with care.** Curated reference answers and gold passages (`data/eval/curated_references.jsonl`) were drafted "
           "by Claude from the gold papers and still need team review; the gold passages were picked from each gold paper "
           "with help from the cross-encoder, so passage-level numbers may slightly favour B3. Synthetic reference answers were written by Qwen3-8B from "
           "one passage, so they share its wording. ROUGE counts word overlap: a correct answer phrased differently scores low, "
           "so read it with token F1, semantic similarity and the grounding numbers, not alone.", "",
           "## S1 · Parsing and chunking (whole corpus)", "", "| Measure | Value |", "|---|---|",
           f"| Papers indexed / with PDF | {s1['papers_indexed']} / {s1['papers_with_pdf']} |",
           f"| Chunks | {s1['chunks']} (median {s1['chunks_per_paper_median']} per paper) |",
           f"| Chunk length p50 / p95 / max (est. tokens) | {s1['chunk_tokens_p50']} / {s1['chunk_tokens_p95']} / {s1['chunk_tokens_max']} |",
           f"| Chunks within the 512-token embedding limit | {s1['share_under_embed_limit']:.1%} |",
           f"| Chunks with a detected section heading | {s1['share_with_detected_section']:.1%} |", ""]
    for name, s in rep["sets"].items():
        a = s["answers"]
        out += [f"## {name} (n = {s['n_questions']})", "",
                "### S2–S3 · Retrieval and reranking (top 10)", "",
                "| Stage | System | Paper Recall@10 | Passage Hit@10 | MRR | nDCG@10 | ROUGE-L recall of reference in top 10 |",
                "|---|---|---|---|---|---|---|"]
        for (_, label, stage) in RETRIEVERS:
            m = s["retrieval"][label]
            out.append(f"| {stage} | {label} | {_f(m['paper_r10'])} | {_f(m['passage_hit10'])} | {_f(m['mrr'])} | "
                       f"{_f(m['ndcg10'])} | {_f(m['rougeL_recall_top10'])} |")
        out += ["", "MRR and nDCG use gold passages where the question has them, else gold papers.", "",
                "### S4–S7 · Evidence pack, answer, grounding, follow-ups (B3 + per-entity, prompt v3)", "",
                "| Stage | Measure | Value |", "|---|---|---|",
                f"| S4 Evidence pack | Gold paper in pack (recall) | {_f(a['pack_gold_paper_recall'])} |",
                f"| S4 Evidence pack | Gold passage in pack | {_f(a['pack_gold_passage_hit'])} |",
                f"| S4 Evidence pack | Context precision (pack passages from gold papers) | {_f(a['pack_context_precision'])} |",
                f"| S4 Evidence pack | ROUGE-L recall of reference in pack (upper bound for the answer) | {_f(a['pack_rougeL_recall'])} |",
                f"| S5 Answer | ROUGE-1 F1 | {_f(a['rouge1_f'])} |",
                f"| S5 Answer | ROUGE-2 F1 | {_f(a['rouge2_f'])} |",
                f"| S5 Answer | ROUGE-L F1 (full answer / short answer only) | {_f(a['rougeL_f'])} / {_f(a['short_rougeL_f'])} |",
                f"| S5 Answer | ROUGE-L recall | {_f(a['rougeL_r'])} |",
                f"| S5 Answer | Token F1 (SQuAD-style) | {_f(a['token_f1'])} |",
                f"| S5 Answer | Semantic similarity to reference (bge-small cosine) | {_f(a['semantic_sim'])} |",
                f"| S5 Answer | Mean answer length (words) | {_f(a['answer_words'])} |",
                f"| S5 Answer | Declined unanswerable / answered answerable | {a['abstention']} / {a['answered_when_answerable']} |",
                f"| S6 Grounding | Citation IDs that exist in the evidence | {_f(a['citation_validity'])} |",
                f"| S6 Grounding | Claim words found in its cited passages (ROUGE-L precision) | {_f(a['claim_rougeL_precision_vs_cited'])} |",
                f"| S6 Grounding | Qwen-judged claims supported | {a['judge_supported'] if a['judge_supported'] is not None else '–'} |"]
        if a["followup_novelty"]["n"]:
            out += [f"| S7 Follow-ups | Suggestions generated | {a['n_followups_total']} |",
                    f"| S7 Follow-ups | Novelty vs original question (1 − ROUGE-L) | {_f(a['followup_novelty'])} |",
                    f"| S7 Follow-ups | Covered by the corpus (top rerank score ≥ {answer.MIN_RERANK_SCORE}) | {_f(a['followup_covered'])} |"]
        out.append("")
        if s.get("examples"):
            out += ["<details><summary>Example reference vs answer</summary>", ""]
            for e in s["examples"][:3]:
                out += [f"**{e['qid']}** (ROUGE-L F1 {e['rougeL_f']})", "", f"- Reference: {e['reference']}",
                        f"- Answer: {e['answer']}"]
                if e["followups"]:
                    out.append("- Follow-ups: " + " · ".join(e["followups"]))
                out.append("")
            out += ["</details>", ""]
    open(OUT_MD, "w").write("\n".join(out))


if __name__ == "__main__":
    main()
