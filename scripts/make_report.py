"""Turn reports/eval_results.json + reports/qasper_results.json into reports/RESULTS.md.

Usage: .venv/bin/python scripts/make_report.py
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import ROOT  # noqa: E402

R = os.path.join(ROOT, "reports")


def fmt(m, key):
    if key not in m:
        return "–"
    lo, hi = m[key]["ci95"]
    return f"{m[key]['mean']:.2f} [{lo:.2f}, {hi:.2f}]" if lo is not None else f"{m[key]['mean']:.2f}"


def main():
    ev = json.load(open(os.path.join(R, "eval_results.json")))
    out = [f"# AI ResearchGraph — evaluation results", "",
           f"Run {ev['run_at']} · corpus {ev['n_papers']} papers, {ev['n_chunks']} chunks · "
           "values are means with 95% bootstrap confidence intervals over questions.", "",
           "**Read with care.** Curated gold papers were drafted from paper titles and still need team review. "
           "Synthetic questions were written by Qwen3-8B from one known passage each, so they echo that passage's wording "
           "and are easier than real questions. With n under 50 the intervals are wide: differences whose intervals overlap "
           "are not established.", ""]
    for name, systems in ev["retrieval"].items():
        n = next(iter(systems.values()))["paper_r10"]["n"]
        out += [f"## Retrieval — {name} (n = {n})", "",
                "| System | Paper Recall@5 | Paper Recall@10 | Paper MRR | Passage Hit@10 | Passage MRR |",
                "|---|---|---|---|---|---|"]
        for label, m in systems.items():
            out.append(f"| {label} | {fmt(m, 'paper_r5')} | {fmt(m, 'paper_r10')} | {fmt(m, 'paper_mrr')} | "
                       f"{fmt(m, 'passage_hit10')} | {fmt(m, 'passage_mrr')} |")
        g = ev.get("graph_ablation", {}).get(name)
        if g:
            out += ["", f"Graph forced on for the {g['n']} multi-hop, comparison and disagreement questions:", "",
                    "| System | Paper Recall@10 | Paper MRR |", "|---|---|---|"]
            for label, m in g["results"].items():
                out.append(f"| {label} | {fmt(m, 'paper_r10')} | {fmt(m, 'paper_mrr')} |")
        pq = ev["per_question"].get(name, [])
        if pq and name == "curated":
            by = defaultdict(lambda: defaultdict(list))
            labels = [k for k in pq[0] if k.endswith("paper R@10")]
            for row in pq:
                for k in labels:
                    by[row["type"]][k].append(row[k])
            out += ["", "Paper Recall@10 by question type:", "",
                    "| Type | n | " + " | ".join(k.replace(" paper R@10", "") for k in labels) + " |",
                    "|---|---|" + "---|" * len(labels)]
            for t, d in sorted(by.items()):
                nn = len(next(iter(d.values())))
                out.append(f"| {t} | {nn} | " + " | ".join(f"{sum(v)/len(v):.2f}" for v in d.values()) + " |")
        out.append("")

    qp = os.path.join(R, "qasper_results.json")
    if os.path.exists(qp):
        q = json.load(open(qp))
        out += [f"## External benchmark — {q['dataset']}", "",
                f"{q['n_questions']} questions over {q['n_papers']} papers (seed {q['seed']}). Each question's own paper is "
                "searched, and success means ranking an annotator-marked evidence paragraph near the top. This measures "
                "within-paper evidence retrieval on NLP papers, not cross-paper comparison or the citation graph. "
                f"Reranker: `{q['reranker']}`.", "",
                "| System | Hit@1 | Hit@3 | Hit@5 | Evidence Recall@5 | MRR |", "|---|---|---|---|---|---|"]
        for s in q["summary"]:
            out.append(f"| {s['system']} | {s['hit1']:.3f} | {s['hit3']:.3f} | {s['hit5']:.3f} | {s['recall5']:.3f} | {s['mrr']:.3f} |")
        out.append("")

    a = ev.get("answers")
    if a:
        out += [f"## End-to-end answers — {a['set']} set, {a['system']} (n = {a['n']})", "",
                "| Measure | Value |", "|---|---|"]
        rows = [("Declined unanswerable questions", a["abstention_on_unanswerable"]),
                ("Answered answerable questions", a["answered_when_answerable"]),
                ("Invalid JSON replies", a["json_failures"]),
                ("Citation IDs that exist in the evidence", a["citation_id_validity"]),
                ("Claims without a valid citation", a["claims_without_valid_citation"]),
                ("Gold papers among cited passages", a["gold_paper_recall_in_citations"]),
                ("Auto-judged: supported / partial / not supported",
                 f"{a['auto_support_supported']} / {a['auto_support_partial']} / {a['auto_support_not_supported']}"),
                ("Latency p50 / p95 (s)", f"{a['latency_p50_s']} / {a['latency_p95_s']}"),
                ("Mean tokens per answer", a["mean_total_tokens"])]
        out += [f"| {k} | {v} |" for k, v in rows]
        out += ["", f"*{a['note']}*", "",
                "Human citation-support labels: fill in `reports/citation_audit_sample.csv` "
                "(column `human_label`), then report the share labelled supported.", ""]
        fails = [r for r in a["per_question"] if (r["expected_answerable"] != r["model_answerable"]) or r["error"]]
        if fails:
            out += ["### Answer failures to analyse", "", "| Question | Type | Expected answerable | Model said | Note |",
                    "|---|---|---|---|---|"]
            for r in fails:
                out.append(f"| {r['qid']} | {r['type']} | {r['expected_answerable']} | {r['model_answerable']} | "
                           f"{(r['error'] or r['short_answer'][:120]).replace('|', '/')} |")
            out.append("")
    path = os.path.join(R, "RESULTS.md")
    open(path, "w").write("\n".join(out))
    print("Wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
