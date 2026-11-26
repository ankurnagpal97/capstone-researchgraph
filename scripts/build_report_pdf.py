"""Build the submission PDF of the technical report: page 1 lists every deliverable link, then reports/REPORT.md.

Relative links in the report become absolute GitHub links so they work in the PDF; the Mermaid architecture
diagram is rendered by mermaid.js inside headless Chrome, which also prints the PDF.

Needs: pip install markdown==3.7, and Google Chrome (or Chromium) installed.
Usage: .venv/bin/python scripts/build_report_pdf.py [--author "Name"] [--out reports/ResearchGraph_Technical_Report.pdf]
       .venv/bin/python scripts/build_report_pdf.py --source RESPONSIBLE_AI.md --no-cover --out Model_Card.pdf
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import tempfile

import markdown

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "https://github.com/ankurnagpal97/capstone-researchgraph"
PAGES = "https://ankurnagpal97.github.io/capstone-researchgraph"
BLOB = f"{REPO}/blob/main"

LINKS = [
    ("GitHub repository (public, reproducible code)", REPO,
     "Code, pinned requirements, Dockerfile, tests, evaluation scripts and result files. Setup and run commands: README."),
    ("Trace dashboard (OpenTelemetry)", f"{PAGES}/traces/",
     "Latency and tokens per span type, agent tool calls and critic verdicts, and a span waterfall for every traced task."),
    ("Demo video (about 10 minutes)", f"{PAGES}/demo/researchgraph_demo.mp4",
     f"Narrated recording of the running system. Also in the repository: {BLOB}/docs/demo/researchgraph_demo.mp4"),
    ("Model card and Responsible AI", f"{BLOB}/reports/RESPONSIBLE_AI.md",
     "Intended use, out-of-scope uses, measured failure modes, refusal behaviour, PII handling, adversarial tests."),
    ("Technical report (online copy of this document)", f"{BLOB}/reports/REPORT.md",
     "Supporting reports: SPEC, RESULTS, STAGES, AGENT, PROMPTS, CHUNKING, INJECTION, CORPUS (in reports/)."),
    ("Project page", f"{PAGES}/", "One page with all of the links above."),
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { box-sizing: border-box; }
body { font: 10.2pt/1.5 -apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif; color: #1b1b1b; }
h1 { font-size: 19pt; margin: 0 0 4pt; line-height: 1.2; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #ddd; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 5pt; break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: #1d5fb4; text-decoration: none; word-break: break-word; }
code { font: 8.8pt/1.4 Menlo, Consolas, monospace; background: #f3f3f1; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f6f4; padding: 8px; border-radius: 6px; overflow: hidden; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 8.8pt; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 1px solid #d8d8d4; padding: 3.5pt 5pt; vertical-align: top; text-align: left; }
th { background: #f1f1ee; }
blockquote { margin: 8pt 0; padding: 6pt 10pt; background: #f6f7fb; border-left: 3px solid #1d5fb4; color: #333; }
.mermaid { text-align: center; margin: 8pt 0; break-inside: avoid; }
.mermaid svg { max-width: 100%; height: auto; max-height: 210mm; }
.cover { page-break-after: always; }
.cover .eyebrow { font-size: 9.5pt; letter-spacing: 2px; color: #b8541a; font-weight: 700; text-transform: uppercase; margin-top: 4mm; }
.cover h1 { font-size: 24pt; margin: 4mm 0 2mm; }
.cover .sub { font-size: 11.5pt; color: #444; margin: 0 0 3mm; }
.cover .by { font-size: 10.5pt; color: #333; margin: 0 0 8mm; }
.cover h2 { border: 0; margin: 0 0 3mm; font-size: 13pt; }
.cover table { font-size: 9.6pt; }
.cover td:first-child { width: 33%; font-weight: 600; }
.cover td .d { color: #555; font-size: 8.6pt; margin-top: 2pt; }
.cover td a { font-weight: 600; }
.cover .note { font-size: 9pt; color: #555; margin-top: 5mm; }
"""


def chrome() -> str:
    for c in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium",
              shutil.which("google-chrome") or "", shutil.which("chromium") or ""):
        if c and os.path.exists(c):
            return c
    raise SystemExit("Google Chrome or Chromium is needed to print the PDF")


def absolutize(md: str) -> str:
    """[text](FILE.md) and [text](../README.md) -> absolute GitHub links."""
    def fix(m):
        text, target = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#)", target):
            return m.group(0)
        path = os.path.normpath(os.path.join("reports", target))
        return f"[{text}]({BLOB}/{path})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix, md)


def render(md: str) -> str:
    blocks = []

    def keep_mermaid(m):
        # A4 portrait: stack the subgraphs top-to-bottom so the diagram is readable at page width
        blocks.append(m.group(1).replace("flowchart LR", "flowchart TB", 1))
        return f"\n\nMERMAIDBLOCK{len(blocks) - 1}\n\n"
    md = re.sub(r"```mermaid\n(.*?)```", keep_mermaid, md, flags=re.S)
    out = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    for i, b in enumerate(blocks):
        out = out.replace(f"<p>MERMAIDBLOCK{i}</p>", f'<div class="mermaid">{html.escape(b)}</div>')
    return out


def cover(author: str) -> str:
    rows = "".join(f'<tr><td>{html.escape(n)}</td><td><a href="{u}">{html.escape(u)}</a>'
                   f'<div class="d">{html.escape(d)}</div></td></tr>' for n, u, d in LINKS)
    by = f'<p class="by">Submitted by: {html.escape(author)}</p>' if author else ""
    return f"""<section class="cover">
<div class="eyebrow">IIT Kharagpur Capstone · Track B · Qwen3-8B · November 2026</div>
<h1>AI ResearchGraph: evidence-grounded RAG over RAG research</h1>
<p class="sub">Technical report. Question answering over 146 research papers in which every claim is cited to the
exact passage it comes from, results from different studies are flagged as not directly comparable, and the system
says so when the papers do not answer the question.</p>
{by}
<h2>Submission links</h2>
<table>{rows}</table>
<p class="note">All links are public. The repository's README has the commands to rebuild the index, run the app,
the tests and every evaluation, and to build and run the Docker image. The trace dashboard is generated from the
trace log by <code>scripts/make_trace_dashboard.py</code>; this PDF is generated by
<code>scripts/build_report_pdf.py</code>.</p>
</section>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="")
    ap.add_argument("--source", default="REPORT.md", help="a Markdown file in reports/")
    ap.add_argument("--no-cover", action="store_true", help="skip the links page")
    ap.add_argument("--title", default="AI ResearchGraph: technical report")
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "ResearchGraph_Technical_Report.pdf"))
    args = ap.parse_args()

    md = open(os.path.join(ROOT, "reports", args.source)).read()
    md = absolutize(md)
    body = render(md)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(args.title)}</title>
<style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>
</head><body>{"" if args.no_cover else cover(args.author)}{body}
<script>mermaid.initialize({{startOnLoad: true, theme: "neutral", flowchart: {{useMaxWidth: true}}}});</script>
</body></html>"""
    tmp = tempfile.mkdtemp(prefix="rg_pdf_")
    src = os.path.join(tmp, "report.html")
    open(src, "w").write(page)
    subprocess.run([chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    "--run-all-compositor-stages-before-draw", f"--print-to-pdf={args.out}", f"file://{src}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {os.path.relpath(args.out, ROOT)} ({os.path.getsize(args.out) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
