"""Record a narrated demo video of the Streamlit app (MP4, 1440x900).

Drives the real app with Playwright (headless Chromium), narrates each scene with a neural text-to-speech voice,
then lays each narration clip at its scene's start time and muxes it onto the screen recording with ffmpeg.
Answers are generated live, so nobody else should use the group API key while this runs (about 4 minutes).

Needs: pip install playwright imageio-ffmpeg edge-tts && python -m playwright install chromium
Voice: Microsoft neural "en-US-AndrewNeural" via edge-tts (online: only the narration text is sent);
       --tts say uses the built-in macOS voice instead (offline).
Usage:  .venv/bin/streamlit run app.py --server.headless true --client.toolbarMode minimal --server.port 8502   (port 8502)
        .venv/bin/python scripts/record_demo.py [--out reports/demo/researchgraph_demo.mp4] [--voice Samantha]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1440, 900
APP = "http://localhost:8502"
TTS_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "researchgraph_tts")
Q_SINGLE = "How does Self-RAG use reflection tokens to decide when to retrieve and to critique its own generations?"
Q_ABSTAIN = "Which RAG method has the lowest carbon footprint per query?"
Q_AGENT = "How do RAGAS and ARES differ in how they evaluate RAG systems?"
Q_AGENT_FU = "What are the limitations of the second one?"
LLM_TIMEOUT = 150_000

CARD = """<html><body style="margin:0;width:{w}px;height:{h}px;background:{bg};color:{fg};font-family:-apple-system,Helvetica,Arial;
display:flex;flex-direction:column;justify-content:center;padding:0 120px;box-sizing:border-box">
<p style="font-size:22px;letter-spacing:4px;color:#F0A877;font-weight:600;margin:0">{eyebrow}</p>
<h1 style="font-size:84px;margin:18px 0 0 0;line-height:1.05">{title}</h1>
<p style="font-size:30px;line-height:1.45;color:{muted};margin:28px 0 0 0;max-width:1100px">{sub}</p></body></html>"""


def card(eyebrow, title, sub):
    return CARD.format(w=W, h=H, bg="#14213D", fg="#F6F4EE", muted="#C9D3E3", eyebrow=eyebrow, title=title, sub=sub)


def speak(text: str, path: str, tts: str, voice: str, rate: str) -> float:
    """Narrate `text` into a mono 44.1 kHz WAV and return its length in seconds.
    tts="edge": Microsoft neural voice via edge-tts (online; only the narration text is sent).
    tts="say":  the built-in macOS voice (offline fallback)."""
    raw = path.replace(".wav", ".mp3" if tts == "edge" else ".aiff")
    if tts == "edge":
        # The free endpoint throttles bursts ("No audio was received"), so: cache finished clips by
        # (voice, rate, text), pause between requests, and back off 5/10/20/40/60 s on failure.
        key = hashlib.sha1(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
        cached = os.path.join(TTS_CACHE, f"{key}.mp3")
        if not (os.path.exists(cached) and os.path.getsize(cached) > 0):
            os.makedirs(TTS_CACHE, exist_ok=True)
            exe = os.path.join(os.path.dirname(sys.executable), "edge-tts")
            for wait in (5, 10, 20, 40, 60, None):
                r = subprocess.run([exe, "--voice", voice, f"--rate={rate}", "--text", text, "--write-media", cached],
                                   capture_output=True)
                if r.returncode == 0 and os.path.getsize(cached) > 0:
                    break
                if wait is None:
                    os.remove(cached) if os.path.exists(cached) else None
                    raise RuntimeError(f"edge-tts failed: {r.stderr.decode()[-300:]}")
                print(f"  voice service busy, retrying in {wait}s")
                time.sleep(wait)
            time.sleep(2)
        shutil.copy(cached, raw)
    else:
        subprocess.run(["say", "-v", voice, "-r", rate, "-o", raw, text], check=True)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-af", "apad=pad_dur=0.25",
                    "-ar", "44100", "-ac", "1", path], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


class Recorder:
    def __init__(self, page, t0):
        self.page, self.t0, self.marks = page, t0, []

    def now(self):
        return time.time() - self.t0

    def scene(self, key, clips, action=None, min_after=0.8):
        """Start narration clip `key`, run `action`, then hold until the narration has finished."""
        start = self.now()
        dur = clips[key]["dur"]
        self.marks.append({"key": key, "start": round(start, 2)})
        print(f"[{start:6.1f}s] {key} ({dur:.1f}s narration)")
        if action:
            action()
        left = start + dur + min_after - self.now()
        if left > 0:
            self.page.wait_for_timeout(int(left * 1000))

    # --- helpers for driving Streamlit ---
    def highlight(self, loc, ms=700):
        try:
            loc.evaluate("(e)=>{e.scrollIntoView({block:'center',behavior:'smooth'});"
                         "e.style.outline='4px solid #B8541A';e.style.outlineOffset='3px';"
                         f"setTimeout(()=>{{e.style.outline='';}},{ms})}}")
            self.page.wait_for_timeout(ms)
        except Exception:
            pass

    def scroll(self, total, steps=8, pause=180):
        self.page.mouse.move(W * 0.45, H * 0.6)
        for _ in range(steps):
            self.page.mouse.wheel(0, total / steps)
            self.page.wait_for_timeout(pause)

    def scroll_top(self):
        self.page.mouse.move(W * 0.45, H * 0.6)
        self.page.mouse.wheel(0, -20000)
        self.page.wait_for_timeout(400)

    def wait_idle(self, timeout=LLM_TIMEOUT):
        """Wait for a Streamlit rerun to start and finish (the 'Running…' status widget)."""
        busy = self.page.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"]')
        try:
            busy.first.wait_for(state="visible", timeout=4000)
        except Exception:
            pass
        # Streamlit switches between spinners (retrieving → asking → follow-ups) with short gaps in
        # between, so only treat the app as idle after 2.5 s with nothing busy on screen.
        deadline, idle_since = time.time() + timeout / 1000, None
        while time.time() < deadline:
            if any(b.is_visible() for b in busy.all()):
                idle_since = None
            elif idle_since is None:
                idle_since = time.time()
            elif time.time() - idle_since >= 2.5:
                break
            self.page.wait_for_timeout(250)

    def ask(self, question):
        box = self.page.get_by_role("textbox", name="Question")
        self.highlight(box, 500)
        box.click()
        box.press("Meta+A")
        box.press("Backspace")
        box.type(question, delay=18)
        box.press("Tab")                     # commit the text so the Ask button enables
        self.wait_idle(10_000)
        btn = self.page.get_by_role("button", name="Ask", exact=True)
        self.highlight(btn, 500)
        btn.click()


NARRATION = {
    "intro": "AI ResearchGraph. An evidence-grounded question answering system over one hundred and forty-six research "
             "papers on retrieval-augmented generation, built for the IIT Kharagpur capstone, track B, with Kwen 3, 8 B.",
    "app": "This is the app. On the right you choose the retrieval system. The default, B3, is hybrid BM25 plus "
           "embedding search, reranked by the MS MARCO MiniLM cross-encoder. Answers come from a single Kwen 3 call "
           "that may use only passages from the indexed papers.",
    "ask1": "First, a single-paper question: how does Self-RAG use reflection tokens? The system retrieves and reranks "
            "passages, packs at most seven of them, and asks Kwen for a JSON list of claims, each citing the passages "
            "that support it. Code then checks every citation.",
    "answer1": "Every claim carries numbered citations, and the source list gives the paper, the section, the page and "
               "the chunk ID.",
    "evidence": "Each evidence card shows the exact passage the model was given, with a link to the paper on archive.",
    "followups": "New in the final version: after each answer, a second, small Kwen call suggests what a researcher "
                 "would most likely ask next. One goes deeper, one compares with a related method, and one asks about "
                 "limitations or evaluation.",
    "click_fu": "Clicking a suggestion asks it straight away. The question trail keeps the chain of questions, and "
                "every new answer brings fresh suggestions, so you can keep exploring the literature.",
    "answer_fu": "The follow-up is answered the same way, with its own citations and its own next questions.",
    "agent_on": "Next, agent mode. A planner, one Kwen call, reads the question and picks tools: search the papers, "
                "read one paper, follow citation neighbours, or look up paper details. The executor runs them, the "
                "answer is generated as before, and a critic decides whether one more round of searching is needed.",
    "agent_steps": "The agent steps panel shows every tool call, its result, and the critic's verdict. Each task is "
                   "capped at two rounds, eight tool calls and twenty thousand tokens, and bad JSON or a failing tool "
                   "is recovered from rather than crashing.",
    "agent_memory": "The agent remembers the conversation. Asked, what are the limitations of the second one, it "
                    "rewrites the question using the earlier turn, so it searches for ARES.",
    "trace": "Every run is traced with OpenTelemetry: the plan, each tool call, each search and rerank, and each "
             "model call with its tokens and timing.",
    "ask_abstain": "Back in the default pipeline, a question the papers cannot answer: which RAG method has the lowest carbon footprint per query?",
    "abstain": "The system says it is not established by the indexed papers, and lists what is missing, instead of "
               "guessing.",
    "graph": "The citation neighbours view shows which corpus papers cite each other. A citation is never treated as "
             "agreement, and because the graph did not measurably improve retrieval, it is off by default.",
    "eval": "Finally, the evaluation tab. The new stage-by-stage report measures accuracy at every step of the "
            "pipeline: chunking; retrieval and reranking with recall, MRR and N D C G; the evidence pack; the answer, "
            "with ROUGE, token F1 and semantic similarity against reference answers; citation grounding; and the "
            "follow-up suggestions.",
    "eval2": "{eval_numbers}",
    "outro": "AI ResearchGraph. Every claim cited, comparability flagged, and an honest answer when the papers do not "
             "support one. Thank you.",
}


def eval_numbers() -> str:
    """One narrated sentence from reports/stage_results.json (curated set), if it exists."""
    p = os.path.join(ROOT, "reports", "stage_results.json")
    if not os.path.exists(p):
        return "The full tables are in the stages report."
    s = json.load(open(p))["sets"].get("curated", {})
    a = s.get("answers", {})
    r = s.get("retrieval", {})
    try:
        b2 = r["B2 Hybrid"]["passage_hit10"]["mean"]
        b3 = r["B3 + per-entity (app default)"]["passage_hit10"]["mean"]
        pct = lambda x: f"{round(100 * x)} percent"
        return (f"On the curated questions, the gold passage is in the top ten for {pct(b2)} of questions with hybrid "
                f"search and {pct(b3)} after reranking. Answers reach a ROUGE-L F1 of {a['rougeL_f']['mean']:.2f} and a "
                f"semantic similarity of {a['semantic_sim']['mean']:.2f} against the reference answers, and "
                f"{pct(a['claim_rougeL_precision_vs_cited']['mean'])} of each claim's words are found in the passages "
                f"it cites.")
    except (KeyError, TypeError):
        return "The full tables are in the stages report."


def agent_numbers() -> str:
    p = os.path.join(ROOT, "reports", "agent_results.json")
    if not os.path.exists(p):
        return ""
    r = json.load(open(p))
    try:
        P, A = r["pipeline"], r["agent"]
        return (f" End to end, the pipeline is right on {round(100 * P['end_to_end_accuracy']['mean'])} percent of the "
                f"{P['n']} curated questions and the agent on {round(100 * A['end_to_end_accuracy']['mean'])} percent, a "
                f"difference within the noise, at about twice the tokens, so the single-call pipeline stays the default.")
    except (KeyError, TypeError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "demo", "researchgraph_demo.mp4"))
    ap.add_argument("--tts", choices=["edge", "say"], default="edge",
                    help="edge = Microsoft neural voice (online, natural); say = macOS voice (offline)")
    ap.add_argument("--voice", default=None, help="default: en-US-AndrewNeural (edge) or Daniel (say)")
    ap.add_argument("--rate", default=None, help="default: -5%% (edge) or 175 words/min (say)")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="rg_demo_")

    NARRATION["eval2"] = eval_numbers() + agent_numbers()
    voice = args.voice or ("en-US-AndrewNeural" if args.tts == "edge" else "Daniel")
    rate = args.rate or ("-5%" if args.tts == "edge" else "175")
    clips = {}
    for k, text in NARRATION.items():
        path = os.path.join(tmp, f"{k}.wav")
        clips[k] = {"path": path, "dur": speak(text, path, args.tts, voice, rate), "text": text}
    print("narration total %.0fs" % sum(c["dur"] for c in clips.values()))

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, record_video_dir=tmp,
                                  record_video_size={"width": W, "height": H})
        page = ctx.new_page()
        t0 = time.time()
        r = Recorder(page, t0)

        page.set_content(card("IIT KGP CAPSTONE · TRACK B · QWEN3-8B", "AI ResearchGraph",
                              "Evidence-grounded answers about RAG research. Every claim is cited to the exact "
                              "passage it comes from."))
        r.scene("intro", clips)

        def load_app():
            page.goto(APP)
            page.get_by_role("textbox", name="Question").wait_for(timeout=60_000)
            page.wait_for_timeout(1500)
            r.highlight(page.locator('[data-testid="stSelectbox"]').first, 900)
        r.scene("app", clips, load_app)

        def ask1():
            r.ask(Q_SINGLE)
            r.wait_idle()
        r.scene("ask1", clips, ask1)
        r.scene("answer1", clips, lambda: r.scroll(450, steps=6, pause=300))

        def evidence():
            page.get_by_text("Evidence given to the model").scroll_into_view_if_needed()
            exp = page.locator('[data-testid="stExpander"] summary', has_text="E1 · ").first
            r.highlight(exp, 500)
            exp.click()
            page.wait_for_timeout(600)
            r.scroll(250, steps=4)
        r.scene("evidence", clips, evidence)

        def followups():
            head = page.get_by_text("You might ask next")
            head.scroll_into_view_if_needed()
            page.mouse.wheel(0, -150)
            for b in page.get_by_role("button", name=re.compile("^→")).all():
                r.highlight(b, 600)
        r.scene("followups", clips, followups)

        def click_fu():
            btn = page.get_by_role("button", name=re.compile("^→")).nth(1)
            r.highlight(btn, 500)
            btn.click()
            r.wait_idle()
            r.scroll_top()
        r.scene("click_fu", clips, click_fu)
        r.scene("answer_fu", clips, lambda: r.scroll(500, steps=6, pause=350))

        def agent_on():
            r.scroll_top()
            tog = page.get_by_text("Agent mode (plan", exact=False)
            r.highlight(tog, 600)
            tog.click()
            r.wait_idle(10_000)
            r.ask(Q_AGENT)
            r.wait_idle()
            r.scroll_top()
        r.scene("agent_on", clips, agent_on)

        def agent_steps():
            exp = page.locator('[data-testid="stExpander"] summary', has_text="Agent steps").first
            exp.scroll_into_view_if_needed()
            r.highlight(exp, 500)
            exp.click()
            page.wait_for_timeout(800)
            r.scroll(350, steps=5, pause=400)
        r.scene("agent_steps", clips, agent_steps)

        def agent_memory():
            r.scroll_top()
            r.ask(Q_AGENT_FU)
            r.wait_idle()
            exp = page.locator('[data-testid="stExpander"] summary', has_text="Agent steps").first
            exp.scroll_into_view_if_needed()
            exp.click()
            page.wait_for_timeout(600)
            r.highlight(page.get_by_text("Question rewritten from the conversation", exact=False), 1200)
        r.scene("agent_memory", clips, agent_memory)

        def trace():
            exp = page.locator('[data-testid="stExpander"] summary', has_text="Trace (OpenTelemetry)").first
            exp.scroll_into_view_if_needed()
            r.highlight(exp, 500)
            exp.click()
            page.wait_for_timeout(800)
            r.scroll(300, steps=4, pause=400)
            r.scroll_top()
            tog = page.get_by_text("Agent mode (plan", exact=False)
            tog.click()                         # back to the default pipeline
            r.wait_idle(10_000)
        r.scene("trace", clips, trace)

        def ask_abstain():
            r.scroll_top()
            r.ask(Q_ABSTAIN)
            r.wait_idle()
            r.scroll_top()
        r.scene("ask_abstain", clips, ask_abstain)
        r.scene("abstain", clips, lambda: r.scroll(380, steps=5, pause=350))

        def graph():
            r.scroll_top()
            tab = page.get_by_role("tab", name="Citation neighbours")
            r.highlight(tab, 400)
            tab.click()
            page.wait_for_timeout(2500)
            r.scroll(300, steps=4, pause=400)
        r.scene("graph", clips, graph)

        def evaluation():
            r.scroll_top()
            tab = page.get_by_role("tab", name="Evaluation")
            r.highlight(tab, 400)
            tab.click()
            page.wait_for_timeout(1500)
            r.scroll(700, steps=10, pause=500)
        r.scene("eval", clips, evaluation)
        r.scene("eval2", clips, lambda: r.scroll(900, steps=12, pause=600))

        page.set_content(card("THANK YOU", "AI ResearchGraph",
                              "Cited claims · abstention · follow-up questions · agent mode with memory · "
                              "tracing · stage-by-stage evaluation"))
        r.scene("outro", clips, min_after=1.5)
        total = r.now()
        video_path = page.video.path()
        ctx.close()
        browser.close()

    # audio track: each clip delayed to its scene start, mixed, padded to the video length
    marks = r.marks
    inputs, filters = [], []
    for i, m in enumerate(marks):
        inputs += ["-i", clips[m["key"]]["path"]]
        ms = int(m["start"] * 1000)
        filters.append(f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(marks)))
    filters.append(f"{mix}amix=inputs={len(marks)}:normalize=0,apad[aout]")
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", video_path, *inputs, "-filter_complex", ";".join(filters),
           "-map", "0:v", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.2f}", "-movflags", "+faststart", args.out]
    subprocess.run(cmd, check=True)
    with open(args.out.replace(".mp4", "_narration.txt"), "w") as f:
        for m in marks:
            f.write(f"[{m['start']:6.1f}s] {clips[m['key']]['text']}\n\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Wrote {os.path.relpath(args.out, ROOT)} ({total:.0f}s)")


if __name__ == "__main__":
    sys.exit(main())
