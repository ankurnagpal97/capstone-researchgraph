"""PDF -> sections -> chunks, keeping page numbers and section headings.

Heuristics tuned for arXiv LaTeX papers (one or two columns):
  - a short text block that looks like "3.2 Method" / "Abstract" / "A Appendix" starts a section
  - the "References" section is kept apart: used only for citation matching, never as evidence
  - the arXiv side-stamp ("arXiv:2310.11511v1 [cs.CL] ...") and bare page numbers are dropped
Chunks are built from whole paragraphs inside one section, about 250-350 estimated tokens,
with ~50 tokens of overlap, and a hard character cap so they stay under the 512-token
embedding limit (bge-small, WordPiece).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

PARSER_VERSION = "pdf-blocks-1"

TARGET_TOKENS = 300      # aim for this many tokens per chunk
OVERLAP_TOKENS = 50
MAX_CHARS = 1500         # hard cap on chunk text, ~400 WordPiece tokens even for dense text
MIN_CHARS = 120          # drop scraps (figure labels, stray numbers)

KNOWN = r"(abstract|introduction|background|related work|preliminaries|method|methods|methodology|approach|" \
        r"experiments?|experimental setup|results|evaluation|analysis|discussion|conclusions?|limitations?|" \
        r"ethics statement|broader impact|acknowledg(e)?ments?|references|bibliography|appendix|appendices)"
NUMBERED = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,2}|[A-H](\.\d{1,2}){0,2})\.?\s+([A-Z][^\n]{1,80})$")
PLAIN = re.compile(rf"^({KNOWN})\s*$", re.I)
REFS = re.compile(r"^(\d{1,2}\.?\s+)?(references|bibliography)\s*$", re.I)
STAMP = re.compile(r"arXiv:\d{4}\.\d{4,5}(v\d+)?\s*\[[^\]]+\](\s*\d{1,2}\s+[A-Z][a-z]{2}\s+\d{4})?")
DATE = re.compile(r"^\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}$", re.I)


def est_tokens(text: str) -> int:
    """Conservative WordPiece token estimate (no tokenizer available): ~3.5 chars per token."""
    return max(len(text.split()), int(len(text) / 3.5))


@dataclass
class Block:
    text: str
    page: int


@dataclass
class Section:
    heading: str
    blocks: list = field(default_factory=list)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)          # ligatures: "ﬁ" -> "fi", full-width digits, etc.
    text = STAMP.sub("", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)        # de-hyphenate line breaks
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _is_heading(text: str) -> bool:
    if len(text) > 90 or len(text.split()) > 12 or text.endswith((".", ",", ";", ":")) and not PLAIN.match(text):
        return False
    if PLAIN.match(text) or REFS.match(text):
        return True
    m = NUMBERED.match(text)
    if not m or DATE.match(text):
        return False
    if re.search(r"\s\d{1,3}$", text):      # table-of-contents line: "A Experimental Details 17"
        return False
    words = m.group(4).split()
    # reject things like "1 We show that" (sentence) or "2 4 6 8" (axis ticks)
    return bool(words) and not any(ch.isdigit() for ch in m.group(4)[:3]) and words[0][0].isupper()


def extract_blocks(pdf_path: str) -> tuple[list[Block], int]:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    blocks = []
    for pno, page in enumerate(doc, start=1):
        for b in page.get_text("blocks"):
            if b[6] != 0:          # 0 = text block, 1 = image
                continue
            text = _clean(b[4])
            if not text or re.fullmatch(r"\d{1,3}", text) or DATE.match(text):
                continue
            blocks.append(Block(text, pno))
    return blocks, len(doc)


def split_sections(blocks: list[Block]) -> tuple[list[Section], str]:
    """Return (body sections, reference-list text)."""
    sections = [Section("Front matter")]
    refs, in_refs = [], False
    for b in blocks:
        if _is_heading(b.text):
            if REFS.match(b.text):
                in_refs = True
                continue
            # an appendix heading after the reference list ends it
            if in_refs and not re.match(r"^([A-H](\.\d+)*\.?\s|appendi)", b.text, re.I):
                refs.append(b.text)
                continue
            in_refs = False
            sections.append(Section(b.text))
            continue
        (refs if in_refs else sections[-1].blocks).append(b if not in_refs else b.text)
    return [s for s in sections if s.blocks], " ".join(refs)


def chunk_sections(paper_id: str, sections: list[Section]) -> list[dict]:
    chunks = []

    def emit(section, parts):
        text = " ".join(p.text for p in parts).strip()
        if len(text) < MIN_CHARS:
            return
        chunks.append({
            "paper_id": paper_id, "section": section.heading,
            "page_start": parts[0].page, "page_end": parts[-1].page, "text": text})

    for s in sections:
        # split over-long blocks at sentence boundaries so no single part exceeds the cap
        parts = []
        for b in s.blocks:
            if len(b.text) <= MAX_CHARS:
                parts.append(b)
                continue
            buf = ""
            for sent in re.split(r"(?<=[.!?])\s+", b.text):
                while len(sent) > MAX_CHARS:                      # no sentence breaks at all
                    if buf:
                        parts.append(Block(buf, b.page)); buf = ""
                    parts.append(Block(sent[:MAX_CHARS], b.page)); sent = sent[MAX_CHARS:]
                if len(buf) + len(sent) + 1 > MAX_CHARS:
                    parts.append(Block(buf, b.page)); buf = sent
                else:
                    buf = f"{buf} {sent}".strip()
            if buf:
                parts.append(Block(buf, b.page))

        cur, cur_len = [], 0
        for p in parts:
            p_tok = est_tokens(p.text)
            too_big = cur and (cur_len + p_tok > TARGET_TOKENS
                               or sum(len(c.text) + 1 for c in cur) + len(p.text) > MAX_CHARS)
            if too_big:
                emit(s, cur)
                # overlap: carry trailing sentences of the last part, up to OVERLAP_TOKENS
                tail, tail_tok = [], 0
                for sent in reversed(re.split(r"(?<=[.!?])\s+", cur[-1].text)):
                    t = est_tokens(sent)
                    if tail_tok + t > OVERLAP_TOKENS:
                        break
                    tail.insert(0, sent); tail_tok += t
                cur = [Block(" ".join(tail), cur[-1].page)] if tail else []
                cur_len = tail_tok
                if cur and sum(len(c.text) + 1 for c in cur) + len(p.text) > MAX_CHARS:
                    cur, cur_len = [], 0
            cur.append(p)
            cur_len += p_tok
        if cur:
            emit(s, cur)

    for i, c in enumerate(chunks):
        c["chunk_index"] = i
        c["chunk_id"] = f"{paper_id}#{i:04d}"
        c["n_words"] = len(c["text"].split())
        c["est_tokens"] = est_tokens(c["text"])
        c["text_sha256"] = hashlib.sha256(c["text"].encode()).hexdigest()
        c["parser_version"] = PARSER_VERSION
    return chunks


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def reference_matches(ref_text: str, corpus: dict[str, str], self_id: str) -> list[str]:
    """IDs of corpus papers cited in this reference list, matched by arXiv ID or by title.

    Titles are matched on their first 6+ normalised words so line breaks and hyphenation in the
    reference list don't matter. Short or generic titles (< 4 words) are matched on the full title.
    """
    norm_refs = _norm(re.sub(r"([a-z])- ([a-z])", r"\1\2", ref_text))   # "cri- tique" -> "critique"
    found = []
    for pid, title in corpus.items():
        if pid == self_id:
            continue
        if pid in ref_text:
            found.append(pid)
            continue
        words = _norm(title).split()
        key = " ".join(words[: max(6, min(len(words), 8))]) if len(words) >= 4 else " ".join(words)
        if len(key) >= 20 and key in norm_refs:
            found.append(pid)
    return found
