"""Serving guards: per-user and global rate limits, an answer cache, and PII scrubbing.

Rate limits protect the shared Track B key (one active request per group): each browser session may ask
SESSION_LIMITS questions, and the whole app GLOBAL_LIMITS, in sliding windows. The API client separately
enforces the provider's 60 requests/min and sends chat calls one at a time.

The answer cache (SQLite) returns a stored answer for the same normalised question + settings for
CACHE_TTL_S, so repeated questions (demo examples, follow-up clicks) cost no LLM call.

PII: questions are scrubbed before they reach the model, the logs, the traces or the cache. Detected
emails, phone numbers, card numbers (Luhn-checked), Aadhaar and PAN numbers, IP addresses and
credentials-like strings are replaced by [EMAIL], [PHONE] ... The paper corpus is public academic text and
is not scrubbed (author names and affiliations are part of the citation record).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque

from . import db

SESSION_LIMITS = ((5, 60), (40, 3600))       # (max questions, window seconds) per browser session
GLOBAL_LIMITS = ((12, 60),)                   # whole app, all sessions together
CACHE_TTL_S = 7 * 24 * 3600


# -- rate limiting -------------------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window limits per key. check() records the request only when it is allowed."""

    def __init__(self, limits, clock=time.monotonic):
        self.limits, self.clock = limits, clock
        self.hist: dict[str, deque] = {}
        self.lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, float]:
        """(allowed, seconds to wait if not)."""
        with self.lock:
            now = self.clock()
            h = self.hist.setdefault(key, deque())
            longest = max(w for _, w in self.limits)
            while h and now - h[0] >= longest:
                h.popleft()
            wait = 0.0
            for n, w in self.limits:
                recent = [t for t in h if now - t < w]
                if len(recent) >= n:
                    wait = max(wait, w - (now - recent[-n]))
            if wait > 0:
                return False, round(wait, 1)
            h.append(now)
            return True, 0.0


class AppLimiter:
    def __init__(self, clock=time.monotonic):
        self.session = RateLimiter(SESSION_LIMITS, clock)
        self.glob = RateLimiter(GLOBAL_LIMITS, clock)

    def check(self, session_id: str) -> tuple[bool, float, str]:
        ok, wait = self.session.check(session_id)
        if not ok:
            return False, wait, "you are asking faster than this demo allows"
        ok, wait = self.glob.check("*")
        if not ok:
            return False, wait, "the shared model key is busy with other users"
        return True, 0.0, ""


# -- answer cache --------------------------------------------------------------------------------------
CACHE_SCHEMA = """CREATE TABLE IF NOT EXISTS answer_cache (
    key TEXT PRIMARY KEY, created REAL, question TEXT, settings TEXT, answer_json TEXT, hits INTEGER DEFAULT 0)"""


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", q.lower())).strip()


def cache_key(question: str, **settings) -> str:
    return hashlib.sha256((_norm_q(question) + "|" + json.dumps(settings, sort_keys=True)).encode()).hexdigest()


def cache_get(key: str, ttl: float = CACHE_TTL_S) -> dict | None:
    con = db.connect()
    con.execute(CACHE_SCHEMA)
    row = con.execute("SELECT created, answer_json FROM answer_cache WHERE key = ?", (key,)).fetchone()
    if not row or time.time() - row[0] > ttl:
        return None
    con.execute("UPDATE answer_cache SET hits = hits + 1 WHERE key = ?", (key,))
    con.commit()
    return json.loads(row[1])


def cache_put(key: str, question: str, settings: dict, ans: dict) -> None:
    if ans.get("error"):            # never cache failures
        return
    con = db.connect()
    con.execute(CACHE_SCHEMA)
    db.upsert(con, "answer_cache", {"key": key, "created": time.time(), "question": question, "settings": settings,
                                    "answer_json": json.dumps(ans, default=str), "hits": 0})
    con.commit()


# -- PII -----------------------------------------------------------------------------------------------
def _luhn(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    return (sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])) % 10 == 0


NB = r"(?<![\w.+])(?<!\d[ -])"            # not glued to a longer number / word
NA = r"(?![ -]?\d)(?![\w.]\w)"
PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("SECRET", re.compile(r"\b(?:sk|pk|api|key|token)[-_][A-Za-z0-9_-]{12,}\b", re.I)),
    ("CARD", re.compile(NB + r"\d(?:[ -]?\d){12,18}" + NA)),
    ("AADHAAR", re.compile(NB + r"[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}" + NA)),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(NB + r"(?:\+\d{1,3}[ -]?)?\(?\d{2,5}\)?(?:[ -]?\d{2,5}){1,3}" + NA)),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]
ARXIV_LIKE = re.compile(r"^\d{4}\.\d{4,5}$")       # 2310.11511 is a paper ID, not a phone number


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Replace personal data with [TYPE] placeholders. Returns (clean text, sorted list of types found)."""
    found = set()
    for name, pat in PII_PATTERNS:
        def sub(m, name=name):
            s = m.group(0)
            digits = re.sub(r"\D", "", s)
            if name == "CARD" and not _luhn(digits):
                return s
            if name == "PHONE" and (ARXIV_LIKE.match(s.strip()) or not 10 <= len(digits) <= 13):
                return s
            if name == "IP" and not all(0 <= int(p) <= 255 for p in s.split(".")):
                return s
            found.add(name)
            return f"[{name}]"
        text = pat.sub(sub, text)
    return text, sorted(found)
