"""Shared client for the Track B chat model (Qwen3-8B) and the embeddings service.

Standard library only. Handles the service limits from the Track B guide:
  - 1 active chat request per group key  -> a process-wide lock sends chat calls one at a time
  - 60 requests/min per key               -> a sliding-window rate limiter
  - 429 / 503 are temporary               -> retry after 2, 4, 8 s, then give up
  - max_tokens must be 1..1024            -> checked before sending
  - embeddings take <= 16 texts           -> embed() splits into batches of 16
Every call is recorded in CALL_LOG (route, status, seconds, tokens) for later analysis.
"""
import http.client, json, os, ssl, threading, time, urllib.error, urllib.request
from collections import deque

RETRY_STATUSES = {429, 503}
RETRY_DELAYS = (2, 4, 8)          # seconds; 3 retries then give up
MAX_OUTPUT_TOKENS = 1024
CONTEXT_LIMIT = 8192
EMBED_BATCH = 16
RATE_PER_MIN = 60

CALL_LOG = []                     # one dict per HTTP attempt


class _NullLock:
    def __enter__(self): return self
    def __exit__(self, *a): return False

_NO_LOCK = _NullLock()


class ApiError(Exception):
    def __init__(self, status, body, route):
        self.status, self.body, self.route = status, body, route
        super().__init__(f"HTTP {status} on {route}: {body[:200]}")


def load_env(path=None):
    """Load KEY=VALUE lines from .env into os.environ (existing vars win)."""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        if os.environ.get("LLM_API_KEY"):      # e.g. Docker env_file: variables already set
            return
        raise FileNotFoundError("No .env found — copy .env.example to .env and fill it in.")
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class RateLimiter:
    """Sliding window: at most `limit` calls in any `window` seconds."""
    def __init__(self, limit=RATE_PER_MIN, window=60.0, clock=time.monotonic, sleep=time.sleep):
        self.limit, self.window, self.clock, self.sleep = limit, window, clock, sleep
        self.calls = deque()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.window:
                self.calls.popleft()
            if len(self.calls) >= self.limit:
                pause = self.window - (now - self.calls[0])
                self.sleep(pause)
                now = self.clock()
                self.calls.popleft()
            self.calls.append(now)


class Endpoint:
    """One HTTPS service (base URL + key + CA cert) with retries and logging."""
    def __init__(self, base_url, api_key, ca_cert, model, name, serialize=False,
                 retry_delays=RETRY_DELAYS, sleep=time.sleep, timeout=120):
        if not os.path.exists(ca_cert):
            raise FileNotFoundError(f"CA certificate not found: {ca_cert}")
        self.base_url, self.api_key, self.model, self.name = base_url.rstrip("/"), api_key, model, name
        self.ctx = ssl.create_default_context(cafile=ca_cert)
        self.retry_delays, self.sleep, self.timeout = retry_delays, sleep, timeout
        self.lock = threading.Lock() if serialize else None
        self.limiter = RateLimiter()

    # -- low level -------------------------------------------------------
    def _open(self, route, body=None, stream=False):
        req = urllib.request.Request(
            self.base_url + route, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        return urllib.request.urlopen(req, context=self.ctx, timeout=self.timeout)

    def request(self, route, body=None, retry=True):
        """Send one request; retry 429/503 with backoff. Returns parsed JSON."""
        delays = list(self.retry_delays) if retry else []
        while True:
            self.limiter.wait()
            entry = {"service": self.name, "route": route}
            try:
                with (self.lock or _NO_LOCK):
                    t = time.time()           # time the request itself, not the wait for the lock
                    entry["time"] = t
                    with self._open(route, body) as r:
                        data = json.load(r)
                entry.update(status=200, seconds=round(time.time() - t, 2), usage=data.get("usage"))
                CALL_LOG.append(entry)
                return data
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")
                entry.update(status=e.code, seconds=round(time.time() - t, 2), error=text[:200])
                CALL_LOG.append(entry)
                if e.code in RETRY_STATUSES and delays:
                    self.sleep(delays.pop(0))
                    continue
                raise ApiError(e.code, text, route) from None
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                # dropped connection, reset, timeout: transient, retry like 429/503.
                # A certificate failure is a configuration error: never retried.
                reason = getattr(e, "reason", e)
                entry.update(status="network", error=f"{type(reason).__name__}: {reason}"[:200])
                CALL_LOG.append(entry)
                if isinstance(reason, ssl.SSLCertVerificationError) or not delays:
                    raise ApiError(0, f"network error: {type(reason).__name__}: {reason}", route) from None
                self.sleep(delays.pop(0))


class Client:
    """High-level API: chat(), stream_chat(), embed(), models()."""
    def __init__(self, env_path=None, sleep=time.sleep):
        load_env(env_path)
        E = os.environ
        root = os.path.dirname(os.path.abspath(env_path or __file__))
        cert = lambda p: p if os.path.isabs(p) else os.path.join(root, p)
        self.llm = Endpoint(E["LLM_BASE_URL"], E["LLM_API_KEY"], cert(E["LLM_CA_CERT"]),
                            E["LLM_MODEL"], "llm", serialize=True, sleep=sleep)
        self.emb = Endpoint(E["EMBED_BASE_URL"], E["EMBED_API_KEY"], cert(E["EMBED_CA_CERT"]),
                            E["EMBED_MODEL"], "embed", sleep=sleep)

    def models(self):
        return [m["id"] for m in self.llm.request("/models")["data"]]

    def chat(self, messages, max_tokens=512, thinking=False, retry=True, **extra):
        """Return the full response JSON. `thinking=False` turns off Qwen3's <think> output."""
        if not 1 <= max_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_tokens must be 1..{MAX_OUTPUT_TOKENS}, got {max_tokens}")
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.llm.model, "messages": messages, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
        return self.llm.request("/chat/completions", body, retry=retry)

    def ask(self, prompt, **kw):
        """Convenience: just the answer text."""
        return self.chat(prompt, **kw)["choices"][0]["message"]["content"]

    def stream_chat(self, messages, max_tokens=512, thinking=False, **extra):
        """Yield text pieces as they arrive (server-sent events). No automatic retry."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        body = {"model": self.llm.model, "messages": messages, "max_tokens": max_tokens, "stream": True,
                "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
        self.llm.limiter.wait()
        with self.llm.lock:
            with self.llm._open("/chat/completions", body) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]

    def embed(self, texts):
        """Embed any number of texts, 16 per request. Returns a list of vectors in input order."""
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for i in range(0, len(texts), EMBED_BATCH):
            data = self.emb.request("/embeddings", {"model": self.emb.model, "input": texts[i:i + EMBED_BATCH]})
            vectors += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return vectors
