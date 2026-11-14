"""Step 1 smoke test: confirm both capstone endpoints answer. Reads .env; prints no secrets."""
import json, os, ssl, sys, time, urllib.request, urllib.error

def load_env(path=".env"):
    if not os.path.exists(path):
        sys.exit("No .env found — copy .env.example to .env and fill it in.")
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

def call(base, key, cert, route, body=None):
    if not os.path.exists(cert):
        return None, f"certificate not found at {cert}"
    ctx = ssl.create_default_context(cafile=cert)
    req = urllib.request.Request(base.rstrip("/") + route, method="POST" if body else "GET",
                                 data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
            return json.load(r), f"HTTP {r.status} in {time.time()-t:.1f}s"
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

load_env()
E = os.environ
ok = True

print("1) Chat model — list models")
data, msg = call(E["LLM_BASE_URL"], E["LLM_API_KEY"], E["LLM_CA_CERT"], "/models")
print("  ", msg, "|", [m["id"] for m in data["data"]] if data else "")
ok &= data is not None

print("2) Chat model — one completion")
data, msg = call(E["LLM_BASE_URL"], E["LLM_API_KEY"], E["LLM_CA_CERT"], "/chat/completions",
                 {"model": E["LLM_MODEL"], "max_tokens": 200,
                  "messages": [{"role": "user", "content": "Explain RAG in three short steps."}]})
print("  ", msg)
if data:
    print("   reply:", data["choices"][0]["message"]["content"][:500].replace("\n", "\n          "))
    print("   tokens:", data.get("usage"))
ok &= data is not None

if E.get("EMBED_API_KEY", "").startswith("YOUR_"):
    print("3) Embeddings — skipped (no key in .env yet)")
else:
    print("3) Embeddings — embed two texts")
    data, msg = call(E["EMBED_BASE_URL"], E["EMBED_API_KEY"], E["EMBED_CA_CERT"], "/embeddings",
                     {"model": E["EMBED_MODEL"], "input": ["Example document chunk", "Example search query"]})
    print("  ", msg, "|", f"{len(data['data'])} vectors × {len(data['data'][0]['embedding'])} dims" if data else "")
    ok &= data is not None

print("\nALL GOOD ✅" if ok else "\nSomething failed — see above.")
