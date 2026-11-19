"""Clients for arXiv (metadata, license, PDF) and OpenAlex (citation lists).

Only documented access routes are used, with polite pacing:
  - arXiv API + OAI-PMH + PDF: at most one request every 3 seconds (arXiv's guidance)
  - OpenAlex: small, sequential requests
No personal contact details are sent; requests carry a generic User-Agent.
"""
from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import json
import xml.etree.ElementTree as ET

UA = "researchgraph-capstone/0.1 (academic project)"
ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
OAI = {"o": "http://www.openarchives.org/OAI/2.0/", "ax": "http://arxiv.org/OAI/arXiv/"}

_last_arxiv_call = [0.0]


def _get(url: str, timeout: int = 60, retries: int = 6, binary: bool = False):
    delay = 5
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
                return data if binary else data.decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
                print(f"    HTTP {e.code} from {urllib.parse.urlparse(url).netloc}; waiting {min(wait, 120)}s")
                time.sleep(min(wait, 120))
                delay = min(delay * 2, 120)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise


def _arxiv_pace():
    wait = 3.0 - (time.time() - _last_arxiv_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call[0] = time.time()


def arxiv_metadata(ids: list[str]) -> dict[str, dict]:
    """Metadata for up to ~100 arXiv IDs in one API call, keyed by ID without version."""
    _arxiv_pace()
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"id_list": ",".join(ids), "max_results": len(ids)})
    root = ET.fromstring(_get(url, retries=1))     # fail fast: ingest falls back to OAI-PMH
    out = {}
    for e in root.findall("a:entry", ATOM):
        abs_url = e.findtext("a:id", "", ATOM).strip()            # http://arxiv.org/abs/2310.11511v1
        full_id = abs_url.rsplit("/abs/", 1)[-1]
        base, _, ver = full_id.rpartition("v")
        if not base or not ver.isdigit():
            base, ver = full_id, ""
        title = " ".join(e.findtext("a:title", "", ATOM).split())
        if not title or title.lower() == "error":
            continue
        out[base] = {
            "paper_id": base,
            "version": f"v{ver}" if ver else None,
            "title": title,
            "abstract": " ".join(e.findtext("a:summary", "", ATOM).split()),
            "authors": [a.findtext("a:name", "", ATOM) for a in e.findall("a:author", ATOM)],
            "published": e.findtext("a:published", "", ATOM)[:10],
            "updated": e.findtext("a:updated", "", ATOM)[:10],
            "categories": [c.get("term") for c in e.findall("a:category", ATOM)],
            "doi": e.findtext("arxiv:doi", None, ATOM),
            "url": f"https://arxiv.org/abs/{base}",
        }
        out[base]["year"] = int(out[base]["published"][:4])
    return out


def arxiv_license(arxiv_id: str) -> str | None:
    """License URL from arXiv's OAI-PMH record, or None if the author chose none."""
    _arxiv_pace()
    url = "https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(
        {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXiv"})
    root = ET.fromstring(_get(url))
    lic = root.find(".//ax:license", OAI)
    return lic.text.strip() if lic is not None and lic.text else None


RAW = {"o": "http://www.openarchives.org/OAI/2.0/", "r": "http://arxiv.org/OAI/arXivRaw/"}


def arxiv_record_oai(arxiv_id: str) -> dict | None:
    """Metadata + license for one paper from arXiv's OAI-PMH (arXivRaw format), in one request.

    Used instead of the arXiv search API when that API is rate-limiting (HTTP 429).
    Returns the same fields as arxiv_metadata() plus "license"."""
    import email.utils
    _arxiv_pace()
    url = "https://oaipmh.arxiv.org/oai?" + urllib.parse.urlencode(
        {"verb": "GetRecord", "identifier": f"oai:arXiv.org:{arxiv_id}", "metadataPrefix": "arXivRaw"})
    root = ET.fromstring(_get(url))
    rec = root.find(".//r:arXivRaw", RAW)
    if rec is None:
        return None
    text = lambda tag: " ".join((rec.findtext(f"r:{tag}", "", RAW) or "").split())
    versions = rec.findall("r:version", RAW)
    dates = [email.utils.parsedate_to_datetime(v.findtext("r:date", "", RAW)).date().isoformat() for v in versions]
    published = dates[0] if dates else ""
    authors = [a.strip() for a in re.split(r",\s*|\s+and\s+", text("authors")) if a.strip()]
    return {"paper_id": arxiv_id, "version": versions[-1].get("version") if versions else None,
            "title": text("title"), "abstract": text("abstract"), "authors": authors,
            "published": published, "updated": dates[-1] if dates else "", "year": int(published[:4]) if published else 0,
            "categories": text("categories").split(), "doi": text("doi") or None,
            "url": f"https://arxiv.org/abs/{arxiv_id}", "license": text("license") or None}


def rights_status(license_url: str | None) -> str:
    """Classify what we may do with the full text.

    CC licenses allow reuse with attribution. arXiv's default non-exclusive licence lets arXiv
    distribute the paper, not us: we index it locally for research and never redistribute it.
    """
    if not license_url:
        return "unknown_quarantine"
    if "creativecommons.org" in license_url:
        return "open_license"
    if "arxiv.org/licenses/nonexclusive-distrib" in license_url:
        return "local_index_only"
    return "unknown_quarantine"


def arxiv_pdf(arxiv_id_with_version: str) -> bytes:
    _arxiv_pace()
    data = _get(f"https://export.arxiv.org/pdf/{arxiv_id_with_version}", timeout=120, binary=True)
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    return data


def openalex_work(arxiv_id: str) -> dict | None:
    """OpenAlex record for the arXiv DOI: its ID and referenced work IDs."""
    url = (f"https://api.openalex.org/works/doi:10.48550/arXiv.{arxiv_id}"
           "?select=id,title,referenced_works,cited_by_count")
    try:
        time.sleep(0.2)
        return json.loads(_get(url, timeout=30))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
