"""OpenTelemetry tracing for the pipeline and the agent.

Spans go to logs/traces.jsonl (one JSON object per finished span), so traces stay on this machine and
can be read back by the app (Trace panel) and by scripts. If OTEL_EXPORTER_OTLP_ENDPOINT is set and the
OTLP exporter package is installed, spans are also sent there (e.g. a local Jaeger or Langfuse).
Set RG_TRACING=0 to turn tracing off. Without the opentelemetry package, span() is a no-op.

Usage:
    with tracing.span("tool.search_papers", query=q) as sp:
        ...
        sp.set("n_hits", 6)
"""
from __future__ import annotations

import contextlib
import json
import os
import threading

from . import ROOT

TRACE_FILE = os.path.join(ROOT, "logs", "traces.jsonl")
ENABLED = os.environ.get("RG_TRACING", "1") != "0"

_tracer = None
_lock = threading.Lock()

try:
    from opentelemetry import trace as _ot
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    HAVE_OTEL = True
except ImportError:          # tracing is optional; the app works without it
    HAVE_OTEL = False


if HAVE_OTEL:
    class JsonlExporter(SpanExporter):
        def __init__(self, path: str = TRACE_FILE):
            self.path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        def export(self, spans):
            with _lock, open(self.path, "a") as f:
                for s in spans:
                    f.write(json.dumps({
                        "trace_id": f"{s.context.trace_id:032x}", "span_id": f"{s.context.span_id:016x}",
                        "parent_id": f"{s.parent.span_id:016x}" if s.parent else None, "name": s.name,
                        "start": s.start_time / 1e9, "duration_ms": round((s.end_time - s.start_time) / 1e6, 1),
                        "status": s.status.status_code.name, "attributes": dict(s.attributes or {})}) + "\n")
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass


def _get_tracer():
    global _tracer
    if _tracer is None and HAVE_OTEL and ENABLED:
        provider = TracerProvider(resource=Resource.create({"service.name": "researchgraph"}))
        provider.add_span_processor(SimpleSpanProcessor(JsonlExporter()))
        if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
            except ImportError:
                pass
        _tracer = provider.get_tracer("researchgraph")
    return _tracer


class _Span:
    def __init__(self, sp=None):
        self.sp = sp

    def set(self, key: str, value) -> None:
        if self.sp is not None:
            if not isinstance(value, (str, bool, int, float)):
                value = json.dumps(value, default=str)[:1000]
            self.sp.set_attribute(key, value)

    @property
    def trace_id(self) -> str | None:
        return f"{self.sp.get_span_context().trace_id:032x}" if self.sp is not None else None


@contextlib.contextmanager
def span(name: str, **attrs):
    tracer = _get_tracer()
    if tracer is None:
        yield _Span()
        return
    with tracer.start_as_current_span(name) as sp:
        w = _Span(sp)
        for k, v in attrs.items():
            w.set(k, v)
        try:
            yield w
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(_ot.Status(_ot.StatusCode.ERROR, str(e)[:200]))
            raise


def read_trace(trace_id: str, path: str = TRACE_FILE) -> list[dict]:
    """All spans of one trace, in start order (for the app's trace panel)."""
    if not trace_id or not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            if trace_id in line:
                s = json.loads(line)
                if s["trace_id"] == trace_id:
                    out.append(s)
    return sorted(out, key=lambda s: s["start"])
