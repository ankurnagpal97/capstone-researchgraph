"""Test configuration: unit tests must not write their fake runs into the real trace log (logs/traces.jsonl)."""
import os

os.environ["RG_TRACING"] = "0"
