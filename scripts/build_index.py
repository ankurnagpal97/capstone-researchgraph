"""Step 3: embed all chunks (cached) and save the vector index. BM25 is built at load time.

Usage: .venv/bin/python scripts/build_index.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from researchgraph import index  # noqa: E402

if __name__ == "__main__":
    index.build()
