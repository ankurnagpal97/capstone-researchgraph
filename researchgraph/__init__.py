"""AI ResearchGraph: evidence-grounded RAG over research papers on RAG and LLM retrieval."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PDF_DIR = os.path.join(DATA, "raw", "pdfs")
INDEX_DIR = os.path.join(DATA, "index")
DB_PATH = os.path.join(DATA, "researchgraph.db")
LOG_DIR = os.path.join(ROOT, "logs")
