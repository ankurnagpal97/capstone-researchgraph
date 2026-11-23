# AI ResearchGraph — Streamlit app in a container.
# Build:  docker build -t researchgraph .
# Run:    docker compose up        (see docker-compose.yml; mounts data/, secrets/ and .env at runtime)
# Nothing secret or licence-restricted is baked into the image: API keys, CA certificates, the paper
# database and the indexes are mounted when the container starts.
FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel.
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY requirements.txt .
RUN grep -vE "^(torch|playwright|imageio-ffmpeg|edge-tts|ipykernel|pytest)==" requirements.txt > req-app.txt \
    && pip install -r req-app.txt

# Bake the public cross-encoder (~90 MB) into the image so the first question is not slow.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY api_client.py app.py ./
COPY researchgraph/ researchgraph/
COPY reports/*.md reports/*.json reports/

RUN useradd --create-home app && mkdir -p /app/data /app/logs /app/secrets && chown -R app /app /opt/hf
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
