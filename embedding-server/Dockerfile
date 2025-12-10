# DriftOS Embedding Server - Slim build
FROM python:3.11-slim

WORKDIR /app

# Install only essential system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than default CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN pip install --no-cache-dir \
    sentence-transformers \
    transformers \
    fastapi \
    uvicorn \
    pydantic \
    spacy

# Download smallest spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY server.py .
COPY preprocessing.py .
COPY nlp_analysis.py .

# Non-root user
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Railway sets PORT dynamically
ENV PORT=8100
EXPOSE 8100

# Health check - use 127.0.0.1 explicitly for IPv4 since we're on dual-stack
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; import os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 8100)}/health')"

# Use shell to expand $PORT
# Listen on :: (IPv6 all interfaces) which also accepts IPv4 connections on dual-stack systems
CMD ["sh", "-c", "uvicorn server:app --host :: --port ${PORT:-8100}"]
