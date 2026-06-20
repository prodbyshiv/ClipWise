FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 clipwise
USER clipwise

WORKDIR /home/clipwise/app

# Copy requirements first (layer caching)
COPY --chown=clipwise requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# Copy application code
COPY --chown=clipwise . .

# ChromaDB persistence directory
RUN mkdir -p chroma_db

# HuggingFace Spaces uses port 7860
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/home/clipwise/app

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
