FROM python:3.13-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libxml2-dev libxslt1-dev && rm -rf /var/lib/apt/lists/*
# This image runs the ingestion pipeline, so it needs the dev requirements, not only the backend runtime
COPY requirements.txt requirements-dev.txt requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN useradd --create-home --shell /bin/false appuser
USER appuser
RUN python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True); from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='BAAI/bge-large-en-v1.5')"
COPY --chown=appuser:appuser . .
CMD ["python", "app/ingestion/download_filings.py"]
