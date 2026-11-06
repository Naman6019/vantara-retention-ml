# Base image
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system build dependencies (libgomp1 for LightGBM/XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (keeps the image ~2 GB smaller than the default CUDA wheel)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# Remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application source, config and the artifacts needed for serving.
# The cleaned transactions are baked in so the scorer can build today's snapshot
# offline (see .dockerignore).
COPY config/ config/
COPY src/ src/
COPY api/ api/
COPY frontend/ frontend/
COPY scripts/ scripts/
COPY models_artifacts/ models_artifacts/
COPY data/interim/cleaned_transactions.parquet data/interim/cleaned_transactions.parquet
COPY pytest.ini .
COPY README.md .

EXPOSE 8000 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
