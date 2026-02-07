# Obraz do uruchomienia API (FastAPI). Wymaga models/best_classifier.pkl.
FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    sentence-transformers \
    scikit-learn \
    joblib

COPY api/ api/
COPY src/ src/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
