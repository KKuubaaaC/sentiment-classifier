# Sentiment Classifier (Allegro Reviews)

Projekt klasyfikacji sentymentu opinii z Allegro (język polski): przypisuje ocenę w skali **1–5 gwiazdek** na podstawie tekstu. Pipeline: preprocessing → embeddingi (sentence-transformers) → klasyfikator (LR / SVM / RF / XGBoost). API w FastAPI z prostą stroną do wklejania opinii i uruchomieniem w Dockerze.

**Tech stack:** Python · Hugging Face (datasets) · sentence-transformers · scikit-learn · XGBoost · FastAPI · Docker · Jupyter

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation & Run](#installation--run)
- [Running with Docker](#running-with-docker)
- [Notebooks](#notebooks)
- [API](#api)
- [Source Code](#source-code)

---

## Project Overview

### Core Features

- **Model training**
  - Zbiór: **Allegro Reviews** (KLEJ) z Hugging Face – opinie po polsku z oceną 1–5.
  - **Preprocessing**: własny moduł (`PolishTextPreprocessor`) – czyszczenie tekstu, usuwanie URL/HTML/emoji, opcjonalnie stopwords.
  - **Embeddingi**: model **sentence-transformers** (paraphrase-multilingual-MiniLM-L12-v2); wektory zapisywane do pliku.
  - **Klasyfikatory**: Logistic Regression, SVC, Random Forest, XGBoost (z obsługą imbalance); wybór najlepszego po F1 macro.
  - **Metryki**: accuracy, F1 macro, **AR score (wMAE)** zgodnie z benchmarkiem KLEJ.

- **Backend**
  - Aplikacja **FastAPI** – endpointy REST do predykcji oraz strona z formularzem do wklejania opinii.
  - Ładowanie pipeline’u (preprocessor + encoder + klasyfikator) przy starcie serwera.

- **Inference**
  - Skrypt `src/predict.py` – przewidywanie z poziomu CLI.
  - Ewaluacja na zbiorze testowym w notatniku `05_evaluation.ipynb`.

### Deployment

- **Containerization**: backend API w **Dockerze** – obraz z Pythonem, zależnościami i kodem; gotowy do uruchomienia lokalnie lub na serwerze.
- Uruchomienie: `docker build` + `docker run` (szczegóły poniżej).

---

## Project Structure

```
sentiment-classifier-pl/
├── .gitignore
├── README.md
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI: /health, /predict, GET / (formularz)
├── data/
│   ├── raw/
│   └── processed/         # train/val CSV, embeddingi .npy
├── models/
│   └── best_classifier.pkl # Zapisany najlepszy klasyfikator
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_training.ipynb
│   └── 05_evaluation.ipynb
├── results/
│   ├── figures/
│   └── metrics.json        # Metryki na val; metrics_test.json na test
└── src/
    ├── __init__.py
    ├── preprocessing.py    # PolishTextPreprocessor
    └── predict.py          # Ładowanie pipeline’u i predict()
```

---

## Installation & Run

Wymagania: **Python 3.13+**, menedżer **uv** (lub pip).

```bash
git clone https://github.com/KKuubaaaC/sentiment-classifier-pl.git
cd sentiment-classifier-pl
uv sync
```

- **Notatniki**: uruchom Jupyter z katalogu projektu i otwórz `notebooks/01_EDA.ipynb` → … → `05_evaluation.ipynb` (kernel = `.venv`).
- **CLI (predykcja)**:
  ```bash
  uv run python -m src.predict "Tekst opinii..."
  ```
- **API lokalnie (bez Dockera)**:
  ```bash
  uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
  ```
  - Strona z formularzem: http://127.0.0.1:8000/
  - Dokumentacja API: http://127.0.0.1:8000/docs

---

## Running with Docker

Przed zbudowaniem obrazu upewnij się, że w katalogu `models/` znajduje się wytrenowany model:

- `models/best_classifier.pkl` (zapis z notatnika `04_training.ipynb`).

**Zbudowanie obrazu** (w katalogu głównym projektu):

```bash
docker build -t sentiment-classifier-pl .
```

**Uruchomienie kontenera**:

```bash
docker run -p 8000:8000 sentiment-classifier-pl
```

- Strona z formularzem (wklej opinię → wynik): http://127.0.0.1:8000/
- Health: http://127.0.0.1:8000/health
- Predykcja JSON: `POST http://127.0.0.1:8000/predict` z body `{"text": "..."}`.

Przy pierwszym uruchomieniu model sentence-transformers może zostać pobrany z sieci (kilkanaście sekund).

---

## Notebooks

| Notebook | Opis |
|----------|------|
| **01_EDA.ipynb** | Eksploracja danych: rozkład ocen i długości (train/val/test), jakość (puste, duplikaty), wnioski. |
| **02_preprocessing.ipynb** | Preprocessing: użycie `PolishTextPreprocessor`, porównanie przed/po, zapis do `data/processed/`. |
| **03_embeddings.ipynb** | Embeddingi: sentence-transformers, encoding train/val, zapis `.npy`. |
| **04_training.ipynb** | Trening: LR, SVC, RF, XGBoost; ewaluacja (accuracy, F1, AR score); zapis `best_classifier.pkl` i metryk. |
| **05_evaluation.ipynb** | Ewaluacja na zbiorze testowym; zapis `results/metrics_test.json`. |

---

## API

- **GET /** – strona HTML z formularzem: pole do wklejenia opisu/opini, przycisk „Sprawdź ocenę”, wyświetlenie oceny 1–5 i gwiazdek.
- **GET /health** – `{"status": "ok"}` (sprawdzenie, czy API i model są gotowe).
- **POST /predict** – body: `{"text": "Tekst opinii..."}` → odpowiedź: `{"rating": 1}` … `{"rating": 5}`.

---

## Source Code

- **src/preprocessing.py** – klasa `PolishTextPreprocessor`: `clean_text`, `tokenize`, `preprocess_pipeline`, `preprocess_series`.
- **src/predict.py** – `load_pipeline()`, `predict(text)` – używane przez API i CLI.
- **api/app.py** – FastAPI: lifespan (ładowanie modelu), endpointy `/`, `/health`, `/predict`.

---


