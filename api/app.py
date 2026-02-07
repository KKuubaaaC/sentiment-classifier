"""
FastAPI: endpointy do przewidywania oceny (1-5) dla tekstu opinii.
Uruchomienie (z katalogu głównego projektu):
  uvicorn api.app:app --reload
"""
import sys
from pathlib import Path

# Katalog główny projektu (nad api/)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.predict import load_pipeline, predict


class PredictRequest(BaseModel):
    text: str = Field(..., description="Tekst opinii do oceny (1-5)")


class PredictResponse(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Przewidywana ocena 1-5")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ładuje pipeline przy starcie aplikacji."""
    app.state.preprocessor, app.state.encoder, app.state.classifier = load_pipeline()
    yield
    # ewentualne zwolnienie zasobów przy shutdown


app = FastAPI(
    title="Sentiment Classifier (Allegro Reviews)",
    description="API do klasyfikacji sentymentu opinii: tekst → ocena 1-5.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Sprawdzenie, czy API działa i model jest załadowany."""
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index():
    """Strona z formularzem: wklej opinię → wyświetl ocenę 1-5."""
    return """
<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sentiment – ocena opinii</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; max-width: 560px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.25rem; margin-bottom: 0.5rem; }
    p { color: #555; font-size: 0.9rem; margin-bottom: 1rem; }
    label { display: block; font-weight: 600; margin-bottom: 0.25rem; }
    textarea { width: 100%; min-height: 120px; padding: 0.5rem; border: 1px solid #ccc; border-radius: 6px; font-size: 1rem; resize: vertical; }
    button { margin-top: 0.75rem; padding: 0.5rem 1.25rem; background: #2563eb; color: white; border: none; border-radius: 6px; font-size: 1rem; cursor: pointer; }
    button:hover { background: #1d4ed8; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #wynik { margin-top: 1.25rem; padding: 1rem; border-radius: 8px; background: #f0f9ff; border: 1px solid #bae6fd; display: none; }
    #wynik.ok { display: block; }
    #wynik .ocena { font-size: 1.5rem; font-weight: 700; color: #0369a1; }
    #wynik .gwiazdki { color: #f59e0b; font-size: 1.5rem; letter-spacing: 0.1em; margin-top: 0.25rem; }
    #wynik .blad { color: #b91c1c; }
  </style>
</head>
<body>
  <h1>Klasyfikacja sentymentu (Allegro Reviews)</h1>
  <p>Wklej tekst opinii – model zwróci ocenę w skali 1–5 gwiazdek.</p>
  <form id="form">
    <label for="text">Opis / opinia</label>
    <textarea id="text" name="text" placeholder="Wklej tutaj tekst opinii..."></textarea>
    <button type="submit" id="btn">Sprawdź ocenę</button>
  </form>
  <div id="wynik">
    <span class="ocena">Ocena: <span id="rating">–</span>/5</span>
    <div class="gwiazdki" id="stars"></div>
    <p id="err" class="blad" style="display:none;"></p>
  </div>
  <script>
    const form = document.getElementById("form");
    const textarea = document.getElementById("text");
    const btn = document.getElementById("btn");
    const wynik = document.getElementById("wynik");
    const ratingEl = document.getElementById("rating");
    const starsEl = document.getElementById("stars");
    const errEl = document.getElementById("err");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const text = (textarea.value || "").trim();
      if (!text) { errEl.style.display = "block"; errEl.textContent = "Wpisz lub wklej opinię."; wynik.classList.add("ok"); return; }
      errEl.style.display = "none";
      btn.disabled = true;
      try {
        const r = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
        const data = await r.json();
        if (!r.ok) { errEl.style.display = "block"; errEl.textContent = data.detail || "Błąd"; wynik.classList.add("ok"); return; }
        const rating = data.rating;
        ratingEl.textContent = rating;
        starsEl.textContent = "★".repeat(rating) + "☆".repeat(5 - rating);
        wynik.classList.add("ok");
      } catch (err) {
        errEl.style.display = "block"; errEl.textContent = "Błąd połączenia: " + err.message; wynik.classList.add("ok");
      }
      btn.disabled = false;
    });
  </script>
</body>
</html>
"""


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    """Przewiduje ocenę 1-5 dla podanego tekstu opinii."""
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Pole 'text' nie może być puste.")

    preprocessor = app.state.preprocessor
    encoder = app.state.encoder
    classifier = app.state.classifier
    rating = predict(text, preprocessor=preprocessor, encoder=encoder, classifier=classifier)

    return PredictResponse(rating=rating)
