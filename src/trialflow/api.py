# src/trialflow/api.py
# загружаем .env (по умолчанию из корня проекта); можно переопределить через ENV_FILE
#load_dotenv(os.getenv("ENV_FILE", ".env"))


from fastapi import FastAPI, Depends, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Union
import time
import os

from .model_io import get_version
from .schemas import Visit, ScoreResponse
from .scoring import score_visits
from .policy import require_api_key

app = FastAPI(title="TrialFlow Demo")

# статика: src/trialflow/static/index.html
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")

@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/ping")
def ping():
    return {"status": "ok", "model_version": get_version()}

@app.post("/score", response_model=ScoreResponse, dependencies=[Depends(require_api_key)])
def score(payload: Union[Visit, List[Visit]], k: int = Query(0, ge=0, le=100), response: Response = None):
    results, meta = score_visits(payload, k_percent=k)
    if response is not None:
        response.headers["X-Model-Version"] = meta.get("model_version", "")
    return ScoreResponse(results=results, meta=meta, timestamp=int(time.time()), signature="trialflow-demo")