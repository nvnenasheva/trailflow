# src/trialflow/serve_baseline.py
from __future__ import annotations
import os, hashlib, hmac, json, time, logging
from typing import List, Optional, Union, Any

import joblib
import pandas as pd
from fastapi import FastAPI, Response, Query, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

from roi_endpoints import router as roi_router

logger = logging.getLogger("uvicorn.error")

# --- ENV настройки ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline_calibrated.pkl")
API_KEY = os.getenv("API_KEY")                  # .env
SIG_SECRET = os.getenv("SIG_SECRET", "devsig")  # .env
MAX_BATCH = int(os.getenv("MAX_BATCH", "5000"))
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*")  # можно указать домены через запятую

# ----- Pydantic схема (как в train) -----
class Visit(BaseModel):
    age: int = Field(..., ge=0, le=120)
    weekday: int = Field(..., ge=0, le=6)
    lead_time_days: int = Field(..., ge=0, le=365)
    is_first_visit: int = Field(..., ge=0, le=1)
    site_id: str
    visit_type: str = "visit"
    sms_received: int = Field(..., ge=0, le=1)
    scholarship: int = Field(..., ge=0, le=1)
    hipertension: int = Field(..., ge=0, le=1)
    diabetes: int = Field(..., ge=0, le=1)
    alcoholism: int = Field(..., ge=0, le=1)
    handicap: int = Field(..., ge=0, le=1)
    gender: str = Field(..., description="M/F/U")

    @validator("gender")
    def _norm_gender(cls, v: str) -> str:
        v = (v or "U").upper()[:1]
        return v if v in {"M", "F", "U"} else "U"

NUM_COLS = ["age", "weekday", "lead_time_days"]
CAT_COLS = [
    "is_first_visit", "site_id", "visit_type", "sms_received",
    "scholarship", "hipertension", "diabetes", "alcoholism", "handicap", "gender",
]
ALL_COLS = NUM_COLS + CAT_COLS

def _model_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def _to_dataframe(items: List[Visit]) -> pd.DataFrame:
    rows = [{k: getattr(it, k) for k in ALL_COLS} for it in items]
    df = pd.DataFrame(rows, columns=ALL_COLS)
    df["weekday"] = df["weekday"].astype(int).clip(0, 6)
    df["lead_time_days"] = df["lead_time_days"].astype(int).clip(lower=0)
    df["is_first_visit"] = df["is_first_visit"].astype(int).clip(0, 1)
    for b in ["sms_received","scholarship","hipertension","diabetes","alcoholism","handicap"]:
        df[b] = df[b].astype(int).clip(0, 1)
    df["age"] = df["age"].astype(int).clip(0, 120)
    df["gender"] = df["gender"].astype(str).str.upper().str[0]
    df["visit_type"] = df["visit_type"].astype(str)
    df["site_id"] = df["site_id"].astype(str)
    return df

def _sign_response(probas: list[float], model_ver: str, ts: int) -> str:
    """
    Короткая HMAC-подпись ответа (водяной знак) — доказывает, что результат пришёл с моего сервера.
    Не хранит приватные данные, но проверяется только тем, у кого есть SIG_SECRET.
    """
    payload = json.dumps({"p": [round(p, 7) for p in probas], "m": model_ver, "t": ts}, separators=(",", ":"))
    return hmac.new(SIG_SECRET.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()[:16]

# ----- App -----
#app = FastAPI(title="TrialFlow baseline scorer", version="0.3.0")
app = FastAPI(title="TrialFlow ROI API")
app.include_router(roi_router)


# CORS (чтобы страница из браузера могла стучаться к API)
origins = [o.strip() for o in ALLOW_ORIGINS.split(",")] if ALLOW_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# статика (красивый UI)
if not os.path.exists("static"):
    os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL = joblib.load(MODEL_PATH)
MODEL_VER = _model_sha256(MODEL_PATH)

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/")
def root():
    # отдаем главную страницу UI
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "UI not found. Put index.html into ./static/"}

@app.get("/ping")
def ping(response: Response):
    response.headers["X-Model-Version"] = MODEL_VER
    return {"status": "ok", "model_version": MODEL_VER}

@app.post("/score", dependencies=[Depends(require_api_key)])
def score(
    payload: Union[Visit, List[Visit]],
    response: Response,
    k: Optional[float] = Query(None, ge=0, le=100, description="Optional Top-K% within this batch"),
):
    items = payload if isinstance(payload, list) else [payload]
    if len(items) == 0:
        raise HTTPException(status_code=400, detail="Empty payload")
    if len(items) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"Batch too large: {len(items)} > {MAX_BATCH}")

    df = _to_dataframe(items)

    proba = MODEL.predict_proba(df)[:, 1].tolist()
    result = [{"proba_no_show": float(p)} for p in proba]

    meta = {"k_percent": None}
    if k is not None and len(result) > 0:
        m = max(1, int(len(result) * (k / 100.0)))
        order = sorted(range(len(result)), key=lambda i: -result[i]["proba_no_show"])
        thr = result[order[m - 1]]["proba_no_show"]
        top_idx = set(order[:m])
        # добавим ранги — удобно в UI
        ranks = [0] * len(result)
        for r, i in enumerate(order, start=1):
            ranks[i] = r
        for i, r in enumerate(result):
            r["is_target"] = i in top_idx
            r["rank"] = ranks[i]
        meta = {"k_percent": k, "threshold_in_batch": thr, "selected_n": m}

    ts = int(time.time())
    sig = _sign_response([r["proba_no_show"] for r in result], MODEL_VER, ts)

    response.headers["X-Model-Version"] = MODEL_VER
    return {"meta": meta, "results": result, "signature": sig, "timestamp": ts}



"""
Подробное описание, что происходит.

Сервис состоит из 2 слоев:
1) Транспорт (FastAPI) — получение http-запросов, валидация json файла (pydentic схема), формирование ответов и метаданных (в т.ч. и заголовок с версией модели).
2) Модель (загружается из файла, обученного train_baseline.py), котрая:
- принмиает на вход pd.DataFrame с признаками визита, которые были при обучении;
- выполняет встроенную предобработку, которая вгита в пайплайн модели (стандартизация числовых, one-hot кодирование категориальных);
- возвращает вероятность no-show (неявки) для каждого визита.

input: JSON-объекты с признаками визита (см. класс Visit).
output: вероятность no-show (неявки) для каждого визита.

Особенности:
- Поддержка одиночного визита и батча (до MAX_BATCH записей).  
- Валидация входных данных с помощью Pydantic (ограничения по типам и диапазонам).
- Опциональный параметр k: для отбора Top-K% визитов с наивысшей вероятностью no-show внутри батча.
- Заголовок ответа X-Model-Version с контрольной суммой модели для отслеживания версии. 

Для запуска сервиса:
    uvicorn src.trialflow.serve_baseline:app --host 0.0.0.0 --port 8000 --reload    
Для тестирования:
    curl -X POST "http://localhost:8000/score?k=10" -H "Content-Type: application/json" -d "[{\"age\":30,\"weekday\":2,\"lead_time_days\":15,\"is_first_visit\":1,\"site_id\":\"A1\",\"visit_type\":\"visit\",\"sms_received\":1,\"scholarship\":0,\"hipertension\":0,\"diabetes\":0,\"alcoholism\":
0,\"handicap\":0,\"gender\":\"M\"}]"


Сценарий работы:
1. Клиент шлёт POST на /score с JSON телом: либо {...} (один визит), либо [{...}, {...}].
2. FastAPI + Pydantic:
    - проверяют структуру и типы;
    - приводят значения в валидный диапазон (через нашу логику _to_dataframe).
3. Собираем DataFrame с точными колонками.
4. Подаём его в MODEL.predict_proba(df) и берём столбец для «положительного» класса.
5. Формируем список результатов [{proba_no_show: ...}, ...].
6. Если передали ?k=10, сортируем по вероятности и отмечаем топ-10% (is_target=True), считаем порог в батче.
7. Возвращаем JSON и заголовок X-Model-Version: <SHA12>.
"""