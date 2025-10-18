# src/trialflow/model_io.py
import os
import hashlib
import threading

from typing import Any, Optional
import joblib
import numpy as np
import math

_MODEL: Optional[Any] = None
_VERSION: str = "unknown"
_LOCK = threading.Lock()
_MODEL_MTIME: Optional[float] = None

def _sha1_of_file(path: str, nbytes: int = 1_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(nbytes)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:8]

def get_model_path() -> str:
    path = os.getenv("MODEL_PATH", "").strip()
    if not path:
        raise RuntimeError("MODEL_PATH is not set in environment")
    if not os.path.isabs(path):
        # путь считаем от корня репозитория (рабочей директории)
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {path}")
    return path

def _load_model_locked():
    global _MODEL, _VERSION, _MODEL_MTIME
    path = get_model_path()
    _MODEL = joblib.load(path)
    # версия: либо из ENV, либо имя файла + короткий sha1
    env_ver = os.getenv("MODEL_VERSION", "").strip()
    if env_ver:
        _VERSION = env_ver
    else:
        _VERSION = f"{os.path.basename(path)}:{_sha1_of_file(path)}"
    _MODEL_MTIME = os.path.getmtime(path)

def get_model():
    """ лениво грузим и авто-перезагружаем при замене файла """
    global _MODEL, _MODEL_MTIME
    path = get_model_path()
    mtime = os.path.getmtime(path)
    with _LOCK:
        if _MODEL is None or _MODEL_MTIME != mtime:
            _load_model_locked()
    return _MODEL

def get_version() -> str:
    if _MODEL is None:
        # ensure loaded to compute version
        get_model()
    return _VERSION

def _sigmoid(x):
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0

def predict_proba_df(df):
    """
    Универсальный вызов predict_proba для sklearn/LightGBM/XGB пайплайнов.
    df — pandas.DataFrame с колонками, совпадающими с теми, что ждал тренинг.
    """
    model = get_model()
    # большинство продакшн-артефактов — это sklearn Pipeline с .predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)
        # берём класс 1
        if isinstance(proba, list) or len(getattr(proba, "shape", [])) == 1:
            proba = np.array(proba)
        if proba.ndim == 2:
            proba = proba[:, -1]
        return np.clip(proba, 1e-6, 1 - 1e-6)

    # если нет predict_proba, но есть decision_function — логистическая калибровка на лету
    if hasattr(model, "decision_function"):
        scores = model.decision_function(df)
        if isinstance(scores, list):
            scores = np.array(scores)
        if scores.ndim > 1:
            scores = scores[:, -1]
        proba = np.vectorize(_sigmoid)(scores)
        return np.clip(proba, 1e-6, 1 - 1e-6)

    # крайний случай — есть только predict (классы); деградируем
    y = model.predict(df)
    return np.clip(np.array(y, dtype=float), 1e-6, 1 - 1e-6)
