# src/trialflow/scoring.py
from typing import List, Union, Tuple
import pandas as pd
from .schemas import Visit, ScoreItem
from .model_io import predict_proba_df, get_version

def _to_dataframe(visits: List[Visit]) -> pd.DataFrame:
    # Pydantic v2: .model_dump()
    rows = [v.model_dump() for v in visits]
    df = pd.DataFrame(rows)
    # при необходимости можно привести типы (int/str), но обычно Pipeline сам справляется
    return df

def score_visits(payload: Union[Visit, List[Visit]], k_percent: int = 0) -> Tuple[List[ScoreItem], dict]:
    visits = payload if isinstance(payload, list) else [payload]
    df = _to_dataframe(visits)
    proba = predict_proba_df(df)  # зовём реальную модель
    items = [ScoreItem(proba_no_show=float(p)) for p in proba]

    # сортировка + ранги
    items.sort(key=lambda r: r.proba_no_show, reverse=True)
    for i, r in enumerate(items, start=1):
        r.rank = i

    # top-K
    selected_n = 0
    threshold = None
    if k_percent and len(items):
        topn = max(0, min(len(items), round(len(items) * k_percent / 100)))
        if topn > 0:
            threshold = items[topn - 1].proba_no_show
            for r in items:
                r.is_target = r.proba_no_show >= threshold
            selected_n = sum(1 for r in items if r.is_target)

    meta = {
        "k_percent": k_percent or None,
        "selected_n": selected_n,
        "threshold_in_batch": threshold,
        "model_version": get_version(),  # добавим в ответ
    }
    return items, meta
