from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Visit(BaseModel):
    age: int
    is_first_visit: bool
    day_of_week: int
    lead_time_days: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(v: Visit):
    # TODO: подгрузить модель; сейчас — фиктивный скор
    risk = 0.2 + 0.1 * (v.day_of_week in (1,5)) + 0.1 * (v.lead_time_days > 30)
    return {"risk": min(0.95, risk)}
