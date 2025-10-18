from pydantic import BaseModel, Field
from typing import List, Optional

class Visit(BaseModel):
    age: int = Field(ge=0, le=120)
    weekday: int = Field(ge=0, le=6)
    lead_time_days: int = Field(ge=0)
    is_first_visit: int = Field(ge=0, le=1)
    site_id: str
    visit_type: str = "visit"
    sms_received: int = Field(ge=0, le=1)
    scholarship: int = Field(ge=0, le=1)
    hipertension: int = Field(ge=0, le=1)
    diabetes: int = Field(ge=0, le=1)
    alcoholism: int = Field(ge=0, le=1)
    handicap: int = Field(ge=0, le=1)
    gender: str = "U"  # "M"/"F"/"U"

class ScoreItem(BaseModel):
    proba_no_show: float
    rank: Optional[int] = None
    is_target: bool = False

class ScoreResponse(BaseModel):
    results: List[ScoreItem]
    meta: dict
    timestamp: int
    signature: str
