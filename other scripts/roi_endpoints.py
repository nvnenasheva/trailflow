# -*- coding: utf-8 -*-
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
from roi import (
    per_target_net_benefit,
    expected_net_benefit,
    optimal_k,
    read_actions_csv,
    pick_best_action,
    Action,
)

router = APIRouter(prefix="/roi", tags=["roi"])

# ---------- Models ----------

class SingleENBRequest(BaseModel):
    n: int = Field(..., gt=0)
    L: float = Field(..., ge=0)                   # €/no-show
    cost: float = Field(..., ge=0)                # €/visit
    uplift: float = Field(..., ge=0, le=1)
    base_p: Optional[float] = Field(None, ge=0, le=1)
    pi: Optional[float] = Field(None, ge=0, le=1) # preferred
    k: Optional[float] = Field(None, ge=0, le=1)
    budget: Optional[float] = Field(None, ge=0)
    optimize_k: bool = True

    @validator("pi", always=True)
    def _check_probs(cls, v, values):
        if v is None and values.get("base_p") is None:
            raise ValueError("Provide pi or base_p")
        return v

class SingleENBResponse(BaseModel):
    per_target_nb: float
    k_used: float
    targeted: int
    spend: float
    enb: float
    roi_x: float

class ActionIn(BaseModel):
    name: str
    uplift: float = Field(..., ge=0, le=1)
    cost: float = Field(..., ge=0)
    pi: Optional[float] = Field(None, ge=0, le=1)
    base_p: Optional[float] = Field(None, ge=0, le=1)

class BestActionRequest(BaseModel):
    n: int = Field(..., gt=0)
    L: float = Field(..., ge=0)
    base_p: Optional[float] = Field(None, ge=0, le=1)
    pi: Optional[float] = Field(None, ge=0, le=1)
    budget: Optional[float] = Field(None, ge=0)
    optimize_k: bool = True
    actions: List[ActionIn]

class BestActionResponse(BaseModel):
    best_action: str
    p_eff: float
    uplift: float
    cost: float
    per_target_nb: float
    k_used: float
    targeted: int
    spend: float
    enb: float
    roi_x: float

# ---------- Endpoints ----------

@router.post("/single", response_model=SingleENBResponse)
def compute_single(req: SingleENBRequest):
    p_eff = req.pi if req.pi is not None else req.base_p  # type: ignore
    pt_nb = per_target_net_benefit(p_eff, req.uplift, req.L, req.cost)
    if req.optimize_k or req.k is None:
        k_used = optimal_k(req.n, pt_nb, req.cost, req.budget)
    else:
        k_used = req.k
    targeted = int(round(req.n * k_used))
    enb = expected_net_benefit(req.n, k_used, p_eff, req.uplift, req.L, req.cost)
    spend = targeted * req.cost
    roi_x = (enb + spend) / spend if spend > 0 else float("inf")
    return SingleENBResponse(
        per_target_nb=pt_nb,
        k_used=k_used,
        targeted=targeted,
        spend=spend,
        enb=enb,
        roi_x=roi_x,
    )

@router.post("/best_action", response_model=BestActionResponse)
def best_action(req: BestActionRequest):
    if not req.actions:
        raise HTTPException(400, "actions list is empty")
    actions = [Action(name=a.name, uplift=a.uplift, cost=a.cost, pi=a.pi, base_p=a.base_p) for a in req.actions]
    a, p_eff, pt_nb, k_used, enb_total = pick_best_action(
        actions=actions,
        n=req.n,
        loss_per_no_show=req.L,
        cli_base_p=req.base_p,
        cli_pi=req.pi,
        k=None,
        budget=req.budget,
        auto_k=req.optimize_k,
    )
    targeted = int(round(req.n * k_used))
    spend = targeted * a.cost
    roi_x = (enb_total + spend) / spend if spend > 0 else float("inf")
    return BestActionResponse(
        best_action=a.name,
        p_eff=p_eff,
        uplift=a.uplift,
        cost=a.cost,
        per_target_nb=pt_nb,
        k_used=k_used,
        targeted=targeted,
        spend=spend,
        enb=enb_total,
        roi_x=roi_x,
    )
