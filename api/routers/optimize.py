from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import ModelRun, Scenario, Session
from api.schemas import ForwardOptimizeRequest, OptimizeResponse, ReverseOptimizeRequest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.optimizer import BudgetOptimizer

router = APIRouter(prefix="/optimize", tags=["optimize"])


def _load_optimizer(run: ModelRun, session: Session):
    if not session.channel_csv_path:
        raise HTTPException(400, "Channel CSV not found for this session")

    df = pd.read_csv(session.channel_csv_path)
    fit_config = json.loads(run.fit_config_json) if run.fit_config_json else {}
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}

    channels = df["channel"].unique().tolist()
    mean_spend = {ch: float(df[df["channel"] == ch]["media_spend"].mean()) for ch in channels}
    contributions = json.loads(run.contributions_json) if run.contributions_json else {}

    # Approximate ROI coefficients from contributions / mean spend
    rois = {}
    for ch in channels:
        spend = mean_spend.get(ch, 1.0)
        contrib = contributions.get(ch, 0.0)
        rois[ch] = contrib / spend if spend > 0 else 0.0

    return channels, rois, mean_spend


def _build_bounds(channels, channel_bounds_list, total_budget=None):
    bounds_map = {b.channel: b for b in channel_bounds_list}
    bounds = []
    for ch in channels:
        b = bounds_map.get(ch)
        if b:
            lo = b.min_spend
            hi = None if b.no_upper_limit else b.max_spend
        else:
            lo = 0.0
            hi = None
        bounds.append((lo, hi))
    return bounds


@router.post("/forward", response_model=OptimizeResponse)
def forward_optimize(req: ForwardOptimizeRequest, db: DBSession = Depends(get_db)):
    run = db.get(ModelRun, req.model_run_id)
    if not run or run.status != "complete":
        raise HTTPException(400, "Model run not found or not complete")
    session = db.get(Session, req.session_id)

    channels, rois, _ = _load_optimizer(run, session)
    bounds = _build_bounds(channels, req.channel_bounds, req.total_budget)

    # Simple proportional allocation weighted by ROI
    from scipy.optimize import minimize

    roi_arr = np.array([rois.get(ch, 0.0) for ch in channels])
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] if b[1] is not None else req.total_budget for b in bounds])

    def neg_conversions(x):
        return -np.sum(roi_arr * x)

    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - req.total_budget}]
    scipy_bounds = list(zip(lo_arr, hi_arr))

    x0 = np.full(len(channels), req.total_budget / len(channels))
    x0 = np.clip(x0, lo_arr, hi_arr)

    from scipy.optimize import minimize
    res = minimize(neg_conversions, x0, method="SLSQP", bounds=scipy_bounds, constraints=constraints)
    allocation = {ch: float(v) for ch, v in zip(channels, res.x)}
    expected_conv = float(-res.fun)

    result = OptimizeResponse(
        channel_allocation=allocation,
        total_spend=req.total_budget,
        expected_conversions=expected_conv,
    )

    # Persist scenario
    scenario = Scenario(
        session_id=req.session_id,
        model_run_id=req.model_run_id,
        name=f"Forward opt budget={req.total_budget:.0f}",
        scenario_type="forward",
        inputs_json=req.model_dump_json(),
        results_json=result.model_dump_json(),
    )
    db.add(scenario)
    db.commit()

    return result


@router.post("/reverse", response_model=OptimizeResponse)
def reverse_optimize(req: ReverseOptimizeRequest, db: DBSession = Depends(get_db)):
    run = db.get(ModelRun, req.model_run_id)
    if not run or run.status != "complete":
        raise HTTPException(400, "Model run not found or not complete")
    session = db.get(Session, req.session_id)

    channels, rois, _ = _load_optimizer(run, session)
    bounds = _build_bounds(channels, req.channel_bounds)

    roi_arr = np.array([rois.get(ch, 1e-9) for ch in channels])
    lo_arr = np.array([b[0] for b in bounds])
    hi_arr = np.array([b[1] if b[1] is not None else 1e9 for b in bounds])

    def total_spend(x):
        return np.sum(x)

    constraints = [
        {"type": "ineq", "fun": lambda x: np.sum(roi_arr * x) - req.target_conversions}
    ]
    scipy_bounds = list(zip(lo_arr, hi_arr))

    # Start from proportional allocation
    if np.sum(roi_arr) > 0:
        weights = roi_arr / np.sum(roi_arr)
        est_spend = req.target_conversions / (np.sum(roi_arr * weights) + 1e-9)
        x0 = weights * est_spend
    else:
        x0 = np.full(len(channels), req.target_conversions / len(channels))
    x0 = np.clip(x0, lo_arr, hi_arr)

    from scipy.optimize import minimize
    res = minimize(total_spend, x0, method="SLSQP", bounds=scipy_bounds, constraints=constraints)
    allocation = {ch: float(v) for ch, v in zip(channels, res.x)}
    total = float(res.fun)

    result = OptimizeResponse(
        channel_allocation=allocation,
        total_spend=total,
        expected_conversions=req.target_conversions,
    )

    scenario = Scenario(
        session_id=req.session_id,
        model_run_id=req.model_run_id,
        name=f"Reverse opt target={req.target_conversions:.0f}",
        scenario_type="reverse",
        inputs_json=req.model_dump_json(),
        results_json=result.model_dump_json(),
    )
    db.add(scenario)
    db.commit()

    return result


@router.get("/scenarios/{session_id}")
def list_scenarios(session_id: int, db: DBSession = Depends(get_db)):
    scenarios = (
        db.query(Scenario)
        .filter_by(session_id=session_id)
        .order_by(Scenario.created_at.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "name": s.name,
            "type": s.scenario_type,
            "inputs": json.loads(s.inputs_json) if s.inputs_json else {},
            "results": json.loads(s.results_json) if s.results_json else {},
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in scenarios
    ]


@router.post("/scenarios/{scenario_id}/save")
def save_scenario(scenario_id: int, name: str, db: DBSession = Depends(get_db)):
    scenario = db.get(Scenario, scenario_id)
    if not scenario:
        raise HTTPException(404, "Scenario not found")
    scenario.name = name
    db.commit()
    return {"id": scenario.id, "name": scenario.name}
