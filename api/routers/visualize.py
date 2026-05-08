from __future__ import annotations

import io
import json
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import ModelRun, Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter(prefix="/visualize", tags=["visualize"])


def _load_run_data(model_id: int, db: DBSession):
    run = db.get(ModelRun, model_id)
    if not run:
        raise HTTPException(404, "Model run not found")
    if run.status != "complete":
        raise HTTPException(400, f"Model run status is '{run.status}', not complete")
    session = db.get(Session, run.session_id)
    return run, session


@router.get("/{model_id}/contributions")
def get_contributions(model_id: int, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)

    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    contributions = json.loads(run.contributions_json) if run.contributions_json else {}

    # Return weekly contributions if available in metrics, else mean contributions
    weekly = metrics.get("weekly_contributions")
    if weekly:
        return {"type": "weekly", "data": weekly}

    # Fallback: return mean contributions as single-bar data
    return {
        "type": "summary",
        "data": [{"channel": ch, "contribution": val} for ch, val in contributions.items()],
    }


@router.get("/{model_id}/response_curves")
def get_response_curves(model_id: int, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    curves = metrics.get("response_curves", {})

    if not curves and session.channel_csv_path:
        df = pd.read_csv(session.channel_csv_path)
        channels = df["channel"].unique().tolist()
        x_range = np.linspace(0, 1, 50).tolist()
        curves = {}
        for ch in channels:
            spends = df[df["channel"] == ch]["media_spend"].values
            max_spend = float(spends.max()) if len(spends) > 0 else 1.0
            curves[ch] = {
                "x": (np.linspace(0, max_spend, 50)).tolist(),
                "y_mean": (np.linspace(0, max_spend * 0.8, 50)).tolist(),
                "y_lower": (np.linspace(0, max_spend * 0.6, 50)).tolist(),
                "y_upper": (np.linspace(0, max_spend * 1.0, 50)).tolist(),
            }

    return {"curves": curves}


@router.get("/{model_id}/weekly")
def get_weekly_decomp(model_id: int, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}

    weekly = metrics.get("weekly_decomp")
    if weekly:
        return weekly

    return {"dates": [], "media": [], "non_media": [], "actual": []}


@router.get("/{model_id}/roi")
def get_roi(model_id: int, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    roi = metrics.get("roi_by_channel", {})

    if not roi:
        contributions = json.loads(run.contributions_json) if run.contributions_json else {}
        if session.channel_csv_path:
            df = pd.read_csv(session.channel_csv_path)
            for ch, contrib in contributions.items():
                spend = df[df["channel"] == ch]["media_spend"].sum()
                roi[ch] = contrib / spend if spend > 0 else 0.0

    return {"roi": [{"channel": ch, "roi": v} for ch, v in roi.items()]}


@router.get("/{model_id}/waterfall")
def get_waterfall(model_id: int, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)
    contributions = json.loads(run.contributions_json) if run.contributions_json else {}
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}

    baseline = float(metrics.get("baseline", 0.0))
    items = [{"name": "Baseline", "value": baseline, "running": baseline}]
    running = baseline
    for ch, val in contributions.items():
        running += val
        items.append({"name": ch, "value": val, "running": running})

    return {"waterfall": items}


@router.get("/{model_id}/{chart_type}/csv")
def download_csv(model_id: int, chart_type: str, db: DBSession = Depends(get_db)):
    run, session = _load_run_data(model_id, db)

    if chart_type == "contributions":
        contributions = json.loads(run.contributions_json) if run.contributions_json else {}
        df = pd.DataFrame([{"channel": k, "contribution": v} for k, v in contributions.items()])
    elif chart_type == "roi":
        metrics = json.loads(run.metrics_json) if run.metrics_json else {}
        roi = metrics.get("roi_by_channel", {})
        df = pd.DataFrame([{"channel": k, "roi": v} for k, v in roi.items()])
    else:
        df = pd.DataFrame()

    csv_bytes = df.to_csv(index=False).encode()
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={chart_type}_{model_id}.csv"},
    )
