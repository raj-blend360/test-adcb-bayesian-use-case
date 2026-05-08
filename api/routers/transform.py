import json
import sys
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import Session
from api.schemas import TransformConfigRequest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.transformations import geometric_adstock_np

router = APIRouter(prefix="/transform", tags=["transform"])


@router.post("/config")
def save_transform_config(req: TransformConfigRequest, db: DBSession = Depends(get_db)):
    session = db.get(Session, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    session.config_json = req.model_dump_json()
    db.commit()
    return {"status": "saved", "session_id": req.session_id}


@router.get("/preview/{session_id}")
def preview_adstocked(session_id: int, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session or not session.channel_csv_path:
        raise HTTPException(404, "Session or channel CSV not found")

    df = pd.read_csv(session.channel_csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    config = json.loads(session.config_json) if session.config_json else {}
    channel_configs = {c["channel"]: c for c in config.get("channels", [])}

    result = {}
    for channel in df["channel"].unique():
        ch_df = df[df["channel"] == channel].sort_values("date")
        spend = ch_df["media_spend"].values.astype(float)
        ch_cfg = channel_configs.get(channel, {})
        adstock_cfg = ch_cfg.get("adstock", {})
        decay = adstock_cfg.get("decay_prior_mean", 0.5)
        max_lag = adstock_cfg.get("max_lag", 8)
        adstocked = geometric_adstock_np(spend, decay, max_lag)
        result[channel] = {
            "dates": ch_df["date"].dt.strftime("%Y-%m-%d").tolist(),
            "raw_spend": spend.tolist(),
            "adstocked_spend": adstocked.tolist(),
        }

    return result
