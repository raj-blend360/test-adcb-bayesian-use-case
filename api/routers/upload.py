import json
import os
import shutil
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import Session
from api.schemas import ColumnMapping, UploadResponse

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CHANNEL_REQUIRED = {"date", "channel", "media_spend", "conversions"}
CAMPAIGN_REQUIRED = {"date", "channel", "campaign", "media_spend"}


def _detect_mapping(columns: list[str]) -> ColumnMapping:
    col_lower = {c.lower(): c for c in columns}
    mapping = ColumnMapping()
    for field, candidates in {
        "date": ["date", "week", "week_date", "period"],
        "channel": ["channel", "media_channel", "channel_name"],
        "sub_channel": ["sub_channel", "subchannel", "sub-channel"],
        "campaign": ["campaign", "campaign_name"],
        "media_spend": ["media_spend", "spend", "cost", "budget"],
        "impressions": ["impressions", "imps"],
        "clicks": ["clicks", "click"],
        "conversions": ["conversions", "conv", "leads", "sales", "signups"],
    }.items():
        for cand in candidates:
            if cand in col_lower:
                setattr(mapping, field, col_lower[cand])
                break
    return mapping


def _normalize_single_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single wide CSV into the long channel-level schema.

    Expected wide patterns per channel:
      - spends_<channel>
      - media_impressions_<channel>
      - media_clicks_<channel>
    plus shared columns like date, conversions, and exogenous_* controls.
    """
    cols = list(df.columns)
    if CHANNEL_REQUIRED.issubset(cols):
        return df

    if "date" not in cols:
        raise HTTPException(400, "Missing required column: date")

    spend_cols = [c for c in cols if c.startswith("spends_")]
    if not spend_cols:
        raise HTTPException(400, "Missing spend columns. Expected columns like spends_channel1")

    if "conversions" not in cols:
        raise HTTPException(400, "Missing required target column: conversions")

    channels = [c.removeprefix("spends_") for c in spend_cols]
    exogenous_cols = [c for c in cols if c.startswith("exogenous_")]

    long_rows = []
    for _, row in df.iterrows():
        for ch in channels:
            spend_col = f"spends_{ch}"
            imp_col = f"media_impressions_{ch}"
            clk_col = f"media_clicks_{ch}"
            rec = {
                "date": row["date"],
                "channel": ch,
                "media_spend": row.get(spend_col, 0.0),
                "impressions": row.get(imp_col, 0.0) if imp_col in cols else 0.0,
                "clicks": row.get(clk_col, 0.0) if clk_col in cols else 0.0,
                "conversions": row["conversions"],
            }
            for exog in exogenous_cols:
                rec[exog] = row.get(exog)
            long_rows.append(rec)

    return pd.DataFrame(long_rows)


@router.post("/channel", response_model=UploadResponse)
async def upload_channel(
    file: UploadFile = File(...),
    session_id: Optional[int] = Form(None),
    db: DBSession = Depends(get_db),
):
    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    df = _normalize_single_csv(df)

    missing = CHANNEL_REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing required columns after normalization: {missing}")

    # Persist file
    save_path = os.path.join(UPLOAD_DIR, f"channel_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    # Create or update session
    if session_id:
        session = db.get(Session, session_id)
        if not session:
            raise HTTPException(404, "Session not found")
    else:
        session = Session()
        db.add(session)
        db.flush()

    session.channel_csv_path = save_path
    db.commit()
    db.refresh(session)

    mapping = _detect_mapping(list(df.columns))
    preview = df.head(10).fillna("").to_dict(orient="records")
    return UploadResponse(
        session_id=session.id,
        columns=list(df.columns),
        preview=preview,
        detected_mapping=mapping,
    )


@router.post("/campaign", response_model=UploadResponse)
async def upload_campaign(
    file: UploadFile = File(...),
    session_id: int = Form(...),
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    missing = CAMPAIGN_REQUIRED - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    save_path = os.path.join(UPLOAD_DIR, f"campaign_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(content)

    session.campaign_csv_path = save_path
    db.commit()
    db.refresh(session)

    mapping = _detect_mapping(list(df.columns))
    preview = df.head(10).fillna("").to_dict(orient="records")
    return UploadResponse(
        session_id=session.id,
        columns=list(df.columns),
        preview=preview,
        detected_mapping=mapping,
    )


@router.get("/columns/{session_id}")
def get_columns(session_id: int, db: DBSession = Depends(get_db)):
    session = db.get(Session, session_id)
    if not session or not session.channel_csv_path:
        raise HTTPException(404, "Session or channel CSV not found")

    df = pd.read_csv(session.channel_csv_path, nrows=0)
    channel_cols = list(df.columns)
    channels = sorted(pd.read_csv(session.channel_csv_path)["channel"].unique().tolist()) if "channel" in channel_cols else []

    campaign_cols = []
    campaigns = []
    if session.campaign_csv_path:
        cdf = pd.read_csv(session.campaign_csv_path)
        campaign_cols = list(cdf.columns)
        campaigns = sorted(cdf["campaign"].unique().tolist()) if "campaign" in campaign_cols else []

    return {
        "channel_columns": channel_cols,
        "channels": channels,
        "campaign_columns": campaign_cols,
        "campaigns": campaigns,
        "detected_mapping": _detect_mapping(channel_cols).model_dump(),
    }
