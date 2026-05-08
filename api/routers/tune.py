import json
import os
import sys
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import ModelRun, Session, TuningConfig
from api.schemas import TuneConfigRequest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

router = APIRouter(prefix="/tune", tags=["tune"])


@router.post("/config")
def save_tune_config(req: TuneConfigRequest, db: DBSession = Depends(get_db)):
    session = db.get(Session, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    tc = TuningConfig(
        session_id=req.session_id,
        holidays_json=json.dumps([h.model_dump() for h in req.holidays]),
        seasonality_json=json.dumps(req.seasonality.model_dump()),
    )
    db.add(tc)
    db.commit()
    db.refresh(tc)
    return {"tuning_config_id": tc.id}


@router.post("/run")
def run_tuned_model(
    req: TuneConfigRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    from api.routers.model import _run_fit, FitRequest

    base_run = db.get(ModelRun, req.base_model_run_id)
    if not base_run:
        raise HTTPException(404, "Base model run not found")

    # Build a new run incrementing iteration_num
    new_iter = base_run.iteration_num + 1
    fit_config = json.loads(base_run.fit_config_json) if base_run.fit_config_json else {}

    # Inject holiday / seasonality overrides into session config
    session = db.get(Session, req.session_id)
    existing_config = json.loads(session.config_json) if session.config_json else {}
    existing_config["holidays"] = [h.model_dump() for h in req.holidays]
    existing_config["seasonality"] = req.seasonality.model_dump()
    session.config_json = json.dumps(existing_config)
    db.commit()

    run = ModelRun(
        session_id=req.session_id,
        model_num=base_run.model_num,
        iteration_num=new_iter,
        status="pending",
        fit_config_json=json.dumps(fit_config),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(_run_fit, run.id, fit_config)
    return {"job_id": run.id, "status": "pending", "iteration_num": new_iter}


@router.get("/history/{session_id}")
def get_tune_history(session_id: int, db: DBSession = Depends(get_db)):
    configs = (
        db.query(TuningConfig)
        .filter_by(session_id=session_id)
        .order_by(TuningConfig.created_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "holidays": json.loads(c.holidays_json),
            "seasonality": json.loads(c.seasonality_json),
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in configs
    ]
