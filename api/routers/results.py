import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import ModelRun
from api.schemas import ModelRunSummary, SaveModelRequest

router = APIRouter(prefix="/results", tags=["results"])


def _to_summary(run: ModelRun) -> dict:
    contributions = json.loads(run.contributions_json) if run.contributions_json else None
    metrics = json.loads(run.metrics_json) if run.metrics_json else {}
    return {
        "id": run.id,
        "model_num": run.model_num,
        "iteration_num": run.iteration_num,
        "name": run.name,
        "status": run.status,
        "adj_r2": run.adj_r2,
        "mape": run.mape,
        "rhat_pass_pct": run.rhat_pass_pct,
        "confidence_width": run.confidence_width,
        "contributions": contributions,
        "metrics": metrics,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("")
def list_results(session_id: int, db: DBSession = Depends(get_db)):
    runs = (
        db.query(ModelRun)
        .filter_by(session_id=session_id)
        .order_by(ModelRun.created_at.desc())
        .all()
    )
    return [_to_summary(r) for r in runs]


@router.get("/{model_id}")
def get_result(model_id: int, db: DBSession = Depends(get_db)):
    run = db.get(ModelRun, model_id)
    if not run:
        raise HTTPException(404, "Model run not found")
    return _to_summary(run)


@router.post("/{model_id}/save")
def save_model_name(
    model_id: int,
    req: SaveModelRequest,
    db: DBSession = Depends(get_db),
):
    run = db.get(ModelRun, model_id)
    if not run:
        raise HTTPException(404, "Model run not found")
    run.name = req.name
    db.commit()
    return {"id": run.id, "name": run.name}
