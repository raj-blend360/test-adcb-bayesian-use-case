from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session as DBSession

from api.deps import get_db
from api.models import ModelRun, Session
from api.schemas import FitRequest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.data_processing import DataConfig, DataProcessor
from src.diagnostics import check_convergence, out_of_sample_validation, generate_diagnostic_report
from src.model import BayesianMMM, ModelConfig

router = APIRouter(prefix="/model", tags=["model"])

IDATA_DIR = "idata"
os.makedirs(IDATA_DIR, exist_ok=True)

FIT_SPEED_PRESETS = {
    "fast": {
        "inference_method": "map",
        "samples": 100,
        "tune": 100,
        "chains": 1,
        "halo_pairs": [],
    },
    "standard": {
        "inference_method": "map",
        "samples": 1000,
        "tune": 1000,
        "chains": 2,
    },
    "thorough": {
        "inference_method": "mcmc",
        "samples": 3000,
        "tune": 2000,
        "chains": 4,
    },
}


def _build_model_config(req: FitRequest, dataset) -> ModelConfig:
    channel_halo = []
    campaign_halo = []
    for pair in req.halo_pairs:
        if pair.type == "channel":
            channel_halo.append((pair.a, pair.b))
        elif pair.type == "campaign":
            campaign_halo.append((pair.a, pair.b))

    return ModelConfig(
        inference=req.inference_method,
        n_samples=req.samples,
        n_tune=req.tune,
        n_chains=req.chains,
        target_accept=req.target_accept,
        halo_pairs=channel_halo,
        campaign_halo_pairs=campaign_halo,
        min_halo_spend=req.min_halo_spend,
        adstock_max_lag=req.adstock_max_lag,
    )


def _subtract_campaign_spends(dataset, req: FitRequest):
    """Subtract halo campaign spends from parent channel spend columns."""
    if dataset.campaign_spend_matrix is None:
        return dataset

    campaign_pairs = [(p.a, p.b) for p in req.halo_pairs if p.type == "campaign" and p.subtract_campaign_spend]
    if not campaign_pairs:
        return dataset

    halo_campaigns = {c for pair in campaign_pairs for c in pair}
    spend_matrix = dataset.spend_matrix.copy()
    spend_raw = dataset.spend_raw.copy()

    if dataset.campaign_names and dataset.campaign_channels:
        camp_lookup = dict(zip(dataset.campaign_names, dataset.campaign_channels))
        ch_idx = {ch: i for i, ch in enumerate(dataset.channel_names)}
        for ci, camp_name in enumerate(dataset.campaign_names):
            if camp_name in halo_campaigns:
                parent_ch = camp_lookup.get(camp_name)
                if parent_ch and parent_ch in ch_idx:
                    pi = ch_idx[parent_ch]
                    camp_spend = dataset.campaign_spend_matrix[:, ci]
                    spend_raw[:, pi] = np.maximum(spend_raw[:, pi] - camp_spend, 0.0)
                    # Re-scale
                    if dataset.spend_scaler is not None:
                        spend_matrix[:, pi] = (spend_raw[:, pi] - dataset.spend_scaler.mean_[pi]) / dataset.spend_scaler.scale_[pi]
                    else:
                        spend_matrix[:, pi] = spend_raw[:, pi]

    dataset.spend_matrix = spend_matrix
    dataset.spend_raw = spend_raw
    return dataset


def _apply_channel_input_metrics(
    channel_df: pd.DataFrame,
    transform_config: dict,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """For each channel, replace media_spend with configured input metric."""
    cfg_channels = transform_config.get("channels", []) or []
    metric_by_channel = {
        ch_cfg.get("channel"): ch_cfg.get("metric", "clicks")
        for ch_cfg in cfg_channels
        if ch_cfg.get("channel")
    }
    if not metric_by_channel:
        return channel_df, {}

    out_df = channel_df.copy()
    for ch, metric in metric_by_channel.items():
        if metric == "media_spend":
            continue
        if metric not in out_df.columns:
            print(f"[Transform] Channel '{ch}': metric '{metric}' missing, keeping clicks/media_spend fallback.")
            continue
        mask = out_df["channel"] == ch
        out_df.loc[mask, "media_spend"] = pd.to_numeric(
            out_df.loc[mask, metric], errors="coerce"
        ).fillna(0.0)
    return out_df, metric_by_channel


def _resolve_fit_request(req: FitRequest) -> FitRequest:
    if not req.fit_speed:
        return req
    preset = FIT_SPEED_PRESETS.get(req.fit_speed)
    if not preset:
        return req
    return req.model_copy(update=preset)


def _run_fit(run_id: int, req_dict: dict):
    from api.database import SessionLocal
    db = SessionLocal()
    try:
        run = db.get(ModelRun, run_id)
        run.status = "running"
        db.commit()

        req = _resolve_fit_request(FitRequest(**req_dict))
        session = db.get(Session, req.session_id)
        if not session or not session.channel_csv_path:
            raise ValueError("Session or channel CSV not found")

        # Load data
        channel_df = pd.read_csv(session.channel_csv_path)
        campaign_df = pd.read_csv(session.campaign_csv_path) if session.campaign_csv_path else None

        # Parse transform config
        transform_config = json.loads(session.config_json) if session.config_json else {}
        channel_df, metric_by_channel = _apply_channel_input_metrics(channel_df, transform_config)
        print("[Transform] Channel input metrics:", metric_by_channel or "default clicks for all channels")
        print(f"[Transform] Global max adstock lag: {req.adstock_max_lag} weeks")
        print("[Transform] Beta prior: Normal(0, 0.3), non-centered parameterization enabled")
        print(
            "[Fit] Effective settings:",
            {
                "fit_speed": req.fit_speed or "custom",
                "draws": req.samples,
                "tune": req.tune,
                "chains": req.chains,
                "inference_method": req.inference_method,
                "halo_count": len(req.halo_pairs),
            },
        )

        dc = DataConfig(
            include_seasonality=transform_config.get("include_seasonality", True),
            test_weeks=transform_config.get("test_weeks", 12),
        )
        processor = DataProcessor(dc)
        dataset = processor.prepare(channel_df, campaign_df)

        # Subtract halo campaign spends from parent channels
        dataset = _subtract_campaign_spends(dataset, req)

        # Build and fit model
        model_cfg = _build_model_config(req, dataset)
        mmm = BayesianMMM(model_cfg)
        results = mmm.fit(dataset)

        # Diagnostics
        conv_df = check_convergence(results)
        oos_dict = out_of_sample_validation(results)
        diag_df = generate_diagnostic_report(results)

        # Contributions
        contribs = results.contributions  # dict channel -> array
        contrib_means = {ch: float(np.mean(v)) for ch, v in contribs.items()} if contribs else {}

        # Metrics from convergence / OOS / diagnostic report
        diag = {}
        if not diag_df.empty:
            for _, row in diag_df.iterrows():
                diag[str(row.get("metric", ""))] = row.get("value")

        adj_r2 = float(oos_dict.get("r2", 0.0))
        mape = float(oos_dict.get("mape", 0.0))
        rhat_pass_count = int(conv_df.get("rhat_ok", pd.Series([True])).sum()) if not conv_df.empty else 1
        total_params = max(len(conv_df), 1)
        rhat_pass = float(rhat_pass_count / total_params * 100)
        conf_width = 0.0  # HDI width can be added later

        # Save idata
        idata_path = os.path.join(IDATA_DIR, f"run_{run_id}.nc")
        try:
            results.idata.to_netcdf(idata_path)
        except Exception:
            idata_path = None

        run.status = "complete"
        run.adj_r2 = adj_r2
        run.mape = mape
        run.rhat_pass_pct = rhat_pass
        run.confidence_width = conf_width
        run.contributions_json = json.dumps(contrib_means)
        run.metrics_json = json.dumps({**diag, **oos_dict})
        run.idata_path = idata_path
        db.commit()

    except Exception as e:
        run = db.get(ModelRun, run_id)
        run.status = "failed"
        run.error_message = traceback.format_exc()
        db.commit()
    finally:
        db.close()


@router.post("/fit")
def fit_model(
    req: FitRequest,
    background_tasks: BackgroundTasks,
    db: DBSession = Depends(get_db),
):
    session = db.get(Session, req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Determine model_num and iteration_num
    existing = db.query(ModelRun).filter_by(session_id=req.session_id).all()
    model_num = len(existing) + 1
    iteration_num = 1

    run = ModelRun(
        session_id=req.session_id,
        model_num=model_num,
        iteration_num=iteration_num,
        status="pending",
        fit_config_json=req.model_dump_json(),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(_run_fit, run.id, req.model_dump())
    return {"job_id": run.id, "status": "pending"}


@router.get("/status/{job_id}")
def get_status(job_id: int, db: DBSession = Depends(get_db)):
    run = db.get(ModelRun, job_id)
    if not run:
        raise HTTPException(404, "Job not found")
    return {
        "job_id": run.id,
        "status": run.status,
        "error": run.error_message,
        "model_num": run.model_num,
        "iteration_num": run.iteration_num,
    }
