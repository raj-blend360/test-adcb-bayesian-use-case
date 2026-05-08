from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel


# ─── Upload ──────────────────────────────────────────────────────────────────

class ColumnMapping(BaseModel):
    date: str = "date"
    channel: str = "channel"
    sub_channel: Optional[str] = None
    campaign: Optional[str] = None
    media_spend: str = "media_spend"
    impressions: Optional[str] = None
    clicks: Optional[str] = None
    conversions: str = "conversions"


class UploadResponse(BaseModel):
    session_id: int
    columns: list[str]
    preview: list[dict]
    detected_mapping: ColumnMapping


# ─── Transform ───────────────────────────────────────────────────────────────

class ChannelAdstockConfig(BaseModel):
    adstock_type: str = "geometric"  # geometric / weibull
    max_lag: int = 8
    decay_prior_mean: float = 0.5

class ChannelSaturationConfig(BaseModel):
    saturation_type: str = "hill"  # hill / logistic / michaelis_menten
    alpha_prior_mean: float = 2.0
    lambda_prior_mean: float = 0.5

class ChannelTransformConfig(BaseModel):
    channel: str
    adstock: ChannelAdstockConfig = ChannelAdstockConfig()
    saturation: ChannelSaturationConfig = ChannelSaturationConfig()
    metric: str = "conversions"  # conversions / impressions / clicks

class TransformConfigRequest(BaseModel):
    session_id: int
    include_seasonality: bool = True
    include_holiday: bool = True
    include_promo: bool = False
    seasonality_periods: list[float] = [52.0, 26.0]
    n_harmonics: int = 2
    test_weeks: int = 12
    channels: list[ChannelTransformConfig] = []


# ─── Model fit ───────────────────────────────────────────────────────────────

class HaloPair(BaseModel):
    type: str  # "channel" or "campaign"
    a: str
    b: str
    subtract_campaign_spend: bool = True  # only relevant when type=="campaign"

class FitRequest(BaseModel):
    session_id: int
    inference_method: str = "map"  # map / advi / mcmc
    samples: int = 1000
    tune: int = 1000
    chains: int = 2
    target_accept: float = 0.9
    halo_pairs: list[HaloPair] = []
    min_halo_spend: float = 0.0
    adstock_max_lag: int = 8
    transform_config_id: Optional[int] = None


# ─── Results ─────────────────────────────────────────────────────────────────

class ModelRunSummary(BaseModel):
    id: int
    model_num: int
    iteration_num: int
    name: Optional[str]
    status: str
    adj_r2: Optional[float]
    mape: Optional[float]
    rhat_pass_pct: Optional[float]
    confidence_width: Optional[float]
    contributions: Optional[dict]
    created_at: str

    class Config:
        from_attributes = True

class SaveModelRequest(BaseModel):
    name: str


# ─── Tune ────────────────────────────────────────────────────────────────────

class HolidayEntry(BaseModel):
    label: str
    start_date: str
    end_date: str

class SeasonalityConfig(BaseModel):
    quarterly: bool = False
    half_yearly: bool = True
    annual: bool = True

class TuneConfigRequest(BaseModel):
    session_id: int
    base_model_run_id: int
    holidays: list[HolidayEntry] = []
    seasonality: SeasonalityConfig = SeasonalityConfig()


# ─── Optimize ────────────────────────────────────────────────────────────────

class ChannelBound(BaseModel):
    channel: str
    min_spend: float = 0.0
    no_upper_limit: bool = True
    max_spend: Optional[float] = None

class ForwardOptimizeRequest(BaseModel):
    session_id: int
    model_run_id: int
    total_budget: float
    channel_bounds: list[ChannelBound] = []

class ReverseOptimizeRequest(BaseModel):
    session_id: int
    model_run_id: int
    target_conversions: float
    channel_bounds: list[ChannelBound] = []

class OptimizeResponse(BaseModel):
    channel_allocation: dict[str, float]
    total_spend: float
    expected_conversions: float
    campaign_breakdown: Optional[dict[str, float]] = None
