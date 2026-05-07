"""
Data processing module for Bayesian MMM.

Responsibilities:
  - Load raw campaign / channel data
  - Validate schema
  - Build Fourier seasonality features
  - Aggregate to channel-level for modelling
  - Train / test split
  - Normalisation helpers
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

REQUIRED_COLS = {"date", "channel", "media_spend", "conversions"}

CONTROL_COLS = [
    "seasonality_sin_52",
    "seasonality_cos_52",
    "seasonality_sin_26",
    "seasonality_cos_26",
    "holiday_flag",
    "promo_flag",
]


# ---------------------------------------------------------------------------
# Fourier seasonality
# ---------------------------------------------------------------------------

def fourier_terms(
    dates: pd.Series,
    periods: list[float] | None = None,
    n_harmonics: int = 2,
) -> pd.DataFrame:
    """Build Fourier sin/cos columns from a DatetimeSeries.

    Args:
        dates: weekly date column.
        periods: list of periods in weeks. Defaults to [52, 26] (annual + semi-annual).
        n_harmonics: number of sin/cos pairs per period.

    Returns:
        DataFrame with Fourier feature columns.
    """
    if periods is None:
        periods = [52.0, 26.0]

    t = np.arange(len(dates))
    cols: dict[str, np.ndarray] = {}
    for p in periods:
        for k in range(1, n_harmonics + 1):
            cols[f"sin_{int(p)}_{k}"] = np.sin(2 * np.pi * k * t / p)
            cols[f"cos_{int(p)}_{k}"] = np.cos(2 * np.pi * k * t / p)
    return pd.DataFrame(cols, index=dates.values)


# ---------------------------------------------------------------------------
# DataConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for data pre-processing."""

    target_col: str = "conversions"
    spend_col: str = "media_spend"
    date_col: str = "date"
    channel_col: str = "channel"

    # Seasonality
    include_seasonality: bool = True
    seasonality_periods: list[float] = field(default_factory=lambda: [52.0, 26.0])
    n_harmonics: int = 2

    # Control variables to pass through
    control_cols: list[str] = field(default_factory=lambda: ["holiday_flag", "promo_flag"])

    # Train / test split
    test_weeks: int = 12

    # Normalisation
    scale_spend: bool = True
    scale_target: bool = True


# ---------------------------------------------------------------------------
# MMMDataset
# ---------------------------------------------------------------------------

@dataclass
class MMMDataset:
    """Container returned by DataProcessor.prepare()."""

    # Time index (sorted unique dates)
    dates: np.ndarray

    # Spend matrix: shape (T, C)
    spend_matrix: np.ndarray
    channel_names: list[str]

    # Target vector: shape (T,)
    target: np.ndarray

    # Seasonality / control regressors: shape (T, K)
    control_matrix: np.ndarray
    control_names: list[str]

    # Raw (unscaled) versions for plotting
    spend_raw: np.ndarray
    target_raw: np.ndarray

    # Scalers (None if scaling disabled)
    spend_scaler: Optional[StandardScaler]
    target_scaler: Optional[StandardScaler]

    # Train / test masks
    train_mask: np.ndarray
    test_mask: np.ndarray

    # Campaign-level data (optional, for hierarchical model)
    campaign_df: Optional[pd.DataFrame] = None

    @property
    def n_time(self) -> int:
        return len(self.dates)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    @property
    def n_controls(self) -> int:
        return len(self.control_names)

    def train_data(self):
        return (
            self.spend_matrix[self.train_mask],
            self.target[self.train_mask],
            self.control_matrix[self.train_mask],
        )

    def test_data(self):
        return (
            self.spend_matrix[self.test_mask],
            self.target[self.test_mask],
            self.control_matrix[self.test_mask],
        )


# ---------------------------------------------------------------------------
# DataProcessor
# ---------------------------------------------------------------------------

class DataProcessor:
    """Prepares raw MMM data for modelling."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        df: pd.DataFrame,
        campaign_df: Optional[pd.DataFrame] = None,
    ) -> MMMDataset:
        """Full preparation pipeline.

        Args:
            df: Channel-level weekly DataFrame (one row per week × channel).
                Must contain at minimum: date, channel, media_spend, conversions.
            campaign_df: Optional campaign-level DataFrame for hierarchical modelling.

        Returns:
            MMMDataset ready for the Bayesian model.
        """
        cfg = self.config
        df = self._validate_and_clean(df)
        df = self._aggregate_to_channel_weekly(df)

        # Pivot to wide: rows=weeks, cols=channels
        spend_wide = df.pivot(
            index=cfg.date_col, columns=cfg.channel_col, values=cfg.spend_col
        ).sort_index()
        channel_names = list(spend_wide.columns)
        spend_raw = spend_wide.values.astype(float)
        dates = spend_wide.index.values

        # Target (sum over channels per week)
        target_wide = df.groupby(cfg.date_col)[cfg.target_col].mean()
        target_raw = target_wide.loc[spend_wide.index].values.astype(float)

        # Control / seasonality features
        control_matrix, control_names = self._build_controls(
            pd.Series(spend_wide.index, name="date"), df
        )

        # Scaling
        spend_scaler = None
        target_scaler = None
        spend_scaled = spend_raw.copy()
        target_scaled = target_raw.copy()

        if cfg.scale_spend:
            spend_scaler = StandardScaler()
            spend_scaled = spend_scaler.fit_transform(spend_raw)

        if cfg.scale_target:
            target_scaler = StandardScaler()
            target_scaled = target_scaler.fit_transform(
                target_raw.reshape(-1, 1)
            ).ravel()

        # Train / test split
        T = len(dates)
        n_test = cfg.test_weeks
        train_mask = np.zeros(T, dtype=bool)
        test_mask = np.zeros(T, dtype=bool)
        train_mask[: T - n_test] = True
        test_mask[T - n_test :] = True

        return MMMDataset(
            dates=dates,
            spend_matrix=spend_scaled,
            channel_names=channel_names,
            target=target_scaled,
            control_matrix=control_matrix,
            control_names=control_names,
            spend_raw=spend_raw,
            target_raw=target_raw,
            spend_scaler=spend_scaler,
            target_scaler=target_scaler,
            train_mask=train_mask,
            test_mask=test_mask,
            campaign_df=campaign_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df[self.config.date_col] = pd.to_datetime(df[self.config.date_col])
        df[self.config.spend_col] = pd.to_numeric(df[self.config.spend_col], errors="coerce").fillna(0.0).clip(lower=0)
        df[self.config.target_col] = pd.to_numeric(df[self.config.target_col], errors="coerce").fillna(0.0).clip(lower=0)

        n_before = len(df)
        df = df.dropna(subset=[self.config.date_col])
        if len(df) < n_before:
            warnings.warn(f"Dropped {n_before - len(df)} rows with invalid dates.")

        return df

    def _aggregate_to_channel_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        agg_cols = {cfg.spend_col: "sum", cfg.target_col: "mean"}
        for col in cfg.control_cols:
            if col in df.columns:
                agg_cols[col] = "first"

        return (
            df.groupby([cfg.date_col, cfg.channel_col])
            .agg(agg_cols)
            .reset_index()
            .sort_values([cfg.date_col, cfg.channel_col])
        )

    def _build_controls(
        self, dates_series: pd.Series, df: pd.DataFrame
    ) -> tuple[np.ndarray, list[str]]:
        cfg = self.config
        parts: list[pd.DataFrame] = []
        names: list[str] = []

        if cfg.include_seasonality:
            fourier_df = fourier_terms(
                dates_series,
                periods=cfg.seasonality_periods,
                n_harmonics=cfg.n_harmonics,
            )
            parts.append(fourier_df)
            names.extend(fourier_df.columns.tolist())

        # Pass-through control columns from data
        for col in cfg.control_cols:
            if col in df.columns:
                col_series = (
                    df.groupby(cfg.date_col)[col].first().sort_index()
                )
                parts.append(col_series.rename(col).to_frame())
                names.append(col)

        if parts:
            control_df = pd.concat(parts, axis=1).fillna(0.0)
            return control_df.values.astype(float), names

        T = len(dates_series)
        return np.zeros((T, 0), dtype=float), []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def inverse_scale_target(
        values: np.ndarray, scaler: Optional[StandardScaler]
    ) -> np.ndarray:
        if scaler is None:
            return values
        return scaler.inverse_transform(values.reshape(-1, 1)).ravel()

    @staticmethod
    def inverse_scale_spend(
        values: np.ndarray, scaler: Optional[StandardScaler]
    ) -> np.ndarray:
        if scaler is None:
            return values
        return scaler.inverse_transform(values)

    @staticmethod
    def compute_roi(spend: np.ndarray, conversions: np.ndarray) -> np.ndarray:
        """Element-wise ROI: conversions / spend."""
        with np.errstate(divide="ignore", invalid="ignore"):
            roi = np.where(spend > 0, conversions / spend, 0.0)
        return roi

    @staticmethod
    def compute_cpc(spend: np.ndarray, conversions: np.ndarray) -> np.ndarray:
        """Cost per conversion."""
        with np.errstate(divide="ignore", invalid="ignore"):
            cpc = np.where(conversions > 0, spend / conversions, 0.0)
        return cpc
