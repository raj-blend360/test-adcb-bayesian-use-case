"""
Halo candidate selection for Bayesian MMM.

Provides a data-driven utility to score and rank cross-channel campaign pairs
as candidates for halo effect modelling, based on:
  - Adstocked spend correlation  (how synchronised campaigns move together)
  - Spend overlap                (fraction of weeks both campaigns are active)
  - Minimum campaign spend       (proxy for statistical identifiability)

Usage
-----
    from src.halo_analysis import rank_halo_candidates, plot_halo_heatmap

    scored = rank_halo_candidates(
        campaign_df=campaign_df,
        channel_decay={"TV": 0.65, "Digital": 0.20, "Radio": 0.40, "OOH": 0.50},
        min_halo_spend=100_000,
        top_n=10,
    )
    print(scored)
"""

from __future__ import annotations

import itertools
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .transformations import geometric_adstock_np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fmt_millions(x, _):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ---------------------------------------------------------------------------
# Step 1: build adstocked spend matrix
# ---------------------------------------------------------------------------


def compute_campaign_adstocked_spend(
    campaign_df: pd.DataFrame,
    channel_decay: dict[str, float],
    max_lag: int = 13,
    date_col: str = "date",
    campaign_col: str = "campaign",
    channel_col: str = "channel",
    spend_col: str = "media_spend",
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    """Pivot campaign spend and apply geometric adstock per campaign.

    Args:
        campaign_df: tidy DataFrame with columns date, campaign, channel, media_spend.
        channel_decay: dict mapping channel name → adstock decay rate.
            Campaigns inherit their parent channel's decay.
        max_lag: adstock look-back window in weeks.
        date_col: date column name.
        campaign_col: campaign column name.
        channel_col: channel column name.
        spend_col: spend column name.

    Returns:
        adstocked_matrix: (T, N_campaigns) float64.
        raw_spend_matrix: (T, N_campaigns) float64 unscaled.
        campaign_names: list of campaign name strings (column order).
        campaign_channels: list of parent channel per campaign (parallel to names).
        dates: sorted date array (T,).
    """
    campaign_df = campaign_df.copy()
    campaign_df[date_col] = pd.to_datetime(campaign_df[date_col])

    # Pivot to (T, N_campaigns)
    pivot = campaign_df.pivot_table(
        index=date_col,
        columns=campaign_col,
        values=spend_col,
        aggfunc="sum",
        fill_value=0.0,
    ).sort_index()

    dates = pivot.index.values
    campaign_names = list(pivot.columns)
    raw_spend_matrix = pivot.values.astype(float)

    # Parent channel lookup
    ch_lookup = (
        campaign_df.groupby(campaign_col)[channel_col].first().to_dict()
    )
    campaign_channels = [ch_lookup.get(c, "") for c in campaign_names]

    # Apply adstock per campaign using parent-channel decay
    adstocked_matrix = np.zeros_like(raw_spend_matrix)
    for i, camp in enumerate(campaign_names):
        ch = campaign_channels[i]
        decay = channel_decay.get(ch, 0.4)
        adstocked_matrix[:, i] = geometric_adstock_np(
            raw_spend_matrix[:, i], decay, max_lag
        )

    return adstocked_matrix, raw_spend_matrix, campaign_names, campaign_channels, dates


# ---------------------------------------------------------------------------
# Step 2: score all cross-channel pairs
# ---------------------------------------------------------------------------


def compute_pairwise_scores(
    adstocked_matrix: np.ndarray,
    raw_spend_matrix: np.ndarray,
    campaign_names: list[str],
    campaign_channels: list[str],
    min_halo_spend: float = 0.0,
    correlation_weight: float = 0.5,
    overlap_weight: float = 0.3,
    spend_weight: float = 0.2,
) -> pd.DataFrame:
    """Score all cross-channel campaign pairs as halo candidates.

    Only pairs where both campaigns belong to **different** channels are
    considered — within-channel campaigns already share a channel coefficient.

    Args:
        adstocked_matrix: (T, N_campaigns) from compute_campaign_adstocked_spend.
        raw_spend_matrix: (T, N_campaigns) raw unscaled spend.
        campaign_names: list of campaign names (column order of the matrices).
        campaign_channels: parent channel for each campaign.
        min_halo_spend: exclude campaigns whose total raw spend is below this.
        correlation_weight: weight for adstock Pearson correlation metric.
        overlap_weight: weight for spend overlap metric.
        spend_weight: weight for minimum-campaign-spend metric.

    Returns:
        DataFrame with one row per cross-channel pair, sorted descending by
        composite_score. Columns:
          campaign_a, channel_a, campaign_b, channel_b,
          adstock_correlation, spend_overlap, min_total_spend, composite_score.
    """
    total_spend = raw_spend_matrix.sum(axis=0)  # (N_campaigns,)

    # Eligibility mask
    eligible_idx = [
        i for i, s in enumerate(total_spend) if s >= min_halo_spend
    ]

    rows = []
    for i, j in itertools.combinations(eligible_idx, 2):
        ch_i = campaign_channels[i]
        ch_j = campaign_channels[j]
        if ch_i == ch_j:
            continue  # skip within-channel pairs

        xi = adstocked_matrix[:, i]
        xj = adstocked_matrix[:, j]

        # Adstocked correlation (clip to [0, 1] — negative synergy not modelled)
        corr_mat = np.corrcoef(xi, xj)
        corr = float(np.clip(corr_mat[0, 1], 0.0, 1.0))

        # Spend overlap: fraction of weeks where both have non-zero raw spend
        overlap = float(
            np.mean(
                (raw_spend_matrix[:, i] > 0) & (raw_spend_matrix[:, j] > 0)
            )
        )

        min_spend = float(min(total_spend[i], total_spend[j]))

        rows.append(
            {
                "campaign_a": campaign_names[i],
                "channel_a": ch_i,
                "campaign_b": campaign_names[j],
                "channel_b": ch_j,
                "adstock_correlation": corr,
                "spend_overlap": overlap,
                "min_total_spend": min_spend,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "campaign_a", "channel_a", "campaign_b", "channel_b",
                "adstock_correlation", "spend_overlap", "min_total_spend",
                "composite_score",
            ]
        )

    df = pd.DataFrame(rows)

    # Normalise each metric to [0, 1]
    def _norm(col):
        mn, mx = col.min(), col.max()
        return (col - mn) / (mx - mn + 1e-8)

    df["composite_score"] = (
        correlation_weight * _norm(df["adstock_correlation"])
        + overlap_weight * _norm(df["spend_overlap"])
        + spend_weight * _norm(df["min_total_spend"])
    )

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def rank_halo_candidates(
    campaign_df: pd.DataFrame,
    channel_decay: dict[str, float],
    min_halo_spend: float = 0.0,
    top_n: int = 20,
    max_lag: int = 13,
    correlation_weight: float = 0.5,
    overlap_weight: float = 0.3,
    spend_weight: float = 0.2,
    date_col: str = "date",
    campaign_col: str = "campaign",
    channel_col: str = "channel",
    spend_col: str = "media_spend",
) -> pd.DataFrame:
    """Score and rank campaign pairs as halo effect candidates.

    Args:
        campaign_df: tidy campaign-level DataFrame with columns
            date, campaign, channel, media_spend.
        channel_decay: dict mapping channel name → adstock decay (0–1).
            Campaigns inherit the decay of their parent channel.
        min_halo_spend: campaigns with total spend below this are excluded.
            Use raw currency units (e.g. £500,000).
        top_n: return this many top-ranked pairs.
        max_lag: adstock look-back window in weeks.
        correlation_weight: weight for adstocked-spend Pearson r.
        overlap_weight: weight for % weeks both campaigns are active.
        spend_weight: weight for min(total_spend_a, total_spend_b).
        date_col, campaign_col, channel_col, spend_col: column name overrides.

    Returns:
        DataFrame with top_n rows sorted descending by composite_score.
    """
    adstocked, raw, names, channels, _ = compute_campaign_adstocked_spend(
        campaign_df,
        channel_decay=channel_decay,
        max_lag=max_lag,
        date_col=date_col,
        campaign_col=campaign_col,
        channel_col=channel_col,
        spend_col=spend_col,
    )

    scored = compute_pairwise_scores(
        adstocked,
        raw,
        names,
        channels,
        min_halo_spend=min_halo_spend,
        correlation_weight=correlation_weight,
        overlap_weight=overlap_weight,
        spend_weight=spend_weight,
    )

    return scored.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_halo_heatmap(
    scored_df: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple = (14, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of composite halo scores for top-N campaign pairs.

    Args:
        scored_df: output from rank_halo_candidates().
        top_n: how many pairs to include in the heatmap.
        figsize: figure size.
        save_path: if provided, saves figure to this path.

    Returns:
        matplotlib Figure.
    """
    import os

    df = scored_df.head(top_n)
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No halo candidates found", ha="center", va="center")
        return fig

    # Collect unique campaign names that appear in top pairs
    camps_a = df["campaign_a"].tolist()
    camps_b = df["campaign_b"].tolist()
    all_camps = sorted(set(camps_a + camps_b))
    n = len(all_camps)
    idx = {c: i for i, c in enumerate(all_camps)}

    # Build symmetric score matrix
    matrix = np.full((n, n), np.nan)
    for _, row in df.iterrows():
        i = idx[row["campaign_a"]]
        j = idx[row["campaign_b"]]
        matrix[i, j] = row["composite_score"]
        matrix[j, i] = row["composite_score"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_camps, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(all_camps, fontsize=8)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if matrix[i, j] < 0.7 else "white",
                )

    plt.colorbar(im, ax=ax, label="Composite Halo Score")
    ax.set_title(
        f"Halo Candidate Scores — Top {len(df)} Cross-Channel Pairs",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------


def suggest_halo_config(
    scored_df: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """Convert top-N scored pairs into a list suitable for halo_config.json.

    Args:
        scored_df: output from rank_halo_candidates().
        top_n: how many pairs to include.

    Returns:
        List of dicts: [{"campaign_a": "...", "campaign_b": "...", "score": ...}]
    """
    rows = []
    for _, row in scored_df.head(top_n).iterrows():
        rows.append(
            {
                "campaign_a": row["campaign_a"],
                "campaign_b": row["campaign_b"],
                "score": round(float(row["composite_score"]), 4),
            }
        )
    return rows
