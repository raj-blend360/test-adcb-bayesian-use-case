"""
Visualization module for Bayesian MMM.

All functions return matplotlib Figure objects (or Plotly figures where noted)
and optionally save to disk.

Functions
---------
plot_contributions()        — stacked area chart of base + channel contributions
plot_response_curves()      — channel response curves with diminishing returns
plot_roi_metrics()          — bar chart of ROI / cost-per-conversion
plot_budget_allocation()    — current vs optimized spend comparison
plot_diagnostics()          — R-hat and ESS histograms
plot_posterior_distributions() — posterior ridgelines for key parameters
plot_efficient_frontier()   — budget vs conversions frontier
plot_campaign_allocation()  — campaign-level spend breakdown
plot_actual_vs_predicted()  — in-sample and out-of-sample fit
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Optional Plotly
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    _PLOTLY = True
except ImportError:
    _PLOTLY = False


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#2196F3", "#FF5722", "#4CAF50", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#607D8B",
    "#795548", "#8BC34A", "#03A9F4", "#FFEB3B",
]


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {path}")


def _fmt_millions(x, _):
    if abs(x) >= 1e6:
        return f"{x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"{x/1e3:.0f}K"
    return f"{x:.0f}"


# ---------------------------------------------------------------------------
# 1. Contributions chart
# ---------------------------------------------------------------------------


def plot_contributions(
    contributions: dict,
    dates: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 7),
) -> plt.Figure:
    """Stacked area chart of base + channel contributions over time.

    Args:
        contributions: output from BayesianMMM.get_contributions().
        dates: date array aligned with contribution rows.
        save_path: if provided, save figure to this path.
    """
    contrib_df = contributions["channels"]
    actual = contributions["actual"]
    predicted = contributions["total_predicted"]

    fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]

    channels = [c for c in contrib_df.columns if c not in ("base", "controls")]
    colors = PALETTE[: len(channels) + 2]

    # Stacked area
    stack_labels = ["base"] + channels
    if "controls" in contrib_df.columns:
        stack_labels.append("controls")

    stack_data = [contrib_df["base"].values] + [
        contrib_df[c].values for c in channels
    ]
    if "controls" in contrib_df.columns:
        stack_data.append(contrib_df["controls"].values)

    ax.stackplot(
        dates,
        stack_data,
        labels=stack_labels,
        colors=colors[: len(stack_labels)],
        alpha=0.85,
    )
    ax.plot(dates, actual, "k-", lw=1.5, label="Actual", zorder=10)
    ax.plot(dates, predicted, "w--", lw=1.2, label="Predicted", zorder=9, alpha=0.8)

    ax.set_title("Channel Contributions to Conversions", fontsize=14, fontweight="bold")
    ax.set_ylabel("Conversions")
    ax.legend(loc="upper left", fontsize=8, ncol=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.grid(axis="y", alpha=0.3)

    # Residuals subplot
    ax2 = axes[1]
    residuals = actual - predicted
    ax2.bar(dates, residuals, color="steelblue", alpha=0.6, width=6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_title("Residuals (Actual − Predicted)", fontsize=10)
    ax2.set_ylabel("Δ Conversions")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 2. Response curves
# ---------------------------------------------------------------------------


def plot_response_curves(
    curves: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Channel response curves with diminishing returns and current-spend marker.

    Args:
        curves: output from BayesianMMM.get_response_curves().
        save_path: optional save path.
    """
    channels = list(curves.keys())
    n = len(channels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (ch, curve) in enumerate(curves.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        color = PALETTE[idx % len(PALETTE)]

        spend = curve["spend"]
        mean_ = curve["conversions_mean"]
        low = curve["conversions_hdi_low"]
        high = curve["conversions_hdi_high"]

        ax.plot(spend, mean_, color=color, lw=2.5, label="Mean")
        ax.fill_between(spend, low, high, alpha=0.25, color=color, label="90% CI")

        # Current spend marker
        curr_sp = curve["current_spend"]
        curr_conv = curve["current_conversions"]
        ax.axvline(curr_sp, color="black", ls="--", lw=1.2, label="Current spend")
        ax.scatter([curr_sp], [curr_conv], color="black", zorder=5, s=60)

        ax.set_title(ch, fontsize=12, fontweight="bold")
        ax.set_xlabel("Spend")
        ax.set_ylabel("Incremental Conversions")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide extra axes
    for extra in range(n, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Channel Response Curves (Diminishing Returns)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 3. ROI / efficiency metrics
# ---------------------------------------------------------------------------


def plot_roi_metrics(
    roi_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Bar charts for ROI and cost-per-conversion per channel."""
    channels = roi_df.index.tolist()
    x = np.arange(len(channels))

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ROI
    axes[0].bar(x, roi_df["roi"], color=PALETTE[: len(channels)], alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(channels, rotation=30, ha="right")
    axes[0].set_title("ROI (Conversions per £ Spent)")
    axes[0].set_ylabel("ROI")
    axes[0].grid(axis="y", alpha=0.3)

    # Cost per conversion
    axes[1].bar(x, roi_df["cost_per_conversion"], color=PALETTE[len(channels):len(channels)*2], alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(channels, rotation=30, ha="right")
    axes[1].set_title("Cost per Conversion")
    axes[1].set_ylabel("£")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    axes[1].grid(axis="y", alpha=0.3)

    # % contribution
    axes[2].bar(x, roi_df["pct_contribution"], color=PALETTE[2:2+len(channels)], alpha=0.85)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(channels, rotation=30, ha="right")
    axes[2].set_title("% Contribution to Total Conversions")
    axes[2].set_ylabel("%")
    axes[2].grid(axis="y", alpha=0.3)

    fig.suptitle("Channel ROI & Efficiency Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 4. Budget allocation comparison
# ---------------------------------------------------------------------------


def plot_budget_allocation(
    opt_result,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Current vs optimized spend comparison (grouped bar chart)."""
    channels = opt_result.channel_names
    current = opt_result.current_spend
    optimal = opt_result.optimal_spend
    x = np.arange(len(channels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    bars1 = ax.bar(x - width / 2, current, width, label="Current", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, optimal, width, label="Optimized", color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=30, ha="right")
    ax.set_title("Spend: Current vs Optimized")
    ax.set_ylabel("Spend")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # % change
    ax2 = axes[1]
    pct = opt_result.spend_change_pct
    colors_ = ["#4CAF50" if v >= 0 else "#F44336" for v in pct]
    ax2.bar(x, pct, color=colors_, alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=30, ha="right")
    ax2.set_title("Spend Change % per Channel")
    ax2.set_ylabel("% Change")
    ax2.grid(axis="y", alpha=0.3)

    total_curr = current.sum()
    total_opt = optimal.sum()
    conv_uplift = opt_result.conversion_uplift_pct

    fig.suptitle(
        f"Budget Optimization  |  Budget: {_fmt_millions(total_curr, None)} → {_fmt_millions(total_opt, None)}"
        f"  |  Conversion Uplift: +{conv_uplift:.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 5. Convergence diagnostics
# ---------------------------------------------------------------------------


def plot_diagnostics(
    conv_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Histogram of R-hat and ESS values across all parameters."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if "r_hat" in conv_df.columns:
        rhats = conv_df["r_hat"].dropna()
        axes[0].hist(rhats, bins=20, color="#2196F3", alpha=0.8, edgecolor="white")
        axes[0].axvline(1.05, color="red", ls="--", lw=1.5, label="Threshold (1.05)")
        axes[0].set_title("R-hat Distribution")
        axes[0].set_xlabel("R-hat")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "R-hat not available", ha="center", va="center")

    if "ess_bulk" in conv_df.columns:
        ess = conv_df["ess_bulk"].dropna()
        axes[1].hist(ess, bins=20, color="#4CAF50", alpha=0.8, edgecolor="white")
        axes[1].axvline(400, color="red", ls="--", lw=1.5, label="Min ESS (400)")
        axes[1].set_title("ESS-bulk Distribution")
        axes[1].set_xlabel("ESS-bulk")
        axes[1].set_ylabel("Count")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "ESS not available", ha="center", va="center")

    fig.suptitle("MCMC Convergence Diagnostics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 6. Posterior distributions
# ---------------------------------------------------------------------------


def plot_posterior_distributions(
    results,
    var_names: Optional[list[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 10),
) -> plt.Figure:
    """Violin plots of posterior distributions for key parameters."""
    import arviz as az

    idata = results.idata
    channel_names = results.dataset.channel_names

    default_vars = ["beta", "alpha_hill", "gamma_hill", "decay"]
    if var_names is None:
        var_names = [v for v in default_vars if v in idata.posterior]

    n_vars = len(var_names)
    if n_vars == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No posterior variables found", ha="center", va="center")
        return fig

    ncols = min(2, n_vars)
    nrows = (n_vars + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_vars == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, var in enumerate(var_names):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        samples = idata.posterior[var].values  # (chains, draws, ...)
        flat = samples.reshape(-1, samples.shape[-1]) if samples.ndim == 3 else samples.reshape(-1, 1)

        vp = ax.violinplot(
            [flat[:, i] for i in range(flat.shape[1])],
            positions=range(flat.shape[1]),
            showmedians=True,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_alpha(0.7)

        labels = channel_names[: flat.shape[1]] if flat.shape[1] <= len(channel_names) else [str(i) for i in range(flat.shape[1])]
        ax.set_xticks(range(flat.shape[1]))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(f"Posterior: {var}", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Hide extra panels
    for extra in range(n_vars, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Posterior Parameter Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 7. Efficient frontier
# ---------------------------------------------------------------------------


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    current_budget: float,
    current_conversions: float,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot optimal conversions vs total budget (efficient frontier)."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        frontier_df["total_budget"],
        frontier_df["optimal_conversions"],
        color="#2196F3",
        lw=2.5,
        label="Efficient Frontier",
    )
    ax.fill_between(
        frontier_df["total_budget"],
        frontier_df["optimal_conversions"],
        alpha=0.15,
        color="#2196F3",
    )

    # Current operating point
    ax.scatter(
        [current_budget],
        [current_conversions],
        color="red",
        s=100,
        zorder=5,
        label="Current",
    )
    ax.annotate(
        " Current",
        (current_budget, current_conversions),
        fontsize=9,
        color="red",
    )

    ax.set_xlabel("Total Budget")
    ax.set_ylabel("Optimal Conversions")
    ax.set_title("Budget Efficient Frontier", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8. Campaign allocation
# ---------------------------------------------------------------------------


def plot_campaign_allocation(
    camp_df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Grouped bar chart of current vs optimized campaign spend."""
    if camp_df is None or camp_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No campaign data available", ha="center", va="center")
        return fig

    df = camp_df.reset_index()
    channels = df["channel"].unique()
    n_ch = len(channels)
    ncols = min(2, n_ch)
    nrows = (n_ch + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_ch == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, ch in enumerate(channels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ch_df = df[df["channel"] == ch]
        x = np.arange(len(ch_df))
        width = 0.35
        ax.bar(x - width / 2, ch_df["current_spend"], width, label="Current", color="#2196F3", alpha=0.85)
        ax.bar(x + width / 2, ch_df["optimal_spend"], width, label="Optimized", color="#FF5722", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(ch_df["campaign"], rotation=40, ha="right", fontsize=8)
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_ylabel("Spend")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    for extra in range(n_ch, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle("Campaign-Level Spend Allocation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 9. Actual vs predicted
# ---------------------------------------------------------------------------


def plot_actual_vs_predicted(
    results,
    oos_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """In-sample fit with out-of-sample overlay."""
    dataset = results.dataset
    contributions = None
    try:
        from .model import BayesianMMM
        mmm = BayesianMMM(results.config)
        contributions = mmm.get_contributions(results)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=figsize)

    dates = dataset.dates
    train_dates = dates[dataset.train_mask]

    if contributions is not None:
        predicted = contributions["total_predicted"]
        actual = contributions["actual"]
        ax.plot(train_dates, actual, "k-", lw=1.5, label="Actual", zorder=4)
        ax.plot(train_dates, predicted, color="#2196F3", lw=1.8, ls="--", label="Predicted (in-sample)", zorder=3)

    # Shade test region
    test_dates = dates[dataset.test_mask]
    if len(test_dates) > 0:
        ax.axvspan(test_dates[0], test_dates[-1], alpha=0.08, color="orange", label="Test period")

    # Overlay OOS predictions
    if oos_metrics is not None:
        ax.plot(
            oos_metrics["test_dates"],
            oos_metrics["observed"],
            "ko",
            ms=4,
            label="Actual (test)",
            zorder=6,
        )
        ax.plot(
            oos_metrics["test_dates"],
            oos_metrics["predicted"],
            color="#FF5722",
            lw=2,
            label=f"Predicted (OOS, MAPE={oos_metrics['mape']:.1f}%)",
            zorder=5,
        )

    ax.set_title("Actual vs Predicted Conversions", fontsize=13, fontweight="bold")
    ax.set_ylabel("Conversions")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 10. Waterfall — spend vs conversions decomposition
# ---------------------------------------------------------------------------


def plot_waterfall_decomposition(
    contributions: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Waterfall chart showing base + incremental by channel."""
    contrib_df = contributions["channels"]
    channels = [c for c in contrib_df.columns if c not in ("base", "controls")]

    totals = {"Base": contrib_df["base"].sum()}
    for ch in channels:
        totals[ch] = contrib_df[ch].sum()
    if "controls" in contrib_df.columns:
        totals["Controls"] = contrib_df["controls"].sum()

    labels = list(totals.keys())
    values = list(totals.values())
    running = 0
    bottoms = []
    bar_colors = []
    for i, v in enumerate(values):
        bottoms.append(running if i > 0 else 0)
        bar_colors.append(PALETTE[i % len(PALETTE)])
        running += v

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(labels))
    ax.bar(x, values, bottom=bottoms, color=bar_colors, alpha=0.85, width=0.6)

    # Cumulative line
    cumulative = np.cumsum(values)
    ax.plot(x, cumulative, "k--o", lw=1.2, ms=5, label="Cumulative")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Total Conversions")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_millions))
    ax.set_title("Conversion Waterfall: Base + Incremental by Channel", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_media_vs_base_contribution(
    contributions: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Two-bar chart for total Media vs Base contribution share."""
    contrib_df = contributions["channels"]
    channels = [c for c in contrib_df.columns if c not in ("base", "controls")]

    base_total = float(contrib_df["base"].sum())
    media_total = float(contrib_df[channels].sum().sum()) if channels else 0.0
    total = base_total + media_total
    if total <= 0:
        base_pct = media_pct = 0.0
    else:
        base_pct = (base_total / total) * 100.0
        media_pct = (media_total / total) * 100.0

    labels = ["Media", "Base"]
    values = [media_pct, base_pct]
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, values, color=[PALETTE[0], PALETTE[1]], alpha=0.9)
    ax.set_ylim(0, max(100, max(values) * 1.2 if values else 100))
    ax.set_ylabel("Contribution Share (%)")
    ax.set_title("Media vs Base Contribution", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_channel_contribution_share(
    contributions: dict,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Bar chart of channel-wise contribution share (% of total media contribution)."""
    contrib_df = contributions["channels"]
    channels = [c for c in contrib_df.columns if c not in ("base", "controls")]

    channel_totals = pd.Series({ch: float(contrib_df[ch].sum()) for ch in channels}).sort_values(ascending=False)
    media_total = float(channel_totals.sum())
    if media_total <= 0:
        pct = pd.Series(0.0, index=channel_totals.index)
    else:
        pct = (channel_totals / media_total) * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(pct))]
    labels = pct.index.tolist()
    values = pct.values.tolist()
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.set_ylabel("Contribution Share (%)")
    ax.set_title("Channel-wise Media Contribution Share", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    _save(fig, save_path)
    return fig
