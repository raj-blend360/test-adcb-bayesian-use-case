"""
Constrained budget optimizer for Bayesian MMM.

Supports:
  1. Annual budget optimization — maximize conversions given total budget.
  2. Reverse optimization  — find spend needed to hit a target conversion count.
  3. Channel → Campaign allocation of optimized channel budgets.
  4. Flexible bounds (±30% default, toggleable).
  5. Channel-level max-increase cap (60% safety rail).
  6. Channel freezing (spend kept at current level).

Uses scipy.optimize.minimize (SLSQP) for continuous constrained optimization
and falls back to cvxpy for convex reformulations when applicable.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult


# ---------------------------------------------------------------------------
# Response curve helpers
# ---------------------------------------------------------------------------


def _hill_conversions(
    spend: np.ndarray,
    alpha: float,
    gamma: float,
    beta: float,
    x_ref_max: float,
) -> np.ndarray:
    """Scalar or array Hill-based conversions from raw spend."""
    x_norm = spend / (x_ref_max + 1e-8)
    sat = x_norm**alpha / (x_norm**alpha + gamma**alpha + 1e-12)
    return beta * sat


def _logistic_conversions(
    spend: np.ndarray,
    lam: float,
    beta: float,
) -> np.ndarray:
    return beta * (1 - np.exp(-lam * spend)) / (1 + np.exp(-lam * spend) + 1e-12)


def _mm_conversions(
    spend: np.ndarray,
    vmax: float,
    km: float,
    beta: float,
) -> np.ndarray:
    return beta * vmax * spend / (km + spend + 1e-8)


# ---------------------------------------------------------------------------
# ChannelParams
# ---------------------------------------------------------------------------


@dataclass
class ChannelParams:
    """Posterior-mean parameters for one channel's response curve."""

    name: str
    saturation_type: str  # hill | logistic | michaelis_menten

    # Hill
    alpha: float = 2.0
    gamma: float = 0.4
    beta: float = 1.0
    x_ref_max: float = 1.0  # reference max spend (unscaled)

    # Logistic
    lam: float = 1.0

    # Michaelis-Menten
    vmax: float = 1.0
    km: float = 1.0

    def conversions(self, spend: np.ndarray) -> np.ndarray:
        """Predicted conversions at given spend levels."""
        if self.saturation_type == "hill":
            return _hill_conversions(spend, self.alpha, self.gamma, self.beta, self.x_ref_max)
        elif self.saturation_type == "logistic":
            return _logistic_conversions(spend, self.lam, self.beta)
        else:
            return _mm_conversions(spend, self.vmax, self.km, self.beta)

    def marginal_conversion(self, spend: float, delta: float = 1.0) -> float:
        """∂conversions / ∂spend at given spend level."""
        hi = float(self.conversions(np.array([spend + delta]))[0])
        lo = float(self.conversions(np.array([spend]))[0])
        return (hi - lo) / delta


# ---------------------------------------------------------------------------
# OptimizerConfig
# ---------------------------------------------------------------------------


@dataclass
class OptimizerConfig:
    """Settings for the budget optimizer."""

    # Bounds settings
    use_bounds: bool = True
    bounds_pct: float = 0.30            # ±30% default

    # Safety cap on increase
    max_increase_pct: float = 0.60      # 60% max increase per channel

    # Channels to freeze (names)
    frozen_channels: list[str] = field(default_factory=list)

    # Optimization solver settings
    method: str = "SLSQP"
    max_iter: int = 1000
    tol: float = 1e-8

    # Campaign allocation method: "proportional" | "response"
    campaign_allocation: str = "proportional"


# ---------------------------------------------------------------------------
# OptimizationResult
# ---------------------------------------------------------------------------


@dataclass
class OptimizationResult:
    """Output from the optimizer."""

    success: bool
    message: str

    # Per-channel results
    channel_names: list[str]
    current_spend: np.ndarray      # baseline spend
    optimal_spend: np.ndarray      # optimized spend
    spend_change_pct: np.ndarray   # % change

    # Conversion predictions
    current_conversions: float
    optimal_conversions: float
    conversion_uplift_pct: float

    # Campaign breakdown (optional)
    campaign_allocation: Optional[pd.DataFrame] = None

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "channel": self.channel_names,
                "current_spend": self.current_spend,
                "optimal_spend": self.optimal_spend,
                "spend_change": self.optimal_spend - self.current_spend,
                "spend_change_pct": self.spend_change_pct,
            }
        )
        df["current_conversions_contribution"] = [
            0.0
        ] * len(df)  # filled in by caller
        return df.set_index("channel")


# ---------------------------------------------------------------------------
# BudgetOptimizer
# ---------------------------------------------------------------------------


class BudgetOptimizer:
    """Constrained budget optimizer for MMM response curves.

    Usage
    -----
    >>> params = optimizer.extract_channel_params(results)
    >>> opt_result = optimizer.optimize_budget(params, total_budget, current_spend)
    >>> rev_result = optimizer.reverse_optimize(params, target_conversions, current_spend)
    """

    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()

    # ------------------------------------------------------------------
    # Extract parameters from MMMResults
    # ------------------------------------------------------------------

    def extract_channel_params(self, results) -> list[ChannelParams]:
        """Pull posterior mean parameters from MMMResults into ChannelParams."""
        idata = results.idata
        dataset = results.dataset
        cfg = results.config
        post = idata.posterior

        channel_params = []
        for c, ch in enumerate(dataset.channel_names):
            beta = float(post["beta"].mean(("chain", "draw")).values[c])
            x_ref_max = float(dataset.spend_raw[:, c].max()) + 1e-8

            if cfg.apply_saturation and "saturation" in post:
                gamma = float(np.asarray(post["saturation"].mean(("chain", "draw")).values).reshape(-1)[0])
                cp = ChannelParams(name=ch, saturation_type="hill", gamma=gamma, beta=beta, x_ref_max=x_ref_max)
            else:
                cp = ChannelParams(name=ch, saturation_type="hill", gamma=1.0, beta=beta, x_ref_max=1.0)
            channel_params.append(cp)

        return channel_params

    # ------------------------------------------------------------------
    # Annual budget optimization
    # ------------------------------------------------------------------

    def optimize_budget(
        self,
        channel_params: list[ChannelParams],
        total_budget: float,
        current_spend: np.ndarray,
        campaign_df: Optional[pd.DataFrame] = None,
    ) -> OptimizationResult:
        """Maximize total conversions for a fixed total budget.

        Args:
            channel_params: list of ChannelParams (one per channel).
            total_budget: total spend budget to allocate.
            current_spend: baseline spend array (one per channel).
            campaign_df: optional DataFrame with campaign-level spend for allocation.

        Returns:
            OptimizationResult.
        """
        cfg = self.config
        n = len(channel_params)
        channel_names = [p.name for p in channel_params]
        frozen_mask = np.array(
            [ch in cfg.frozen_channels for ch in channel_names], dtype=bool
        )

        # Variable channels only
        free_idx = np.where(~frozen_mask)[0]
        frozen_idx = np.where(frozen_mask)[0]
        frozen_spend = current_spend[frozen_idx].sum()
        free_budget = total_budget - frozen_spend

        if free_budget < 0:
            warnings.warn("Frozen channels consume more than total_budget. Capping.")
            free_budget = 0.0

        free_current = current_spend[free_idx]
        free_params = [channel_params[i] for i in free_idx]

        bounds = self._build_bounds(free_params, free_current, reverse=False)
        x0 = np.clip(free_current, [b[0] for b in bounds], [b[1] for b in bounds])

        def neg_conversions(x: np.ndarray) -> float:
            return -sum(p.conversions(np.array([xi]))[0] for p, xi in zip(free_params, x))

        def neg_conversions_grad(x: np.ndarray) -> np.ndarray:
            grad = np.zeros(len(x))
            delta = total_budget * 1e-5
            for i in range(len(x)):
                xi_hi = x.copy()
                xi_hi[i] += delta
                xi_lo = x.copy()
                xi_lo[i] -= delta
                grad[i] = (neg_conversions(xi_hi) - neg_conversions(xi_lo)) / (2 * delta)
            return grad

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: x.sum() - free_budget,
                "jac": lambda x: np.ones(len(x)),
            }
        ]

        result: OptimizeResult = minimize(
            neg_conversions,
            x0,
            jac=neg_conversions_grad,
            method=cfg.method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
        )

        # Rebuild full spend vector
        optimal_full = current_spend.copy().astype(float)
        optimal_full[free_idx] = np.clip(
            result.x, [b[0] for b in bounds], [b[1] for b in bounds]
        )

        change_pct = (optimal_full - current_spend) / (current_spend + 1e-8) * 100

        current_conv = sum(
            p.conversions(np.array([current_spend[i]]))[0]
            for i, p in enumerate(channel_params)
        )
        optimal_conv = sum(
            p.conversions(np.array([optimal_full[i]]))[0]
            for i, p in enumerate(channel_params)
        )
        uplift = (optimal_conv - current_conv) / (current_conv + 1e-8) * 100

        campaign_alloc = None
        if campaign_df is not None:
            campaign_alloc = self._allocate_to_campaigns(
                channel_names, optimal_full, campaign_df, channel_params
            )

        return OptimizationResult(
            success=result.success,
            message=result.message,
            channel_names=channel_names,
            current_spend=current_spend.astype(float),
            optimal_spend=optimal_full,
            spend_change_pct=change_pct,
            current_conversions=float(current_conv),
            optimal_conversions=float(optimal_conv),
            conversion_uplift_pct=float(uplift),
            campaign_allocation=campaign_alloc,
        )

    # ------------------------------------------------------------------
    # Reverse optimization
    # ------------------------------------------------------------------

    def reverse_optimize(
        self,
        channel_params: list[ChannelParams],
        target_conversions: float,
        current_spend: np.ndarray,
        campaign_df: Optional[pd.DataFrame] = None,
    ) -> OptimizationResult:
        """Minimize total spend to achieve a target conversion count.

        No upper bounds on spend (only lower bounds = 0), so the optimizer
        can freely increase channels as needed to hit the target.

        Args:
            channel_params: list of ChannelParams.
            target_conversions: desired total conversions.
            current_spend: baseline spend (used as starting point and for lower bounds).
            campaign_df: optional campaign-level data for allocation.

        Returns:
            OptimizationResult with total spend required.
        """
        cfg = self.config
        n = len(channel_params)
        channel_names = [p.name for p in channel_params]

        # For reverse optimization: no upper bounds
        bounds = [(0.0, None) for _ in range(n)]
        x0 = current_spend.copy().astype(float)

        def total_spend(x: np.ndarray) -> float:
            return x.sum()

        def spend_grad(x: np.ndarray) -> np.ndarray:
            return np.ones(n)

        def conv_constraint(x: np.ndarray) -> float:
            return (
                sum(p.conversions(np.array([xi]))[0] for p, xi in zip(channel_params, x))
                - target_conversions
            )

        result: OptimizeResult = minimize(
            total_spend,
            x0,
            jac=spend_grad,
            method=cfg.method,
            bounds=bounds,
            constraints=[{"type": "eq", "fun": conv_constraint}],
            options={"maxiter": cfg.max_iter, "ftol": cfg.tol},
        )

        optimal_full = np.clip(result.x, 0, None)
        change_pct = (optimal_full - current_spend) / (current_spend + 1e-8) * 100

        current_conv = sum(
            p.conversions(np.array([current_spend[i]]))[0]
            for i, p in enumerate(channel_params)
        )
        optimal_conv = sum(
            p.conversions(np.array([optimal_full[i]]))[0]
            for i, p in enumerate(channel_params)
        )
        uplift = (optimal_conv - current_conv) / (current_conv + 1e-8) * 100

        campaign_alloc = None
        if campaign_df is not None:
            campaign_alloc = self._allocate_to_campaigns(
                channel_names, optimal_full, campaign_df, channel_params
            )

        return OptimizationResult(
            success=result.success,
            message=result.message,
            channel_names=channel_names,
            current_spend=current_spend.astype(float),
            optimal_spend=optimal_full,
            spend_change_pct=change_pct,
            current_conversions=float(current_conv),
            optimal_conversions=float(optimal_conv),
            conversion_uplift_pct=float(uplift),
            campaign_allocation=campaign_alloc,
        )

    # ------------------------------------------------------------------
    # Campaign allocation
    # ------------------------------------------------------------------

    def _allocate_to_campaigns(
        self,
        channel_names: list[str],
        channel_spend: np.ndarray,
        campaign_df: pd.DataFrame,
        channel_params: list[ChannelParams],
    ) -> pd.DataFrame:
        """Distribute optimized channel spend to campaigns.

        Two methods:
          - 'proportional': keep historical spend ratios within the channel.
          - 'response': weight by each campaign's marginal ROI.

        Args:
            channel_names: list of channel names in order.
            channel_spend: optimized spend per channel.
            campaign_df: campaign-level historical spend DataFrame.
                         Expected columns: ['channel', 'campaign', 'media_spend'].
            channel_params: ChannelParams list for response-based weighting.

        Returns:
            DataFrame with campaign-level optimal spend.
        """
        cfg = self.config
        rows = []

        for ch, opt_spend in zip(channel_names, channel_spend):
            ch_campaigns = campaign_df[campaign_df["channel"] == ch]
            if ch_campaigns.empty:
                continue

            camp_spend = (
                ch_campaigns.groupby("campaign")["media_spend"].sum()
            )
            camp_names = camp_spend.index.tolist()
            hist_spend = camp_spend.values.astype(float)

            if cfg.campaign_allocation == "proportional":
                weights = hist_spend / (hist_spend.sum() + 1e-8)
            else:
                # Response-based: weight by marginal conversion at current spend
                cp = next((p for p in channel_params if p.name == ch), None)
                if cp is None:
                    weights = hist_spend / (hist_spend.sum() + 1e-8)
                else:
                    marginals = np.array(
                        [cp.marginal_conversion(float(s)) for s in hist_spend]
                    )
                    marginals = np.clip(marginals, 0, None)
                    weights = marginals / (marginals.sum() + 1e-8)

            allocated = opt_spend * weights

            for camp, alloc, hist in zip(camp_names, allocated, hist_spend):
                rows.append(
                    {
                        "channel": ch,
                        "campaign": camp,
                        "current_spend": hist,
                        "optimal_spend": alloc,
                        "spend_change": alloc - hist,
                        "spend_change_pct": (alloc - hist) / (hist + 1e-8) * 100,
                    }
                )

        return pd.DataFrame(rows).set_index(["channel", "campaign"])

    # ------------------------------------------------------------------
    # Bounds builder
    # ------------------------------------------------------------------

    def _build_bounds(
        self,
        params: list[ChannelParams],
        current_spend: np.ndarray,
        reverse: bool = False,
    ) -> list[tuple[float, Optional[float]]]:
        cfg = self.config
        bounds = []

        for i, (p, cs) in enumerate(zip(params, current_spend)):
            if reverse:
                # No upper bound for reverse optimization
                bounds.append((0.0, None))
                continue

            lb = 0.0
            ub = None

            if cfg.use_bounds:
                lb = max(0.0, cs * (1 - cfg.bounds_pct))
                ub = cs * (1 + cfg.bounds_pct)

            # Safety cap: never increase by more than max_increase_pct
            max_ub = cs * (1 + cfg.max_increase_pct)
            if ub is None:
                ub = max_ub
            else:
                ub = min(ub, max_ub)

            bounds.append((lb, ub))

        return bounds

    # ------------------------------------------------------------------
    # Efficient frontier
    # ------------------------------------------------------------------

    def efficient_frontier(
        self,
        channel_params: list[ChannelParams],
        current_spend: np.ndarray,
        budget_range: Optional[np.ndarray] = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        """Compute optimal conversions across a range of total budgets.

        Args:
            channel_params: ChannelParams list.
            current_spend: baseline spend.
            budget_range: array of total budget values to sweep.
                          Defaults to 50% – 150% of current total spend.
            n_points: number of budget points.

        Returns:
            DataFrame with columns: total_budget, optimal_conversions, channel allocations.
        """
        current_total = current_spend.sum()
        if budget_range is None:
            budget_range = np.linspace(
                current_total * 0.5, current_total * 1.5, n_points
            )

        rows = []
        for budget in budget_range:
            try:
                result = self.optimize_budget(channel_params, budget, current_spend)
                row = {"total_budget": budget, "optimal_conversions": result.optimal_conversions}
                for ch, sp in zip(result.channel_names, result.optimal_spend):
                    row[f"spend_{ch}"] = sp
                rows.append(row)
            except Exception as e:
                warnings.warn(f"Frontier point budget={budget:.0f} failed: {e}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Marginal ROI table
    # ------------------------------------------------------------------

    def marginal_roi_table(
        self,
        channel_params: list[ChannelParams],
        spend: np.ndarray,
    ) -> pd.DataFrame:
        """Compute marginal ROI (dConversions/dSpend) at given spend levels."""
        rows = []
        for p, s in zip(channel_params, spend):
            marginal = p.marginal_conversion(float(s), delta=max(s * 0.01, 1.0))
            total_conv = float(p.conversions(np.array([s]))[0])
            roi = total_conv / (s + 1e-8)
            rows.append(
                {
                    "channel": p.name,
                    "spend": s,
                    "total_conversions": total_conv,
                    "roi": roi,
                    "marginal_roi": marginal,
                    "cost_per_conversion": s / (total_conv + 1e-8),
                }
            )
        return pd.DataFrame(rows).set_index("channel")
