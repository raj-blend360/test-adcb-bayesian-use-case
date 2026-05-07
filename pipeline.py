"""
End-to-end Bayesian MMM pipeline.

This script demonstrates the full workflow:

  1. Generate synthetic data
  2. Pre-process into MMMDataset
  3. Fit Bayesian MMM (PyMC)
  4. Compute contributions, response curves, ROI
  5. Run diagnostics
  6. Optimize budget (annual + reverse)
  7. Generate all visualizations

Usage
-----
    python pipeline.py                     # full MCMC (slow, accurate)
    python pipeline.py --fast              # MAP point-estimate (seconds)
    python pipeline.py --samples 200       # fewer MCMC samples
    python pipeline.py --no-plots          # skip saving plots
    python pipeline.py --target 50000      # reverse-optimize for 50 000 conversions

Assumptions / design choices
-----------------------------
- Weekly data aggregated to channel level for modelling.
- Hill saturation + geometric adstock (configurable via ModelConfig).
- Two Fourier harmonics for annual + semi-annual seasonality.
- Train / test split: last 12 weeks held out.
- Budget optimization: ±30% bounds, 60% safety cap on any single channel increase.
- Frozen channels: none by default (pass --freeze TV to freeze TV).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path for relative imports when running as script
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_synthetic_data import (
    generate_synthetic_data,
    generate_channel_weekly,
)
from src.data_processing import DataConfig, DataProcessor
from src.model import BayesianMMM, ModelConfig
from src.diagnostics import generate_diagnostic_report, out_of_sample_validation
from src.optimizer import BudgetOptimizer, OptimizerConfig
from src import visualization as viz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bayesian MMM end-to-end pipeline")
    p.add_argument("--fast", action="store_true", help="Use MAP inference (fast, no uncertainty)")
    p.add_argument("--advi", action="store_true", help="Use ADVI variational inference")
    p.add_argument("--samples", type=int, default=500, help="MCMC draw count per chain")
    p.add_argument("--tune", type=int, default=500, help="MCMC tuning steps")
    p.add_argument("--chains", type=int, default=2, help="MCMC chain count")
    p.add_argument("--weeks", type=int, default=104, help="Synthetic dataset weeks")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-plots", dest="no_plots", action="store_true", help="Skip saving plots")
    p.add_argument("--no-bounds", dest="no_bounds", action="store_true", help="Disable ±30% bounds")
    p.add_argument("--target", type=float, default=None, help="Target conversions for reverse optimization")
    p.add_argument("--freeze", nargs="*", default=[], help="Channels to freeze (e.g. --freeze TV OOH)")
    p.add_argument("--halo", nargs="*", default=["TV,Digital"], help="Halo pairs e.g. TV,Digital Radio,Digital")
    p.add_argument("--output-dir", default="outputs", help="Directory for saved outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _parse_halo_pairs(raw: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for item in raw:
        parts = item.split(",")
        if len(parts) == 2:
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_generate_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    _section("STEP 1: Generate Synthetic Data")
    campaign_df = generate_synthetic_data(n_weeks=args.weeks, seed=args.seed)
    channel_df = generate_channel_weekly(n_weeks=args.weeks, seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    campaign_df.to_csv(os.path.join(args.output_dir, "synthetic_campaign_data.csv"), index=False)
    channel_df.to_csv(os.path.join(args.output_dir, "synthetic_channel_data.csv"), index=False)

    print(f"  Campaign rows : {len(campaign_df):,}")
    print(f"  Channel rows  : {len(channel_df):,}")
    print(f"  Date range    : {channel_df['date'].min().date()} → {channel_df['date'].max().date()}")
    print(f"  Channels      : {sorted(channel_df['channel'].unique().tolist())}")

    total_spend = channel_df["media_spend"].sum()
    total_conv = channel_df.groupby("date")["conversions"].mean().sum()
    print(f"  Total spend   : £{total_spend:,.0f}")
    print(f"  Total conv    : {total_conv:,.0f}")
    return campaign_df, channel_df


def step_preprocess(channel_df: pd.DataFrame, campaign_df: pd.DataFrame, args) -> "MMMDataset":
    _section("STEP 2: Data Pre-processing")

    cfg = DataConfig(
        test_weeks=12,
        scale_spend=True,
        scale_target=True,
        include_seasonality=True,
        seasonality_periods=[52.0, 26.0],
        n_harmonics=2,
        control_cols=["holiday_flag", "promo_flag"],
    )
    processor = DataProcessor(cfg)
    dataset = processor.prepare(channel_df, campaign_df=campaign_df)

    print(f"  Time steps    : {dataset.n_time}")
    print(f"  Channels      : {dataset.n_channels}  → {dataset.channel_names}")
    print(f"  Controls      : {dataset.n_controls}  → {dataset.control_names[:4]}…")
    print(f"  Train weeks   : {dataset.train_mask.sum()}")
    print(f"  Test weeks    : {dataset.test_mask.sum()}")
    return dataset


def step_fit_model(dataset, args) -> "MMMResults":
    _section("STEP 3: Fit Bayesian MMM")

    halo_pairs = _parse_halo_pairs(args.halo)
    inference = "map" if args.fast else ("advi" if args.advi else "mcmc")

    cfg = ModelConfig(
        adstock_type="geometric",
        saturation_type="hill",
        adstock_max_lag=13,
        halo_pairs=halo_pairs,
        n_samples=args.samples,
        n_tune=args.tune,
        n_chains=args.chains,
        target_accept=0.90,
        random_seed=args.seed,
        inference=inference,
    )

    print(f"  Inference     : {inference}")
    print(f"  Adstock       : geometric (max_lag=13)")
    print(f"  Saturation    : Hill")
    print(f"  Halo pairs    : {halo_pairs}")
    if inference == "mcmc":
        print(f"  Samples       : {args.samples} × {args.chains} chains")

    mmm = BayesianMMM(cfg)
    t0 = time.time()
    results = mmm.fit(dataset)
    elapsed = time.time() - t0
    print(f"  Fit time      : {elapsed:.1f}s")

    return results, mmm


def step_contributions(results, mmm, args) -> dict:
    _section("STEP 4: Channel Contributions")

    contributions = mmm.get_contributions(results)
    channel_pct = contributions["channel_pct"]

    print("  Channel contribution %:")
    for ch, pct in sorted(channel_pct.items(), key=lambda x: -x[1]):
        print(f"    {ch:<20} {pct:>6.1f}%")

    roi_df = mmm.get_roi_metrics(results)
    print("\n  ROI metrics:")
    print(roi_df[["total_spend", "total_conversions", "roi", "cost_per_conversion"]].to_string())

    roi_df.to_csv(os.path.join(args.output_dir, "roi_metrics.csv"))
    contributions["channels"].to_csv(os.path.join(args.output_dir, "contributions.csv"))

    return contributions, roi_df


def step_response_curves(results, mmm, args) -> dict:
    _section("STEP 5: Response Curves")

    curves = mmm.get_response_curves(results, n_points=100, spend_multiplier=2.0)
    for ch, c in curves.items():
        curr_sp = c["current_spend"]
        curr_conv = c["current_conversions"]
        print(f"  {ch:<20}  spend={curr_sp:>10,.0f}  conv={curr_conv:>8.2f}")

    return curves


def step_diagnostics(results, args) -> dict:
    _section("STEP 6: Model Diagnostics")

    conv_df = generate_diagnostic_report(results)

    try:
        oos = out_of_sample_validation(results)
        print(f"\n  OOS MAPE: {oos['mape']:.2f}%  |  R²: {oos['r2']:.4f}")
    except Exception as e:
        oos = None
        print(f"  OOS validation skipped: {e}")

    conv_df.to_csv(os.path.join(args.output_dir, "convergence_summary.csv"))
    return conv_df, oos


def step_optimize(results, mmm, dataset, campaign_df, args) -> tuple:
    _section("STEP 7: Budget Optimization")

    channel_params = mmm.extract_channel_params(results)

    optimizer = BudgetOptimizer(
        OptimizerConfig(
            use_bounds=not args.no_bounds,
            bounds_pct=0.30,
            max_increase_pct=0.60,
            frozen_channels=args.freeze,
            campaign_allocation="proportional",
        )
    )

    current_spend = dataset.spend_raw.mean(axis=0)  # average weekly spend per channel
    # Annualise
    annual_spend = current_spend * 52
    total_budget = annual_spend.sum()

    print(f"  Total annual budget : £{total_budget:,.0f}")
    print(f"  Frozen channels     : {args.freeze or 'none'}")
    print(f"  Bounds ±30%         : {'ON' if not args.no_bounds else 'OFF'}")

    # Annual optimization
    opt_result = optimizer.optimize_budget(
        channel_params,
        total_budget,
        annual_spend,
        campaign_df=campaign_df if not campaign_df.empty else None,
    )

    print(f"\n  Optimization success : {opt_result.success}")
    print(f"  Message              : {opt_result.message}")
    print(f"  Current conversions  : {opt_result.current_conversions:,.1f}")
    print(f"  Optimal conversions  : {opt_result.optimal_conversions:,.1f}")
    print(f"  Uplift               : +{opt_result.conversion_uplift_pct:.1f}%")

    print("\n  Per-channel allocation:")
    alloc_df = opt_result.to_dataframe()
    print(alloc_df[["current_spend", "optimal_spend", "spend_change_pct"]].to_string())
    alloc_df.to_csv(os.path.join(args.output_dir, "budget_allocation.csv"))

    # Reverse optimization
    rev_result = None
    target_conv = args.target
    if target_conv is None:
        target_conv = opt_result.current_conversions * 1.20  # +20%
        print(f"\n  Running reverse optimization for +20% target: {target_conv:,.0f} conversions")
    else:
        print(f"\n  Running reverse optimization for target: {target_conv:,.0f} conversions")

    rev_result = optimizer.reverse_optimize(
        channel_params,
        target_conv,
        annual_spend,
    )
    print(f"  Required total spend : £{rev_result.optimal_spend.sum():,.0f}")
    print(f"  Achieved conversions : {rev_result.optimal_conversions:,.1f}")
    rev_alloc_df = rev_result.to_dataframe()
    rev_alloc_df.to_csv(os.path.join(args.output_dir, "reverse_allocation.csv"))

    # Marginal ROI table
    marginal_df = optimizer.marginal_roi_table(channel_params, annual_spend)
    print("\n  Marginal ROI at current spend:")
    print(marginal_df[["spend", "roi", "marginal_roi", "cost_per_conversion"]].to_string())
    marginal_df.to_csv(os.path.join(args.output_dir, "marginal_roi.csv"))

    # Efficient frontier
    frontier_df = optimizer.efficient_frontier(
        channel_params, annual_spend, n_points=15
    )
    frontier_df.to_csv(os.path.join(args.output_dir, "efficient_frontier.csv"), index=False)

    return opt_result, rev_result, channel_params, frontier_df, optimizer


def step_visualizations(
    results,
    contributions,
    roi_df,
    curves,
    conv_df,
    oos,
    opt_result,
    frontier_df,
    dataset,
    args,
) -> None:
    if args.no_plots:
        print("\n  Plots skipped (--no-plots).")
        return

    _section("STEP 8: Visualizations")
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def sp(name):
        return os.path.join(plot_dir, name)

    print("  Plotting contributions …")
    train_dates = dataset.dates[dataset.train_mask]
    fig = viz.plot_contributions(
        contributions, train_dates, save_path=sp("contributions.png")
    )
    plt.close(fig)

    print("  Plotting response curves …")
    fig = viz.plot_response_curves(curves, save_path=sp("response_curves.png"))
    plt.close(fig)

    print("  Plotting ROI metrics …")
    fig = viz.plot_roi_metrics(roi_df, save_path=sp("roi_metrics.png"))
    plt.close(fig)

    print("  Plotting budget allocation …")
    fig = viz.plot_budget_allocation(opt_result, save_path=sp("budget_allocation.png"))
    plt.close(fig)

    print("  Plotting diagnostics …")
    fig = viz.plot_diagnostics(conv_df, save_path=sp("diagnostics.png"))
    plt.close(fig)

    print("  Plotting posterior distributions …")
    try:
        fig = viz.plot_posterior_distributions(results, save_path=sp("posteriors.png"))
        plt.close(fig)
    except Exception as e:
        print(f"    Posterior plot skipped: {e}")

    print("  Plotting efficient frontier …")
    current_conv = opt_result.current_conversions
    current_budget = opt_result.current_spend.sum()
    fig = viz.plot_efficient_frontier(
        frontier_df,
        current_budget=current_budget,
        current_conversions=current_conv,
        save_path=sp("efficient_frontier.png"),
    )
    plt.close(fig)

    if opt_result.campaign_allocation is not None:
        print("  Plotting campaign allocation …")
        fig = viz.plot_campaign_allocation(
            opt_result.campaign_allocation, save_path=sp("campaign_allocation.png")
        )
        plt.close(fig)

    print("  Plotting actual vs predicted …")
    fig = viz.plot_actual_vs_predicted(
        results, oos_metrics=oos, save_path=sp("actual_vs_predicted.png")
    )
    plt.close(fig)

    print("  Plotting waterfall decomposition …")
    fig = viz.plot_waterfall_decomposition(contributions, save_path=sp("waterfall.png"))
    plt.close(fig)

    print(f"\n  All plots saved to: {plot_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*shape.*")

    print("\n" + "=" * 60)
    print("  BAYESIAN MMM — END-TO-END PIPELINE")
    print("=" * 60)
    print(f"  Weeks       : {args.weeks}")
    print(f"  Seed        : {args.seed}")
    print(f"  Inference   : {'MAP' if args.fast else ('ADVI' if args.advi else 'MCMC')}")
    print(f"  Output dir  : {args.output_dir}/")

    t_start = time.time()

    campaign_df, channel_df = step_generate_data(args)
    dataset = step_preprocess(channel_df, campaign_df, args)
    results, mmm = step_fit_model(dataset, args)
    contributions, roi_df = step_contributions(results, mmm, args)
    curves = step_response_curves(results, mmm, args)
    conv_df, oos = step_diagnostics(results, args)
    opt_result, rev_result, channel_params, frontier_df, optimizer = step_optimize(
        results, mmm, dataset, campaign_df, args
    )
    step_visualizations(
        results,
        contributions,
        roi_df,
        curves,
        conv_df,
        oos,
        opt_result,
        frontier_df,
        dataset,
        args,
    )

    elapsed_total = time.time() - t_start
    _section(f"PIPELINE COMPLETE  ({elapsed_total:.1f}s)")
    print(f"  Outputs → {args.output_dir}/")
    print()


if __name__ == "__main__":
    main()
