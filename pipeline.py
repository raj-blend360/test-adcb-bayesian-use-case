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
import json
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
    p.add_argument("--no-adstock", action="store_true", help="Disable adstock transformation")
    p.add_argument("--no-saturation", action="store_true", help="Disable saturation transformation")
    p.add_argument("--tune", type=int, default=500, help="MCMC tuning steps")
    p.add_argument("--chains", type=int, default=2, help="MCMC chain count")
    p.add_argument("--cores", type=int, default=None, help="CPU cores for parallel chains (default: PyMC auto)")
    p.add_argument("--nuts-sampler", default="numpyro", choices=["numpyro", "blackjax", "pymc"], dest="nuts_sampler",
                   help="NUTS backend for MCMC (numpyro is usually fastest)")
    p.add_argument("--nuts-init", default="jitter+adapt_diag", dest="nuts_init",
                   help="NUTS initialization strategy")
    p.add_argument("--weeks", type=int, default=104, help="Synthetic dataset weeks")
    p.add_argument("--channel-inputs", nargs="*", default=[], dest="channel_inputs",
                   help="Per-channel input metric as Channel:metric (metric in {impressions,clicks,spends})")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-plots", dest="no_plots", action="store_true", help="Skip saving plots")
    p.add_argument("--no-bounds", dest="no_bounds", action="store_true", help="Disable ±30%% bounds")
    p.add_argument("--target", type=float, default=None, help="Target conversions for reverse optimization")
    p.add_argument(
        "--optimization-level",
        choices=["monthly", "annual"],
        default="annual",
        dest="optimization_level",
        help="Budget optimization horizon used for spend scaling and optimizer outputs.",
    )
    p.add_argument("--freeze", nargs="*", default=[], help="Channels to freeze (e.g. --freeze TV OOH)")
    p.add_argument("--halo", nargs="*", default=None, help="Channel-level halo pairs e.g. TV,Digital Radio,Digital (overridden by --halo-config)")
    p.add_argument("--halo-config", default=None, dest="halo_config", help="Path to halo_config.json with channel- and campaign-level halo pairs")
    p.add_argument("--min-halo-spend", type=float, default=0.0, dest="min_halo_spend", help="Minimum total campaign spend to be eligible as halo candidate (raw currency units)")
    p.add_argument("--halo-top-n", type=int, default=10, dest="halo_top_n", help="Number of top halo candidates to print in analysis step")
    p.add_argument("--output-dir", default="outputs", help="Directory for saved outputs")
    p.add_argument("--input-csv", default=None, dest="input_csv",
                   help="Path to user input CSV (single file, wide or long format; skips synthetic data generation)")
    p.add_argument("--channel-csv", default=None, dest="channel_csv",
                   help="[Deprecated] Path to channel-level CSV. Use --input-csv instead.")
    p.add_argument("--campaign-csv", default=None, dest="campaign_csv",
                   help="Optional campaign-level CSV (only needed for campaign-level halo effects)")
    p.add_argument(
        "--date-format",
        default="%d-%m-%Y",
        dest="date_format",
        help="Date parsing format for input CSV files (used with --input-csv/--campaign-csv)",
    )
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


def _load_halo_config(path: str) -> dict:
    import json
    with open(path) as f:
        return json.load(f)


def _resolve_halo_from_args(args) -> tuple[list, list, float]:
    """Parse CLI args / halo config into (channel_pairs, campaign_pairs, min_spend)."""
    if args.halo_config:
        config = _load_halo_config(args.halo_config)
        ch_pairs = [
            (p["channel_a"], p["channel_b"])
            for p in config.get("channel_halo_pairs", [])
        ]
        camp_pairs = [
            (p["campaign_a"], p["campaign_b"])
            for p in config.get("campaign_halo_pairs", [])
        ]
        min_spend = config.get("min_halo_spend", args.min_halo_spend)
        return ch_pairs, camp_pairs, float(min_spend)
    else:
        ch_pairs = _parse_halo_pairs(args.halo or ["TV,Digital"])
        return ch_pairs, [], args.min_halo_spend


def _parse_channel_inputs(raw: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    valid = {"impressions", "clicks", "spends"}
    for item in raw:
        if ":" not in item:
            continue
        channel, metric = item.split(":", 1)
        channel = channel.strip()
        metric = metric.strip().lower()
        if channel and metric in valid:
            mapping[channel] = metric
    return mapping


def _apply_channel_inputs(channel_df: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    df = channel_df.copy()
    used: dict[str, str] = {}
    if "media_input" not in df.columns:
        df["media_input"] = df["media_spend"]

    col_map = {"spends": "media_spend", "clicks": "clicks", "impressions": "impressions"}
    for ch in sorted(df["channel"].unique()):
        metric = mapping.get(ch, "clicks")
        src_col = col_map[metric]
        if src_col not in df.columns:
            print(f"  [warn] Channel '{ch}' requested '{metric}' but '{src_col}' is missing. Falling back to clicks/spends.")
            if "clicks" in df.columns:
                metric = "clicks"
                src_col = "clicks"
            else:
                metric = "spends"
                src_col = "media_spend"
        mask = df["channel"] == ch
        df.loc[mask, "media_input"] = (
            pd.to_numeric(df.loc[mask, src_col], errors="coerce").fillna(0.0).clip(lower=0.0)
        )
        used[ch] = metric

    return df, used


def _normalize_channel_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Accept either long-format or wide-format input and return channel long-format.

    Supported wide columns include patterns such as:
      - spends_<channel>
      - media_impressions_<channel>
      - media_clicks_<channel>
      - exogenous_<control_name>

    Example expected row shape:
      date, spends_channel1, media_impressions_channel1, media_clicks_channel1,
      exogenous_holiday_flag, exogenous_event1
    """
    df = raw_df.copy()
    if "date" not in df.columns:
        raise ValueError("Input CSV must include a 'date' column.")

    # Already long format
    if {"channel", "media_spend"}.issubset(df.columns):
        return df

    spend_cols = [c for c in df.columns if c.startswith("spends_")]
    if not spend_cols:
        raise ValueError(
            "Unsupported channel CSV schema. Provide long format "
            "(date, channel, media_spend, ...), or wide columns like 'spends_<channel>'."
        )

    def _get_numeric_column_or_default(col_name: str) -> pd.Series:
        if col_name in df.columns:
            return pd.to_numeric(df[col_name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=df.index)

    channel_rows: list[pd.DataFrame] = []
    exo_cols = [c for c in df.columns if c.startswith("exogenous_")]
    for spend_col in spend_cols:
        channel = spend_col.replace("spends_", "", 1)
        temp = pd.DataFrame(
            {
                "date": df["date"],
                "channel": channel,
                "media_spend": pd.to_numeric(df[spend_col], errors="coerce").fillna(0.0),
                "impressions": _get_numeric_column_or_default(f"media_impressions_{channel}")
                + _get_numeric_column_or_default(f"impressions_{channel}"),
                "clicks": _get_numeric_column_or_default(f"media_clicks_{channel}")
                + _get_numeric_column_or_default(f"clicks_{channel}"),
            }
        )
        channel_rows.append(temp)

    long_df = pd.concat(channel_rows, ignore_index=True)

    # Copy exogenous controls to the row level (same weekly value across channels)
    for exo_col in exo_cols:
        clean_name = exo_col.replace("exogenous_", "", 1)
        long_df[clean_name] = long_df["date"].map(df.set_index("date")[exo_col].to_dict())

    # Alias common user-provided exogenous names to model default control names
    if "holiday_flag" not in long_df.columns and "holiday_flag" in [c.replace("exogenous_", "", 1) for c in exo_cols]:
        pass
    if "promo_flag" not in long_df.columns and "event1" in long_df.columns:
        long_df["promo_flag"] = long_df["event1"]

    # Conversions are required by downstream pipeline; default to 0 if not provided.
    if "conversions" in df.columns:
        long_df["conversions"] = long_df["date"].map(df.set_index("date")["conversions"].to_dict())
    else:
        long_df["conversions"] = 0.0

    return long_df


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


def step_load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    _section("STEP 1: Load Real Data")
    input_csv = args.input_csv or args.channel_csv
    if not input_csv:
        raise ValueError("Provide --input-csv (preferred) or --channel-csv.")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    raw_input_df = pd.read_csv(input_csv)
    print("  Input columns :")
    for col in raw_input_df.columns:
        print(f"    - {col}")

    channel_df = _normalize_channel_dataframe(raw_input_df)
    print("  Normalized columns :")
    for col in channel_df.columns:
        print(f"    - {col}")
    channel_df["date"] = pd.to_datetime(channel_df["date"], format=args.date_format)

    campaign_df = None
    if args.campaign_csv:
        if not os.path.exists(args.campaign_csv):
            raise FileNotFoundError(f"Campaign CSV not found: {args.campaign_csv}")
        campaign_df = pd.read_csv(args.campaign_csv)
        campaign_df["date"] = pd.to_datetime(campaign_df["date"], format=args.date_format)
        print(f"  Campaign rows : {len(campaign_df):,}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"  Channel rows  : {len(channel_df):,}")
    print(f"  Date range    : {channel_df['date'].min().date()} → {channel_df['date'].max().date()}")
    print(f"  Channels      : {sorted(channel_df['channel'].unique().tolist())}")

    total_spend = channel_df["media_spend"].sum()
    total_conv = channel_df.groupby("date")["conversions"].mean().sum()
    print(f"  Total spend   : {total_spend:,.0f}")
    print(f"  Total conv    : {total_conv:,.0f}")
    return campaign_df, channel_df


def step_preprocess(channel_df: pd.DataFrame, campaign_df: pd.DataFrame, args) -> "MMMDataset":
    _section("STEP 2: Data Pre-processing")

    requested_inputs = _parse_channel_inputs(args.channel_inputs)
    channel_df, used_inputs = _apply_channel_inputs(channel_df, requested_inputs)

    requested_controls: list[str] = []
    reserved_cols = {"date", "channel", "media_spend", "media_input", "conversions", "impressions", "clicks", "campaign"}
    candidate_control_cols = [
        c for c in channel_df.columns
        if c not in reserved_cols and (c.endswith("_flag") or c.startswith("exogenous_") or c.startswith("control_"))
    ]
    for control_col in candidate_control_cols:
        control_values = pd.to_numeric(channel_df[control_col], errors="coerce").fillna(0.0)
        if float(control_values.abs().sum()) > 0.0:
            requested_controls.append(control_col)

    print("  Requested control flags:", requested_controls or "none")

    cfg = DataConfig(
        spend_col="media_input",
        test_weeks=12,
        scale_spend=True,
        scale_target=True,
        include_seasonality=True,
        seasonality_periods=[52.0, 26.0],
        n_harmonics=2,
        control_cols=requested_controls,
    )
    processor = DataProcessor(cfg)
    dataset = processor.prepare(channel_df, campaign_df=campaign_df)

    print(f"  Time steps    : {dataset.n_time}")
    print(f"  Channels      : {dataset.n_channels}  → {dataset.channel_names}")
    print(f"  Controls      : {dataset.n_controls}  → {dataset.control_names[:4]}…")
    print(f"  Train weeks   : {dataset.train_mask.sum()}")
    print(f"  Test weeks    : {dataset.test_mask.sum()}")
    print("  Input metric by channel:")
    for ch in dataset.channel_names:
        print(f"    {ch:<20} {used_inputs.get(ch, 'clicks')}")
    print("  Transformations applied:")
    print(f"    spend scaling         : {'ON' if cfg.scale_spend else 'OFF'}")
    print(f"    target scaling        : {'ON' if cfg.scale_target else 'OFF'}")
    print(f"    seasonality features  : {'ON' if cfg.include_seasonality else 'OFF'}")
    return dataset


def step_halo_analysis(dataset, args) -> pd.DataFrame:
    _section("STEP 2b: Halo Candidate Analysis")

    from src.halo_analysis import rank_halo_candidates, plot_halo_heatmap

    if dataset.campaign_spend_matrix is None:
        print("  No campaign spend data available. Skipping halo analysis.")
        return pd.DataFrame()

    # Use typical adstock decay rates as a prior for the scoring step
    channel_decay = {ch: 0.4 for ch in dataset.channel_names}

    scored_df = rank_halo_candidates(
        campaign_df=dataset.campaign_df,
        channel_decay=channel_decay,
        min_halo_spend=args.min_halo_spend,
        top_n=args.halo_top_n,
    )

    if scored_df.empty:
        print(f"  No candidates found (min_halo_spend=£{args.min_halo_spend:,.0f}).")
    else:
        print(f"\n  Top {len(scored_df)} halo candidates (cross-channel pairs):\n")
        display_cols = [
            "campaign_a", "channel_a", "campaign_b", "channel_b",
            "adstock_correlation", "spend_overlap", "min_total_spend", "composite_score",
        ]
        print(scored_df[display_cols].to_string(index=False))

        if not args.no_plots:
            heatmap_path = os.path.join(args.output_dir, "plots", "halo_candidates.png")
            fig = plot_halo_heatmap(scored_df, save_path=heatmap_path)
            plt.close(fig)

    return scored_df


def step_fit_model(dataset, args) -> "MMMResults":
    _section("STEP 3: Fit Bayesian MMM")

    ch_halo_pairs, camp_halo_pairs, min_halo_spend = _resolve_halo_from_args(args)
    inference = "map" if args.fast else ("advi" if args.advi else "mcmc")

    cfg = ModelConfig(
        apply_adstock=not args.no_adstock,
        apply_saturation=not args.no_saturation,
        adstock_max_lag=8,
        beta_prior_sigma=0.3,
        halo_pairs=ch_halo_pairs,
        campaign_halo_pairs=camp_halo_pairs,
        min_halo_spend=min_halo_spend,
        n_samples=args.samples,
        n_tune=args.tune,
        n_chains=args.chains,
        target_accept=0.90,
        random_seed=args.seed,
        cores=args.cores,
        nuts_sampler=args.nuts_sampler,
        nuts_init=args.nuts_init,
        inference=inference,
    )

    print(f"  Inference          : {inference}")
    print(f"  Adstock            : {'ON (geometric)' if cfg.apply_adstock else 'OFF'}")
    print(f"  Saturation         : {'ON (Hill)' if cfg.apply_saturation else 'OFF'}")
    print(f"  Channel halo pairs : {ch_halo_pairs}")
    print(f"  Campaign halo pairs: {camp_halo_pairs}")
    if inference == "mcmc":
        print(f"  Samples            : {args.samples} × {args.chains} chains")
        print(f"  NUTS backend       : {args.nuts_sampler}")

    mmm = BayesianMMM(cfg)
    t0 = time.time()
    results = mmm.fit(dataset)
    elapsed = time.time() - t0
    print(f"  Fit time           : {elapsed:.1f}s")

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
        print(f"  Train WMAPE: {oos['train_wmape']:.2f}%")
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

    current_spend_weekly = dataset.spend_raw.mean(axis=0)  # average weekly spend per channel
    period_factor = 12 if args.optimization_level == "monthly" else 52
    period_label = "monthly" if args.optimization_level == "monthly" else "annual"
    current_spend_period = current_spend_weekly * period_factor
    total_budget = current_spend_period.sum()

    print(f"  Total {period_label} budget : £{total_budget:,.0f}")
    print(f"  Optimization level  : {period_label}")
    print(f"  Frozen channels     : {args.freeze or 'none'}")
    print(f"  Bounds ±30%         : {'ON' if not args.no_bounds else 'OFF'}")

    # Annual optimization
    opt_result = optimizer.optimize_budget(
        channel_params,
        total_budget,
        current_spend_period,
        campaign_df=campaign_df if campaign_df is not None and not campaign_df.empty else None,
    )

    print(f"\n  Optimization success : {opt_result.success}")
    print(f"  Message              : {opt_result.message}")
    print(f"  Current conversions  : {opt_result.current_conversions:,.1f}")
    print(f"  Optimal conversions  : {opt_result.optimal_conversions:,.1f}")
    print(f"  Uplift               : +{opt_result.conversion_uplift_pct:.1f}%")

    print("\n  Per-channel allocation:")
    alloc_df = opt_result.to_dataframe()
    channel_current_conv = np.array(
        [p.conversions(np.array([s]))[0] for p, s in zip(channel_params, opt_result.current_spend)]
    )
    channel_opt_conv = np.array(
        [p.conversions(np.array([s]))[0] for p, s in zip(channel_params, opt_result.optimal_spend)]
    )
    conv_change_pct = (channel_opt_conv - channel_current_conv) / (channel_current_conv + 1e-8) * 100
    alloc_df["current_conversions"] = channel_current_conv
    alloc_df["optimal_conversions"] = channel_opt_conv
    alloc_df["conversion_change_pct"] = conv_change_pct
    print(alloc_df[["current_spend", "optimal_spend", "spend_change_pct"]].to_string())
    alloc_df.to_csv(os.path.join(args.output_dir, "budget_allocation.csv"))
    excel_df = pd.DataFrame(
        {
            "Channel": opt_result.channel_names,
            "Avg Monthly Spend": current_spend_weekly * 4.34524,
            "Optimised Spends": opt_result.optimal_spend,
            "% Change in Spend": opt_result.spend_change_pct,
            "Avg Conversions": channel_current_conv,
            "Optimised Conversions": channel_opt_conv,
            "% Change in Conversions": conv_change_pct,
            "Optimization Level": args.optimization_level,
        }
    )
    excel_path = os.path.join(args.output_dir, "optimization_results.xlsx")
    with pd.ExcelWriter(excel_path) as writer:
        excel_df.to_excel(writer, index=False, sheet_name="summary")
        alloc_df.reset_index().to_excel(writer, index=False, sheet_name="allocation_detail")

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
        current_spend_period,
    )
    print(f"  Required total spend : £{rev_result.optimal_spend.sum():,.0f}")
    print(f"  Achieved conversions : {rev_result.optimal_conversions:,.1f}")
    rev_alloc_df = rev_result.to_dataframe()
    rev_alloc_df.to_csv(os.path.join(args.output_dir, "reverse_allocation.csv"))

    # Marginal ROI table
    marginal_df = optimizer.marginal_roi_table(channel_params, current_spend_period)
    print("\n  Marginal ROI at current spend:")
    print(marginal_df[["spend", "roi", "marginal_roi", "cost_per_conversion"]].to_string())
    marginal_df.to_csv(os.path.join(args.output_dir, "marginal_roi.csv"))

    # Efficient frontier
    frontier_df = optimizer.efficient_frontier(
        channel_params, current_spend_period, n_points=15
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

    print("  Plotting media vs base contribution …")
    fig = viz.plot_media_vs_base_contribution(
        contributions, save_path=sp("media_vs_base_contribution.png")
    )
    plt.close(fig)

    print("  Plotting channel-wise contribution share …")
    fig = viz.plot_channel_contribution_share(
        contributions, save_path=sp("channel_wise_contribution_share.png")
    )
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

    if args.input_csv or args.channel_csv:
        # User-provided --input-csv is only used as input for the Bayesian MMM flow.
        # We intentionally skip synthetic generation and optional halo candidate discovery
        # so the provided file directly drives model fitting and downstream outputs.
        campaign_df, channel_df = step_load_data(args)
    else:
        campaign_df, channel_df = step_generate_data(args)
    dataset = step_preprocess(channel_df, campaign_df, args)
    if not (args.input_csv or args.channel_csv):
        step_halo_analysis(dataset, args)
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
    summary = {
        "status": "ok",
        "inference": "map" if args.fast else ("advi" if args.advi else "mcmc"),
        "weeks": args.weeks,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_total, 2),
        "output_dir": args.output_dir,
        "used_real_data": bool(args.input_csv or args.channel_csv),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "pipeline_run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    _section(f"PIPELINE COMPLETE  ({elapsed_total:.1f}s)")
    print(f"  Outputs → {args.output_dir}/")
    print(f"  Run summary → {args.output_dir}/pipeline_run_summary.json")
    print()


if __name__ == "__main__":
    main()
