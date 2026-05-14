"""
Model diagnostics for Bayesian MMM.

Provides:
  - Convergence checks (R-hat, ESS)
  - Posterior predictive checks
  - Out-of-sample validation metrics
  - MAPE, RMSE, R² evaluation
  - Trace plot generation
  - Summary report
"""

from __future__ import annotations

from typing import Optional

import arviz as az
import numpy as np
import pandas as pd

from .data_processing import MMMDataset
from .model import MMMResults


def _safe_mape(observed: np.ndarray, predicted: np.ndarray, min_denom_ratio: float = 0.01) -> float:
    """MAPE with denominator floor to avoid exploding errors near zero actuals."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    scale = float(np.mean(np.abs(observed))) if observed.size else 0.0
    eps = max(1e-8, min_denom_ratio * scale)
    denom = np.maximum(np.abs(observed), eps)
    return float(np.mean(np.abs(observed - predicted) / denom) * 100)


# ---------------------------------------------------------------------------
# Convergence diagnostics
# ---------------------------------------------------------------------------


def check_convergence(
    results: MMMResults,
    rhat_threshold: float = 1.05,
    ess_threshold: int = 400,
) -> pd.DataFrame:
    """Compute R-hat and ESS for all sampled variables.

    For MAP inference, R-hat and ESS are not applicable (single point estimate);
    the function returns a summary with NaN for those columns and prints a note.

    Args:
        results: fitted MMMResults.
        rhat_threshold: flag variables with R-hat > threshold.
        ess_threshold: flag variables with ESS < threshold.

    Returns:
        DataFrame with convergence statistics.
    """
    idata = results.idata
    is_map = results.config.inference == "map"

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        summary = az.summary(idata, round_to=4)

    if is_map:
        print("  [NOTE] MAP inference: R-hat and ESS are not applicable for point estimates.")
        print("         Use inference='mcmc' or inference='advi' for convergence diagnostics.")

    rhat_col = "r_hat" if "r_hat" in summary.columns else None
    ess_col = "ess_bulk" if "ess_bulk" in summary.columns else None

    summary["rhat_ok"] = True
    summary["ess_ok"] = True

    if rhat_col:
        summary["rhat_ok"] = summary[rhat_col] <= rhat_threshold
    if ess_col:
        summary["ess_ok"] = summary[ess_col] >= ess_threshold

    summary["converged"] = summary["rhat_ok"] & summary["ess_ok"]

    n_bad = (~summary["converged"]).sum()
    if n_bad > 0:
        print(
            f"[WARNING] {n_bad} parameter(s) did not converge "
            f"(R-hat > {rhat_threshold} or ESS < {ess_threshold})."
        )
    else:
        print("[OK] All parameters converged.")

    return summary


# ---------------------------------------------------------------------------
# Posterior predictive checks
# ---------------------------------------------------------------------------


def posterior_predictive_check(
    results: MMMResults,
    n_samples: int = 200,
) -> dict:
    """Compare posterior predictive distribution to observed data.

    Returns:
        dict with keys:
          - 'ppc_mean': posterior predictive mean (T,)
          - 'ppc_hdi_low': 5th percentile (T,)
          - 'ppc_hdi_high': 95th percentile (T,)
          - 'observed': observed target (T,)
          - 'coverage_94': fraction of obs within 94% HDI
    """
    idata = results.idata
    dataset = results.dataset

    if "posterior_predictive" not in idata:
        raise ValueError(
            "No posterior predictive samples found. "
            "Re-fit with inference='mcmc' and ensure pm.sample_posterior_predictive was called."
        )

    ppc = idata.posterior_predictive["y"].values  # (chains, draws, T)
    ppc_flat = ppc.reshape(-1, ppc.shape[-1])

    # Subset to n_samples for speed
    if ppc_flat.shape[0] > n_samples:
        idx = np.random.choice(ppc_flat.shape[0], n_samples, replace=False)
        ppc_flat = ppc_flat[idx]

    ppc_mean = ppc_flat.mean(axis=0)
    ppc_low = np.percentile(ppc_flat, 3, axis=0)
    ppc_high = np.percentile(ppc_flat, 97, axis=0)

    train_sp, target_train, _ = dataset.train_data()
    observed = target_train

    in_interval = ((observed >= ppc_low) & (observed <= ppc_high))
    coverage = in_interval.mean()

    return {
        "ppc_mean": ppc_mean,
        "ppc_hdi_low": ppc_low,
        "ppc_hdi_high": ppc_high,
        "observed": observed,
        "coverage_94": float(coverage),
    }


# ---------------------------------------------------------------------------
# Out-of-sample validation
# ---------------------------------------------------------------------------


def out_of_sample_validation(
    results: MMMResults,
) -> dict:
    """Evaluate model on the held-out test set.

    Uses posterior mean parameters to generate predictions.

    Returns:
        dict with MAPE, RMSE, R², MAE, and arrays of predicted vs actual.
    """
    from .transformations import (
        geometric_adstock_np,
        weibull_adstock_np,
        hill_saturation_np,
        logistic_saturation_np,
        michaelis_menten_np,
    )

    idata = results.idata
    dataset = results.dataset
    cfg = results.config
    post = idata.posterior

    spend_test, target_test, control_test = dataset.test_data()
    n_channels = dataset.n_channels

    # Posterior means
    beta_mean = post["beta"].mean(("chain", "draw")).values  # (C,)
    base_mean = float(post["base"].mean(("chain", "draw")).values)

    if cfg.saturation_type == "hill":
        gamma_mean = post["gamma_hill"].mean(("chain", "draw")).values
    elif cfg.saturation_type == "logistic":
        lam_mean = post["lam"].mean(("chain", "draw")).values
    else:
        vmax_mean = post["vmax"].mean(("chain", "draw")).values
        km_mean = post["km"].mean(("chain", "draw")).values

    if cfg.adstock_type == "geometric":
        decay_mean = post["decay"].mean(("chain", "draw")).values
    else:
        wb_shape_mean = post["wb_shape"].mean(("chain", "draw")).values
        wb_scale_mean = post["wb_scale"].mean(("chain", "draw")).values

    gamma_ctrl_mean = None
    if dataset.n_controls > 0 and "gamma_ctrl" in post:
        gamma_ctrl_mean = post["gamma_ctrl"].mean(("chain", "draw")).values

    # Need full time-series for adstock (use all weeks, predict on test slice)
    spend_full = dataset.spend_matrix
    test_mask = dataset.test_mask

    channel_preds = np.zeros((spend_test.shape[0], n_channels))

    for c in range(n_channels):
        x_full_raw = spend_full[:, c]
        # Mirror the model's min-shift to keep values non-negative
        x_full = x_full_raw - x_full_raw.min()

        if cfg.adstock_type == "geometric":
            decay_value = float(decay_mean) if np.ndim(decay_mean) == 0 else float(decay_mean[c])
            x_ad_full = geometric_adstock_np(x_full, decay_value)
        else:
            x_ad_full = weibull_adstock_np(
                x_full,
                float(wb_shape_mean[c]),
                float(wb_scale_mean[c]),
                variant="pdf" if cfg.adstock_type == "weibull_pdf" else "cdf",
            )

        x_ad_full = np.clip(x_ad_full, 0, None)
        x_ad_test = x_ad_full[test_mask]

        if cfg.saturation_type == "hill":
            # Normalise using training max (after shift)
            x_train_max = x_full[dataset.train_mask].max() + 1e-8
            x_norm = x_ad_test / x_train_max
            gh = float(gamma_mean[c])
            sat = x_norm / (x_norm + gh + 1e-12)
        elif cfg.saturation_type == "logistic":
            sat = logistic_saturation_np(x_ad_test, float(lam_mean[c]))
        else:
            sat = michaelis_menten_np(
                x_ad_test, float(vmax_mean[c]), float(km_mean[c])
            )

        channel_preds[:, c] = float(beta_mean[c]) * sat

    ctrl_contrib = np.zeros(spend_test.shape[0])
    if gamma_ctrl_mean is not None:
        ctrl_contrib = control_test @ gamma_ctrl_mean

    predicted_scaled = base_mean + channel_preds.sum(axis=1) + ctrl_contrib
    observed_scaled = target_test

    # Unscale
    scaler = dataset.target_scaler
    if scaler is not None:
        predicted = scaler.inverse_transform(predicted_scaled.reshape(-1, 1)).ravel()
        observed = scaler.inverse_transform(observed_scaled.reshape(-1, 1)).ravel()
    else:
        predicted = predicted_scaled
        observed = observed_scaled

    mape = _safe_mape(observed, predicted)
    rmse = float(np.sqrt(np.mean((observed - predicted) ** 2)))
    mae = float(np.mean(np.abs(observed - predicted)))
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-8))


    train_actual = dataset.target_raw[dataset.train_mask]
    from .model import BayesianMMM
    train_pred = BayesianMMM(results.config).get_contributions(results)["total_predicted"]
    train_mape = _safe_mape(train_actual, train_pred)
    train_ss_res = np.sum((train_actual - train_pred) ** 2)
    train_ss_tot = np.sum((train_actual - train_actual.mean()) ** 2)
    train_r2 = float(1 - train_ss_res / (train_ss_tot + 1e-8))

    return {
        "mape": mape,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predicted": predicted,
        "observed": observed,
        "test_dates": dataset.dates[test_mask],
        "train_mape": train_mape,
        "train_r2": train_r2,
    }


# ---------------------------------------------------------------------------
# Full diagnostic report
# ---------------------------------------------------------------------------


def generate_diagnostic_report(results: MMMResults) -> pd.DataFrame:
    """Print and return a consolidated diagnostics summary."""

    print("=" * 60)
    print("BAYESIAN MMM — DIAGNOSTIC REPORT")
    print("=" * 60)

    # Convergence
    print("\n[1] Convergence Diagnostics")
    conv_df = check_convergence(results)
    n_params = len(conv_df)
    n_converged = conv_df["converged"].sum()
    print(f"    Parameters: {n_params}  |  Converged: {n_converged}/{n_params}")
    if "r_hat" in conv_df.columns:
        worst_rhat = conv_df["r_hat"].max()
        print(f"    Worst R-hat: {worst_rhat:.4f}")
    if "ess_bulk" in conv_df.columns:
        worst_ess = conv_df["ess_bulk"].min()
        print(f"    Worst ESS-bulk: {worst_ess:.0f}")

    # OOS validation
    print("\n[2] Out-of-sample Validation (held-out test set)")
    try:
        oos = out_of_sample_validation(results)
        print(f"    MAPE : {oos['mape']:.2f}%")
        print(f"    RMSE : {oos['rmse']:.2f}")
        print(f"    MAE  : {oos['mae']:.2f}")
        print(f"    R²   : {oos['r2']:.4f}")
        print(f"    Train MAPE : {oos['train_mape']:.2f}%")
        print(f"    Train R²   : {oos['train_r2']:.4f}")
    except Exception as e:
        print(f"    Could not compute OOS metrics: {e}")

    # PPC
    print("\n[3] Posterior Predictive Check")
    try:
        ppc = posterior_predictive_check(results)
        print(f"    94% interval coverage: {ppc['coverage_94']*100:.1f}%")
    except Exception as e:
        print(f"    Could not compute PPC: {e}")

    print("\n[4] Posterior Summary (top parameters)")
    summary_cols = ["mean", "sd"]
    hdi_cols = [c for c in conv_df.columns if c.startswith("hdi_")]
    if hdi_cols:
        summary_cols.extend(hdi_cols[:2])
    available_cols = [c for c in summary_cols if c in conv_df.columns]
    if available_cols:
        print(conv_df[available_cols].head(20).to_string())
    else:
        print("    No posterior summary columns available.")
    print("=" * 60)

    return conv_df
