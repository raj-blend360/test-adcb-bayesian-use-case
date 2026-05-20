"""
Media transformations for Bayesian MMM.

Implements:
  - Geometric adstock  (closed-form, vectorised)
  - Weibull adstock    (PDF and CDF variants)
  - Hill saturation
  - Logistic saturation
  - Michaelis-Menten saturation

All functions are implemented in both:
  - NumPy flavour  → for data generation, optimizer response curves
  - PyTensor / PyMC flavour  → for embedding inside a PyMC model graph
"""

from __future__ import annotations

import numpy as np
import pytensor
import pytensor.tensor as pt


# ===========================================================================
# NumPy implementations (offline / optimizer use)
# ===========================================================================


def geometric_adstock_np(
    x: np.ndarray,
    decay: float,
    max_lag: int = 13,
) -> np.ndarray:
    """Geometric adstock with decay factor.

    y[t] = x[t] + decay * x[t-1] + decay^2 * x[t-2] + ...

    Args:
        x: spend array, shape (T,).
        decay: retention rate in [0, 1).
        max_lag: maximum look-back window (weeks).

    Returns:
        Adstocked array, shape (T,).
    """
    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        for lag in range(min(t + 1, max_lag + 1)):
            out[t] += (decay**lag) * x[t - lag]
    return out


def weibull_adstock_np(
    x: np.ndarray,
    shape: float,
    scale: float,
    max_lag: int = 13,
    variant: str = "pdf",
) -> np.ndarray:
    """Weibull adstock — more flexible decay shape than geometric.

    Args:
        x: spend array, shape (T,).
        shape: Weibull shape parameter (k > 0).
        scale: Weibull scale parameter (lambda > 0).
        max_lag: maximum look-back window.
        variant: 'pdf' uses the PDF as weights; 'cdf' uses 1 - CDF.

    Returns:
        Adstocked array, shape (T,).
    """
    lags = np.arange(max_lag + 1, dtype=float)
    if variant == "pdf":
        weights = (shape / scale) * (lags / scale) ** (shape - 1) * np.exp(-((lags / scale) ** shape))
    else:  # cdf
        weights = np.exp(-((lags / scale) ** shape))
    weights[0] = 1.0  # current week always gets full weight
    weights = weights / weights.sum()

    T = len(x)
    out = np.zeros(T)
    for t in range(T):
        for lag in range(min(t + 1, max_lag + 1)):
            out[t] += weights[lag] * x[t - lag]
    return out


def hill_saturation_np(
    x: np.ndarray,
    alpha: float,
    gamma: float,
) -> np.ndarray:
    """Hill (power) saturation function.

    S(x) = x^alpha / (x^alpha + gamma^alpha)

    Normalises x to [0, 1] before applying so that gamma is interpretable
    as a fraction of the observed maximum spend.

    Args:
        x: adstocked spend, shape (T,).
        alpha: Hill slope (steepness of diminishing returns).
        gamma: half-saturation point as fraction of max(x) in [0, 1].

    Returns:
        Saturated values in [0, 1], shape (T,).
    """
    x_norm = x / (x.max() + 1e-8)
    return x_norm**alpha / (x_norm**alpha + gamma**alpha)


def logistic_saturation_np(
    x: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Logistic saturation: S(x) = (1 - exp(-lambda * x)) / (1 + exp(-lambda * x)).

    Args:
        x: adstocked spend, shape (T,).
        lam: growth rate (>0).

    Returns:
        Saturated values in (0, 1), shape (T,).
    """
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def michaelis_menten_np(
    x: np.ndarray,
    vmax: float,
    km: float,
) -> np.ndarray:
    """Michaelis-Menten saturation: S(x) = Vmax * x / (Km + x).

    Args:
        x: adstocked spend.
        vmax: maximum saturation value.
        km: half-saturation constant (same units as x).
    """
    return vmax * x / (km + x + 1e-8)




def create_fourier_features(
    t: np.ndarray,
    period: float,
    order: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    """Vectorized Fourier basis features for seasonality modeling."""
    t = np.asarray(t, dtype=float)
    ks = np.arange(1, order + 1, dtype=float)
    angles = 2.0 * np.pi * t[:, None] * ks[None, :] / float(period)
    sin_feats = np.sin(angles)
    cos_feats = np.cos(angles)
    features = np.concatenate([sin_feats, cos_feats], axis=1)
    names = [f"{prefix}_sin_{int(period)}_{k}" for k in ks.astype(int)] + [
        f"{prefix}_cos_{int(period)}_{k}" for k in ks.astype(int)
    ]
    return features, names


def build_laplace_seasonality_np(
    t: np.ndarray,
    omega: float,
    decay_lambda: float,
) -> np.ndarray:
    """Laplace-like decaying sinusoid feature in NumPy for diagnostics/prediction."""
    t = np.asarray(t, dtype=float)
    return np.exp(-decay_lambda * t) * np.sin(omega * t)


def build_laplace_seasonality_pt(
    t: pt.TensorVariable,
    omega: pt.TensorVariable,
    decay_lambda: pt.TensorVariable,
) -> pt.TensorVariable:
    """PyTensor Laplace-like decaying sinusoid for Bayesian estimation."""
    return pt.exp(-decay_lambda * t) * pt.sin(omega * t)

# ===========================================================================
# PyTensor / PyMC implementations (model graph use)
# ===========================================================================


def geometric_adstock_pt(
    x: pt.TensorVariable,
    decay: pt.TensorVariable,
    max_lag: int = 13,
) -> pt.TensorVariable:
    """Geometric adstock in PyTensor for use inside a PyMC model.

    Uses scan to build the recurrence y[t] = x[t] + decay * y[t-1].

    Args:
        x: spend tensor, shape (T,).
        decay: scalar tensor in [0, 1).
        max_lag: ignored (kept for API symmetry); recurrence is exact.

    Returns:
        Adstocked tensor, shape (T,).
    """
    def step(x_t, y_prev, decay_):
        return x_t + decay_ * y_prev

    result, _ = pytensor.scan(
        fn=step,
        sequences=[x],
        outputs_info=[pt.zeros(())],
        non_sequences=[decay],
    )
    return result


def weibull_adstock_pt(
    x: pt.TensorVariable,
    shape: pt.TensorVariable,
    scale: pt.TensorVariable,
    max_lag: int = 13,
    variant: str = "pdf",
) -> pt.TensorVariable:
    """Weibull adstock in PyTensor.

    Computes lag weights symbolically, then applies them via a scan.
    """
    lags = pt.arange(max_lag + 1, dtype="float64")
    if variant == "pdf":
        raw_w = (shape / scale) * (lags / scale) ** (shape - 1) * pt.exp(-((lags / scale) ** shape))
    else:
        raw_w = pt.exp(-((lags / scale) ** shape))

    # Force weight[0] = 1.0 and normalise
    raw_w = pt.set_subtensor(raw_w[0], 1.0)
    weights = raw_w / raw_w.sum()

    def step(t_idx, weights_, x_):
        indices = pt.maximum(t_idx - pt.arange(max_lag + 1), 0)
        x_lags = x_[indices]
        return pt.dot(weights_, x_lags)

    T = x.shape[0]
    result, _ = pytensor.scan(
        fn=step,
        sequences=[pt.arange(T)],
        non_sequences=[weights, x],
    )
    return result


def hill_saturation_pt(
    x: pt.TensorVariable,
    alpha: pt.TensorVariable,
    gamma: pt.TensorVariable,
) -> pt.TensorVariable:
    """Hill saturation in PyTensor.

    Normalises x by its maximum to keep gamma interpretable.
    """
    x_norm = x / (x.max() + 1e-8)
    return x_norm**alpha / (x_norm**alpha + gamma**alpha)


def logistic_saturation_pt(
    x: pt.TensorVariable,
    lam: pt.TensorVariable,
) -> pt.TensorVariable:
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))


def michaelis_menten_pt(
    x: pt.TensorVariable,
    vmax: pt.TensorVariable,
    km: pt.TensorVariable,
) -> pt.TensorVariable:
    return vmax * x / (km + x + 1e-8)


# ===========================================================================
# Response curve utilities (NumPy, for optimizer)
# ===========================================================================


def response_curve_hill(
    spend_range: np.ndarray,
    alpha: float,
    gamma: float,
    channel_coef: float,
    x_ref_max: float,
) -> np.ndarray:
    """Expected conversions from Hill saturation at given spend levels.

    Args:
        spend_range: 1-D array of spend values to evaluate.
        alpha: Hill slope.
        gamma: half-saturation (fraction of x_ref_max).
        channel_coef: posterior mean channel coefficient.
        x_ref_max: reference maximum spend (used for normalisation).

    Returns:
        Predicted conversions array.
    """
    x_norm = spend_range / (x_ref_max + 1e-8)
    sat = x_norm**alpha / (x_norm**alpha + gamma**alpha)
    return channel_coef * sat


def marginal_roi_hill(
    spend: float,
    alpha: float,
    gamma: float,
    channel_coef: float,
    x_ref_max: float,
    delta: float = 1.0,
) -> float:
    """Marginal ROI: d(conversions)/d(spend) approximated numerically."""
    conv_hi = response_curve_hill(
        np.array([spend + delta]), alpha, gamma, channel_coef, x_ref_max
    )[0]
    conv_lo = response_curve_hill(
        np.array([spend]), alpha, gamma, channel_coef, x_ref_max
    )[0]
    return (conv_hi - conv_lo) / delta


# ===========================================================================
# Transformation registry
# ===========================================================================


ADSTOCK_FN_NP = {
    "geometric": geometric_adstock_np,
    "weibull_pdf": lambda x, **kw: weibull_adstock_np(x, variant="pdf", **kw),
    "weibull_cdf": lambda x, **kw: weibull_adstock_np(x, variant="cdf", **kw),
}

SATURATION_FN_NP = {
    "hill": hill_saturation_np,
    "logistic": logistic_saturation_np,
    "michaelis_menten": michaelis_menten_np,
}

ADSTOCK_FN_PT = {
    "geometric": geometric_adstock_pt,
    "weibull_pdf": lambda x, **kw: weibull_adstock_pt(x, variant="pdf", **kw),
    "weibull_cdf": lambda x, **kw: weibull_adstock_pt(x, variant="cdf", **kw),
}

SATURATION_FN_PT = {
    "hill": hill_saturation_pt,
    "logistic": logistic_saturation_pt,
    "michaelis_menten": michaelis_menten_pt,
}
