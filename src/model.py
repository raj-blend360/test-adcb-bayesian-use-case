"""
Bayesian Marketing Mix Model (MMM) — PyMC implementation.

Architecture
------------
  target[t] = base
             + Σ_c  β_c · sat(adstock(spend_c[t]))    # channel contributions
             + Σ_c Σ_c'  δ_{cc'} · halo(c, c')[t]    # cross-channel halo
             + Σ_k  γ_k · control_k[t]               # seasonality / controls
             + ε[t]                                    # observation noise

Hierarchical structure
  Campaign parameters are nested within channels via partial pooling:
    β_camp ~ Normal(β_channel, σ_channel)

Configurable
  - Adstock type: geometric | weibull_pdf | weibull_cdf
  - Saturation type: hill | logistic | michaelis_menten
  - Halo pairs: list of (ch_a, ch_b) tuples
  - Seasonality / control inclusion
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from .data_processing import MMMDataset
from .transformations import (
    hill_saturation_pt,
    logistic_saturation_pt,
    michaelis_menten_pt,
)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the Bayesian MMM."""

    # Transformations
    adstock_type: str = "geometric"       # geometric | weibull_pdf | weibull_cdf
    saturation_type: str = "hill"         # hill | logistic | michaelis_menten
    adstock_max_lag: int = 13
    precompute_adstock: bool = True

    # Channel-level halo pairs  [(channel_a, channel_b), ...]
    halo_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Campaign-level halo pairs  [(campaign_a, campaign_b), ...]  cross-channel only
    campaign_halo_pairs: list[tuple[str, str]] = field(default_factory=list)

    # Minimum total raw spend for a campaign to be eligible for halo modelling
    min_halo_spend: float = 0.0

    # Hierarchical campaign model
    include_campaign_hierarchy: bool = False

    # MCMC settings
    n_samples: int = 1000
    n_tune: int = 1000
    n_chains: int = 2
    target_accept: float = 0.90
    random_seed: int = 42
    cores: Optional[int] = None
    nuts_sampler: str = "numpyro"  # numpyro | blackjax | pymc
    nuts_init: str = "jitter+adapt_diag"

    # Inference method: "mcmc" | "map" | "advi"
    inference: str = "mcmc"

    # Prior scales
    beta_prior_sigma: float = 0.3
    gamma_hill_alpha: float = 3.0
    gamma_hill_beta: float = 3.0


# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------

@dataclass
class MMMResults:
    """Container for fitted model results."""

    idata: az.InferenceData
    model: pm.Model
    config: ModelConfig
    dataset: MMMDataset

    # Cached posterior summaries
    _channel_contributions: Optional[np.ndarray] = field(default=None, repr=False)
    _posterior_predictive: Optional[np.ndarray] = field(default=None, repr=False)

    def posterior_mean(self, var: str) -> np.ndarray:
        return self.idata.posterior[var].mean(("chain", "draw")).values

    def posterior_hdi(self, var: str, hdi_prob: float = 0.94) -> np.ndarray:
        return az.hdi(self.idata, var_names=[var], hdi_prob=hdi_prob)[var].values

    @property
    def channel_names(self) -> list[str]:
        return self.dataset.channel_names

    @property
    def summary(self) -> "pd.DataFrame":
        import pandas as pd
        vars_of_interest = ["beta", "gamma_hill", "decay"]
        vars_present = [v for v in vars_of_interest if v in self.idata.posterior]
        return az.summary(self.idata, var_names=vars_present, round_to=4)


# ---------------------------------------------------------------------------
# BayesianMMM
# ---------------------------------------------------------------------------

class BayesianMMM:
    """Bayesian Marketing Mix Model.

    Usage
    -----
    >>> mmm = BayesianMMM(config)
    >>> results = mmm.fit(dataset)
    >>> contributions = mmm.get_contributions(results)
    >>> curves = mmm.get_response_curves(results)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: MMMDataset) -> MMMResults:
        """Build and sample the Bayesian MMM.

        Args:
            dataset: prepared MMMDataset from DataProcessor.

        Returns:
            MMMResults with InferenceData and model reference.
        """
        cfg = self.config
        spend_train, target_train, control_train = dataset.train_data()

        # Resolve halo pairs before entering the model context
        ch_halo_idx = self._resolve_halo_pairs(cfg.halo_pairs, dataset.channel_names)
        camp_halo_idx = self._resolve_campaign_halo_pairs(
            cfg.campaign_halo_pairs,
            dataset.campaign_names or [],
            dataset.campaign_channels or [],
            dataset.campaign_spend_matrix,
            cfg.min_halo_spend,
        )

        has_campaign_halo = bool(camp_halo_idx) and dataset.campaign_spend_matrix is not None

        # Campaign → parent channel index mapping (for borrowing adstock decay)
        camp_to_ch_idx: dict[int, int] = {}
        if has_campaign_halo and dataset.campaign_names and dataset.campaign_channels:
            ch_name_to_idx = {n: i for i, n in enumerate(dataset.channel_names)}
            for ci, (camp_name, ch_name) in enumerate(
                zip(dataset.campaign_names, dataset.campaign_channels)
            ):
                camp_to_ch_idx[ci] = ch_name_to_idx.get(ch_name, 0)

        adstocked_np = self._precompute_adstock_matrix(spend_train, cfg)

        with pm.Model() as model:
            # ---- Data containers ----------------------------------------
            spend_data = pm.Data("spend", spend_train)
            target_obs = pm.Data("target_obs", target_train)
            if control_train.shape[1] > 0:
                control_data = pm.Data("controls", control_train)

            if has_campaign_halo:
                camp_spend_train = dataset.campaign_spend_matrix[dataset.train_mask]
                campaign_spend_data = pm.Data("campaign_spend", camp_spend_train)

            n_channels = dataset.n_channels
            n_controls = dataset.n_controls

            # ---- Channel-level priors -----------------------------------
            beta_z = pm.Normal("beta_z", mu=0.0, sigma=1.0, shape=n_channels)
            beta = pm.Deterministic("beta", 0.0 + cfg.beta_prior_sigma * beta_z)

            # Adstock parameters
            if cfg.adstock_type == "geometric":
                decay = pm.Beta("decay", alpha=3, beta=3)
            else:
                wb_shape = pm.Gamma("wb_shape", alpha=2, beta=1, shape=n_channels)
                wb_scale = pm.Gamma("wb_scale", alpha=3, beta=1, shape=n_channels)

            # Saturation parameters
            if cfg.saturation_type == "hill":
                gamma_hill = pm.Beta("gamma_hill", alpha=cfg.gamma_hill_alpha, beta=cfg.gamma_hill_beta, shape=n_channels)
            elif cfg.saturation_type == "logistic":
                lam = pm.HalfNormal("lam", sigma=1.0, shape=n_channels)
            else:  # michaelis_menten
                vmax = pm.HalfNormal("vmax", sigma=2.0, shape=n_channels)
                km = pm.HalfNormal("km", sigma=1.0, shape=n_channels)

            # ---- Control / seasonality priors ---------------------------
            if n_controls > 0:
                gamma_ctrl = pm.Normal("gamma_ctrl", mu=0, sigma=0.5, shape=n_controls)

            # ---- Base conversions (intercept) ---------------------------
            base = pm.Normal("base", mu=0, sigma=1.0)

            # ---- Observation noise --------------------------------------
            sigma = pm.HalfNormal("sigma", sigma=0.5)

            # ---- Halo effect priors (channel-level) ---------------------
            if ch_halo_idx:
                delta_halo = pm.HalfNormal("delta_halo", sigma=0.3, shape=len(ch_halo_idx))

            # ---- Halo effect priors (campaign-level) --------------------
            if has_campaign_halo and camp_halo_idx:
                # Tighter prior — campaign interactions are smaller in magnitude
                delta_halo_campaign = pm.HalfNormal(
                    "delta_halo_campaign", sigma=0.2, shape=len(camp_halo_idx)
                )

            # ---- Media transformations (vectorized, no Python loops) -----
            adstocked = pm.Data("adstocked", adstocked_np)

            if cfg.saturation_type == "hill":
                ad_norm = adstocked / (pt.max(adstocked, axis=0, keepdims=True) + 1e-8)
                sat_matrix = ad_norm / (ad_norm + gamma_hill + 1e-8)
            elif cfg.saturation_type == "logistic":
                sat_matrix = logistic_saturation_pt(adstocked, lam)
            else:
                sat_matrix = michaelis_menten_pt(adstocked, vmax, km)

            contrib_stack = sat_matrix * beta
            media_total = contrib_stack.sum(axis=1)

            # ---- Halo effects (channel-level) ---------------------------
            halo_total = pt.zeros(spend_data.shape[0])
            if ch_halo_idx:
                ch_halo_idx_arr = np.asarray(ch_halo_idx, dtype=np.int64)
                ch_idx_a = pt.as_tensor_variable(ch_halo_idx_arr[:, 0])
                ch_idx_b = pt.as_tensor_variable(ch_halo_idx_arr[:, 1])
                ch_pair_products = adstocked[:, ch_idx_a] * adstocked[:, ch_idx_b]
                halo_total = halo_total + pt.sum(ch_pair_products * delta_halo, axis=1)

            # ---- Halo effects (campaign-level) --------------------------
            if has_campaign_halo and camp_halo_idx:
                camp_halo_idx_arr = np.asarray(camp_halo_idx, dtype=np.int64)
                camp_idx_a = pt.as_tensor_variable(camp_halo_idx_arr[:, 0])
                camp_idx_b = pt.as_tensor_variable(camp_halo_idx_arr[:, 1])
                camp_pair_products = (
                    campaign_spend_data[:, camp_idx_a] * campaign_spend_data[:, camp_idx_b]
                )
                halo_total = halo_total + pt.sum(
                    camp_pair_products * delta_halo_campaign,
                    axis=1,
                )

            # ---- Control regressors -------------------------------------
            ctrl_total = pt.zeros(spend_data.shape[0])
            if n_controls > 0:
                ctrl_total = pt.dot(control_data, gamma_ctrl)

            # ---- Expected outcome ---------------------------------------
            mu = base + media_total + halo_total + ctrl_total

            # ---- Likelihood ---------------------------------------------
            pm.Normal("y", mu=mu, sigma=sigma, observed=target_obs)

            # ---- Deterministic channel contributions (for decomposition) --
            pm.Deterministic("channel_contribs", contrib_stack)

        # ---- Inference --------------------------------------------------
        with model:
            if cfg.inference == "mcmc":
                sample_kwargs = dict(
                    draws=cfg.n_samples,
                    tune=cfg.n_tune,
                    chains=cfg.n_chains,
                    cores=cfg.cores,
                    target_accept=cfg.target_accept,
                    random_seed=cfg.random_seed,
                    progressbar=True,
                    return_inferencedata=True,
                    init=cfg.nuts_init,
                )
                if cfg.nuts_sampler in {"numpyro", "blackjax"}:
                    sample_kwargs["nuts_sampler"] = cfg.nuts_sampler

                try:
                    idata = pm.sample(**sample_kwargs)
                except ImportError as exc:
                    # PyMC external samplers (NumPyro/BlackJAX) require JAX/JAXLIB.
                    # On Windows, mismatched wheels often fail with a DLL load error.
                    err_msg = str(exc).lower()
                    using_external_sampler = sample_kwargs.get("nuts_sampler") in {"numpyro", "blackjax"}
                    if using_external_sampler and ("dll load failed" in err_msg or "while importing _jax" in err_msg):
                        sample_kwargs.pop("nuts_sampler", None)
                        print(
                            "[WARN] Falling back to PyMC's native NUTS because JAX/JAXLIB failed to load "
                            f"({exc})."
                        )
                        idata = pm.sample(**sample_kwargs)
                    else:
                        raise
                pm.sample_posterior_predictive(idata, extend_inferencedata=True)
            elif cfg.inference == "advi":
                approx = pm.fit(
                    n=50_000,
                    method="advi",
                    random_seed=cfg.random_seed,
                    progressbar=True,
                )
                idata = approx.sample(cfg.n_samples * cfg.n_chains)
                idata = az.convert_to_inference_data(idata)
            else:  # map
                map_est = pm.find_MAP()
                # pm.find_MAP returns transformed variables (beta_log__ etc.).
                # We recover untransformed values by evaluating model variables at the MAP.
                untransformed = {}
                for rv in model.free_RVs:
                    name = rv.name
                    try:
                        val = model.rvs_to_values[rv]
                        # Get the untransformed variable name (strip transform suffix)
                        raw_name = name.split("_")[0] if "__" not in name else name
                    except Exception:
                        pass
                # Use model.compile_logp to evaluate deterministics at MAP
                point = {k: np.atleast_1d(v) for k, v in map_est.items()}
                try:
                    point_on_model = model.compute_initial_point()
                    point_on_model.update(map_est)
                    untransformed = {
                        v.name: model.rvs_to_transforms[v].backward(
                            map_est[model.rvs_to_values[v].name], *v.owner.inputs[1:]
                        ).eval()
                        if v in model.rvs_to_transforms
                        else map_est.get(v.name, map_est.get(model.rvs_to_values[v].name))
                        for v in model.free_RVs
                    }
                except Exception:
                    untransformed = {}

                # Merge: prefer untransformed names where available
                merged = {k: np.array([[v]]) for k, v in map_est.items()}
                for k, v in untransformed.items():
                    if v is not None:
                        merged[k] = np.array([[v]])

                # NOTE:
                # arviz.from_dict expects posterior variables via the keyword arg
                # `posterior=...`. Passing a dict with a top-level "posterior" key
                # creates a *single* variable called "posterior", which then breaks
                # downstream access like idata.posterior["base"].
                idata = az.from_dict(posterior=merged)

        return MMMResults(
            idata=idata,
            model=model,
            config=cfg,
            dataset=dataset,
        )


    def _precompute_adstock_matrix(self, spend: np.ndarray, cfg: ModelConfig) -> np.ndarray:
        """Precompute adstock outside PyMC to avoid expensive scan ops."""
        if not cfg.precompute_adstock:
            return spend
        if cfg.adstock_type != "geometric":
            warnings.warn(
                "precompute_adstock=True only supports geometric adstock. "
                "Non-geometric adstock will fall back to raw scaled spend. "
                "Set precompute_adstock=False to model non-geometric adstock."
            )
            return spend

        x = spend - spend.min(axis=0, keepdims=True)
        max_lag = max(0, int(cfg.adstock_max_lag))
        if max_lag == 0:
            return np.clip(x, 0.0, np.inf)

        alpha = 0.35
        out = np.zeros_like(x)

        # Finite-lag geometric accumulation:
        # out[t] = sum_{k=0..max_lag} alpha^k * x[t-k]
        for t in range(x.shape[0]):
            k_max = min(max_lag, t)
            for k in range(k_max + 1):
                out[t] += (alpha**k) * x[t - k]

        return np.clip(out, 0.0, np.inf)

    # ------------------------------------------------------------------
    # get_contributions
    # ------------------------------------------------------------------

    def get_contributions(self, results: MMMResults) -> dict:
        """Decompose total conversions into base + channel contributions.

        Returns a dict with keys:
          - 'base': scalar mean base contribution
          - 'channels': DataFrame (T × C) of channel contributions
          - 'total_predicted': 1-D array of predicted conversions
          - 'actual': 1-D array of observed conversions (unscaled)
          - 'channel_pct': dict of channel → % of total
        """
        import pandas as pd

        idata = results.idata
        dataset = results.dataset

        # channel_contribs is computed on the training set only → shape (T_train, C)
        train_mask = dataset.train_mask
        train_dates = dataset.dates[train_mask]
        T_train = train_mask.sum()

        if "channel_contribs" in idata.posterior:
            contrib_mean = idata.posterior["channel_contribs"].mean(
                ("chain", "draw")
            ).values  # (T_train, C)
        else:
            contrib_mean = np.zeros((T_train, dataset.n_channels))

        base_mean = float(idata.posterior["base"].mean(("chain", "draw")).values)

        # Control contribution aligned to training rows
        ctrl_contrib = np.zeros(T_train)
        if dataset.n_controls > 0 and "gamma_ctrl" in idata.posterior:
            gamma_mean = idata.posterior["gamma_ctrl"].mean(
                ("chain", "draw")
            ).values
            ctrl_contrib = dataset.control_matrix[train_mask] @ gamma_mean

        # Keep every additive component in model-space first (scaled y-space).
        # This avoids mixing scaled and unscaled variables.
        base_scaled = np.full(T_train, base_mean, dtype=float)
        media_scaled = contrib_mean
        controls_scaled = ctrl_contrib
        pred_scaled = base_scaled + media_scaled.sum(axis=1) + controls_scaled

        # Unscale if needed.
        # StandardScaler: y_scaled = (y_raw - y_mean) / y_std
        # So: y_raw = y_scaled * y_std + y_mean
        # For decomposition:
        #   base_raw[t] = base_scaled[t] * y_std + y_mean
        #   channel_raw[t, c] = channel_scaled[t, c] * y_std
        #   controls_raw[t] = controls_scaled[t] * y_std
        # and predictions satisfy exactly:
        #   pred_raw[t] = base_raw[t] + sum_c(channel_raw[t, c]) + controls_raw[t]
        scaler = dataset.target_scaler
        if scaler is not None:
            y_std = float(scaler.scale_[0])
            y_mean = float(scaler.mean_[0])

            contrib_unscaled = media_scaled * y_std
            base_unscaled_vec = base_scaled * y_std + y_mean
            ctrl_unscaled = controls_scaled * y_std
            total_pred_unscaled = pred_scaled * y_std + y_mean
            base_unscaled = base_unscaled_vec
        else:
            total_pred_unscaled = pred_scaled
            contrib_unscaled = media_scaled
            base_unscaled = base_scaled
            ctrl_unscaled = controls_scaled

        # Numerical consistency check for decomposition identity.
        reconstructed = base_unscaled + contrib_unscaled.sum(axis=1) + ctrl_unscaled
        if not np.allclose(reconstructed, total_pred_unscaled, atol=1e-6, rtol=1e-6):
            warnings.warn(
                "Contribution decomposition mismatch: base + channels + controls "
                "does not reconstruct predictions within tolerance."
            )

        contrib_df = pd.DataFrame(
            contrib_unscaled,
            columns=dataset.channel_names,
            index=train_dates,
        )
        contrib_df["base"] = base_unscaled
        contrib_df["controls"] = ctrl_unscaled

        total_unscaled = dataset.target_raw[train_mask]
        channel_totals = contrib_unscaled.sum(axis=0)
        overall_total = channel_totals.sum() + float(np.mean(base_unscaled)) * T_train

        channel_pct = {
            ch: float(channel_totals[i] / (overall_total + 1e-8) * 100)
            for i, ch in enumerate(dataset.channel_names)
        }

        return {
            "base": base_unscaled,
            "channels": contrib_df,
            "total_predicted": total_pred_unscaled,
            "actual": total_unscaled,
            "channel_pct": channel_pct,
            "ctrl_contrib": ctrl_unscaled,
        }

    # ------------------------------------------------------------------
    # get_response_curves
    # ------------------------------------------------------------------

    def get_response_curves(
        self,
        results: MMMResults,
        n_points: int = 100,
        spend_multiplier: float = 2.0,
    ) -> dict:
        """Compute channel response curves from posterior parameters.

        Returns a dict keyed by channel name, each value a dict with:
          - 'spend': spend grid (unscaled)
          - 'conversions_mean': predicted conversions at each spend level
          - 'conversions_hdi_low': 5th percentile
          - 'conversions_hdi_high': 95th percentile
          - 'current_spend': mean observed spend for this channel
          - 'current_conversions': conversions at current spend
        """
        from .transformations import (
            geometric_adstock_np,
            hill_saturation_np,
            logistic_saturation_np,
            michaelis_menten_np,
        )

        idata = results.idata
        dataset = results.dataset
        cfg = results.config

        # Pull posterior samples
        post = idata.posterior
        beta_samples = post["beta"].values  # (chains, draws, C)
        _, _, n_ch = beta_samples.shape
        beta_flat = beta_samples.reshape(-1, n_ch)

        # Flatten parameter tensors once to avoid repeated reshape + Python loops.
        gamma_hill_flat = lam_flat = vmax_flat = km_flat = None
        if cfg.saturation_type == "hill":
            gamma_hill_flat = post["gamma_hill"].values.reshape(-1, n_ch)
        elif cfg.saturation_type == "logistic":
            lam_flat = post["lam"].values.reshape(-1, n_ch)
        else:
            vmax_flat = post["vmax"].values.reshape(-1, n_ch)
            km_flat = post["km"].values.reshape(-1, n_ch)

        curves = {}
        for c, ch in enumerate(dataset.channel_names):
            # Unscaled spend for this channel
            raw_spend = dataset.spend_raw[:, c]
            x_max = raw_spend.max() * spend_multiplier
            spend_grid = np.linspace(0, x_max, n_points)

            # Vectorized across posterior samples (rows) and spend grid points (cols).
            if cfg.saturation_type == "hill":
                x_ref = raw_spend.max() + 1e-8
                x_norm = spend_grid / x_ref  # (n_points,)
                gh = gamma_hill_flat[:, c][:, None]  # (n_samples, 1)
                sat = x_norm[None, :] / (x_norm[None, :] + gh + 1e-12)
            elif cfg.saturation_type == "logistic":
                lm = lam_flat[:, c][:, None]  # (n_samples, 1)
                sat = 1.0 / (1.0 + np.exp(-lm * spend_grid[None, :]))
            else:
                vm = vmax_flat[:, c][:, None]  # (n_samples, 1)
                km_ = km_flat[:, c][:, None]  # (n_samples, 1)
                sat = (vm * spend_grid[None, :]) / (km_ + spend_grid[None, :] + 1e-12)

            conv_samples = beta_flat[:, c][:, None] * sat

            # Unscale: multiply by target scale factor if applicable
            if dataset.target_scaler is not None:
                scale = dataset.target_scaler.scale_[0]
                conv_samples = conv_samples * scale

            curves[ch] = {
                "spend": spend_grid,
                "conversions_mean": conv_samples.mean(axis=0),
                "conversions_hdi_low": np.percentile(conv_samples, 5, axis=0),
                "conversions_hdi_high": np.percentile(conv_samples, 95, axis=0),
                "current_spend": float(raw_spend.mean()),
                "current_conversions": float(
                    conv_samples.mean(axis=0)[
                        np.argmin(np.abs(spend_grid - raw_spend.mean()))
                    ]
                ),
            }

        return curves

    # ------------------------------------------------------------------
    # ROI / efficiency metrics
    # ------------------------------------------------------------------

    def get_roi_metrics(self, results: MMMResults) -> "pd.DataFrame":
        """Compute ROI and cost-per-conversion per channel."""
        import pandas as pd

        contribs = self.get_contributions(results)
        contrib_df = contribs["channels"]
        dataset = results.dataset

        rows = []
        for c, ch in enumerate(dataset.channel_names):
            total_spend = dataset.spend_raw[:, c].sum()
            total_conv = contrib_df[ch].sum()
            roi = total_conv / (total_spend + 1e-8)
            cpc = total_spend / (total_conv + 1e-8)
            rows.append(
                {
                    "channel": ch,
                    "total_spend": total_spend,
                    "total_conversions": total_conv,
                    "roi": roi,
                    "cost_per_conversion": cpc,
                    "pct_contribution": contribs["channel_pct"].get(ch, 0.0),
                }
            )

        return pd.DataFrame(rows).set_index("channel")

    # ------------------------------------------------------------------
    # Convenience: extract optimizer params directly from the model
    # ------------------------------------------------------------------

    def extract_channel_params(self, results: MMMResults) -> list:
        """Delegate to BudgetOptimizer.extract_channel_params for convenience."""
        from .optimizer import BudgetOptimizer
        return BudgetOptimizer().extract_channel_params(results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_halo_pairs(
        halo_pairs: list[tuple[str, str]],
        channel_names: list[str],
    ) -> list[tuple[int, int]]:
        """Convert channel-name pairs to index pairs."""
        name_to_idx = {n: i for i, n in enumerate(channel_names)}
        resolved = []
        for ca, cb in halo_pairs:
            if ca in name_to_idx and cb in name_to_idx:
                resolved.append((name_to_idx[ca], name_to_idx[cb]))
            else:
                warnings.warn(
                    f"Halo pair ({ca}, {cb}) — one or both channels not found. Skipping."
                )
        return resolved

    @staticmethod
    def _resolve_campaign_halo_pairs(
        campaign_halo_pairs: list[tuple[str, str]],
        campaign_names: list[str],
        campaign_channels: list[str],
        campaign_spend_matrix: Optional[np.ndarray],
        min_halo_spend: float = 0.0,
    ) -> list[tuple[int, int]]:
        """Validate and convert campaign-name halo pairs to index pairs.

        Applies three filters:
          1. Both campaigns must exist in campaign_names.
          2. Campaigns must belong to different channels (cross-channel only).
          3. Both campaigns must meet the min_halo_spend threshold.
        """
        if not campaign_names:
            return []

        name_to_idx = {n: i for i, n in enumerate(campaign_names)}
        ch_lookup = dict(zip(campaign_names, campaign_channels))

        # Total spend per campaign for threshold filtering
        total_spend: dict[int, float] = {}
        if campaign_spend_matrix is not None:
            for i in range(campaign_spend_matrix.shape[1]):
                total_spend[i] = float(campaign_spend_matrix[:, i].sum())

        resolved = []
        for ca, cb in campaign_halo_pairs:
            if ca not in name_to_idx or cb not in name_to_idx:
                warnings.warn(
                    f"Campaign halo pair ({ca}, {cb}) — one or both campaigns not found. Skipping."
                )
                continue
            ia, ib = name_to_idx[ca], name_to_idx[cb]

            if ch_lookup.get(ca) == ch_lookup.get(cb):
                warnings.warn(
                    f"Campaign halo pair ({ca}, {cb}) are in the same channel "
                    f"({ch_lookup.get(ca)}). Skipping — within-channel campaigns "
                    f"share a channel coefficient."
                )
                continue

            if min_halo_spend > 0:
                spend_a = total_spend.get(ia, 0.0)
                spend_b = total_spend.get(ib, 0.0)
                if spend_a < min_halo_spend or spend_b < min_halo_spend:
                    warnings.warn(
                        f"Campaign halo pair ({ca}, {cb}) — one or both campaigns "
                        f"have total spend below min_halo_spend={min_halo_spend:.0f}. Skipping."
                    )
                    continue

            resolved.append((ia, ib))
        return resolved
