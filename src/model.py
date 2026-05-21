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
    create_fourier_features,
    build_laplace_seasonality_pt,
)


def format_posterior_label(
    param: str,
    dataset: MMMDataset,
    index: int | tuple[int, ...] | None = None,
    config: Optional["ModelConfig"] = None,
) -> str:
    """Create readable posterior labels with business names instead of array indices."""
    if index is None:
        return param

    if isinstance(index, tuple):
        # Time-varying innovations/rw tensors are shaped (time_step, channel).
        if (
            param in {"beta_tvc_innov", "beta_tvc_rw"}
            and len(index) == 2
        ):
            tvc_channel_names = dataset.channel_names
            if config and config.tvc_channels:
                dataset_idx = {ch.lower(): i for i, ch in enumerate(dataset.channel_names)}
                resolved_idx = [dataset_idx[ch.lower()] for ch in config.tvc_channels if ch.lower() in dataset_idx]
                if resolved_idx:
                    tvc_channel_names = [dataset.channel_names[i] for i in resolved_idx]
            if 0 <= index[1] < len(tvc_channel_names):
                return f"{param}[t={index[0]}, channel={tvc_channel_names[index[1]]}]"
        return f"{param}[{','.join(str(i) for i in index)}]"

    if param == "beta" and 0 <= index < len(dataset.channel_names):
        return f"beta_{dataset.channel_names[index]}"

    if param == "gamma_ctrl" and 0 <= index < len(dataset.control_names):
        control_name = dataset.control_names[index]
        if control_name.endswith("_flag") and len(dataset.dates) > index:
            week_start = np.datetime_as_string(dataset.dates[index], unit="D")
            return f"{control_name}_week_{week_start}"
        return f"gamma_ctrl_{control_name}"

    if param == "channel_contribs":
        t = index // max(len(dataset.channel_names), 1)
        c = index % max(len(dataset.channel_names), 1)
        if 0 <= c < len(dataset.channel_names):
            week_start = np.datetime_as_string(dataset.dates[min(t, len(dataset.dates) - 1)], unit="D")
            return f"channel_contribs_{dataset.channel_names[c]}_week_{week_start}"

    return f"{param}[{index}]"


def relabel_summary_index(summary_df, dataset: MMMDataset, config: Optional["ModelConfig"] = None):
    new_index = []
    for label in summary_df.index:
        if "[" in label and label.endswith("]"):
            base, idx_txt = label[:-1].split("[", 1)
            idx_parts = [p.strip() for p in idx_txt.split(",")]
            if idx_parts and all(p.lstrip("-").isdigit() for p in idx_parts):
                parsed = tuple(int(p) for p in idx_parts)
                if len(parsed) == 1:
                    parsed = parsed[0]
                new_index.append(format_posterior_label(base, dataset, parsed, config=config))
                continue
        new_index.append(label)
    summary_df = summary_df.copy()
    summary_df.index = new_index
    return summary_df


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Configuration for the Bayesian MMM."""

    # Transformations
    apply_adstock: bool = True
    apply_saturation: bool = True
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
    saturation_alpha: float = 3.0
    saturation_beta: float = 3.0
    decay_alpha: float = 3.0
    decay_beta: float = 3.0
    channel_beta_prior_sigma: dict[str, float] = field(default_factory=dict)
    channel_decay_prior: dict[str, tuple[float, float]] = field(default_factory=dict)
    channel_saturation_prior: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Time-varying media coefficients
    use_time_varying_media: bool = True
    tvc_channels: Optional[list[str]] = None
    tvc_frequency: int = 1
    use_dynamic_intercept: bool = True
    rw_sigma_rate: float = 5.0
    shared_rw_sigma: bool = True

    # Extra seasonality
    use_fourier_seasonality: bool = True
    fourier_weekly_order: int = 2
    fourier_monthly_order: int = 2
    fourier_yearly_order: int = 3
    use_laplace_seasonality: bool = False


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
        vars_of_interest = ["beta", "saturation", "decay"]
        vars_present = [v for v in vars_of_interest if v in self.idata.posterior]
        summary = az.summary(self.idata, var_names=vars_present, round_to=4)
        return relabel_summary_index(summary, self.dataset, self.config)


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
        n_time = spend_train.shape[0]

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

            # Time-varying coefficients capture media effectiveness drift over time
            # while Gaussian random walks enforce smooth, non-jumpy temporal evolution.
            beta = self.create_time_varying_beta(
                n_time=n_time,
                n_channels=n_channels,
                cfg=cfg,
                channel_names=dataset.channel_names,
            )

            # Single optional media-transformation parameters
            decay_alpha, decay_beta = self._resolve_channel_beta_priors(
                channel_names=dataset.channel_names,
                per_channel=cfg.channel_decay_prior,
                default_alpha=cfg.decay_alpha,
                default_beta=cfg.decay_beta,
            )
            sat_alpha, sat_beta = self._resolve_channel_beta_priors(
                channel_names=dataset.channel_names,
                per_channel=cfg.channel_saturation_prior,
                default_alpha=cfg.saturation_alpha,
                default_beta=cfg.saturation_beta,
            )

            if cfg.apply_adstock:
                decay = pm.Beta("decay", alpha=decay_alpha, beta=decay_beta, shape=n_channels)

            if cfg.apply_saturation:
                saturation = pm.Beta("saturation", alpha=sat_alpha, beta=sat_beta, shape=n_channels)

            # ---- Control / seasonality priors ---------------------------
            if n_controls > 0:
                gamma_ctrl = pm.Normal("gamma_ctrl", mu=0, sigma=0.5, shape=n_controls)

            extra_fourier = np.zeros((n_time, 0))
            if cfg.use_fourier_seasonality:
                extra_fourier, _ = self.build_fourier_seasonality(n_time, cfg)
            if extra_fourier.shape[1] > 0:
                fourier_data = pm.Data("fourier_features", extra_fourier)
                gamma_fourier = pm.Normal("gamma_fourier", mu=0.0, sigma=0.3, shape=extra_fourier.shape[1])

            if cfg.use_laplace_seasonality:
                t_idx = pt.arange(n_time, dtype="float64")
                laplace_lambda = pm.Exponential("laplace_lambda", lam=2.0)
                laplace_omega = pm.HalfNormal("laplace_omega", sigma=0.5)
                laplace_amp = pm.Normal("laplace_amp", mu=0.0, sigma=0.3)
                laplace_term = laplace_amp * build_laplace_seasonality_pt(t_idx, laplace_omega, laplace_lambda)
            else:
                laplace_term = pt.zeros(n_time)

            # ---- Base conversions (intercept) ---------------------------
            base = pm.Normal("base", mu=0, sigma=1.0)
            if cfg.use_dynamic_intercept:
                base_rw_sigma = pm.Exponential("base_rw_sigma", lam=cfg.rw_sigma_rate)
                base_t = pm.GaussianRandomWalk("base_t", sigma=base_rw_sigma, shape=n_time)
            else:
                base_t = pt.zeros(n_time)

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

            if cfg.apply_saturation:
                ad_norm = adstocked / (pt.max(adstocked, axis=0, keepdims=True) + 1e-8)
                sat_matrix = ad_norm / (ad_norm + saturation + 1e-8)
            else:
                sat_matrix = adstocked

            contrib_stack, media_total = self.apply_time_varying_coefficients(sat_matrix, beta)

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
            fourier_total = pt.zeros(spend_data.shape[0])
            if cfg.use_fourier_seasonality and extra_fourier.shape[1] > 0:
                fourier_total = pt.dot(fourier_data, gamma_fourier)

            mu = base + base_t + media_total + halo_total + ctrl_total + fourier_total + laplace_term

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


    def create_time_varying_beta(
        self,
        n_time: int,
        n_channels: int,
        cfg: ModelConfig,
        channel_names: Optional[list[str]] = None,
    ) -> pt.TensorVariable:
        """Create smooth channel-time betas using Gaussian random walks."""
        ch_to_idx = {name: i for i, name in enumerate(channel_names or [])}
        if cfg.tvc_channels is None:
            tvc_idx = list(range(n_channels))
        else:
            tvc_idx = sorted(
                {
                    ch_to_idx[ch]
                    for ch in cfg.tvc_channels
                    if ch in ch_to_idx
                }
            )
        static_idx = [i for i in range(n_channels) if i not in tvc_idx]

        beta_sigma = self._resolve_channel_prior_sigma(
            channel_names=channel_names or [str(i) for i in range(n_channels)],
            default_sigma=cfg.beta_prior_sigma,
            per_channel=cfg.channel_beta_prior_sigma,
        )

        beta_init = pt.zeros((n_channels,))
        if static_idx:
            beta_init_static = pm.HalfNormal(
                "beta_init_static",
                sigma=beta_sigma[static_idx],
                shape=len(static_idx),
            )
            beta_init = pt.set_subtensor(beta_init[static_idx], beta_init_static)
        if tvc_idx:
            beta_init_tvc = pm.Normal(
                "beta_init_tvc",
                mu=0.0,
                sigma=beta_sigma[tvc_idx],
                shape=len(tvc_idx),
            )
            beta_init = pt.set_subtensor(beta_init[tvc_idx], beta_init_tvc)
        beta_init = pm.Deterministic("beta_init", beta_init)

        if channel_names and len(channel_names) == n_channels:
            for i, ch in enumerate(channel_names):
                safe_name = str(ch).replace(" ", "_").replace("/", "_")
                pm.Deterministic(f"beta_init[{safe_name}]", beta_init[i])

        if not cfg.use_time_varying_media:
            beta_static = pm.Deterministic("beta_static", beta_init)
            return pt.repeat(beta_static.dimshuffle("x", 0), n_time, axis=0)

        beta_full = pt.repeat(beta_init.dimshuffle("x", 0), n_time, axis=0)
        if not tvc_idx:
            return pm.Deterministic("beta", beta_full)

        n_steps = int(np.ceil(n_time / max(1, int(cfg.tvc_frequency))))
        if cfg.shared_rw_sigma:
            rw_sigma = pm.HalfNormal("rw_sigma", sigma=0.05)
        else:
            rw_sigma = pm.HalfNormal("rw_sigma", sigma=0.05, shape=len(tvc_idx))

        # NOTE: Avoid pm.GaussianRandomWalk here because some PyMC/PyTensor
        # builds can raise `NotImplementedError: Logprob method not implemented
        # for CumOp{-1, add}` during logp graph construction. Build the same
        # random-walk process explicitly from Normal innovations + cumsum.
        beta_tvc_innov = pm.Normal(
            "beta_tvc_innov",
            mu=0.0,
            sigma=rw_sigma,
            shape=(n_steps, len(tvc_idx)),
        )
        beta_tvc_rw = pm.Deterministic("beta_tvc_rw", pt.cumsum(beta_tvc_innov, axis=0))
        beta_tvc_steps = pm.Deterministic("beta_tvc_steps", pt.softplus(beta_tvc_rw + beta_init[tvc_idx]))
        repeat_idx = np.minimum(np.arange(n_time) // max(1, int(cfg.tvc_frequency)), n_steps - 1)
        beta_tvc_full = beta_tvc_steps[repeat_idx]
        beta_full = pt.set_subtensor(beta_full[:, tvc_idx], beta_tvc_full)
        if static_idx:
            beta_full = pt.set_subtensor(beta_full[:, static_idx], pt.repeat(beta_init[static_idx].dimshuffle("x", 0), n_time, axis=0))
        return pm.Deterministic("beta", beta_full)

    @staticmethod
    def _resolve_channel_prior_sigma(
        channel_names: list[str],
        default_sigma: float,
        per_channel: Optional[dict[str, float]] = None,
    ) -> np.ndarray:
        sigma = np.full((len(channel_names),), float(default_sigma), dtype=float)
        for i, ch in enumerate(channel_names):
            if per_channel and ch in per_channel:
                sigma[i] = max(1e-6, float(per_channel[ch]))
        return sigma

    @staticmethod
    def _resolve_channel_beta_priors(
        channel_names: list[str],
        per_channel: Optional[dict[str, tuple[float, float]]],
        default_alpha: float,
        default_beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = np.full((len(channel_names),), float(default_alpha), dtype=float)
        beta = np.full((len(channel_names),), float(default_beta), dtype=float)
        for i, ch in enumerate(channel_names):
            if per_channel and ch in per_channel:
                a, b = per_channel[ch]
                alpha[i] = max(1e-6, float(a))
                beta[i] = max(1e-6, float(b))
        return alpha, beta

    @staticmethod
    def apply_time_varying_coefficients(
        transformed_media: pt.TensorVariable,
        beta_t: pt.TensorVariable,
    ) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        """Apply time-varying coefficients in vectorized form."""
        contrib_stack = transformed_media * beta_t
        media_total = contrib_stack.sum(axis=1)
        return contrib_stack, media_total

    @staticmethod
    def build_fourier_seasonality(n_time: int, cfg: ModelConfig) -> tuple[np.ndarray, list[str]]:
        t = np.arange(n_time, dtype=float)
        feats = []
        names = []
        if cfg.fourier_weekly_order > 0:
            f, n = create_fourier_features(t, period=7.0, order=cfg.fourier_weekly_order, prefix="weekly")
            feats.append(f); names.extend(n)
        if cfg.fourier_monthly_order > 0:
            f, n = create_fourier_features(t, period=30.4, order=cfg.fourier_monthly_order, prefix="monthly")
            feats.append(f); names.extend(n)
        if cfg.fourier_yearly_order > 0:
            f, n = create_fourier_features(t, period=365.25, order=cfg.fourier_yearly_order, prefix="yearly")
            feats.append(f); names.extend(n)
        if not feats:
            return np.zeros((n_time, 0)), []
        return np.concatenate(feats, axis=1), names

    def _precompute_adstock_matrix(self, spend: np.ndarray, cfg: ModelConfig) -> np.ndarray:
        """Precompute adstock outside PyMC to avoid expensive scan ops."""
        if not cfg.apply_adstock or not cfg.precompute_adstock:
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

        # Soft positivity for channel reporting: keep displayed channel contributions
        # non-negative (negative values are shown as zero).
        contrib_display = np.clip(contrib_unscaled, 0.0, np.inf)

        contrib_df = pd.DataFrame(
            contrib_display,
            columns=dataset.channel_names,
            index=train_dates,
        )
        contrib_df["base"] = base_unscaled
        contrib_df["controls"] = ctrl_unscaled

        total_unscaled = dataset.target_raw[train_mask]
        channel_totals = contrib_display.sum(axis=0)
        positive_totals = np.clip(channel_totals, 0.0, np.inf)
        media_total = float(positive_totals.sum())

        # Report media mix shares as a normalized partition over channels only.
        # This avoids >100% totals when baseline/control contributions are negative.
        if media_total <= 1e-8:
            channel_pct = {ch: 0.0 for ch in dataset.channel_names}
        else:
            channel_pct = {
                ch: float(positive_totals[i] / media_total * 100)
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
        beta_samples = post["beta"].values
        # beta can be either (chains, draws, channels) for static coefficients
        # or (chains, draws, time, channels) when time-varying coefficients are enabled.
        if beta_samples.ndim == 3:
            _, _, n_ch = beta_samples.shape
            beta_flat = beta_samples.reshape(-1, n_ch)
        elif beta_samples.ndim == 4:
            # Aggregate across time to produce a single response curve per channel.
            _, _, _, n_ch = beta_samples.shape
            beta_flat = beta_samples.mean(axis=2).reshape(-1, n_ch)
        else:
            raise ValueError(
                f"Unexpected beta posterior shape {beta_samples.shape}; expected 3D or 4D tensor."
            )
        # Soft positivity for response curves: hide negative beta draws by flooring at zero.
        beta_flat = np.clip(beta_flat, 0.0, np.inf)

        # Flatten parameter tensors once to avoid repeated reshape + Python loops.
        saturation_flat = None
        if cfg.apply_saturation and "saturation" in post:
            saturation_samples = post["saturation"].values
            # Keep saturation samples aligned with beta_flat rows. Saturation can be
            # (chains, draws) for static terms or include time/channel dimensions
            # when those parameters are modeled as dynamic; in those cases collapse
            # extra axes so each posterior draw maps to a single scalar here.
            if saturation_samples.ndim == 2:
                saturation_flat = saturation_samples.reshape(-1, 1)
            elif saturation_samples.ndim >= 3:
                reduce_axes = tuple(range(2, saturation_samples.ndim))
                saturation_flat = saturation_samples.mean(axis=reduce_axes).reshape(-1, 1)
            else:
                raise ValueError(
                    f"Unexpected saturation posterior shape {saturation_samples.shape}; expected at least 2D tensor."
                )

        curves = {}
        for c, ch in enumerate(dataset.channel_names):
            # Unscaled spend for this channel
            raw_spend = dataset.spend_raw[:, c]
            x_max = raw_spend.max() * spend_multiplier
            spend_grid = np.linspace(0, x_max, n_points)

            # Vectorized across posterior samples (rows) and spend grid points (cols).
            if cfg.apply_saturation and saturation_flat is not None:
                x_ref = raw_spend.max() + 1e-8
                x_norm = spend_grid / x_ref
                gh = saturation_flat
                sat = x_norm[None, :] / (x_norm[None, :] + gh + 1e-12)
            else:
                sat = np.broadcast_to(spend_grid[None, :], (beta_flat.shape[0], n_points))

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
