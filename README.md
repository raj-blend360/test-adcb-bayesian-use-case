# Bayesian Marketing Mix Model (MMM) + Budget Optimizer

A production-grade, modular Bayesian MMM system built with **PyMC** and **SciPy**, featuring hierarchical channel modelling, adstock transformations, saturation functions, cross-channel halo effects, and a constrained budget optimizer.

---

## Architecture

```
├── data/
│   └── generate_synthetic_data.py   # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_processing.py           # Data loading, cleaning, feature engineering
│   ├── transformations.py           # Adstock & saturation (NumPy + PyTensor)
│   ├── model.py                     # Bayesian MMM (PyMC)
│   ├── diagnostics.py               # R-hat, ESS, OOS validation
│   ├── optimizer.py                 # Constrained budget optimizer
│   └── visualization.py            # Matplotlib charts
├── pipeline.py                      # End-to-end orchestration
├── outputs/                         # Generated CSVs + plots
└── requirements.txt
```

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run full pipeline (MCMC — accurate but slow)
```bash
python pipeline.py --samples 500 --tune 500 --chains 2
```

### Fast MAP point-estimate run (seconds)
```bash
python pipeline.py --fast
```

### Reverse-optimize for a target conversion count
```bash
python pipeline.py --fast --target 60000
```

### Freeze specific channels
```bash
python pipeline.py --fast --freeze TV OOH
```

### Disable ±30% bounds (unconstrained)
```bash
python pipeline.py --fast --no-bounds
```

---

## Model Design

### Likelihood
```
target[t] = base
           + Σ_c  β_c · Hill(Adstock(spend_c[t]))
           + Σ_{c,c'} δ_{cc'} · halo(c, c')[t]
           + Σ_k  γ_k · control_k[t]
           + ε[t],    ε ~ Normal(0, σ)
```

### Priors
| Parameter | Prior | Notes |
|-----------|-------|-------|
| `β` (channel coef) | HalfNormal(σ=1) | Positivity constraint |
| `decay` (adstock) | Beta(α=3, β=3) | ~0.5 a priori |
| `α_hill` (slope) | TruncatedNormal(μ=2, σ=0.5, 0.5–10) | Diminishing returns shape |
| `γ_hill` (half-sat) | Beta(α=3, β=3) | Fraction of max spend |
| `δ_halo` | HalfNormal(σ=0.3) | Cross-channel spillover |
| `γ_ctrl` | Normal(0, 0.5) | Seasonality / control coefficients |
| `σ` | HalfNormal(0.5) | Observation noise |

### Adstock options
- **Geometric**: `y[t] = x[t] + decay · y[t-1]` (default)
- **Weibull PDF**: flexible asymmetric decay
- **Weibull CDF**: slow-start then fast decay

### Saturation options
- **Hill** (default): `S(x) = xᵅ / (xᵅ + γᵅ)`
- **Logistic**: `S(x) = (1 − e^{−λx}) / (1 + e^{−λx})`
- **Michaelis-Menten**: `S(x) = Vmax · x / (Km + x)`

### Seasonality
Fourier terms for annual (52-week) and semi-annual (26-week) cycles with configurable harmonics.

---

## Optimizer Design

### Annual optimization
Maximizes total conversions subject to:
- Fixed total budget constraint
- Per-channel bounds (±30% of current spend, toggleable)
- 60% max-increase safety cap
- Channel freezing support

### Reverse optimization
Minimizes total spend to reach a target conversion count:
- No upper bounds (freely increases channels as needed)
- Lower bound = 0 (cannot go negative)

### Campaign allocation
Distributes optimized channel spend to campaigns via:
- **Proportional**: preserve historical spend ratios
- **Response**: weight by marginal conversion rate

---

## Outputs

| File | Description |
|------|-------------|
| `outputs/synthetic_*.csv` | Generated input data |
| `outputs/contributions.csv` | Weekly channel contributions |
| `outputs/roi_metrics.csv` | ROI, CPC, % contribution per channel |
| `outputs/budget_allocation.csv` | Current vs optimized annual spend |
| `outputs/reverse_allocation.csv` | Spend required for target conversions |
| `outputs/marginal_roi.csv` | Marginal ROI at current spend |
| `outputs/efficient_frontier.csv` | Conversions across budget range |
| `outputs/convergence_summary.csv` | R-hat, ESS per parameter |
| `outputs/plots/*.png` | All visualizations |

### Plots generated
1. `contributions.png` — Stacked area chart (base + channels) vs actual
2. `response_curves.png` — Diminishing returns curves with 90% CI
3. `roi_metrics.png` — ROI, cost-per-conversion, % contribution
4. `budget_allocation.png` — Current vs optimized spend per channel
5. `diagnostics.png` — R-hat and ESS histograms
6. `posteriors.png` — Posterior violin plots
7. `efficient_frontier.png` — Budget vs conversions frontier
8. `campaign_allocation.png` — Campaign-level spend breakdown
9. `actual_vs_predicted.png` — In-sample fit + OOS predictions
10. `waterfall.png` — Waterfall decomposition

---

## API Reference

```python
from src.data_processing import DataConfig, DataProcessor
from src.model import BayesianMMM, ModelConfig
from src.optimizer import BudgetOptimizer, OptimizerConfig
from src import visualization as viz

# 1. Prepare data
processor = DataProcessor(DataConfig())
dataset = processor.prepare(channel_df)

# 2. Fit model
mmm = BayesianMMM(ModelConfig(n_samples=1000, n_chains=2))
results = mmm.fit(dataset)

# 3. Decompose
contributions = mmm.get_contributions(results)
curves = mmm.get_response_curves(results)
roi_df = mmm.get_roi_metrics(results)

# 4. Optimize
optimizer = BudgetOptimizer(OptimizerConfig(use_bounds=True))
channel_params = optimizer.extract_channel_params(results)
opt = optimizer.optimize_budget(channel_params, total_budget, current_spend)
rev = optimizer.reverse_optimize(channel_params, target_conversions, current_spend)

# 5. Visualize
viz.plot_contributions(contributions, dataset.dates)
viz.plot_response_curves(curves)
viz.plot_budget_allocation(opt)
```

---

## Assumptions

1. **Weekly granularity**: all modelling is at weekly resolution.
2. **Additive model**: contributions sum linearly (log-normal link not used).
3. **Scaling**: spend and target are z-scored before modelling; contributions are unscaled post-hoc.
4. **Adstock max lag**: 13 weeks (one quarter) — configurable.
5. **Hill gamma interpretation**: fraction of the training-period maximum spend, not absolute units.
6. **Campaign allocation**: uses average historical spend proportions; does not re-optimize at campaign level.
7. **Halo effects**: modelled as the product of adstocked spend for two channels — captures synergistic amplification.
8. **Out-of-sample**: last `test_weeks` (default 12) are held out and never used in model fitting.
