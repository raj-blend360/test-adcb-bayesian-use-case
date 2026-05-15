# Pipeline Walkthrough (`pipeline.py` + related modules)

This document explains the end-to-end execution flow of the Bayesian MMM project, centered on `pipeline.py`, and maps every stage to the relevant supporting files.

---

## 1) Repository map for the pipeline

The orchestration layer is `pipeline.py`, which delegates work to:

- `data/generate_synthetic_data.py` for synthetic data creation.
- `src/data_processing.py` for dataset normalization, feature engineering, scaling, and train/test splits.
- `src/model.py` for Bayesian MMM fitting and post-fit analytics (contributions, response curves, ROI).
- `src/halo_analysis.py` for pre-model halo candidate scoring/visualization.
- `src/diagnostics.py` for convergence + out-of-sample checks.
- `src/optimizer.py` for constrained spend optimization, reverse optimization, and efficient frontier generation.
- `src/visualization.py` for plotting outputs.

Optional serving layers that wrap similar logic:

- `api/main.py` and `api/routers/*.py` expose pipeline/model steps over HTTP.
- `webapp/src/pages/*.tsx` consume API endpoints and provide UI flows for upload, transform, tune, model, optimize, results, and visualization.

---

## 2) How `pipeline.py` starts

`pipeline.py` does three startup tasks before any modeling:

1. Configures Matplotlib to `Agg` so plots can render in headless/CI/server environments.
2. Adds the project root to `sys.path` to allow local imports when running as a script.
3. Imports all core primitives from `data/` and `src/` modules.

Then `main()`:

- Parses CLI args with `parse_args()`.
- Prints a run header (weeks, seed, inference mode, output dir).
- Chooses **real data path** (`--input-csv`) or **synthetic path**.
- Executes each numbered pipeline step in sequence.
- Writes `pipeline_run_summary.json` at the end.

---

## 3) CLI interface (`parse_args`)

The CLI is intentionally broad so you can run the same script for experimentation and production-like batch jobs.

### Inference controls

- `--fast` → MAP estimate (very fast, no posterior uncertainty)
- `--advi` → variational inference
- default (neither above) → full MCMC/NUTS
- `--samples`, `--tune`, `--chains`, `--cores`, `--nuts-sampler`, `--nuts-init`

### Transform/model controls

- `--no-adstock`
- `--no-saturation`
- `--halo` and `--halo-config`
- `--min-halo-spend`, `--halo-top-n`

### Data controls

- `--weeks`, `--seed` for synthetic generation
- `--input-csv` for user-provided channel file (wide or long)
- `--campaign-csv` for optional campaign-level halo usage
- `--date-format` for parsing custom date formats
- `--channel-inputs Channel:metric` lets each channel choose `impressions`, `clicks`, or `spends` as model input

### Optimization/output controls

- `--target` (raw conversions target for reverse optimization)
- `--optimization-level` (`monthly` or `annual` budget scaling)
- `--freeze` channels
- `--no-bounds`
- `--share-bounds-json` for per-channel share caps/floors
- `--no-plots`
- `--output-dir`

---

## 4) Input normalization path

When `--input-csv` is provided, `step_load_data()` is used.

### `_normalize_channel_dataframe(raw_df)`

This helper accepts either:

- **Long format** (already has `date`, `channel`, `media_spend`, ...), or
- **Wide format** with patterns such as:
  - `spends_<channel>`
  - `media_impressions_<channel>` / `impressions_<channel>`
  - `media_clicks_<channel>` / `clicks_<channel>`
  - `exogenous_*`

It converts wide data into a long channel-week table and preserves exogenous controls. It also creates backward-compatible aliases (`holiday_flag`, `promo_flag`) and ensures `conversions` exists (defaults to `0.0` if absent).

### `_apply_channel_inputs(channel_df, mapping)`

This helper creates/overwrites `media_input` per channel using your chosen metric from `--channel-inputs`. If requested metric columns are missing, it falls back to clicks/spend and warns.

---

## 5) Step-by-step pipeline execution

## Step 1 — data acquisition

- `step_generate_data(args)` creates synthetic campaign + channel data and saves CSVs in `output-dir`.
- `step_load_data(args)` loads real CSV inputs, prints schema, normalizes shape, and validates date parsing.

## Step 2 — preprocessing

`step_preprocess(channel_df, campaign_df, args)`:

- discovers non-empty control columns (`*_flag`, `exogenous_*`, `control_*`)
- constructs `DataConfig` (scaling + seasonality + holdout setup)
- runs `DataProcessor.prepare(...)` to create model-ready tensors/matrices
- logs channel/control counts and train/test split details

## Step 2b — halo candidate analysis (synthetic path only)

`step_halo_analysis(dataset, args)`:

- uses campaign spend matrix when available
- scores candidate pairs via `rank_halo_candidates(...)`
- prints top candidates and optionally saves heatmap plot

## Step 3 — model fitting

`step_fit_model(dataset, args)`:

- resolves halo pairs from CLI/config
- builds `ModelConfig`
- instantiates `BayesianMMM`
- calls `mmm.fit(dataset)` and returns `(results, mmm)`

## Step 4 — contribution decomposition + ROI

`step_contributions(results, mmm, args)`:

- gets contribution time series/percentages
- gets ROI table
- saves `contributions.csv` and `roi_metrics.csv`

## Step 5 — response curves

`step_response_curves(results, mmm, args)`:

- computes channel response curves up to 2x current spend
- prints current spend/conversion anchor values

## Step 6 — diagnostics

`step_diagnostics(results, args)`:

- writes convergence diagnostics
- computes out-of-sample metrics when possible (MAPE, R², adjusted R², WMAPE)

## Step 7 — optimization

`step_optimize(results, mmm, dataset, campaign_df, args)`:

- extracts channel response params from model outputs
- configures optimizer constraints (bounds, freeze list, custom share bounds)
- runs constrained budget optimization
- runs reverse optimization to hit target conversions
- computes marginal ROI + efficient frontier
- writes CSV + Excel artifacts

Important: reverse optimization target (`--target`) is interpreted in **raw conversion units**, then converted to model-scaled space internally using `_raw_to_model_conversions(...)` and mapped back with `_model_to_raw_conversions(...)` for reporting.

## Step 8 — plotting

`step_visualizations(...)` saves all charts unless `--no-plots` is set.

Common outputs include:

- contributions stack
- response curves
- ROI chart
- budget allocation
- diagnostics/posteriors
- efficient frontier
- actual vs predicted
- waterfall and channel share visuals

---

## 6) Output contract

After a successful run, `output-dir` typically contains:

- data snapshots (`synthetic_*.csv` if synthetic path)
- model outputs (`contributions.csv`, `roi_metrics.csv`)
- optimizer outputs (`budget_allocation.csv`, `reverse_allocation.csv`, `marginal_roi.csv`, `efficient_frontier.csv`, `optimization_results.xlsx`)
- diagnostics (`convergence_summary.csv`)
- plots (`plots/*.png`)
- run metadata (`pipeline_run_summary.json`)

This makes every run reproducible and reviewable without re-running the model.

---

## 7) API and webapp relationship

Even though `pipeline.py` is script-first, the API and webapp mirror the same lifecycle:

1. Upload / transform data
2. Tune or choose model settings
3. Fit model
4. Inspect contributions + diagnostics
5. Optimize and visualize

So the script can be treated as a canonical, linear reference for understanding the entire system behavior.

---

## 8) Typical run commands

```bash
# Full probabilistic run
python pipeline.py --samples 500 --tune 500 --chains 2

# Fast debug iteration
python pipeline.py --fast

# Real data input (wide or long)
python pipeline.py --input-csv my_channels.csv --campaign-csv my_campaigns.csv --date-format "%Y-%m-%d"

# Target-based reverse optimization
python pipeline.py --fast --target 50000

# Custom per-channel input metrics
python pipeline.py --channel-inputs TV:impressions Search:clicks Meta:spends
```

---

## 9) Practical reading order for new contributors

If you want to understand the codebase quickly, read in this order:

1. `pipeline.py` (global orchestration)
2. `src/data_processing.py` (what exactly is fed to the model)
3. `src/model.py` (likelihood + priors + inference + decomposition helpers)
4. `src/optimizer.py` (business decision layer)
5. `src/diagnostics.py` and `src/visualization.py` (evaluation/reporting)
6. `api/routers/*.py` and `webapp/src/pages/*.tsx` (productized interfaces)

That sequence mirrors runtime dependency flow and minimizes context switching.
