# Bayesian MMM — Documentation

A production-grade Bayesian Media Mix Model system with a browser-based UI and a CLI pipeline. Both interfaces work with your own data or the built-in synthetic data generator.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Running the CLI Pipeline](#running-the-cli-pipeline)
3. [Running the React Webapp](#running-the-react-webapp)
4. [CSV Data Format](#csv-data-format)
5. [File Structure](#file-structure)
6. [Data Persistence](#data-persistence)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Node.js | ≥ 18 | [nodejs.org](https://nodejs.org) (webapp only) |
| npm | ≥ 9 | Bundled with Node.js (webapp only) |

Install Python dependencies:

```bash
# Core MMM packages (PyMC, NumPy, SciPy, etc.)
pip install -r requirements.txt

# FastAPI backend packages (webapp only)
pip install -r api/requirements.txt
```

---

## Running the CLI Pipeline

`pipeline.py` is the command-line entry point. It runs the full MMM workflow end-to-end: data loading → preprocessing → model fitting → diagnostics → optimization → plots.

### With your own data (Anaconda prompt / terminal)

```bash
# Activate your conda environment first
conda activate <your-env>
cd path\to\test-adcb-bayesian-use-case
pip install -r requirements.txt
```

**MAP inference (fast, ~30s) — channel data only:**
```bash
python pipeline.py --fast --channel-csv path\to\your_channel_data.csv
```

**MAP inference with campaign data (enables halo effects):**
```bash
python pipeline.py --fast \
  --channel-csv path\to\your_channel_data.csv \
  --campaign-csv path\to\your_campaign_data.csv
```

**Full MCMC run (full posterior uncertainty):**
```bash
python pipeline.py \
  --channel-csv path\to\your_channel_data.csv \
  --samples 1000 --tune 1000 --chains 2
```

**With halo effects from a config file:**
```bash
python pipeline.py --fast \
  --channel-csv path\to\your_channel_data.csv \
  --campaign-csv path\to\your_campaign_data.csv \
  --halo-config halo_config.json \
  --min-halo-spend 100000
```

### With synthetic data (default, no CSV needed)

```bash
python pipeline.py --fast
```

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--channel-csv PATH` | None | Path to channel-level CSV. When provided, skips synthetic data generation. |
| `--campaign-csv PATH` | None | Path to campaign-level CSV (optional, enables campaign halo effects). |
| `--fast` | off | Use MAP inference (~30s point estimate). |
| `--advi` | off | Use ADVI variational inference (~2min). |
| `--samples N` | 500 | MCMC draw count per chain. |
| `--tune N` | 500 | MCMC tuning steps. |
| `--chains N` | 2 | Number of MCMC chains. |
| `--no-plots` | off | Skip saving plots (faster for testing). |
| `--target N` | None | Target conversions for reverse optimization. |
| `--freeze CH …` | [] | Channels to freeze at current spend (e.g. `--freeze TV OOH`). |
| `--halo CH,CH …` | None | Channel-level halo pairs (e.g. `--halo TV,Digital`). |
| `--halo-config PATH` | None | Path to `halo_config.json` with channel and campaign halo pairs. |
| `--min-halo-spend N` | 0 | Minimum total campaign spend to qualify as a halo candidate. |
| `--halo-top-n N` | 10 | Number of top halo candidates to print. |
| `--output-dir PATH` | outputs | Directory for all saved outputs. |
| `--weeks N` | 104 | Weeks of synthetic data to generate (ignored when `--channel-csv` is set). |
| `--seed N` | 42 | Random seed for synthetic data generation. |

---

## Running the React Webapp

The webapp provides a browser UI covering the same 7-step workflow as the CLI.

### Quick start (one command)

```bash
./start.sh
```

This installs all dependencies and starts both servers:

- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API docs (Swagger)**: http://localhost:8000/docs

Press `Ctrl+C` to stop both.

### Manual start (two terminals)

**Terminal 1 — Backend:**
```bash
pip install -r requirements.txt
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd webapp
npm install
npm run dev
```

### Webapp workflow

| Step | Page | What you do |
|------|------|-------------|
| 1 | **Upload** | Drag-and-drop channel CSV (required) and campaign CSV (optional) |
| 2 | **Transform** | Configure adstock type/lag and saturation type/params per channel; preview adstocked spend |
| 3 | **Model** | Choose MAP/ADVI/MCMC; add halo pairs; click Run Model |
| 4 | **Results** | Compare model runs by Adj R², MAPE, R-hat; save model names |
| 5 | **Tune** | Add holiday periods (date pickers), toggle seasonality, re-run model |
| 6 | **Visualize** | View contributions, response curves, ROI, waterfall, weekly decomp; download CSV/PNG |
| 7 | **Optimize** | Forward (budget → max conversions) or reverse (target → min spend); save scenarios |

---

## CSV Data Format

### Channel CSV (required)

One row per **week × channel**.

| Column | Required | Description |
|--------|----------|-------------|
| `date` | Yes | Week start date (any parseable format, e.g. `2023-01-02`) |
| `channel` | Yes | Channel name (e.g. `TV`, `Digital`, `Radio`, `OOH`) |
| `media_spend` | Yes | Spend in that channel for that week |
| `conversions` | Yes | Conversions attributed to that week |
| `impressions` | No | Impressions (used if selected as target metric in webapp) |
| `clicks` | No | Clicks |
| `holiday_flag` | No | 1 if a holiday week, 0 otherwise |
| `promo_flag` | No | 1 if a promotional week, 0 otherwise |

**Example with control variables:**
```
date,channel,media_spend,conversions,holiday_flag,promo_flag
2023-01-02,TV,500000,1200,0,0
2023-01-02,Digital,300000,800,0,0
2023-01-02,Radio,80000,200,0,0
2023-01-09,TV,480000,1150,0,0
2023-01-09,Digital,310000,820,0,0
2023-01-09,Radio,75000,190,0,0
2023-12-18,TV,420000,950,1,0
2023-12-18,Digital,280000,700,1,0
2023-12-18,Radio,65000,160,1,0
2023-12-25,TV,350000,750,1,0
2023-12-25,Digital,200000,550,1,0
2023-12-25,Radio,50000,130,1,0
2024-02-14,TV,520000,1250,0,1
2024-02-14,Digital,350000,950,0,1
2024-02-14,Radio,90000,230,0,1
```

**How to interpret:**
- Week of 2023-12-18 and 2023-12-25: `holiday_flag=1` (Christmas season)
- Week of 2024-02-14: `promo_flag=1` (Valentine's Day promotional push)
- All other weeks: both flags = 0

---

## Adding Custom Control Variables

You can add **any numeric column** to your CSV and include it as a control variable. Examples:

```
date,channel,media_spend,conversions,holiday_flag,promo_flag,competitor_promo,major_event,economic_index
2023-01-02,TV,500000,1200,0,0,0,0,1.05
2023-01-02,Digital,300000,800,0,0,0,0,1.05
2023-12-25,TV,350000,750,1,0,0,0,0.98
2023-12-25,Digital,200000,550,1,0,0,0,0.98
2024-02-14,TV,520000,1250,0,1,1,0,1.02
2024-02-14,Digital,350000,950,0,1,1,0,1.02
2024-06-15,TV,480000,1100,0,0,0,1,1.08
2024-06-15,Digital,320000,880,0,0,0,1,1.08
```

Then tell the system to include them in the model:

**Via CLI:**
```python
from src.data_processing import DataConfig, DataProcessor

cfg = DataConfig(
    control_cols=["holiday_flag", "promo_flag", "competitor_promo", "major_event", "economic_index"]
)
processor = DataProcessor(cfg)
dataset = processor.prepare(channel_df)
```

**Via webapp (Transform page):**
- These custom columns won't have checkboxes (the UI only shows the default ones)
- To enable them, edit `pipeline.py` or use the CLI with a modified `DataConfig`

**What you can add:**

| Column Name | Type | Example | What it captures |
|---|---|---|---|
| `holiday_flag` | 0/1 | Christmas, Eid, New Year | Major holidays that affect behavior |
| `promo_flag` | 0/1 | Sales event, clearance | Your brand's promotional activities |
| `competitor_promo` | 0/1 | Competitor running ads | Competitive activity in the market |
| `major_event` | 0/1 | Sports event, concert | External events (World Cup, Olympics, etc.) |
| `economic_index` | float | 0.98–1.05 | Economic conditions (GDP growth, inflation) |
| `weather_score` | float | 0–1 | Extreme weather (heatwave, snowstorm) |
| `marketing_push` | 0/1 | Brand campaign | Internal marketing intensity |

---

## How Control Variables Work in the Model

The model learns a coefficient for each control variable:

```
Conversions = Base + 
              (Channel 1 effect) + (Channel 2 effect) + ... +
              (Adstock × Saturation for each channel) +
              (Holiday coefficient × holiday_flag) +
              (Promo coefficient × promo_flag) +
              (Other controls) +
              Noise
```

For example:
- If `holiday_flag=1` is consistently followed by **lower** conversions, the model learns a **negative holiday coefficient**
- If `economic_index=0.95` (recession) correlates with **lower** conversions, the model captures that relationship
- The model separates **media spend effects** from **external factors**

---

## Rules for Control Variables

1. **One value per week, shared across channels** — all channels see the same `holiday_flag` or `major_event` that week
2. **Numeric only** — use 0/1 for flags, floats for indices (0.98, 1.05, etc.)
3. **No NaN/missing values** — if you don't have data, use 0
4. **Don't include columns you'll pass elsewhere** — e.g., don't put `date` or `channel` in control_cols
5. **Optional columns** — if a column is missing from your CSV, it's simply ignored

**Minimal example:**
```
date,channel,media_spend,conversions
2023-01-02,TV,500000,1200
2023-01-02,Digital,300000,800
2023-01-02,Radio,80000,200
2023-01-09,TV,480000,1150
2023-01-09,Digital,310000,820
2023-01-09,Radio,75000,190
```

### Campaign CSV (optional)

One row per **week × campaign**. Required only for campaign-level halo effects.

| Column | Required | Description |
|--------|----------|-------------|
| `date` | Yes | Week start date |
| `channel` | Yes | Parent channel (must match channel CSV) |
| `campaign` | Yes | Campaign name |
| `media_spend` | Yes | Campaign spend for that week |
| `conversions` | Yes | Conversions |
| `sub_channel` | No | Sub-channel grouping |
| `impressions` | No | Impressions |
| `clicks` | No | Clicks |
| `holiday_flag` | No | Holiday indicator |
| `promo_flag` | No | Promo indicator |

---

## File Structure

```
repo/
├── pipeline.py             # CLI entry point
├── requirements.txt        # Core MMM dependencies (PyMC, NumPy, SciPy, etc.)
├── halo_config.json        # Example halo pairs config
├── start.sh                # One-command webapp launcher
├── WEBAPP_README.md        # This file
│
├── src/                    # Python MMM library
│   ├── data_processing.py  # DataProcessor, MMMDataset, StandardScaler
│   ├── model.py            # BayesianMMM, ModelConfig
│   ├── transformations.py  # Adstock and saturation functions (NumPy + PyTensor)
│   ├── diagnostics.py      # R-hat, ESS, OOS validation
│   ├── optimizer.py        # BudgetOptimizer (SLSQP forward + reverse)
│   ├── visualization.py    # Matplotlib plot helpers
│   └── halo_analysis.py    # Campaign halo candidate scoring
│
├── api/                    # FastAPI backend (webapp)
│   ├── main.py             # App entry point + CORS
│   ├── database.py         # SQLite + SQLAlchemy setup
│   ├── models.py           # ORM tables: sessions, model_runs, scenarios, tuning_configs
│   ├── schemas.py          # Pydantic request/response schemas
│   ├── deps.py             # DB session dependency
│   ├── requirements.txt    # Backend-only dependencies
│   └── routers/
│       ├── upload.py       # CSV upload + column detection
│       ├── transform.py    # Adstock/saturation config + preview
│       ├── model.py        # Async model fitting + job polling
│       ├── results.py      # Model run listing + metrics
│       ├── tune.py         # Holidays, seasonality, re-run
│       ├── visualize.py    # Chart data + CSV download
│       └── optimize.py     # Forward/reverse optimizer + scenarios
│
├── webapp/                 # React frontend
│   ├── src/
│   │   ├── pages/          # Upload, Transform, Model, Results, Tune, Visualize, Optimize
│   │   ├── components/     # Layout, PageHeader, Spinner, StatusBadge
│   │   ├── lib/            # api.ts (Axios client), utils.ts
│   │   └── store/          # Zustand global state (persisted to localStorage)
│   ├── package.json
│   └── vite.config.ts      # Dev server with proxy to :8000
│
└── data/
    └── generate_synthetic_data.py  # Synthetic data generator
```

---

## Data Persistence

**CLI pipeline** — outputs written to `outputs/` (git-ignored):
- `contributions.csv`, `roi_metrics.csv`, `budget_allocation.csv`
- `convergence_summary.csv`, `efficient_frontier.csv`
- `plots/` — all matplotlib figures as PNG

**Webapp** — stored in SQLite (`mmm_sessions.db`, git-ignored):
- Uploaded CSV paths, transform configs
- All model runs with metrics (Adj R², MAPE, R-hat, contributions)
- Optimization scenarios with inputs and results
- Tuning history (holidays, seasonality toggles)

Uploaded CSVs → `uploads/` (git-ignored). Fitted inference data → `idata/` (git-ignored).

The frontend persists `sessionId` and `activeModelId` in **browser localStorage** so you can refresh without losing your place.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pymc'`**
```bash
pip install -r requirements.txt
```

**Port already in use (webapp)**
```bash
# macOS / Linux
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9

# Windows (Anaconda prompt)
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**CORS errors in the browser**

Ensure the backend is running on port 8000. The Vite dev server proxies all API calls — do not call the API directly from a different port.

**Model fit stuck on "Running"**

Check the backend terminal for Python tracebacks. Common causes:
- PyMC / PyTensor version mismatch → `pip install -r requirements.txt`
- CSV missing required columns → check the Upload page shows all 4 required columns

**`npm install` fails**

Ensure Node.js ≥ 18: `node --version`. Behind a corporate proxy: `npm config set proxy http://your-proxy:port`.

**`NotImplementedError: Masked arrays or arrays with nan entries are not supported`**

This means your CSV has weeks where a channel has no spend row, producing NaN after the pivot. This is handled automatically — missing channel-weeks are filled with 0. If you still see this error, check your CSV for:
- Completely empty `media_spend` or `conversions` columns
- Non-numeric values in numeric columns (e.g. `"-"` or `"N/A"` instead of `0`)

**Pipeline crashes on your CSV**

Run with `--fast --no-plots` first to isolate data issues quickly:
```bash
python pipeline.py --fast --no-plots --channel-csv your_data.csv
```
Check that `date`, `channel`, `media_spend`, `conversions` columns all exist and contain no fully-null weeks.
