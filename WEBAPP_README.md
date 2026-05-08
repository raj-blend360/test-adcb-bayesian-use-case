# Bayesian MMM — React Webapp

A browser-based UI for data scientists to upload data, configure and run Bayesian MMM models, visualize results, and optimize budgets.

---

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Node.js | ≥ 18 | [nodejs.org](https://nodejs.org) |
| npm | ≥ 9 | Bundled with Node.js |

---

## Quick Start (one command)

From the repo root:

```bash
./start.sh
```

This installs all dependencies and starts both the backend and frontend:

- **Backend API**: http://localhost:8000
- **Frontend UI**: http://localhost:5173
- **API docs (Swagger)**: http://localhost:8000/docs

Press `Ctrl+C` to stop both servers.

---

## Manual Setup

If you prefer to start the backend and frontend separately:

### 1. Backend (FastAPI)

```bash
# From the repo root
pip install -r api/requirements.txt

uvicorn api.main:app --reload --port 8000
```

The backend also requires the core MMM packages (PyMC, NumPy, etc.):

```bash
pip install -r requirements.txt
```

### 2. Frontend (React)

Open a second terminal:

```bash
cd webapp
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

---

## Usage Workflow

The app follows a linear 7-step workflow. Navigate between steps using the sidebar.

### Step 1 — Upload
- Drag and drop or click to upload your **channel-level CSV** (required).
- Optionally upload a **campaign-level CSV** to enable campaign-level halo effects.
- Required columns for channel CSV: `date`, `channel`, `media_spend`, `conversions`
- Optional: `impressions`, `clicks`, `sub_channel`, `campaign`

### Step 2 — Transform
- Select global features: seasonality, holiday flag, promo flag.
- Expand each channel to configure:
  - **Adstock**: Geometric or Weibull, max lag (1–13 weeks), decay prior.
  - **Saturation**: Hill, Logistic, or Michaelis-Menten, with parameter priors.
  - **Target metric**: conversions / impressions / clicks.
- Click **Preview Adstock** to see a live chart of raw vs adstocked spend.
- Click **Save & Continue** when done.

### Step 3 — Model
- Choose inference method: **MAP** (~30s), **ADVI** (~2min), or **MCMC** (~10min+).
- Set sampling parameters: samples, tune, chains, target acceptance rate.
- Optionally add **halo effect pairs**:
  - Toggle between channel-level or campaign-level pairs.
  - For campaign pairs, the "Subtract from channel" checkbox removes that campaign's spend from its parent channel's main effect to avoid double-counting.
- Click **Run Model**. A status indicator shows progress; the page polls automatically.

### Step 4 — Results
- Compare all model runs in a table: Adj R², MAPE, R-hat pass%, Confidence Width.
- Click the eye icon to set a model as active for visualization.
- Click the save icon to give a model a name.
- Check multiple rows to see a side-by-side comparison chart.

### Step 5 — Tune
- Add named **holiday periods** with date ranges (e.g. Eid Al-Adha, Ramadan).
- Toggle **seasonality components**: quarterly, half-yearly, annual.
- Click **Re-run with Tuning** to create a new iteration of the active model.
- All tuning history is saved and shown in a collapsible list.

### Step 6 — Visualize
- Select any completed model from the dropdown.
- Switch between 5 chart tabs:
  - **Contributions** — mean contribution per channel (bar chart)
  - **Response Curves** — saturation curve with confidence band per channel
  - **Weekly Decomp** — media vs non-media stacked area over time
  - **ROI** — return on investment per channel (horizontal bar)
  - **Waterfall** — baseline → channel-by-channel cumulative build-up
- Every chart has a **Download CSV** button.

### Step 7 — Optimize
- Toggle between **Forward** (budget → maximize conversions) and **Reverse** (target conversions → minimize spend).
- Set per-channel minimum spend and optionally cap the maximum.
- Click **Run Optimization** to get the optimal allocation.
- Results show: total spend, expected conversions, pie chart of allocation, comparison bar vs current.
- Scenarios are saved automatically and listed at the bottom of the page.

---

## File Structure

```
repo/
├── api/                    # FastAPI backend
│   ├── main.py             # App entry point
│   ├── database.py         # SQLite + SQLAlchemy setup
│   ├── models.py           # ORM tables (sessions, model_runs, scenarios)
│   ├── schemas.py          # Pydantic request/response models
│   ├── deps.py             # DB session dependency
│   └── routers/
│       ├── upload.py
│       ├── transform.py
│       ├── model.py        # Async model fitting
│       ├── results.py
│       ├── tune.py
│       ├── visualize.py
│       └── optimize.py
├── webapp/                 # React frontend
│   ├── src/
│   │   ├── pages/          # Upload, Transform, Model, Results, Tune, Visualize, Optimize
│   │   ├── components/     # Layout, PageHeader, Spinner, StatusBadge
│   │   ├── lib/            # api.ts (Axios client), utils.ts
│   │   └── store/          # Zustand global state (persisted to localStorage)
│   ├── package.json
│   └── vite.config.ts      # Dev server with proxy to :8000
├── src/                    # Existing Python MMM code (unchanged)
├── start.sh                # One-command launcher
└── WEBAPP_README.md        # This file
```

---

## Data Persistence

The backend stores all session data in **SQLite** (`mmm_sessions.db` in the repo root, git-ignored). This includes:

- Uploaded CSV paths
- Transform configs
- All model runs with metrics
- Optimization scenarios
- Tuning history

Uploaded CSVs are saved to `uploads/` (git-ignored). Fitted model inference data is saved to `idata/` (git-ignored).

The frontend persists session ID and active model selection in **browser localStorage**, so you can refresh without losing your place.

---

## Troubleshooting

**Port already in use**

```bash
# Kill whatever is using port 8000 or 5173
lsof -ti:8000 | xargs kill -9
lsof -ti:5173 | xargs kill -9
```

**CORS errors in the browser**

Ensure the backend is running on port 8000. The Vite dev server proxies all API calls automatically — do not access the API directly from the browser on a different port.

**Model fit stuck on "Running"**

Check the backend terminal for Python tracebacks. Common causes:
- PyMC / PyTensor version mismatch → run `pip install -r requirements.txt` again.
- Invalid CSV column names → check the Upload page shows all required columns.

**`npm install` fails**

Ensure Node.js ≥ 18 is installed: `node --version`. If behind a corporate proxy, configure npm: `npm config set proxy http://your-proxy:port`.
