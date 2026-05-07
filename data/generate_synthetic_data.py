"""
Synthetic data generator for Bayesian MMM.

Generates weekly media spend, impressions, and conversions across a
Channel → Sub-channel → Campaign hierarchy, including control variables.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Hierarchy definition
# ---------------------------------------------------------------------------

HIERARCHY = {
    "TV": {
        "sub_channels": {
            "TV_National": ["TV_Nat_Brand", "TV_Nat_Promo"],
            "TV_Regional": ["TV_Reg_Awareness", "TV_Reg_Direct"],
        }
    },
    "Digital": {
        "sub_channels": {
            "Paid_Search": ["Search_Brand", "Search_Generic", "Search_Competitor"],
            "Social": ["Social_Facebook", "Social_Instagram", "Social_TikTok"],
            "Display": ["Display_Prospecting", "Display_Retargeting"],
        }
    },
    "Radio": {
        "sub_channels": {
            "Radio_AM": ["Radio_AM_Drive", "Radio_AM_Midday"],
            "Radio_FM": ["Radio_FM_Drive"],
        }
    },
    "OOH": {
        "sub_channels": {
            "Billboard": ["Billboard_Highway", "Billboard_Urban"],
            "Transit": ["Transit_Bus", "Transit_Metro"],
        }
    },
}

# True ground-truth parameters (used to generate realistic data)
TRUE_PARAMS = {
    # adstock decay per channel
    "adstock_decay": {
        "TV": 0.65,
        "Digital": 0.20,
        "Radio": 0.40,
        "OOH": 0.50,
    },
    # Hill saturation: (alpha, gamma) – alpha=slope, gamma=half-saturation
    "hill_alpha": {"TV": 2.5, "Digital": 1.8, "Radio": 2.0, "OOH": 2.2},
    "hill_gamma": {"TV": 0.45, "Digital": 0.35, "Radio": 0.40, "OOH": 0.42},
    # base channel coefficient (contribution scale)
    "channel_coef": {"TV": 0.55, "Digital": 0.80, "Radio": 0.25, "OOH": 0.20},
    # cross-channel halo: TV lifts Search
    "halo": {("TV", "Digital"): 0.08},
    # base conversions (intercept)
    "base_conversions": 4000,
    "noise_std": 200,
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        out[t] = x[t] + (decay * out[t - 1] if t > 0 else 0.0)
    return out


def _hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    x_norm = x / (x.max() + 1e-8)
    return x_norm**alpha / (x_norm**alpha + gamma**alpha)


def _fourier_seasonality(n_weeks: int, periods: list[float], n_harmonics: int = 2) -> pd.DataFrame:
    t = np.arange(n_weeks)
    cols = {}
    for p in periods:
        for k in range(1, n_harmonics + 1):
            cols[f"sin_{p}_{k}"] = np.sin(2 * np.pi * k * t / p)
            cols[f"cos_{p}_{k}"] = np.cos(2 * np.pi * k * t / p)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(
    n_weeks: int = 104,
    start_date: str = "2022-01-03",
    seed: int = 42,
    include_holidays: bool = True,
    include_promotions: bool = True,
) -> pd.DataFrame:
    """Return a tidy DataFrame with one row per (week, campaign)."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start=start_date, periods=n_weeks, freq="W-MON")

    # --- Seasonality components (annual + semi-annual) --------------------
    season_df = _fourier_seasonality(n_weeks, periods=[52, 26], n_harmonics=2)
    season_signal = (
        0.15 * season_df["sin_52_1"]
        + 0.08 * season_df["cos_52_1"]
        + 0.05 * season_df["sin_26_1"]
    ).values

    # --- Holiday indicator ------------------------------------------------
    holiday_weeks = []
    for yr in [2022, 2023]:
        # Christmas / New Year region
        holiday_weeks += list(
            pd.date_range(f"{yr}-12-19", periods=3, freq="W-MON")
        )
        # Ramadan proxy (varies; use a fixed window for synthetic)
        holiday_weeks += list(
            pd.date_range(f"{yr}-04-04", periods=4, freq="W-MON")
        )
    holiday_flag = np.array(
        [1 if d in holiday_weeks else 0 for d in dates], dtype=float
    )

    # --- Promotion indicator ----------------------------------------------
    promo_starts = [
        pd.Timestamp("2022-03-07"),
        pd.Timestamp("2022-09-05"),
        pd.Timestamp("2023-03-06"),
        pd.Timestamp("2023-09-04"),
    ]
    promo_flag = np.zeros(n_weeks)
    for s in promo_starts:
        for i, d in enumerate(dates):
            if s <= d < s + pd.Timedelta(weeks=3):
                promo_flag[i] = 1.0

    # --- Spend generation per channel -------------------------------------
    channel_spend: dict[str, np.ndarray] = {}
    channel_impressions: dict[str, np.ndarray] = {}

    spend_profiles = {
        "TV": {"mean": 500_000, "std": 80_000, "cpm": 12.0},
        "Digital": {"mean": 300_000, "std": 60_000, "cpm": 4.0},
        "Radio": {"mean": 80_000, "std": 15_000, "cpm": 8.0},
        "OOH": {"mean": 120_000, "std": 20_000, "cpm": 6.0},
    }

    total_conversions = np.full(n_weeks, float(TRUE_PARAMS["base_conversions"]))
    total_conversions += season_signal * 800

    if include_holidays:
        total_conversions += holiday_flag * 600

    if include_promotions:
        total_conversions += promo_flag * 400

    adstocked: dict[str, np.ndarray] = {}

    for ch, prof in spend_profiles.items():
        raw_spend = rng.normal(prof["mean"], prof["std"], n_weeks).clip(0)
        # Inject seasonal uplift in spend
        raw_spend *= 1 + 0.2 * season_signal
        raw_spend = raw_spend.clip(0)
        channel_spend[ch] = raw_spend
        channel_impressions[ch] = raw_spend / prof["cpm"] * 1000  # impressions

        ad = _geometric_adstock(raw_spend, TRUE_PARAMS["adstock_decay"][ch])
        sat = _hill_saturation(
            ad,
            TRUE_PARAMS["hill_alpha"][ch],
            TRUE_PARAMS["hill_gamma"][ch],
        )
        adstocked[ch] = sat
        total_conversions += TRUE_PARAMS["channel_coef"][ch] * sat * 3000

    # Halo effect: TV → Digital
    if ("TV", "Digital") in TRUE_PARAMS["halo"]:
        halo_coef = TRUE_PARAMS["halo"][("TV", "Digital")]
        total_conversions += halo_coef * adstocked["TV"] * adstocked["Digital"] * 2000

    # Add noise
    total_conversions += rng.normal(0, TRUE_PARAMS["noise_std"], n_weeks)
    total_conversions = total_conversions.clip(0)

    # --- Build campaign-level rows ----------------------------------------
    rows = []
    for ch, sub_dict in HIERARCHY.items():
        ch_spend = channel_spend[ch]
        ch_impr = channel_impressions[ch]

        sub_channels = sub_dict["sub_channels"]
        # Allocate channel spend to sub-channels using fixed weights + noise
        n_sub = len(sub_channels)
        sub_weights_base = rng.dirichlet(np.ones(n_sub) * 3)

        for si, (sub_ch, campaigns) in enumerate(sub_channels.items()):
            n_camp = len(campaigns)
            camp_weights_base = rng.dirichlet(np.ones(n_camp) * 3)

            for wi in range(n_weeks):
                # Slightly vary weights over time for realism
                sub_noise = rng.dirichlet(np.ones(n_sub) * 20)
                sub_w = 0.85 * sub_weights_base + 0.15 * sub_noise
                sub_w /= sub_w.sum()

                camp_noise = rng.dirichlet(np.ones(n_camp) * 20)
                camp_w = 0.85 * camp_weights_base + 0.15 * camp_noise
                camp_w /= camp_w.sum()

                sub_spend_w = ch_spend[wi] * sub_w[si]
                sub_impr_w = ch_impr[wi] * sub_w[si]

                for ci, camp in enumerate(campaigns):
                    rows.append(
                        {
                            "date": dates[wi],
                            "week_number": wi + 1,
                            "channel": ch,
                            "sub_channel": sub_ch,
                            "campaign": camp,
                            "media_spend": sub_spend_w * camp_w[ci],
                            "impressions": sub_impr_w * camp_w[ci],
                            "conversions": total_conversions[wi]
                            / sum(len(c) for s in HIERARCHY.values() for c in s["sub_channels"].values()),
                            # Control variables (same for all campaigns in same week)
                            "seasonality_sin_52": season_df["sin_52_1"].iloc[wi],
                            "seasonality_cos_52": season_df["cos_52_1"].iloc[wi],
                            "seasonality_sin_26": season_df["sin_26_1"].iloc[wi],
                            "seasonality_cos_26": season_df["cos_26_1"].iloc[wi],
                            "holiday_flag": holiday_flag[wi],
                            "promo_flag": promo_flag[wi],
                        }
                    )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "channel", "sub_channel", "campaign"]).reset_index(
        drop=True
    )
    return df


def generate_channel_weekly(
    n_weeks: int = 104,
    start_date: str = "2022-01-03",
    seed: int = 42,
    include_holidays: bool = True,
    include_promotions: bool = True,
) -> pd.DataFrame:
    """Return a channel-level weekly DataFrame (aggregated from campaign rows)."""
    df = generate_synthetic_data(
        n_weeks=n_weeks,
        start_date=start_date,
        seed=seed,
        include_holidays=include_holidays,
        include_promotions=include_promotions,
    )
    agg = (
        df.groupby(["date", "week_number", "channel"])
        .agg(
            media_spend=("media_spend", "sum"),
            impressions=("impressions", "sum"),
            conversions=("conversions", "sum"),
            seasonality_sin_52=("seasonality_sin_52", "first"),
            seasonality_cos_52=("seasonality_cos_52", "first"),
            seasonality_sin_26=("seasonality_sin_26", "first"),
            seasonality_cos_26=("seasonality_cos_26", "first"),
            holiday_flag=("holiday_flag", "first"),
            promo_flag=("promo_flag", "first"),
        )
        .reset_index()
    )
    return agg


def generate_wide_channel_weekly(
    n_weeks: int = 104,
    start_date: str = "2022-01-03",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a wide (one row per week) DataFrame with channel spend as columns."""
    long = generate_channel_weekly(n_weeks=n_weeks, start_date=start_date, seed=seed)
    control_cols = [
        "seasonality_sin_52", "seasonality_cos_52",
        "seasonality_sin_26", "seasonality_cos_26",
        "holiday_flag", "promo_flag",
    ]
    controls = (
        long[["date", "week_number"] + control_cols]
        .drop_duplicates("date")
        .set_index("date")
    )

    spend_wide = long.pivot(index="date", columns="channel", values="media_spend")
    spend_wide.columns = [f"spend_{c}" for c in spend_wide.columns]

    conv_wide = (
        long.groupby("date")["conversions"].sum().rename("total_conversions")
    )

    wide = pd.concat([spend_wide, conv_wide, controls], axis=1).reset_index()
    return wide


if __name__ == "__main__":
    import os

    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating campaign-level data …")
    df_camp = generate_synthetic_data(n_weeks=104)
    df_camp.to_csv(os.path.join(out_dir, "synthetic_campaign_data.csv"), index=False)
    print(f"  Saved {len(df_camp):,} rows → outputs/synthetic_campaign_data.csv")

    print("Generating channel-level data …")
    df_ch = generate_channel_weekly(n_weeks=104)
    df_ch.to_csv(os.path.join(out_dir, "synthetic_channel_data.csv"), index=False)
    print(f"  Saved {len(df_ch):,} rows → outputs/synthetic_channel_data.csv")

    print("Generating wide format data …")
    df_wide = generate_wide_channel_weekly(n_weeks=104)
    df_wide.to_csv(os.path.join(out_dir, "synthetic_wide_data.csv"), index=False)
    print(f"  Saved {len(df_wide):,} rows → outputs/synthetic_wide_data.csv")

    print("\nSample (wide):")
    print(df_wide.head(3).to_string())
    print("\nTrue parameters used:")
    for k, v in TRUE_PARAMS.items():
        print(f"  {k}: {v}")
