"""
features.py
-----------
Feature engineering: turn a plain hourly series into columns an ML model can use.

Design rule (very important!):
    The model forecasts DAY-AHEAD, so every feature must be knowable 24 hours
    in advance.  That is why all lag features start at lag_24 - never lag_1.
    Using yesterday's data is fine, using "one hour ago" would be cheating for a
    day-ahead forecast ("data leakage").

Features
    calendar   hour, dayofweek, month, is_weekend, is_holiday
    cyclical   sin/cos of hour, weekday, month (so 23:00 is close to 00:00)
    weather    temperature, cooling_degree, heating_degree   (if available)
    history    lag_24, lag_48, lag_168 (same hour yesterday/2 days/last week),
               rolling mean/std of the 24 h that ended 24 h ago, weekly mean
"""
import numpy as np
import pandas as pd

from src import config

BASIC_FEATURES = ["hour", "dayofweek"]       # the two features used in the original brief


def build_features(energy: pd.Series, temperature: pd.Series | None = None) -> pd.DataFrame:
    """Build the feature table for a *regular hourly* energy series."""
    idx = energy.index
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    month = idx.month.to_numpy()

    f = pd.DataFrame(index=idx)
    f["hour"] = hour
    f["dayofweek"] = dow
    f["month"] = month
    f["is_weekend"] = (dow >= 5).astype(int)
    f["is_holiday"] = np.fromiter(
        ((m, d) in config.HOLIDAYS for m, d in zip(month, idx.day)), dtype=int, count=len(idx)
    )

    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    f["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    f["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    if temperature is not None:
        t = temperature.to_numpy()
        f["temperature"] = t
        f["cooling_degree"] = np.clip(t - 24, 0, None)
        f["heating_degree"] = np.clip(15 - t, 0, None)

    f["lag_24"] = energy.shift(24)
    f["lag_48"] = energy.shift(48)
    f["lag_168"] = energy.shift(168)
    past = energy.shift(24)                      # only use data that is >= 24 h old
    f["roll_mean_24"] = past.rolling(24).mean()
    f["roll_std_24"] = past.rolling(24).std()
    f["roll_mean_168"] = past.rolling(168).mean()
    return f


def make_supervised(clean_df: pd.DataFrame):
    """Return X (features) and y (target), with the warm-up rows removed."""
    temp = clean_df[config.TEMP_COL] if config.TEMP_COL in clean_df.columns else None
    y_all = clean_df[config.TARGET_COL]
    X_all = build_features(y_all, temp)
    valid = X_all.notna().all(axis=1)
    return X_all[valid], y_all[valid]
