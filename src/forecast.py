"""
forecast.py
-----------
Forecast the FUTURE (beyond the last known reading).

How: "recursive" forecasting.  We predict hour 1, write the prediction into the
history, then use it as a lag when predicting hour 25, and so on.  Because all
features are >= 24 h old, we can walk forward hour by hour.

Weather assumption: tomorrow's temperature is unknown to us, so we use the
historical average for that (month, hour) - a "climatology" forecast.  In a real
deployment you would plug in a weather-service forecast instead.
"""
import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import build_features


def temperature_climatology(history: pd.DataFrame):
    """Average temperature for every (month, hour) pair, or None if no weather data."""
    if config.TEMP_COL not in history.columns:
        return None
    t = history[config.TEMP_COL]
    return t.groupby([t.index.month, t.index.hour]).mean()


def recursive_forecast(model, feature_cols, history: pd.DataFrame, horizon: int,
                       residual_std: float | None = None) -> pd.DataFrame:
    """Forecast `horizon` hours after the last row of `history`."""
    L = config.LOOKBACK_HOURS
    hist = history.iloc[-L:]
    last = hist.index[-1]
    future_idx = pd.date_range(last + pd.Timedelta(hours=1), periods=horizon, freq="h")
    full_idx = hist.index.append(future_idx)

    energy = pd.Series(
        np.concatenate([hist[config.TARGET_COL].to_numpy(), np.full(horizon, np.nan)]),
        index=full_idx,
    )
    temp = None
    if config.TEMP_COL in history.columns:
        clim = temperature_climatology(history)
        future_t = clim.reindex(
            pd.MultiIndex.from_arrays([future_idx.month, future_idx.hour])
        ).to_numpy()
        temp = pd.Series(
            np.concatenate([hist[config.TEMP_COL].to_numpy(), future_t]), index=full_idx
        )

    for i in range(horizon):
        pos = L + i
        win = slice(pos - L, pos + 1)
        feats = build_features(energy.iloc[win], temp.iloc[win] if temp is not None else None)
        row = feats.iloc[[-1]][feature_cols]
        energy.iloc[pos] = float(model.predict(row)[0])

    out = pd.DataFrame({"Forecast": energy.iloc[L:].to_numpy()}, index=future_idx)
    out.index.name = config.DATETIME_COL
    if residual_std is not None:                 # approximate 95 % band
        out["Lower"] = out["Forecast"] - 1.96 * residual_std
        out["Upper"] = out["Forecast"] + 1.96 * residual_std
    return out.round(2)


class EnergyForecaster:
    """Loads the saved model + history; used by the Flask API and by main.py."""

    def __init__(self, model_path=config.MODEL_PATH, data_path=config.CLEAN_DATA_PATH):
        self.bundle = joblib.load(model_path)
        self.history = pd.read_csv(data_path, parse_dates=[config.DATETIME_COL],
                                   index_col=config.DATETIME_COL)
        self.history.index.freq = "h"

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return self.history.index[-1]

    def forecast(self, hours: int) -> pd.DataFrame:
        if not 1 <= hours <= config.MAX_API_HORIZON_HOURS:
            raise ValueError(f"hours must be between 1 and {config.MAX_API_HORIZON_HOURS}")
        return recursive_forecast(
            self.bundle["model"], self.bundle["feature_columns"], self.history,
            hours, self.bundle["residual_std"],
        )

    def predict_at(self, timestamp) -> float:
        """Predicted kWh for one hour (past hours use real lags, future ones recurse)."""
        ts = pd.Timestamp(timestamp).floor("h")
        if ts > self.last_timestamp:
            hours = int((ts - self.last_timestamp) / pd.Timedelta(hours=1))
            return float(self.forecast(hours)["Forecast"].iloc[-1])
        window = self.history.loc[:ts].iloc[-(config.LOOKBACK_HOURS + 1):]
        if len(window) < config.LOOKBACK_HOURS + 1:
            raise ValueError("timestamp is too early: not enough history before it")
        temp = window[config.TEMP_COL] if self.bundle["has_temperature"] else None
        feats = build_features(window[config.TARGET_COL], temp)
        row = feats.iloc[[-1]][self.bundle["feature_columns"]]
        return float(self.bundle["model"].predict(row)[0])
