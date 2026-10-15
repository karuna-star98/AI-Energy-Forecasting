"""
data_loader.py
--------------
Load the raw CSV and turn it into a clean, regular hourly time series.

Cleaning steps (each one is counted and reported):
    1. parse timestamps and sort
    2. drop duplicated timestamps
    3. resample to a strict hourly grid  (gaps become NaN)
    4. flag sensor spikes (readings > 2x the typical value for that weekday+hour)
    5. fill short gaps by interpolation, long gaps with "same hour last week"
"""
import numpy as np
import pandas as pd

from src import config


def load_raw(path=config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {config.DATETIME_COL, config.TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} must contain the columns {config.DATETIME_COL!r} and "
            f"{config.TARGET_COL!r} (optional: {config.TEMP_COL!r}). Missing: {missing}"
        )
    return df


def _fill(series: pd.Series, max_gap: int) -> pd.Series:
    filled = series.interpolate(method="time", limit=max_gap, limit_area="inside")
    filled = filled.fillna(filled.shift(168))      # same hour, previous week
    return filled.ffill().bfill()


def clean_data(df: pd.DataFrame, spike_ratio: float = 2.0, max_gap: int = 6):
    """Return (clean_df, report).  clean_df has a regular hourly DatetimeIndex."""
    report = {"rows_raw": int(len(df))}

    df = df.copy()
    df[config.DATETIME_COL] = pd.to_datetime(df[config.DATETIME_COL])
    df = df.sort_values(config.DATETIME_COL)

    report["duplicate_rows_removed"] = int(df.duplicated(subset=config.DATETIME_COL).sum())
    df = df.drop_duplicates(subset=config.DATETIME_COL, keep="first")

    keep = [c for c in (config.TARGET_COL, config.TEMP_COL) if c in df.columns]
    df = df.set_index(config.DATETIME_COL)[keep].resample("h").mean()
    report["rows_hourly_grid"] = int(len(df))
    report["missing_hours_before"] = int(df[config.TARGET_COL].isna().sum())

    # Sensor spikes: compare every reading with the typical value for its
    # (weekday, hour).  A reading more than `spike_ratio` times larger than
    # normal is not real consumption - it is a faulty measurement.
    s = df[config.TARGET_COL]
    typical = s.groupby([s.index.dayofweek, s.index.hour]).transform("median")
    is_outlier = (s / typical) > spike_ratio
    report["outliers_flagged"] = int(is_outlier.sum())
    df.loc[is_outlier, config.TARGET_COL] = np.nan

    for col in keep:
        df[col] = _fill(df[col], max_gap)
    report["missing_hours_after"] = int(df[config.TARGET_COL].isna().sum())
    report["start"] = str(df.index[0])
    report["end"] = str(df.index[-1])
    report["rows_clean"] = int(len(df))
    return df, report
