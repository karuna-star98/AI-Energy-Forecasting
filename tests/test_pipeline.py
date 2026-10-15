"""Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd

from src import config
from src.data_generation import generate_energy_data
from src.data_loader import clean_data
from src.features import build_features, make_supervised
from src.forecast import recursive_forecast
from src.models import SeasonalNaive
from src.train import chronological_split

SMALL = dict(start="2023-01-01", end="2023-03-31 23:00")


def _clean():
    return clean_data(generate_energy_data(**SMALL))[0]


def test_generated_data_is_messy_but_cleaning_fixes_it():
    raw = generate_energy_data(**SMALL)
    assert raw[config.TARGET_COL].isna().any()
    clean, report = clean_data(raw)
    assert clean.isna().sum().sum() == 0
    assert clean.index.is_monotonic_increasing and clean.index.is_unique
    assert report["duplicate_rows_removed"] > 0
    assert (clean.index.to_series().diff().dropna() == pd.Timedelta("1h")).all()


def test_features_do_not_leak_the_future():
    clean = _clean()
    f = build_features(clean[config.TARGET_COL], clean[config.TEMP_COL])
    # every history feature must come from >= 24 hours ago
    assert f["lag_24"].iloc[100] == clean[config.TARGET_COL].iloc[100 - 24]
    assert f["lag_168"].iloc[300] == clean[config.TARGET_COL].iloc[300 - 168]
    assert "lag_1" not in f.columns


def test_split_is_chronological():
    X, y = make_supervised(_clean())
    X_tr, X_te, y_tr, y_te = chronological_split(X, y, 0.2)
    assert X_tr.index.max() < X_te.index.min()
    assert len(X_tr) + len(X_te) == len(X)


def test_recursive_forecast_shape_and_values():
    clean = _clean()
    X, y = make_supervised(clean)
    model = SeasonalNaive()
    fc = recursive_forecast(model, ["lag_168"], clean, horizon=48)
    assert len(fc) == 48
    assert fc.index[0] == clean.index[-1] + pd.Timedelta(hours=1)
    assert np.isfinite(fc["Forecast"]).all()
    # a seasonal-naive forecast of hour 1 equals the value one week earlier
    assert fc["Forecast"].iloc[0] == round(clean[config.TARGET_COL].iloc[-168], 2)
