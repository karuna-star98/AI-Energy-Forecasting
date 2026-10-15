"""
models.py
---------
The candidate models we compare.  Starting with a trivial baseline is a habit
of good data scientists: a fancy model only matters if it beats the baseline.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.features import BASIC_FEATURES


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Baseline: 'next Monday 10:00 will look like last Monday 10:00'."""

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.asarray(X["lag_168"], dtype=float)


def _mlp(**kw):
    return make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 64),
            max_iter=500,
            early_stopping=True,
            random_state=config.RANDOM_STATE,
            **kw,
        ),
    )


def get_model_specs(feature_cols: list[str]) -> dict:
    """Return {name: (untrained_model, columns_it_uses)}."""
    rs = config.RANDOM_STATE
    return {
        "Seasonal Naive (last week)": (SeasonalNaive(), ["lag_168"]),
        "Linear Regression": (make_pipeline(StandardScaler(), LinearRegression()), feature_cols),
        # The model from the original brief: MLP using only hour + weekday.
        "MLP (hour + day only)": (_mlp(), BASIC_FEATURES),
        "MLP (all features)": (_mlp(), feature_cols),
        "Random Forest": (
            RandomForestRegressor(
                n_estimators=150, min_samples_leaf=3, max_features=0.6, n_jobs=-1, random_state=rs
            ),
            feature_cols,
        ),
        "Gradient Boosting": (
            HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=rs),
            feature_cols,
        ),
    }
