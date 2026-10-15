"""
train.py
--------
Chronological train/test split, model training and comparison.

IMPORTANT: for time series we NEVER shuffle.  The model trains on the past and
is tested on the future - exactly like in real life.  (A random split would
let the model peek at the neighbours of every test hour.)
"""
import time

import joblib
import pandas as pd
from sklearn.base import clone

from src import config
from src.evaluate import metrics_table, regression_metrics
from src.models import get_model_specs


def chronological_split(X: pd.DataFrame, y: pd.Series, test_fraction: float = config.TEST_FRACTION):
    cut = int(len(X) * (1 - test_fraction))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


def train_all(X_train, X_test, y_train, y_test, logger=None):
    """Train every candidate; return (metrics_table, fitted_models, predictions, specs)."""
    log = logger.info if logger else print
    specs = get_model_specs(list(X_train.columns))
    fitted, predictions, scores = {}, {}, {}
    for name, (model, cols) in specs.items():
        t0 = time.time()
        model.fit(X_train[cols], y_train)
        pred = pd.Series(model.predict(X_test[cols]), index=X_test.index, name=name)
        scores[name] = regression_metrics(y_test, pred)
        fitted[name], predictions[name] = model, pred
        log(
            f"trained {name:<28} RMSE={scores[name]['RMSE']:7.2f}  "
            f"MAE={scores[name]['MAE']:6.2f}  R2={scores[name]['R2']:.4f}  ({time.time()-t0:.1f}s)"
        )
    return metrics_table(scores), fitted, predictions, specs


def save_best_model(best_name, specs, X_all, y_all, residual_std, clean_df, path=config.MODEL_PATH):
    """Refit the winning model on ALL data (more recent history = better forecasts)
    and store it together with everything needed to use it later."""
    model, cols = specs[best_name]
    final = clone(model).fit(X_all[cols], y_all)
    bundle = {
        "model_name": best_name,
        "model": final,
        "feature_columns": cols,
        "has_temperature": config.TEMP_COL in clean_df.columns,
        "residual_std": float(residual_std),
        "trained_until": str(X_all.index[-1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return bundle
