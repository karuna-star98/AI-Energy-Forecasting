"""evaluate.py - error metrics used to judge the forecasts."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    """MAE / RMSE in kWh, R2 (1.0 is perfect) and MAPE in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "MAPE_%": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """results = {model_name: metrics_dict}  ->  DataFrame sorted by RMSE."""
    table = pd.DataFrame(results).T.sort_values("RMSE")
    return table.round(3)
