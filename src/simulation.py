"""
simulation.py
-------------
Virtual business simulation: "What is a good forecast worth in money?"

Scenario: a building manager pays a time-of-use tariff (peak hours cost more)
plus a demand charge on the highest hourly load of each day.
Part of the load is flexible (pre-cooling, battery charging, batch jobs,
EV charging).  Each evening the manager looks at TOMORROW's forecast, finds the
most expensive-looking hours and moves a slice of flexible load to the quietest hours.

We then replay what REALLY happened and compare strategies:
    No shifting      - the baseline bill
    Naive forecast   - decisions based on "same as last week"
    AI forecast      - decisions based on the best ML model
    Perfect foresight- decisions based on the actual values (upper bound)
"""
import numpy as np
import pandas as pd

from src import config


def _shift_day(actual: np.ndarray, guide: np.ndarray, k: int, frac: float) -> np.ndarray:
    """Move `frac` of the load in the k hours `guide` thinks are highest into the k lowest."""
    out = actual.copy()
    top = np.argsort(guide)[-k:]
    bottom = np.argsort(guide)[:k]
    moved = frac * actual[top]
    out[top] -= moved
    out[bottom] += moved.sum() / k
    return out


def run_simulation(actual: pd.Series, predictions: dict, k: int = config.SHIFT_HOURS,
                   frac: float = config.SHIFT_FRACTION):
    """predictions = {strategy_name: pd.Series aligned with `actual`}.
    Returns a summary DataFrame (one row per strategy)."""
    price = config.tariff_per_kwh()
    days = [d for d, g in actual.groupby(actual.index.date) if len(g) == 24]

    def evaluate(guide: pd.Series | None):
        cost, peak, hits = 0.0, [], []
        for d in days:
            a_day = actual[actual.index.date == d]
            a = a_day.to_numpy()
            if guide is None:
                shifted = a
            else:
                g = guide.loc[a_day.index].to_numpy()
                shifted = _shift_day(a, g, k, frac)
                hits.append(len(set(np.argsort(g)[-k:]) & set(np.argsort(a)[-k:])) / k)
            cost += float((shifted * price).sum()) + config.DEMAND_CHARGE_PER_KW_DAY * shifted.max()
            peak.append(shifted.max())
        return cost, float(np.mean(peak)), (float(np.mean(hits)) if hits else np.nan)

    base_cost, base_peak, _ = evaluate(None)
    rows = [{"Strategy": "No load shifting", "Total_cost": base_cost, "Avg_daily_peak_kWh": base_peak,
             "Peak_hour_hit_rate_%": np.nan}]
    for name, guide in {**predictions, "Perfect foresight": actual}.items():
        cost, peak, hit = evaluate(guide)
        rows.append({"Strategy": name, "Total_cost": cost, "Avg_daily_peak_kWh": peak,
                     "Peak_hour_hit_rate_%": hit * 100})
    df = pd.DataFrame(rows)
    df["Savings"] = base_cost - df["Total_cost"]
    df["Savings_%"] = df["Savings"] / base_cost * 100
    df["Peak_reduction_%"] = (base_peak - df["Avg_daily_peak_kWh"]) / base_peak * 100
    df.attrs["days"] = len(days)
    return df.round(2)
