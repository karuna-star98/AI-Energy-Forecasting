"""
visualize.py
------------
All charts of the project.  Every function saves a PNG into images/ and returns
the file path.  Numbered file names keep the GitHub image folder tidy.
"""
import matplotlib

matplotlib.use("Agg")                      # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from src import config

sns.set_theme(style="whitegrid", context="notebook")
BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _save(fig, name):
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.IMAGES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overview(df):
    e = df[config.TARGET_COL]
    fig, ax = plt.subplots(2, 1, figsize=(14, 8))
    ax[0].plot(e.index, e, color=BLUE, lw=0.3, alpha=0.5, label="hourly")
    ax[0].plot(e.resample("D").mean(), color=ORANGE, lw=1.4, label="daily average")
    ax[0].set(title="Building electricity consumption - full history", ylabel="kWh per hour")
    ax[0].legend(loc="upper left")
    last = e.iloc[-24 * 14:]
    ax[1].plot(last.index, last, color=BLUE, lw=1.2)
    ax[1].set(title="Zoom: last 14 days (daily rhythm + quiet weekends)", ylabel="kWh per hour")
    return _save(fig, "01_energy_overview.png")


def plot_hourly_profile(df):
    e = df[config.TARGET_COL]
    weekend = e.index.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(10, 5))
    for mask, label, color in [(~weekend, "Weekday", BLUE), (weekend, "Weekend", ORANGE)]:
        g = e[mask].groupby(e[mask].index.hour)
        mean, std = g.mean(), g.std()
        ax.plot(mean.index, mean, color=color, lw=2.2, label=label)
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
    ax.set(title="Average daily load profile", xlabel="Hour of day", ylabel="kWh per hour",
           xticks=range(0, 24, 2))
    ax.legend()
    return _save(fig, "02_hourly_profile.png")


def plot_temperature_relationship(df):
    if config.TEMP_COL not in df.columns:
        return None
    work = df[(df.index.hour.isin(range(10, 17))) & (df.index.dayofweek < 5)]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    hb = ax.hexbin(work[config.TEMP_COL], work[config.TARGET_COL], gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="number of hours")
    ax.set(title="Weekday working hours: temperature vs consumption (air-conditioning effect)",
           xlabel="Outdoor temperature (°C)", ylabel="kWh per hour")
    return _save(fig, "03_temperature_vs_energy.png")


def plot_heatmap(df):
    e = df[config.TARGET_COL]
    pivot = e.groupby([e.index.dayofweek, e.index.hour]).mean().unstack()
    pivot.index = DAYS
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "avg kWh"})
    ax.set(title="When is energy used?  (weekday x hour)", xlabel="Hour of day", ylabel="")
    return _save(fig, "04_weekday_hour_heatmap.png")


def plot_model_comparison(metrics):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    order = metrics.sort_values("RMSE", ascending=False)
    colors = [GREEN if n == metrics.index[0] else BLUE for n in order.index]
    ax[0].barh(order.index, order["RMSE"], color=colors)
    ax[0].set(title="Test RMSE (kWh) - lower is better", xlabel="RMSE")
    ax[1].barh(order.index, order["R2"], color=colors)
    ax[1].set(title="Test R² - higher is better", xlabel="R²")
    ax[1].set_yticklabels([])
    for i, v in enumerate(order["RMSE"]):
        ax[0].text(v, i, f" {v:.1f}", va="center")
    for i, v in enumerate(order["R2"]):
        ax[1].text(v, i, f" {v:.3f}", va="center")
    return _save(fig, "05_model_comparison.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name, days=7):
    n = 24 * days
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_true.index[-n:], y_true.iloc[-n:], color=BLUE, lw=1.8, label="Actual")
    ax.plot(y_true.index[-n:], np.asarray(y_pred)[-n:], color=ORANGE, lw=1.8, ls="--", label="Predicted")
    ax.set(title=f"Actual vs predicted - last {days} days of the test set ({model_name})",
           ylabel="kWh per hour")
    ax.legend()
    return _save(fig, "06_actual_vs_predicted.png")


def plot_scatter(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true, y_pred, s=6, alpha=0.3, color=BLUE)
    lim = [min(y_true.min(), np.min(y_pred)), max(y_true.max(), np.max(y_pred))]
    ax.plot(lim, lim, color="red", lw=1.2, label="perfect prediction")
    ax.set(title=f"Predicted vs actual ({model_name})", xlabel="Actual kWh", ylabel="Predicted kWh")
    ax.legend()
    return _save(fig, "07_prediction_scatter.png")


def plot_residuals(y_true, y_pred):
    resid = y_true - np.asarray(y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.histplot(resid, bins=50, kde=True, ax=ax[0], color=BLUE)
    ax[0].axvline(0, color="red", lw=1)
    ax[0].set(title="Forecast errors (actual - predicted)", xlabel="kWh")
    mae = resid.abs().groupby(resid.index.hour).mean()
    ax[1].bar(mae.index, mae.values, color=ORANGE)
    ax[1].set(title="Average absolute error by hour of day", xlabel="Hour", ylabel="MAE (kWh)")
    return _save(fig, "08_residual_analysis.png")


def plot_feature_importance(model, X_test, y_test, cols, n_rows=3000):
    sample = X_test[cols].sample(min(n_rows, len(X_test)), random_state=config.RANDOM_STATE)
    res = permutation_importance(model, sample, y_test.loc[sample.index], n_repeats=5,
                                 random_state=config.RANDOM_STATE, scoring="neg_root_mean_squared_error")
    imp = pd.Series(res.importances_mean, index=cols).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(imp.index, imp.values, color=GREEN)
    ax.set(title="What does the model rely on? (permutation importance)",
           xlabel="Increase in RMSE (kWh) when the feature is shuffled")
    return _save(fig, "09_feature_importance.png")


def plot_forecast(history, forecast, days_back=7):
    tail = history[config.TARGET_COL].iloc[-24 * days_back:]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(tail.index, tail, color=BLUE, lw=1.6, label="Recent actual")
    ax.plot(forecast.index, forecast["Forecast"], color=ORANGE, lw=1.8, label="Forecast (next 7 days)")
    if "Lower" in forecast:
        ax.fill_between(forecast.index, forecast["Lower"], forecast["Upper"], color=ORANGE,
                        alpha=0.2, label="≈95 % range")
    ax.axvline(tail.index[-1], color=GREY, ls=":", lw=1.2)
    ax.set(title="Future energy demand forecast", ylabel="kWh per hour")
    ax.legend(loc="upper left")
    return _save(fig, "10_future_forecast.png")


def plot_simulation(sim: pd.DataFrame, currency: str = config.CURRENCY):
    sim = sim.copy()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    palette = [GREY if s == "No load shifting" else (GREEN if "Perfect" in s else BLUE)
               for s in sim["Strategy"]]
    ax[0].barh(sim["Strategy"], sim["Savings"], color=palette)
    ax[0].set(title=f"Money saved over the test period ({currency})", xlabel=currency)
    for i, (v, p) in enumerate(zip(sim["Savings"], sim["Savings_%"])):
        ax[0].text(v, i, f" {v:,.0f} ({p:.2f}%)", va="center")
    ax[0].invert_yaxis()
    ax[1].barh(sim["Strategy"], sim["Peak_reduction_%"], color=palette)
    ax[1].set(title="Average daily peak demand reduction (%)", xlabel="%")
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    return _save(fig, "11_simulation_savings.png")
