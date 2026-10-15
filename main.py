"""
main.py - run the complete AI Energy Forecasting pipeline.

    python main.py                       # generate virtual data, train, forecast, simulate
    python main.py --data my_data.csv    # use your own CSV (Datetime, Energy[, Temperature])
    python main.py --regenerate          # create a fresh virtual dataset
"""
import argparse

import pandas as pd

from src import config, visualize
from src.data_generation import generate_energy_data, save_raw_data
from src.data_loader import clean_data, load_raw
from src.features import make_supervised
from src.forecast import recursive_forecast
from src.simulation import run_simulation
from src.train import chronological_split, save_best_model, train_all
from src.utils import get_logger


def main():
    parser = argparse.ArgumentParser(description="AI-powered energy consumption forecasting")
    parser.add_argument("--data", default=None, help="path to your own CSV file")
    parser.add_argument("--regenerate", action="store_true", help="regenerate the virtual dataset")
    parser.add_argument("--horizon", type=int, default=config.FORECAST_HORIZON_HOURS,
                        help="hours to forecast into the future (default 168 = 7 days)")
    args = parser.parse_args()

    config.ensure_dirs()
    logger = get_logger()
    log = logger.info

    # ---- Phase 2: dataset --------------------------------------------------
    raw_path = config.RAW_DATA_PATH
    if args.data:
        raw_path = args.data
    elif args.regenerate or not raw_path.exists():
        log("STEP 1/7  Generating virtual smart-meter data ...")
        save_raw_data(generate_energy_data())
    log(f"STEP 1/7  Loading data from {raw_path}")
    raw = load_raw(raw_path)
    log("Dataset preview:\n" + raw.head(8).to_string())

    # ---- Phase 3: cleaning -------------------------------------------------
    log("STEP 2/7  Cleaning data ...")
    clean, report = clean_data(raw)
    clean.to_csv(config.CLEAN_DATA_PATH, index_label=config.DATETIME_COL)
    pd.Series(report).to_csv(config.OUTPUTS_DIR / "cleaning_report.csv", header=["value"])
    for k, v in report.items():
        log(f"   {k:<24}{v}")

    # ---- Phase 4: features -------------------------------------------------
    log("STEP 3/7  Engineering features ...")
    X, y = make_supervised(clean)
    log(f"   {X.shape[1]} features, {len(X):,} usable hours")
    log("   features: " + ", ".join(X.columns))
    X_train, X_test, y_train, y_test = chronological_split(X, y)
    log(f"   train: {X_train.index[0]} -> {X_train.index[-1]}  ({len(X_train):,} h)")
    log(f"   test : {X_test.index[0]} -> {X_test.index[-1]}  ({len(X_test):,} h)")

    # ---- Phase 5/6: training + evaluation ---------------------------------
    log("STEP 4/7  Training and evaluating models (test = unseen future) ...")
    metrics, fitted, preds, specs = train_all(X_train, X_test, y_train, y_test, logger)
    metrics.to_csv(config.OUTPUTS_DIR / "model_metrics.csv", index_label="Model")
    log("\n" + metrics.to_string())
    ranked = metrics.drop(index="Seasonal Naive (last week)", errors="ignore")
    best = ranked.index[0]
    log(f"BEST MODEL: {best}")
    best_pred = preds[best]
    pd.DataFrame({"Actual": y_test, "Predicted": best_pred.round(2)}).to_csv(
        config.OUTPUTS_DIR / "test_predictions.csv", index_label=config.DATETIME_COL)

    # ---- Phase 7: forecasting the future ----------------------------------
    log("STEP 5/7  Saving model and forecasting the future ...")
    residual_std = float((y_test - best_pred).std())
    bundle = save_best_model(best, specs, X, y, residual_std, clean)
    forecast = recursive_forecast(bundle["model"], bundle["feature_columns"], clean,
                                  args.horizon, residual_std)
    forecast.to_csv(config.OUTPUTS_DIR / "forecast_next_7_days.csv")
    log(f"   forecast {forecast.index[0]} -> {forecast.index[-1]}; "
        f"peak {forecast['Forecast'].max():.1f} kWh at {forecast['Forecast'].idxmax()}")

    # ---- Phase 8: visualisation -------------------------------------------
    log("STEP 6/7  Creating charts in images/ ...")
    visualize.plot_overview(clean)
    visualize.plot_hourly_profile(clean)
    visualize.plot_temperature_relationship(clean)
    visualize.plot_heatmap(clean)
    visualize.plot_model_comparison(metrics)
    visualize.plot_actual_vs_predicted(y_test, best_pred, best)
    visualize.plot_scatter(y_test, best_pred, best)
    visualize.plot_residuals(y_test, best_pred)
    if best in fitted and specs[best][1] != ["lag_168"]:
        visualize.plot_feature_importance(fitted[best], X_test, y_test, specs[best][1])
    visualize.plot_forecast(clean, forecast)

    # ---- Virtual business simulation --------------------------------------
    log("STEP 7/7  Running the virtual business simulation ...")
    sim = run_simulation(y_test, {"Naive forecast": preds["Seasonal Naive (last week)"],
                                  "AI forecast": best_pred})
    sim.to_csv(config.OUTPUTS_DIR / "simulation_summary.csv", index=False)
    visualize.plot_simulation(sim)
    log(f"Simulation over {sim.attrs['days']} test days:\n" + sim.to_string(index=False))
    log("DONE. Charts: images/   Tables: outputs/   Model: models/best_model.joblib")


if __name__ == "__main__":
    main()
