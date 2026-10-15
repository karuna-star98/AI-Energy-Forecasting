# Architecture

## Block diagram

```
 ┌──────────────────────┐
 │  data/raw/energy.csv │   Datetime, Energy (kWh), Temperature (°C)
 │  (virtual smart      │   - made by src/data_generation.py
 │   meter OR your CSV) │   - or any real hourly dataset
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   src/data_loader.py
 │  1. CLEANING         │   sort · drop duplicates · hourly grid · remove spikes
 └──────────┬───────────┘   · fill gaps  ->  data/processed/energy_clean.csv
            ▼
 ┌──────────────────────┐   src/features.py
 │  2. FEATURES         │   calendar · cyclical sin/cos · temperature
 └──────────┬───────────┘   · lag_24/48/168 · rolling stats (all >= 24 h old)
            ▼
 ┌──────────────────────┐   src/train.py + src/models.py
 │  3. TRAIN & COMPARE  │   chronological 80/20 split
 │                      │   Seasonal Naive · Linear · MLP x2 · RF · Gradient Boosting
 └──────────┬───────────┘   metrics: MAE · RMSE · R² · MAPE  ->  outputs/model_metrics.csv
            ▼
 ┌──────────────────────┐
 │  4. BEST MODEL       │   refit on all data -> models/best_model.joblib
 └───────┬──────────┬───┘
         ▼          ▼
 ┌─────────────┐ ┌──────────────────────┐
 │ 5. FORECAST │ │ 6. SERVING (app.py)  │   Flask: /health  /forecast  /predict
 │ next 7 days │ │ REST API             │
 │ (recursive) │ └──────────────────────┘
 └──────┬──────┘
        ▼
 ┌──────────────────────┐   src/simulation.py
 │ 7. BUSINESS          │   time-of-use tariff + demand charge
 │    SIMULATION        │   -> money saved & peak reduced
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐   src/visualize.py
 │ 8. CHARTS & TABLES   │   images/*.png   outputs/*.csv
 └──────────────────────┘
```

## Modules

| File | Responsibility |
|---|---|
| `src/config.py` | paths, constants, tariff and simulation assumptions |
| `src/data_generation.py` | builds the virtual building dataset (with deliberate data problems) |
| `src/data_loader.py` | load CSV, validate columns, clean, report what was fixed |
| `src/features.py` | one function, `build_features`, used for training **and** forecasting (no train/serve skew) |
| `src/models.py` | candidate models incl. the naive baseline |
| `src/train.py` | chronological split, training loop, saving the winner |
| `src/evaluate.py` | MAE, RMSE, R², MAPE |
| `src/forecast.py` | recursive multi-day forecast + `EnergyForecaster` class used by the API |
| `src/simulation.py` | "what is the forecast worth?" business replay |
| `src/visualize.py` | all 11 charts |
| `app.py` | Flask REST API |
| `main.py` | runs everything |

## Data flow in one sentence

Raw CSV → clean hourly series → feature table `X` and target `y` → train on the past, score on the future →
refit winner → walk forward hour by hour to forecast the unseen week → feed forecasts to a cost simulation.

## Key design decisions (good interview material)

1. **Chronological split, never shuffled.** A random split lets the model see the hours right before and after each test hour.
2. **Day-ahead-safe features.** The smallest lag is 24 h. A model using "one hour ago" looks brilliant in a notebook and is useless for planning tomorrow.
3. **One feature function** for training and prediction, so the API cannot disagree with the training pipeline.
4. **Baseline first.** The seasonal-naive model ("same hour last week") is the bar every ML model must clear.
5. **Honest uncertainty.** The forecast ships with an approximate 95 % band built from test-set residuals.

## Known limitations

* Data is simulated, so scores show that the pipeline works, **not** how it would do on a real building.
* Future temperature is a historical (month, hour) average. A real system would use a weather-service forecast.
* The prediction band assumes roughly constant, normally distributed error.
