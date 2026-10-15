# Interview preparation

**Q: Explain the project in 30 seconds.**
I built an end-to-end system that forecasts a building's hourly electricity use for the next 7 days. It cleans smart-meter style data,
engineers calendar, weather and lag features, compares six models on a chronological hold-out set, serves the best one through a Flask API
and simulates the money a facility manager would save by shifting load away from forecast peaks.

**Q: Why simulated data?**
I had no access to real utility data, so I simulated a building with known physics (occupancy, air-conditioning, holidays, growth)
and deliberately injected missing values, duplicates and spikes. It is stated clearly in the README. The code accepts any CSV with
`Datetime, Energy[, Temperature]`, so it can be run on a real dataset.

**Q: Which model won and why?**
An MLP on the full feature set (RMSE ≈ 7.0 kWh, R² ≈ 0.985), narrowly ahead of gradient boosting and random forest.
The differences among the top three are small, so I would pick on speed and stability in production too (gradient boosting trains in about a second).

**Q: The original idea used only hour and day-of-week. What changed?**
That MLP is in the comparison (RMSE ≈ 19 kWh) and is worse than the naive "same hour last week" baseline (≈ 15).
Adding temperature, holidays and day-ahead-safe lags cut the error by roughly 60 %.

**Q: How did you avoid data leakage?**
Chronological split instead of `train_test_split(shuffle=True)`, and no lag shorter than 24 hours because the forecast is day-ahead.
A unit test asserts that `lag_24` is really the value 24 hours earlier and that no `lag_1` exists.

**Q: How do you forecast a whole week when the lags are only 24 h?**
Recursively: predict an hour, write it into the history, and use it as a lag for the hour 24 h later.
Errors can accumulate, which is why the plot has an uncertainty band.

**Q: What are the weaknesses?**
Simulated data; future weather is a climatology average, not a real forecast; one building; the uncertainty band is a simple normal approximation.

**Q: What would you do next?**
Use a real dataset, add real weather forecasts, try quantile regression for proper prediction intervals, add LSTM/temporal models only if they beat boosting,
monitor drift and retrain on a schedule.

**Q: Metrics?**
MAE (average kWh error), RMSE (punishes big misses, matters for peaks), R² (share of variance explained), MAPE (percentage error).
