"""
app.py - Flask REST API that serves the trained model (Step 6 of the brief).

Start:    python app.py
Endpoints
    GET  /health                    -> is the service alive? up to which date is history known?
    GET  /forecast?hours=24         -> hourly forecast for the next N hours (max 336)
    POST /predict  {"datetime": "2025-01-03 14:00"}  -> one predicted value in kWh

Run `python main.py` first so that models/best_model.joblib exists.
"""
from flask import Flask, jsonify, request

from src import config
from src.forecast import EnergyForecaster

app = Flask(__name__)
forecaster = EnergyForecaster()


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        model=forecaster.bundle["model_name"],
        history_ends=str(forecaster.last_timestamp),
        unit="kWh per hour",
    )


@app.get("/forecast")
def forecast():
    try:
        hours = int(request.args.get("hours", 24))
        df = forecaster.forecast(hours)
    except ValueError as err:
        return jsonify(error=str(err)), 400
    records = [
        {"datetime": str(ts), "forecast_kwh": r.Forecast, "lower_kwh": r.Lower, "upper_kwh": r.Upper}
        for ts, r in df.iterrows()
    ]
    return jsonify(model=forecaster.bundle["model_name"], hours=hours, forecast=records)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    if "datetime" not in payload:
        return jsonify(error='send JSON like {"datetime": "2025-01-03 14:00"}'), 400
    try:
        value = forecaster.predict_at(payload["datetime"])
    except ValueError as err:
        return jsonify(error=str(err)), 400
    return jsonify(datetime=payload["datetime"], predicted_energy_kwh=round(value, 2))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
