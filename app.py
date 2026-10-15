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

@app.route("/")
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Energy Forecasting API</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .card { background: #1e293b; padding: 40px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); max-width: 600px; width: 100%; border: 1px solid #334155; }
            h1 { color: #38bdf8; margin-top: 0; font-size: 24px; }
            .status { display: inline-block; padding: 6px 14px; background: #059669; color: white; border-radius: 20px; font-size: 13px; font-weight: bold; margin-bottom: 20px; letter-spacing: 0.5px; }
            p { color: #94a3b8; font-size: 14px; }
            ul { list-style: none; padding: 0; margin-top: 20px; }
            li { background: #334155; margin: 12px 0; padding: 14px 18px; border-radius: 8px; display: flex; justify-content: space-between; align-items: center; border: 1px solid #475569; }
            .endpoint-name { font-weight: 600; color: #e2e8f0; font-size: 14px; }
            a { color: #38bdf8; text-decoration: none; font-weight: bold; background: #0f172a; padding: 6px 12px; border-radius: 6px; font-size: 13px; border: 1px solid #1e3a8a; }
            a:hover { background: #1e3a8a; color: #ffffff; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>⚡ AI Energy Forecasting API</h1>
            <div class="status">● SYSTEM ACTIVE & LIVE</div>
            <p>Production machine learning model serving hourly energy predictions via REST endpoints.</p>
            <ul>
                <li><span class="endpoint-name">System Health & Metadata</span> <a href="/health" target="_blank">View /health</a></li>
                <li><span class="endpoint-name">Next 24h Energy Forecast</span> <a href="/forecast?hours=24" target="_blank">View /forecast</a></li>
                <li><span class="endpoint-name">Single Prediction (POST)</span> <span style="color: #cbd5e1; font-size: 13px; font-family: monospace;">/predict</span></li>
            </ul>
        </div>
    </body>
    </html>
    """
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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
