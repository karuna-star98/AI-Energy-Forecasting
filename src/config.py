"""
config.py
---------
Central place for every path, constant and "business assumption" used in the
project.  Change a value here and the whole pipeline follows.
"""
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
IMAGES_DIR = BASE_DIR / "images"

RAW_DATA_PATH = RAW_DIR / "energy.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "energy_clean.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

# ----------------------------------------------------------------------------
# Column names expected in the input CSV
# ----------------------------------------------------------------------------
DATETIME_COL = "Datetime"
TARGET_COL = "Energy"        # electricity consumption in kWh per hour
TEMP_COL = "Temperature"     # optional: outdoor temperature in degrees C

# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_FRACTION = 0.20             # last 20 % of the timeline is the test set
FORECAST_HORIZON_HOURS = 168     # forecast the next 7 days
MAX_API_HORIZON_HOURS = 336      # the API refuses to look further than 14 days

# Look-back window the feature builder needs: lag_24 + rolling(168) + safety.
LOOKBACK_HOURS = 24 + 168 + 24

# Fixed-date public holidays (month, day).  Edit for your own country.
HOLIDAYS = {(1, 1), (1, 26), (8, 15), (10, 2), (12, 25)}

# ----------------------------------------------------------------------------
# Business simulation assumptions (see docs/simulation.md)
# ----------------------------------------------------------------------------
CURRENCY = "INR"


def tariff_per_kwh() -> np.ndarray:
    """Time-of-use electricity tariff, one price per hour of the day."""
    price = np.full(24, 5.0)            # off-peak  22:00-07:00
    price[7:11] = 8.0                   # shoulder  07:00-11:00
    price[17:22] = 8.0                  # shoulder  17:00-22:00
    price[11:17] = 12.0                 # peak      11:00-17:00
    return price


DEMAND_CHARGE_PER_KW_DAY = 60.0   # extra bill per kW of the daily peak (INR)
SHIFT_FRACTION = 0.10   # share of load in the forecast-peak hours that is flexible
SHIFT_HOURS = 4         # number of peak hours (and valley hours) used per day


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
