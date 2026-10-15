"""
data_generation.py
------------------
Creates a realistic *virtual* smart-meter dataset for a commercial building.

Why simulate?  Students rarely get access to real utility data.  A simulator
lets us control the "physics" (office hours, weekends, holidays, air-conditioning
driven by temperature, yearly growth) and then check that the AI discovers them.

The generated file looks like a real raw export on purpose: it contains missing
readings, duplicated rows and a few sensor spikes, so the cleaning step has
real work to do.

Columns produced
    Datetime     hourly timestamp
    Energy       electricity consumption in kWh
    Temperature  outdoor temperature in degrees C
"""
import numpy as np
import pandas as pd

from src import config


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma: float) -> np.ndarray:
    """Auto-correlated noise: today's noise remembers yesterday's noise."""
    shocks = rng.normal(0.0, sigma, n)
    out = np.zeros(n)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + shocks[i]
    return out


def generate_energy_data(
    start: str = "2022-01-01 00:00",
    end: str = "2024-12-31 23:00",
    seed: int = config.RANDOM_STATE,
    messy: bool = True,
) -> pd.DataFrame:
    """Return an hourly building-energy DataFrame (see module docstring)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()

    # 1) Weather: yearly season + daily cycle + slow random weather fronts ----
    temperature = (
        22
        + 9 * np.sin(2 * np.pi * (doy - 110) / 365)      # summer / winter
        + 5 * np.sin(2 * np.pi * (hour - 9) / 24)        # warmest around 15:00
        + _ar1(rng, n, rho=0.97, sigma=0.6)
    )

    # 2) Human behaviour: office occupancy over the day --------------------
    hours_pts = [0, 5, 7, 9, 12, 14, 17, 19, 21, 24]
    occ_pts = [0.05, 0.05, 0.40, 0.90, 1.00, 0.95, 0.80, 0.35, 0.15, 0.05]
    occupancy = np.interp(hour, hours_pts, occ_pts)
    is_holiday = np.fromiter(
        ((m, d) in config.HOLIDAYS for m, d in zip(idx.month, idx.day)), dtype=bool, count=n
    )
    working_day = (dow < 5) & ~is_holiday
    occupancy = np.where(working_day, occupancy, 0.25 * occupancy)   # quiet weekends

    # 3) Energy = always-on load + people + air-conditioning / heating -------
    base_load = 80.0                                   # servers, lifts, lights
    people_load = 120.0 * occupancy
    cooling = 4.5 * np.clip(temperature - 24, 0, None) * (0.3 + 0.7 * occupancy)
    heating = 3.0 * np.clip(15 - temperature, 0, None) * (0.3 + 0.7 * occupancy)
    growth = 1.0 + 0.03 * ((idx - idx[0]).days.to_numpy() / 365.0)   # +3 % per year
    energy = (base_load + people_load + cooling + heating) * growth
    energy = energy + _ar1(rng, n, rho=0.7, sigma=3.0)               # random noise
    energy = np.clip(energy, 20, None)

    df = pd.DataFrame(
        {
            config.DATETIME_COL: idx,
            config.TARGET_COL: energy.round(2),
            config.TEMP_COL: temperature.round(2),
        }
    )

    # 4) Make the raw file messy, like a real meter export ------------------
    if messy:
        e = df[config.TARGET_COL].to_numpy().copy()
        e[rng.random(n) < 0.004] = np.nan                         # random drop-outs
        for _ in range(6):                                        # communication gaps
            s = rng.integers(0, n - 20)
            e[s : s + rng.integers(3, 13)] = np.nan
        spikes = rng.choice(n, size=10, replace=False)            # sensor spikes
        e[spikes] = e[spikes] * rng.uniform(2.5, 3.5, size=10)
        df[config.TARGET_COL] = e
        dupes = df.sample(5, random_state=seed)                   # duplicated rows
        df = pd.concat([df, dupes]).sort_values(config.DATETIME_COL, kind="stable")
        df = df.reset_index(drop=True)
    return df


def save_raw_data(df: pd.DataFrame, path=config.RAW_DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    data = generate_energy_data()
    save_raw_data(data)
    print(f"Saved {len(data):,} rows to {config.RAW_DATA_PATH}")
