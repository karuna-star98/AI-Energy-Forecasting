# Virtual simulation

## How the dataset stands in for the real world

`src/data_generation.py` builds a three-year hourly record of a commercial building from simple physical rules:

| Real-world effect | How it is simulated |
|---|---|
| Always-on equipment (servers, lifts, lights) | constant 80 kWh base load |
| Office occupancy | smooth 0-1 occupancy curve, peak 09:00-17:00, 25 % on weekends/holidays |
| Air-conditioning | grows with every degree above 24 °C, scaled by occupancy |
| Heating | grows with every degree below 15 °C |
| Weather | yearly season + daily cycle + slow random weather fronts |
| Business growth | +3 % per year |
| Measurement noise | auto-correlated random noise |
| Messy meter data | ~160 missing hours, 5 duplicated rows, 10 sensor spikes |

The AI is never told these rules. If it forecasts well, it has *discovered* them from data - which is what it would
have to do with a real smart-meter export.

## Business simulation: "what is a better forecast worth?"

**Scenario.** A facility manager pays

* a time-of-use tariff - 5 INR/kWh at night, 8 shoulder, 12 between 11:00 and 17:00, and
* a demand charge - 60 INR for every kW of the highest hour of each day.

About 10 % of the load in the busiest hours is flexible (pre-cooling, battery charging, batch jobs, EV charging).

**Steps** (repeated for each of the 217 test days):

1. **Step 1 - Forecast.** The evening before, the model predicts the 24 hours of tomorrow.
2. **Step 2 - Decide.** Pick the 4 hours forecast to be highest and the 4 forecast to be lowest.
3. **Step 3 - Act.** Move 10 % of the load from the 4 peak hours into the 4 valley hours.
4. **Step 4 - Replay reality.** Apply that decision to what *actually* happened that day.
5. **Step 5 - Bill.** Compute energy cost + demand charge, and the daily peak.
6. **Step 6 - Compare** against: no action, the naive "same as last week" forecast, and perfect foresight (upper bound).

**Result of the committed run** (`outputs/simulation_summary.csv`):

| Strategy | Peak hours correctly identified | Bill saved | Peak demand reduced |
|---|---|---|---|
| No load shifting | - | 0 % | 0 % |
| Naive forecast (last week) | 77 % | 2.42 % | 3.72 % |
| **AI forecast** | **86 %** | **2.68 %** | **4.53 %** |
| Perfect foresight (ceiling) | 100 % | 2.87 % | 5.25 % |

Reading the table: the AI forecast captures about 93 % of the savings that perfect foresight could ever achieve and
earns about 11 % more savings than the naive method. All numbers depend on the assumed tariff and flexibility in `src/config.py`;
change them and re-run to see how the value moves.

## Graphs to generate (all created by `python main.py`)

`images/01` overview · `02` daily profile · `03` temperature effect · `04` heatmap · `05` model comparison ·
`06` actual vs predicted · `07` scatter · `08` residuals · `09` feature importance · `10` 7-day forecast · `11` savings.

## Outputs to save

`outputs/model_metrics.csv`, `test_predictions.csv`, `forecast_next_7_days.csv`, `simulation_summary.csv`,
`cleaning_report.csv`, `training_log.txt`.
