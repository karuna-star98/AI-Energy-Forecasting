# GitHub guide: publish, commits, 7-day plan, proof checklist

## 1. Repository details

* **Name:** `ai-energy-consumption-forecasting`
* **Description:** `Hourly electricity demand forecasting for buildings with Python & scikit-learn - cleaning, feature engineering, model comparison, 7-day forecast, Flask API and a cost-saving simulation.`
* **Topics (tags):** `machine-learning` `time-series-forecasting` `energy` `smart-grid` `scikit-learn` `python` `data-science` `flask` `climate-tech` `mlp-regressor`
* **License:** MIT (edit your name in `LICENSE`).  **Visibility:** Public.

## 2. Create and push

1. On github.com click **New repository**, use the name above, leave "Add README / .gitignore / license" **unchecked** (we already have them).
2. In the project folder:

```bash
git init
git branch -M main
git add .
git commit -m "feat: initial project structure"
git remote add origin https://github.com/<your-username>/ai-energy-consumption-forecasting.git
git push -u origin main
```

If Git asks for a password, use a **Personal Access Token** (GitHub → Settings → Developer settings) or sign in with GitHub CLI (`gh auth login`).

## 3. Make the repo look professional

* Keep `README.md` at the top with images (already included, they live in `images/`).
* Pin the repo on your profile (Profile → Customize your pins).
* Add the description and topics (gear icon next to *About*).
* Create a release `v1.0.0` (Releases → Draft a new release).
* Keep commits small with clear messages (see below). Do not commit `.venv/`.

## 4. Commit plan (Conventional Commits)

| # | Commit message | What it contains |
|---|---|---|
| 1 | `chore: project skeleton, requirements, gitignore, license` | folders, `requirements.txt`, `.gitignore` |
| 2 | `feat(data): virtual smart-meter data generator` | `src/config.py`, `src/data_generation.py`, `data/raw/energy.csv` |
| 3 | `feat(data): cleaning pipeline with report` | `src/data_loader.py`, `outputs/cleaning_report.csv` |
| 4 | `feat(features): calendar, weather and lag features` | `src/features.py` |
| 5 | `feat(model): candidate models and chronological training` | `src/models.py`, `src/train.py`, `src/evaluate.py` |
| 6 | `docs(results): model comparison table and metrics` | `outputs/model_metrics.csv`, `images/05_*` |
| 7 | `feat(forecast): recursive 7-day forecast` | `src/forecast.py`, `images/10_*` |
| 8 | `feat(viz): all project charts` | `src/visualize.py`, `images/*.png` |
| 9 | `feat(sim): time-of-use cost simulation` | `src/simulation.py`, `images/11_*` |
| 10 | `feat(api): Flask prediction service` | `app.py` |
| 11 | `test: unit tests for cleaning, leakage and forecasting` | `tests/` |
| 12 | `docs: README, architecture, simulation, interview prep` | `README.md`, `docs/` |

> **Be honest in your history.** Commit as you really work through each piece. Don't back-date commits or push everything on one night and pretend otherwise;
> recruiters can see commit dates and interviewers will ask about the code. Read and run every file before you commit it.

## 5. 7-day proof-building plan

| Day | Goal | Do | Commit | Proof to capture |
|---|---|---|---|---|
| 1 | Setup | Install Python, create `.venv`, `pip install -r requirements.txt`, create the repo, push the skeleton | 1 | screenshot of the empty repo + `pip list` |
| 2 | Dataset | Run `python -m src.data_generation`; open the CSV; read `docs/simulation.md` | 2 | dataset preview (`head`), row count, missing-value count |
| 3 | Preprocessing | Run cleaning; compare before/after; read `data_loader.py` line by line | 3, 4 | `outputs/cleaning_report.csv`, `images/01`, `02` |
| 4 | Model | Run `python main.py`; read `models.py`/`train.py`; change one hyper-parameter and observe | 5 | training log (`outputs/training_log.txt`) |
| 5 | Evaluation | Study the metrics table; explain why the naive baseline matters | 6 | `images/05`, `06`, `07`, `08` |
| 6 | Visualization & simulation | Forecast chart, feature importance, savings chart; try different tariffs in `config.py` | 7, 8, 9 | `images/09`, `10`, `11` |
| 7 | Upload | Start the API, test `/predict`; run tests; finish README; push; release v1.0.0 | 10, 11, 12 | API screenshot, final repo page |

## 6. Screenshots and demo assets

Save into `images/` with the names below (the pipeline already creates the charts; screenshots are for terminal and GitHub views).

| Asset | File name | How |
|---|---|---|
| Dataset preview | `images/screenshot_dataset_preview.png` | terminal: `head -n 8 data/raw/energy.csv` or the first lines of `python main.py` |
| Preprocessing output | `images/screenshot_cleaning_report.png` | step 2 of the `main.py` log |
| Training logs | `images/screenshot_training_log.png` | step 4 of the log (6 models + metrics table) |
| Prediction graph | `images/10_future_forecast.png` | auto-generated |
| Actual vs predicted | `images/06_actual_vs_predicted.png` | auto-generated |
| Error metrics | `outputs/model_metrics.csv` + `images/05_model_comparison.png` | auto-generated |
| API demo | `images/screenshot_api_predict.png` | browser or `curl` (see README) |
| GitHub repo preview | `images/screenshot_github_repo.png` | full-page screenshot of the repo |

Reference them in the README with `![caption](images/file.png)`. Use lower-case names with underscores, never spaces.

## 7. Final proof checklist

- [ ] Repo is public with description and topics
- [ ] README renders correctly with images
- [ ] `python main.py` works on a fresh clone (test it!)
- [ ] `python -m pytest -q` passes
- [ ] Results table in README matches `outputs/model_metrics.csv`
- [ ] README states clearly that the data is simulated
- [ ] API screenshot added
- [ ] No secrets, no `.venv/`
- [ ] Release `v1.0.0` created, repo pinned on profile
- [ ] Repo link added to resume and LinkedIn (Projects section)
- [ ] You can explain every file and answer `docs/interview_prep.md` without reading it

## 8. Resume bullets (edit to match your own work)

* Built an end-to-end hourly energy-demand forecasting system in Python/scikit-learn on a simulated 3-year building dataset; best model (MLP) reached R² 0.985 and RMSE 7.0 kWh, 53 % lower error than a seasonal-naive baseline.
* Engineered day-ahead-safe calendar, weather and lag features with a leakage-free chronological evaluation; deployed the model via a Flask REST API.
* Simulated a time-of-use tariff with demand charges to show forecast-driven load shifting cuts the daily peak by 4.5 % and the bill by 2.7 %.
