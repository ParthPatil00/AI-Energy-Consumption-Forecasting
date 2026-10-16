# AI-Powered Energy Consumption Forecasting System

A beginner-friendly machine-learning project that demonstrates how hourly electricity consumption can be cleaned, analyzed, forecast, and evaluated.

> **Important:** If no dataset is present, the project creates simulated hourly data for demonstration. The results are not real measurements and should not be used for operational decisions.

## What it does
- Creates demo hourly energy data when needed
- Cleans timestamps and missing values
- Builds calendar, lag, and rolling-average features
- Trains a Random Forest regression model using a time-ordered train/test split
- Reports MAE, RMSE, and R²
- Forecasts the next 24 hours
- Saves CSV outputs, a reusable model, and charts

## Requirements
- Python 3.10–3.12 recommended
- pip
- Git (for publishing to GitHub)

## Setup and run

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

## Output files
- `data/energy.csv` — generated demo data on first run
- `outputs/metrics.json` — test-set metrics
- `outputs/test_predictions.csv` — actual and predicted values
- `outputs/next_24_hours_forecast.csv` — 24-hour forecast
- `models/energy_forecast_model.joblib` — saved model
- `images/energy_history.png` — historical series
- `images/actual_vs_predicted.png` — holdout comparison
- `images/prediction_errors.png` — residual histogram
- `images/next_24_hours_forecast.png` — forecast chart

## Dataset schema
A custom dataset must be saved as `data/energy.csv` with columns `timestamp` and `energy_kwh`, one hourly record per row. See `data/README.md`.

## Project architecture
```text
CSV data -> cleaning -> time/lag features -> Random Forest
         -> time-ordered evaluation -> metrics + prediction CSVs
         -> recursive 24-hour forecast -> charts + saved model
```

## GitHub publishing
Create an empty GitHub repository named `ai-energy-consumption-forecasting`. Then from this project folder:

```bash
git init
git add .
git commit -m "Initial project: energy forecasting pipeline"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/ai-energy-consumption-forecasting.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. Never commit passwords, tokens, or private data.

## Limitations and next improvements
- Synthetic data is only for demonstrating the workflow.
- A real deployment needs representative meter data, baselines, monitoring, retraining, and business validation.
- Useful next steps: compare with a naive forecast, add weather data, test other models, and optionally build a Streamlit dashboard.
