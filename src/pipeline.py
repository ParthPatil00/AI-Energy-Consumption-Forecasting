from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_demo_dataset(path: Path, days: int = 180, seed: int = 42) -> pd.DataFrame:
    """Create simulated hourly electricity data when no dataset exists."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    hour = timestamps.hour.to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    day_of_year = timestamps.dayofyear.to_numpy()

    daily_pattern = 1.8 + 0.9 * np.sin((hour - 7) * 2 * np.pi / 24)
    evening_peak = 1.4 * np.exp(-((hour - 19) / 3.0) ** 2)
    workday_effect = np.where(day_of_week < 5, 0.8, -0.2)
    seasonal_effect = 0.35 * np.sin(2 * np.pi * day_of_year / 365)
    noise = rng.normal(0, 0.25, len(timestamps))
    energy_kwh = np.maximum(
        0.3, 4.5 + daily_pattern + evening_peak + workday_effect + seasonal_effect + noise
    )

    df = pd.DataFrame({"timestamp": timestamps, "energy_kwh": energy_kwh.round(3)})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        print(f"No dataset found at {data_path}. Creating a simulated demo dataset...")
        df = make_demo_dataset(data_path)
    else:
        df = pd.read_csv(data_path)

    required = {"timestamp", "energy_kwh"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Dataset must contain columns named 'timestamp' and 'energy_kwh'. "
            "See data/README.md for the expected format."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
    df = df.dropna(subset=["timestamp", "energy_kwh"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
    df = df[df["energy_kwh"] >= 0].copy()
    df = df.set_index("timestamp").asfreq("h")
    df["energy_kwh"] = df["energy_kwh"].interpolate(limit=3).ffill().bfill()
    df = df.dropna()
    if len(df) < 24 * 14:
        raise ValueError("Please provide at least 14 days of hourly data (336 rows).")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["hour"] = idx.hour
    out["day_of_week"] = idx.dayofweek
    out["day_of_year"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["lag_1h"] = out["energy_kwh"].shift(1)
    out["lag_24h"] = out["energy_kwh"].shift(24)
    out["rolling_mean_24h"] = out["energy_kwh"].shift(1).rolling(24).mean()
    return out.dropna()


def run_pipeline(project_root: Path) -> None:
    data_path = project_root / "data" / "energy.csv"
    output_dir = project_root / "outputs"
    model_dir = project_root / "models"
    image_dir = project_root / "images"
    for folder in (output_dir, model_dir, image_dir):
        folder.mkdir(parents=True, exist_ok=True)

    df = load_and_clean_data(data_path)
    print("\n--- Dataset preview ---")
    print(df.head())
    print(f"\nRows after cleaning: {len(df):,}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    plt.figure(figsize=(12, 4))
    df["energy_kwh"].plot()
    plt.title("Hourly Electricity Consumption (Demo / Input Dataset)")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy (kWh)")
    plt.tight_layout()
    plt.savefig(image_dir / "energy_history.png", dpi=150)
    plt.close()

    featured = add_features(df)
    feature_cols = [
        "hour", "day_of_week", "day_of_year", "is_weekend",
        "lag_1h", "lag_24h", "rolling_mean_24h"
    ]
    X = featured[feature_cols]
    y = featured["energy_kwh"]

    # Time-ordered split: earlier rows train the model, later rows test it.
    split_at = int(len(featured) * 0.8)
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]

    model = RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    metrics = {
        "model": "RandomForestRegressor",
        "test_rows": int(len(y_test)),
        "MAE_kWh": mae,
        "RMSE_kWh": rmse,
        "R2": r2,
        "note": "Metrics are from a time-ordered holdout set. Demo data is simulated unless replaced."
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\n--- Evaluation metrics ---")
    print(json.dumps(metrics, indent=2))

    results = pd.DataFrame({
        "timestamp": y_test.index,
        "actual_kwh": y_test.to_numpy(),
        "predicted_kwh": predictions
    })
    results.to_csv(output_dir / "test_predictions.csv", index=False)

    plt.figure(figsize=(12, 5))
    plt.plot(results["timestamp"], results["actual_kwh"], label="Actual", linewidth=1.5)
    plt.plot(results["timestamp"], results["predicted_kwh"], label="Predicted", linewidth=1.5)
    plt.title("Actual vs Predicted Electricity Consumption")
    plt.xlabel("Timestamp")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(image_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    residuals = results["actual_kwh"] - results["predicted_kwh"]
    plt.figure(figsize=(8, 4))
    plt.hist(residuals, bins=30)
    plt.title("Prediction Error Distribution")
    plt.xlabel("Actual - Predicted (kWh)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(image_dir / "prediction_errors.png", dpi=150)
    plt.close()

    model_path = model_dir / "energy_forecast_model.joblib"
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)

    # Demonstration forecast for the next 24 hours, recursively using each prediction.
    history = df["energy_kwh"].copy()
    future_rows = []
    for step in range(1, 25):
        next_time = history.index[-1] + pd.Timedelta(hours=1)
        row = {
            "hour": next_time.hour,
            "day_of_week": next_time.dayofweek,
            "day_of_year": next_time.dayofyear,
            "is_weekend": int(next_time.dayofweek >= 5),
            "lag_1h": float(history.iloc[-1]),
            "lag_24h": float(history.iloc[-24]),
            "rolling_mean_24h": float(history.iloc[-24:].mean())
        }
        pred = max(0.0, float(model.predict(pd.DataFrame([row], columns=feature_cols))[0]))
        future_rows.append({"timestamp": next_time, "forecast_kwh": pred})
        history.loc[next_time] = pred

    forecast = pd.DataFrame(future_rows)
    forecast.to_csv(output_dir / "next_24_hours_forecast.csv", index=False)

    plt.figure(figsize=(11, 4))
    plt.plot(forecast["timestamp"], forecast["forecast_kwh"], marker="o")
    plt.title("Next 24 Hours: Simulated Electricity Demand Forecast")
    plt.xlabel("Timestamp")
    plt.ylabel("Forecast energy (kWh)")
    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(image_dir / "next_24_hours_forecast.png", dpi=150)
    plt.close()

    print(f"\nSaved model: {model_path.relative_to(project_root)}")
    print(f"Saved predictions and metrics in: {output_dir.relative_to(project_root)}/")
    print(f"Saved graphs in: {image_dir.relative_to(project_root)}/")
    print("\nNext 24-hour forecast (first 5 rows):")
    print(forecast.head().to_string(index=False))
    print("\nDone! Open the images/ and outputs/ folders to inspect the results.")
