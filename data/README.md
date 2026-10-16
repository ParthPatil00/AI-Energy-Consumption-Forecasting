# Dataset format

The project creates `energy.csv` automatically the first time it runs if this file is absent.

For your own dataset, save a CSV at `data/energy.csv` with exactly these columns:

```csv
timestamp,energy_kwh
2025-01-01 00:00:00,4.25
2025-01-01 01:00:00,4.10
```

Use one row per hour, non-negative numeric consumption in kWh, and at least 14 days of records. The included automatic dataset is simulated teaching data, not readings from a real building.
