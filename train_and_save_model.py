import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from joblib import dump
import os

os.makedirs("models", exist_ok=True)

def train_and_save_arima(
    df,
    filters,
    model_path,
    order=(1,1,1),
    stable_start="2013-01-31"
):
    # Apply filters
    data = df.copy()
    for col, val in filters.items():
        data = data[data[col] == val]

    if data.empty:
        raise ValueError(f"No data after applying filters: {filters}")

    # Create monthly time series
    ts = (
        data
        .set_index("Open_Time")          # ✅ FIXED
        .sort_index()                    # ✅ IMPORTANT
        .resample("M")
        .size()
        .asfreq("M")
        .fillna(0)
    )

    # Stable regime
    #ts = ts[ts.index >= stable_start]

    if len(ts) < 12:
        raise ValueError(f"Not enough data for model {model_path}")

    # Log transform
    ts_log = np.log1p(ts)

    # Train ARIMA
    model = ARIMA(ts_log, order=order)
    model_fit = model.fit()

    # Save FITTED model
    dump(model_fit, model_path)

    print(f"✅ Saved model: {model_path}")

# -----------------------------
# Model definitions
# -----------------------------
MODEL_CONFIGS = {
    "model_overall_forecast.pkl": {
        "filters": {
            "Category": "incident"
        },
        "order": (0, 1, 3)
    },

    "model_CI_1.pkl": {
        "filters": {
            "Category": "incident",
            "CI_Cat": "application"
        },
        "order": (0, 1, 0)
    },

    "model_CI_2.pkl": {
        "filters": {
            "Category": "incident",
            "CI_Cat": "subapplication"
        },
        "order": (0, 1, 3)
    },

    "model_CI_3.pkl": {
        "filters": {
            "Category": "incident",
            "CI_Subcat": "Server Based Application"
        },
        "order": (3, 1, 1)
    },

    "model_CI_4.pkl": {
        "filters": {
            "Category": "incident",
            "CI_Subcat": "Web Based Application"
        },
        "order": (2, 1, 3)
    }
}


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    df = pd.read_csv("Basic_Cleand_data.csv")
    df["Open_Time"] = pd.to_datetime(df["Open_Time"])

    for model_name, config in MODEL_CONFIGS.items():
        train_and_save_arima(
            df=df,
            filters=config["filters"],
            order=config["order"],
            model_path=f"models/{model_name}"
        )
