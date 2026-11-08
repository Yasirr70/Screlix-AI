import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

FEATURES = [
    "RSI","EMA9","EMA20","EMA50","EMA200","SMA20","SMA50",
    "MACD","MACD_signal","ATR","return_1","return_2","return_5",
    "volatility_10","volume_change"
]

def predict_next_two_days(df):
    data = df.copy().dropna(subset=["Close"])
    if len(data) < 260:
        raise ValueError("At least 260 daily observations are required for a basic model.")
    # Predict future log returns rather than raw price levels.
    for horizon in (1,2):
        data[f"target_{horizon}"] = np.log(data["Close"].shift(-horizon) / data["Close"])
    latest = data.iloc[[-1]][FEATURES]
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(
            n_estimators=250, max_depth=7, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        ))
    ])
    last_close = float(data["Close"].iloc[-1])
    days = []
    for horizon in (1,2):
        train = data.dropna(subset=[f"target_{horizon}"])
        if len(train) < 200:
            raise ValueError("Not enough clean training rows.")
        # Chronological holdout used only for an indicative historical error band.
        split = int(len(train) * 0.8)
        X_train, y_train = train[FEATURES].iloc[:split], train[f"target_{horizon}"].iloc[:split]
        X_test, y_test = train[FEATURES].iloc[split:], train[f"target_{horizon}"].iloc[split:]
        eval_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(n_estimators=150, max_depth=7, min_samples_leaf=4, random_state=42, n_jobs=-1))
        ])
        eval_model.fit(X_train, y_train)
        errors = y_test.to_numpy() - eval_model.predict(X_test)
        error_band = float(np.nanquantile(np.abs(errors), 0.8)) if len(errors) else 0.02
        model.fit(train[FEATURES], train[f"target_{horizon}"])
        predicted_return = float(model.predict(latest)[0])
        predicted_price = last_close * float(np.exp(predicted_return))
        pct = (predicted_price / last_close - 1) * 100
        # Convert the 80th-percentile absolute holdout error to an indicative price band.
        band = max(error_band, 0.005)
        low = last_close * float(np.exp(predicted_return - band))
        high = last_close * float(np.exp(predicted_return + band))
        days.append({
            "label": f"Day {horizon}",
            "prediction": round(predicted_price, 2),
            "change_points": round(predicted_price-last_close, 2),
            "change_pct": round(pct, 2),
            "low": round(low, 2),
            "high": round(high, 2),
            "direction": "Bullish" if pct > 0.15 else ("Bearish" if pct < -0.15 else "Neutral"),
            "mae_log_return": round(mean_absolute_error(y_test, eval_model.predict(X_test)), 5) if len(X_test) else None
        })
    return {"days": days, "reference_close": round(last_close,2)}
