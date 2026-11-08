# Screlix AI — Indian Stock Market Forecasting Dashboard

A Streamlit starter application for Indian equities with stock-universe discovery, historical charts, technical indicators, and experimental two-trading-day forecasts.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload the contents of this ZIP.
2. Open https://share.streamlit.io/ and connect GitHub.
3. Select the repository, branch, and `app.py`.
4. Deploy. No secrets are required for the default historical-data mode.

## Data and universe refresh

- Historical prices are requested from Yahoo Finance through `yfinance`. It is not an official exchange feed and may be delayed, incomplete, rate-limited, or unavailable.
- The app attempts to fetch the NSE equity list and BSE scrip list when the stock universe is loaded. The official sites can block automated requests or change file formats. If that happens, the app falls back to a small sample list.
- Streamlit's cache is set to one hour. Restart the app or clear Streamlit cache to force a fresh universe fetch.
- A newly listed company may not immediately have a usable Yahoo Finance symbol or historical price history.

## Forecast model

The baseline model predicts log returns for the next one and two trading sessions using a Random Forest and technical features. It is an educational prototype, not a validated trading system. The displayed error band is derived from a chronological holdout and is not a formal probability guarantee.

Before production use, add:
- Walk-forward backtesting across multiple market regimes.
- Benchmark comparison (naive last-close and drift models).
- Corporate-action and survivorship-bias handling.
- Licensed, exchange-compliant market data.
- Robust symbol mapping and automated scheduled universe refresh.
- Monitoring for model drift and data outages.

## Important

This software does not provide investment advice. Never trade solely on these predictions. Real-time NSE/BSE data usually requires a licensed vendor and may involve subscription fees.
