import pandas as pd
import yfinance as yf

INDEX_TICKERS = {
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
}

def get_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError(f"No historical data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing fields from data provider: {missing}")
    df = df[required].dropna(subset=["Close"]).copy()
    return df

def get_quote_snapshot(ticker: str) -> dict:
    df = get_history(ticker, "5d")
    if len(df) < 2:
        raise ValueError("Not enough data to calculate daily change.")
    close = float(df.Close.iloc[-1])
    previous = float(df.Close.iloc[-2])
    return {
        "close": close,
        "previous_close": previous,
        "change_pct": (close / previous - 1) * 100,
        "date": str(df.index[-1].date())
    }
