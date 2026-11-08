import pandas as pd

def add_indicators(df):
    x = df.copy()
    close = x["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    x["RSI"] = 100 - (100 / (1 + rs))
    x["EMA9"] = close.ewm(span=9, adjust=False).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False).mean()
    x["EMA50"] = close.ewm(span=50, adjust=False).mean()
    x["EMA200"] = close.ewm(span=200, adjust=False).mean()
    x["SMA20"] = close.rolling(20).mean()
    x["SMA50"] = close.rolling(50).mean()
    x["MACD"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    x["MACD_signal"] = x["MACD"].ewm(span=9, adjust=False).mean()
    x["ATR"] = pd.concat([
        x["High"] - x["Low"],
        (x["High"] - close.shift()).abs(),
        (x["Low"] - close.shift()).abs()
    ], axis=1).max(axis=1).rolling(14).mean()
    x["return_1"] = close.pct_change()
    x["return_2"] = close.pct_change(2)
    x["return_5"] = close.pct_change(5)
    x["volatility_10"] = x["return_1"].rolling(10).std()
    x["volume_change"] = x["Volume"].pct_change().replace([float("inf"),-float("inf")], 0)
    return x.replace([float("inf"),-float("inf")], pd.NA)
