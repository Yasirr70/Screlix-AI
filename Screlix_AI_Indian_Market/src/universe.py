import io
import pandas as pd
import requests
import streamlit as st

NSE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
BSE_URL = "https://www.bseindia.com/downloads1/List_of_Scrips.csv"

FALLBACK = [
    ("RELIANCE","Reliance Industries","NSE"),("TCS","Tata Consultancy Services","NSE"),
    ("HDFCBANK","HDFC Bank","NSE"),("ICICIBANK","ICICI Bank","NSE"),
    ("INFY","Infosys","NSE"),("ITC","ITC","NSE"),("SBIN","State Bank of India","NSE"),
    ("BHARTIARTL","Bharti Airtel","NSE"),("LT","Larsen & Toubro","NSE"),
    ("AXISBANK","Axis Bank","NSE"),("KOTAKBANK","Kotak Mahindra Bank","NSE"),
    ("BAJFINANCE","Bajaj Finance","NSE"),("MARUTI","Maruti Suzuki","NSE"),
    ("SUNPHARMA","Sun Pharmaceutical","NSE"),("TATAMOTORS","Tata Motors","NSE"),
    ("TATASTEEL","Tata Steel","NSE"),("NTPC","NTPC","NSE"),("ONGC","ONGC","NSE"),
    ("WIPRO","Wipro","NSE"),("HCLTECH","HCL Technologies","NSE")
]

def _fetch_csv(url):
    headers = {"User-Agent":"Mozilla/5.0 ScrelixAI/1.0", "Accept":"text/csv,*/*"}
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))

def load_stock_universe():
    rows = []
    try:
        df = _fetch_csv(NSE_URL)
        sym_col = next(c for c in df.columns if c.strip().upper() in ["SYMBOL","SYMBOL "])
        name_col = next((c for c in df.columns if "NAME" in c.strip().upper()), None)
        for _, r in df.iterrows():
            symbol = str(r[sym_col]).strip()
            if symbol and symbol.lower() != "nan":
                name = str(r[name_col]).strip() if name_col else symbol
                rows.append((symbol, name, "NSE"))
    except Exception:
        pass
    try:
        df = _fetch_csv(BSE_URL)
        cols = {c.strip().lower(): c for c in df.columns}
        code_col = next((v for k,v in cols.items() if "security code" in k or k == "scrip code"), None)
        name_col = next((v for k,v in cols.items() if "security name" in k or "issuer name" in k), None)
        if code_col and name_col:
            for _, r in df.iterrows():
                symbol = str(r[code_col]).strip()
                name = str(r[name_col]).strip()
                if symbol and symbol.lower() != "nan":
                    rows.append((symbol, name, "BSE"))
    except Exception:
        pass
    if not rows:
        rows = FALLBACK
    out = pd.DataFrame(rows, columns=["symbol","company","exchange"]).drop_duplicates(["symbol","exchange"])
    out["yahoo_symbol"] = out.apply(
        lambda r: f"{r['symbol']}.NS" if r["exchange"] == "NSE" else f"{r['symbol']}.BO", axis=1
    )
    return out.sort_values(["exchange","symbol"]).reset_index(drop=True)
