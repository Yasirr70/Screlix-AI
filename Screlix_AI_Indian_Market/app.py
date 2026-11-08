import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.market_data import get_history, get_quote_snapshot, INDEX_TICKERS
from src.universe import load_stock_universe
from src.indicators import add_indicators
from src.predictor import predict_next_two_days
from src.storage import init_db, save_prediction, get_recent_predictions

st.set_page_config(page_title="Screlix AI | Indian Markets", page_icon="📈", layout="wide")
init_db()

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(120,130,150,.08); padding: 14px; border-radius: 12px;}
div[data-testid="stDataFrame"] {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Screlix AI")
st.caption("Indian equity research • Two-trading-day machine-learning forecasts")
with st.sidebar:
    st.header("Market terminal")
    page = st.radio("Navigate", ["Market Overview", "Stock Analysis", "AI Predictions", "Prediction History", "About & Data"])
    st.divider()
    st.caption("Data may be delayed or unavailable depending on the provider.")

@st.cache_data(ttl=3600, show_spinner=False)
def stock_list():
    return load_stock_universe()

if page == "Market Overview":
    st.subheader("Indian market overview")
    cols = st.columns(3)
    for col, (name, ticker) in zip(cols, list(INDEX_TICKERS.items())[:3]):
        with col:
            try:
                snap = get_quote_snapshot(ticker)
                col.metric(name, f"{snap['close']:,.2f}", f"{snap['change_pct']:+.2f}%")
                st.caption(f"Previous close: {snap['previous_close']:,.2f} · {snap['date']}")
            except Exception as e:
                col.metric(name, "Unavailable")
                col.caption("Provider did not return usable data.")
    st.divider()
    st.subheader("NIFTY 50 constituents — latest available data")
    universe = stock_list()
    nifty = universe[universe["symbol"].isin([
        "RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","ITC","SBIN","BHARTIARTL",
        "LT","HINDUNILVR","AXISBANK","KOTAKBANK","BAJFINANCE","MARUTI","SUNPHARMA",
        "TATAMOTORS","TATASTEEL","NTPC","POWERGRID","ONGC","WIPRO","HCLTECH",
        "ULTRACEMCO","ASIANPAINT","TITAN","ADANIENT","ADANIPORTS","COALINDIA",
        "JSWSTEEL","NESTLEIND","TECHM","INDUSINDBK","GRASIM","BAJAJFINSV",
        "DRREDDY","CIPLA","APOLLOHOSP","EICHERMOT","DIVISLAB","BRITANNIA",
        "HEROMOTOCO","HINDALCO","BPCL","TATACONSUM","SBILIFE","HDFCLIFE",
        "BEL","TRENT","SHRIRAMFIN"
    ])]
    if not nifty.empty:
        st.dataframe(nifty[["company","symbol","exchange"]], use_container_width=True, hide_index=True)
    st.info("For a full searchable NSE/BSE universe, open Stock Analysis. The symbol list refreshes from official listing files when available.")

elif page == "Stock Analysis":
    st.subheader("Search and analyse an Indian stock")
    universe = stock_list()
    options = universe.apply(lambda r: f"{r['symbol']} — {r['company']} ({r['exchange']})", axis=1).tolist()
    selected = st.selectbox("Search company or symbol", options, index=0 if options else None)
    if selected:
        row = universe.iloc[options.index(selected)]
        symbol = row["symbol"]
        ticker = row["yahoo_symbol"]
        st.markdown(f"### {row['company']} · `{symbol}`")
        period = st.selectbox("Historical period", ["1y","2y","5y","10y"], index=2)
        try:
            hist = get_history(ticker, period)
            hist = add_indicators(hist)
            last = hist.iloc[-1]
            prev = hist.iloc[-2]
            a,b,c,d = st.columns(4)
            a.metric("Latest close", f"₹{last.Close:,.2f}", f"{(last.Close/prev.Close-1)*100:+.2f}%")
            b.metric("RSI (14)", f"{last.RSI:,.1f}")
            c.metric("EMA 20", f"₹{last.EMA20:,.2f}")
            d.metric("Volume", f"{last.Volume:,.0f}")
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=hist.index, open=hist.Open, high=hist.High, low=hist.Low, close=hist.Close, name="Price"))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.EMA20, name="EMA 20", line=dict(width=1)))
            fig.add_trace(go.Scatter(x=hist.index, y=hist.EMA50, name="EMA 50", line=dict(width=1)))
            fig.update_layout(height=500, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10,r=10,t=25,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Fundamental snapshot")
            try:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                fields = {
                    "Market capitalization": info.get("marketCap"),
                    "P/E ratio": info.get("trailingPE"),
                    "EPS": info.get("trailingEps"),
                    "52-week high": info.get("fiftyTwoWeekHigh"),
                    "52-week low": info.get("fiftyTwoWeekLow"),
                    "Sector": info.get("sector"),
                }
                st.json({k: (f"₹{v:,.2f}" if isinstance(v,(int,float)) and k not in ["Market capitalization"] else v) for k,v in fields.items() if v is not None})
            except Exception:
                st.caption("Fundamental snapshot unavailable from the selected data provider.")
        except Exception as e:
            st.error(f"Could not load price history for {symbol}. Try again later. Details: {e}")

elif page == "AI Predictions":
    st.subheader("AI forecast — next two trading days")
    universe = stock_list()
    search = st.text_input("Enter NSE/BSE symbol or index", value="NIFTY")
    choices = {
        "NIFTY":"^NSEI", "SENSEX":"^BSESN", "BANKNIFTY":"^NSEBANK",
        "FINNIFTY":"NIFTY_FIN_SERVICE.NS"
    }
    match = universe[universe["symbol"].str.upper() == search.strip().upper()]
    ticker = choices.get(search.strip().upper(), match.iloc[0]["yahoo_symbol"] if not match.empty else search.strip().upper()+".NS")
    if st.button("Generate 2-day forecast", type="primary"):
        with st.spinner("Loading historical data and training the forecasting model..."):
            try:
                hist = add_indicators(get_history(ticker, "5y"))
                result = predict_next_two_days(hist)
                last_close = float(hist.Close.iloc[-1])
                st.caption(f"Reference close: ₹{last_close:,.2f} · Latest data date: {hist.index[-1].date()} · Model: Random Forest regression")
                c1,c2 = st.columns(2)
                for col, item in zip([c1,c2], result["days"]):
                    with col:
                        st.markdown(f"### {item['label']}")
                        col.metric("Predicted close", f"₹{item['prediction']:,.2f}", f"{item['change_pct']:+.2f}%")
                        st.write(f"Expected change: **{item['change_points']:+,.2f} points**")
                        st.write(f"Indicative range: ₹{item['low']:,.2f} – ₹{item['high']:,.2f}")
                        st.write(f"Direction: **{item['direction']}**")
                st.warning("Forecasts are experimental model estimates, not investment advice. The range is an indicative model-error band, not a guaranteed confidence interval.")
                save_prediction(ticker, hist.index[-1].date().isoformat(), result)
            except Exception as e:
                st.error(f"Prediction could not be generated: {e}")

elif page == "Prediction History":
    st.subheader("Saved forecast runs")
    rows = get_recent_predictions()
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Run time","Ticker","Data date","Forecast JSON"]), use_container_width=True, hide_index=True)
    else:
        st.info("No forecasts saved yet. Generate a forecast from AI Predictions.")

else:
    st.subheader("About Screlix AI")
    st.write("Screlix AI is an educational research dashboard for Indian equities. It combines market history, technical indicators and a baseline machine-learning forecast.")
    st.markdown("""
**Data and coverage**
- Historical prices: Yahoo Finance via `yfinance` (community interface; not an exchange feed).
- Stock universe: official NSE/BSE listing-file URLs where accessible, with a small fallback universe.
- New listings: refresh the universe manually or enable scheduled refresh on a hosted worker.
- Real-time quotes: not guaranteed. Use a licensed market-data vendor for production/live trading use.

**Important limitations**
- No model can reliably know the next two closing prices.
- NSE/BSE websites may block automated downloads or change file formats.
- A stock being listed does not guarantee a valid Yahoo Finance ticker.
- Backtest and validate every model before relying on its output.
""")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
