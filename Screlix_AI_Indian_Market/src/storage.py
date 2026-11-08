import sqlite3, json
from datetime import datetime

DB = "screlix_ai.db"

def init_db():
    with sqlite3.connect(DB) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT, ticker TEXT, data_date TEXT, forecast_json TEXT
        )""")

def save_prediction(ticker, data_date, result):
    with sqlite3.connect(DB) as con:
        con.execute(
            "INSERT INTO predictions(created_at,ticker,data_date,forecast_json) VALUES(?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), ticker, data_date, json.dumps(result))
        )

def get_recent_predictions(limit=100):
    with sqlite3.connect(DB) as con:
        return con.execute(
            "SELECT created_at,ticker,data_date,forecast_json FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,)
        ).fetchall()
