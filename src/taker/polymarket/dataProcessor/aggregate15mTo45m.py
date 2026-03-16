"""
Aggregate 15m OHLCV data into 45m bars.
45m = 3 x 15m bars, grouped by snapping timestamp to 45m grid.
"""

import json
import pandas as pd
from pathlib import Path

FIFTEEN_MIN_MS = 900_000
FORTYFIVE_MIN_MS = 2_700_000  # 45 * 60 * 1000

DATA_PATH = Path(__file__).parents[2] / "data" / "bigData" / "BTCUSDT-15m.json"
OUT_PATH  = Path(__file__).parents[2] / "data" / "bigData" / "BTCUSDT-45m.json"

# --- Load ---
with open(DATA_PATH) as f:
    file = json.load(f)

df = pd.DataFrame(file["ohlcv"], columns=["timestamp", "open", "high", "low", "close", "vol"])
print(f"15m bars loaded: {len(df):,}")

# --- Snap to 45m grid ---
df["ts_45m"] = (df["timestamp"] // FORTYFIVE_MIN_MS) * FORTYFIVE_MIN_MS

# --- Aggregate ---
agg = df.groupby("ts_45m", sort=True).agg(
    open  = ("open",  "first"),
    high  = ("high",  "max"),
    low   = ("low",   "min"),
    close = ("close", "last"),
    vol   = ("vol",   "sum"),
    bar_count = ("timestamp", "count"),
).reset_index().rename(columns={"ts_45m": "timestamp"})

# Flag incomplete bars (< 3 x 15m bars) — usually only first/last bar
incomplete = agg[agg["bar_count"] < 3]
if len(incomplete):
    print(f"Incomplete 45m bars (< 3 x 15m): {len(incomplete)} — dropping")
    agg = agg[agg["bar_count"] == 3]

agg = agg.drop(columns=["bar_count"]).reset_index(drop=True)
print(f"45m bars produced: {len(agg):,}")
print(f"Expected ~{len(df) // 3:,} (15m ÷ 3)")

# --- Save ---
ohlcv_list = agg[["timestamp", "open", "high", "low", "close", "vol"]].values.tolist()
out = {"symbol": file.get("symbol", "BTCUSDT"), "timeframe": "45m", "ohlcv": ohlcv_list}

with open(OUT_PATH, "w") as f:
    json.dump(out, f)

print(f"Saved → {OUT_PATH}")
print(agg.tail(3).to_string())
