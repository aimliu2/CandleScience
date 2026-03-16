"""
Aggregate BTCUSDT-5m.json → 45m and compare OHLC against BTCUSDT-45m.json (from 15m).
Counts how many bars differ between the two aggregation paths.
"""

import json
import pandas as pd
from pathlib import Path

FORTYFIVE_MIN_MS = 2_700_000

def find_project_root(marker=".git"):
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError("Project root not found")

root = find_project_root()

# --- Load 5m ---
with open(root / "data" / "bigData" / "BTCUSDT-5m.json") as f:
    file = json.load(f)
df_5m = pd.DataFrame(file["ohlcv"], columns=["timestamp", "open", "high", "low", "close", "vol"])
df_5m[["open", "high", "low", "close", "vol"]] = df_5m[["open", "high", "low", "close", "vol"]].astype("float32")
del file

# --- Load 45m (aggregated from 15m) ---
with open(root / "data" / "bigData" / "BTCUSDT-45m.json") as f:
    file = json.load(f)
df_45m = pd.DataFrame(file["ohlcv"], columns=["timestamp", "open", "high", "low", "close", "vol"])
df_45m[["open", "high", "low", "close", "vol"]] = df_45m[["open", "high", "low", "close", "vol"]].astype("float32")
del file

# --- Aggregate 5m → 45m ---
df_5m["ts_45m"] = (df_5m["timestamp"] // FORTYFIVE_MIN_MS) * FORTYFIVE_MIN_MS
agg_5m = df_5m.groupby("ts_45m", sort=True).agg(
    open      = ("open",      "first"),
    high      = ("high",      "max"),
    low       = ("low",       "min"),
    close     = ("close",     "last"),
    vol       = ("vol",       "sum"),
    bar_count = ("timestamp", "count"),
).reset_index().rename(columns={"ts_45m": "timestamp"})

incomplete = agg_5m[agg_5m["bar_count"] != 9]
if len(incomplete):
    print(f"Incomplete 45m bars (< 9 x 5m): {len(incomplete)} — dropping")
agg_5m = agg_5m[agg_5m["bar_count"] == 9].drop(columns=["bar_count"]).reset_index(drop=True)

print(f"45m from  5m: {len(agg_5m):,} bars")
print(f"45m from 15m: {len(df_45m):,} bars")

# --- Compare on common timestamps ---
merged = pd.merge(
    agg_5m.rename(columns={"open":"o5","high":"h5","low":"l5","close":"c5","vol":"v5"}),
    df_45m[["timestamp","open","high","low","close"]].rename(columns={"open":"o15","high":"h15","low":"l15","close":"c15"}),
    on="timestamp", how="inner"
)
print(f"Common bars : {len(merged):,}\n")

TOL = 0.01  # float32 rounding tolerance
all_ok = True
for col, a, b in [("open","o5","o15"), ("high","h5","h15"), ("low","l5","l15"), ("close","c5","c15")]:
    diff = (merged[a] - merged[b]).abs()
    n_diff = (diff > TOL).sum()
    max_diff = diff.max()
    status = "OK" if n_diff == 0 else "MISMATCH"
    print(f"  {col:<6}: {n_diff:>5,} bars differ  (max diff: {max_diff:.4f})  [{status}]")
    if n_diff > 0:
        all_ok = False

print()
print("Result:", "MATCH — 45m data is consistent" if all_ok else "MISMATCH — investigate differences")