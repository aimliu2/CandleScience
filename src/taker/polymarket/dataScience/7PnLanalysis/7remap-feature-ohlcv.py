"""
match-feature-ohlcv.py
Join raw OHLCV prices onto each test-set feature row.

Uses index-based lookup on the sorted test sequence to handle gaps:
  entry_open  : open  of the next actual bar after signal  (index i+1)
  exit_close  : close of the 3rd actual bar after signal   (index i+3)

This means entry/exit always refer to the next real trading bar regardless
of any calendar gap (exchange downtime, weekend) between bars.

Source : data/mlData/raw/BTCUSDT-5m-v10.jsonl
Anchor : data/mlData/trainData/xxx.jsonl
Output : data/mlData/processed/xxx-ohlc-mapped.jsonl
"""

import json
from pathlib import Path
from datetime import datetime, timezone

# ── config ────────────────────────────────────────────────────────────────
EXIT_BARS = 3   # fixed hold period (bars)
# ─────────────────────────────────────────────────────────────────────────

def find_project_root(marker=".git"):
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError("Project root not found")

ROOT      = find_project_root()
RAW_PATH  = ROOT / "data/mlData/raw/BTCUSDT-5m-vX.jsonl"
TEST_PATH = ROOT / "data/mlData/trainData/202603-vX-test-regime-mapped.jsonl"
OUT_PATH  = ROOT / "data/mlData/processed/202603-vX-test-regime-mapped-ohlc.jsonl"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fmt_ts = lambda t: datetime.fromtimestamp(t / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ── 1. load OHLCV lookup table keyed by timestamp ─────────────────────────
print("Loading OHLCV data...")
ohlcv = {}
with open(RAW_PATH) as f:
    for line in f:
        row = json.loads(line)
        ohlcv[row["timestamp"]] = {
            "open":  row["open"],
            "close": row["close"],
        }
print(f"  Loaded {len(ohlcv):,} OHLCV bars")

# ── 2. load and sort test rows by timestamp ───────────────────────────────
print("\nLoading and sorting test rows...")
test_rows = []
with open(TEST_PATH) as f:
    for line in f:
        test_rows.append(json.loads(line))
test_rows.sort(key=lambda r: r["timestamp"])
print(f"  Loaded {len(test_rows):,} test rows")

# ── 3. index-based enrichment ─────────────────────────────────────────────
# For signal at index i, entry = row[i+1].open, exit = row[i+EXIT_BARS].close
# Rows within EXIT_BARS of the tail are skipped (no i+3 exists).
print(f"\nMatching OHLCV using index-based lookup (i+1, i+{EXIT_BARS})...")
written   = 0
skipped   = 0
gap_warns = []   # rows where i+1 or i+EXIT_BARS timestamp is missing from OHLCV

with open(OUT_PATH, "w") as dst:
    for i, row in enumerate(test_rows):
        if i + EXIT_BARS >= len(test_rows):
            skipped += 1
            continue

        ts_signal = row["timestamp"]
        ts_entry  = test_rows[i + 1]["timestamp"]
        ts_exit   = test_rows[i + EXIT_BARS]["timestamp"]

        bar_t     = ohlcv.get(ts_signal)
        bar_entry = ohlcv.get(ts_entry)
        bar_exit  = ohlcv.get(ts_exit)

        if bar_t is None or bar_entry is None or bar_exit is None:
            gap_warns.append({
                "ts":             ts_signal,
                "missing_signal": bar_t     is None,
                "missing_entry":  bar_entry is None,
                "missing_exit":   bar_exit  is None,
            })
            skipped += 1
            continue

        row["open_t"]     = bar_t["open"]
        row["close_t"]    = bar_t["close"]
        row["entry_open"] = bar_entry["open"]
        row["exit_close"] = bar_exit["close"]

        dst.write(json.dumps(row) + "\n")
        written += 1

# ── 4. report ─────────────────────────────────────────────────────────────
print(f"  Total test rows  : {len(test_rows):,}")
print(f"  Written          : {written:,}")
print(f"  Tail-dropped     : {min(EXIT_BARS, len(test_rows)):,}  (i+{EXIT_BARS} beyond end)")
print(f"  OHLCV misses     : {len(gap_warns):,}")

if gap_warns:
    print("\n  OHLCV miss details (first 5):")
    for g in gap_warns[:5]:
        missing = [k for k, v in g.items() if k != "ts" and v]
        print(f"    {fmt_ts(g['ts'])}  missing: {missing}")

# ── 5. gap report — flag non-contiguous consecutive test rows ──────────────
BAR_MS   = 5 * 60 * 1000
gaps     = []
for i in range(len(test_rows) - 1):
    diff = test_rows[i + 1]["timestamp"] - test_rows[i]["timestamp"]
    if diff != BAR_MS:
        gaps.append((test_rows[i]["timestamp"], test_rows[i + 1]["timestamp"], diff // BAR_MS))

print(f"\n  Non-contiguous bar pairs (gaps in test sequence): {len(gaps)}")
if gaps:
    print("  These rows use the next actual bar as entry/exit (index-based, intentional):")
    for ts_a, ts_b, bars in gaps[:5]:
        print(f"    {fmt_ts(ts_a)}  →  {fmt_ts(ts_b)}  ({bars} bars apart)")

print(f"\nOutput : {OUT_PATH}")
