"""
Feature Engineering Pipeline
Group I + Group L · 10 Features · Triple-Barrier LSTM
-----------------------------------------------------
Correct order:
    1. Compute raw features
    2. Rolling Z-score  (DELTA_1, DELTA_3, BUY_RATIO)
    3. Drop warmup rows (first 500)
    4. Temporal split   (train / val / test)
    5. Global Z-score   (fit on train only)
    6. Winsorise        (clip at ±3σ, per split)
    7. Build LSTM sequences
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

ROLLING_WINDOW  = 500       # bars for rolling Z-score (~41 hours on 5-min bars)
WINSOR_CLIP     = 3.0       # clip at ±3σ after Z-score
SEQ_LEN         = 20        # LSTM input sequence length
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
# TEST_RATIO    = 0.15      # implicit — whatever remains

ROLLING_FEATURES = ["DELTA_1", "DELTA_3", "BUY_RATIO"]
GLOBAL_FEATURES  = ["ROC_3", "ROC_5", "ROC_10", "MOM_3", "RETURNS_1",
                     "VOL_SPIKE", "DELTA_DIV"]
ALL_FEATURES     = GLOBAL_FEATURES + ROLLING_FEATURES   # final column order

# label map: original {-1, 0, +1} → PyTorch class index {0, 1, 2}
LABEL_MAP = {-1: 0, 0: 1, 1: 2}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — COMPUTE RAW FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df to have columns:
        open, high, low, close, volume,
        taker_buy_vol,   ← Binance kline field 'V'
        atr_42,          ← pre-computed 42-bar ATR
        label            ← triple-barrier label {-1, 0, +1}

    Returns df with 10 new feature columns appended.
    Original columns are preserved.
    """
    df = df.copy()

    # ── Group I — Rate of Change ──────────────────────────────────────────────
    df["ROC_3"]     = df["close"].pct_change(3)
    df["ROC_5"]     = df["close"].pct_change(5)
    df["ROC_10"]    = df["close"].pct_change(10)
    df["MOM_3"]     = (df["close"] - df["close"].shift(3)) / df["atr_42"]
    df["RETURNS_1"] = df["close"].pct_change(1)

    # ── Group L — Order Flow ──────────────────────────────────────────────────
    taker_sell_vol  = df["volume"] - df["taker_buy_vol"]

    df["DELTA_1"]   = df["taker_buy_vol"] - taker_sell_vol
    df["DELTA_3"]   = df["DELTA_1"].rolling(3).sum()
    df["BUY_RATIO"] = df["taker_buy_vol"] / df["volume"]
    df["VOL_SPIKE"] = df["volume"] / df["volume"].rolling(5).mean()

    # DELTA_DIV: price direction contradicts flow direction
    df["DELTA_DIV"] = (
        (np.sign(df["ROC_3"]) != np.sign(df["DELTA_3"]))
        .astype(int)
    )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — ROLLING Z-SCORE (DELTA_1, DELTA_3, BUY_RATIO)
# ─────────────────────────────────────────────────────────────────────────────

def apply_rolling_zscore(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Replaces DELTA_1, DELTA_3, BUY_RATIO with their rolling Z-score versions.
    Rolling mean and std use only past 'window' bars — fully causal.

    First 'window' rows will have NaN rolling stats → dropped in Step 3.
    """
    df = df.copy()

    for feat in ROLLING_FEATURES:
        roll_mean = df[feat].rolling(window).mean()
        roll_std  = df[feat].rolling(window).std()

        # guard against zero std (extremely unlikely but possible in flat markets)
        roll_std  = roll_std.replace(0, np.nan)

        df[feat]  = (df[feat] - roll_mean) / roll_std

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DROP WARMUP ROWS
# ─────────────────────────────────────────────────────────────────────────────

def drop_warmup(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Drop first 'window' rows where rolling stats are unstable.
    Also drops any remaining NaNs from feature construction (ROC_10 needs 10 bars,
    DELTA_3 needs 3, etc.) — the rolling window dominates so this is a no-op.
    Reset index so downstream iloc splits work correctly.
    """
    df = df.iloc[window:].copy()
    df = df.dropna(subset=ALL_FEATURES + ["label"])
    df = df.reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame):
    """
    Strict temporal split — NO shuffling.
    Shuffling causes look-ahead leakage: future bars bleed into train sequences.

    Returns:
        train_df, val_df, test_df
    """
    n        = len(df)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    train_df = df.iloc[:n_train].copy()
    val_df   = df.iloc[n_train : n_train + n_val].copy()
    test_df  = df.iloc[n_train + n_val :].copy()

    print(f"Split sizes — train: {len(train_df):,}  "
          f"val: {len(val_df):,}  "
          f"test: {len(test_df):,}")

    return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — GLOBAL Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────

def fit_global_scaler(train_df: pd.DataFrame, save_path: str = None):
    """
    Fit StandardScaler on GLOBAL_FEATURES using train set only.
    ROLLING_FEATURES are already Z-scored — do not refit on them.

    Optionally save scaler to disk for inference pipeline.
    """
    scaler = StandardScaler()
    scaler.fit(train_df[GLOBAL_FEATURES])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(scaler, save_path)
        print(f"Scaler saved to {save_path}")

    return scaler


def apply_global_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Apply pre-fitted scaler to GLOBAL_FEATURES only.
    Call this separately on train, val, and test — same scaler instance each time.
    """
    df = df.copy()
    df[GLOBAL_FEATURES] = scaler.transform(df[GLOBAL_FEATURES])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — WINSORISE
# ─────────────────────────────────────────────────────────────────────────────

def winsorise(df: pd.DataFrame, clip: float = WINSOR_CLIP) -> pd.DataFrame:
    """
    Clip all features at ±clip standard deviations.
    Applied AFTER Z-scoring — values are already in σ units so
    clipping at ±3 directly removes extreme outliers.

    Applied independently per split — no cross-split contamination.
    """
    df = df.copy()
    df[ALL_FEATURES] = df[ALL_FEATURES].clip(lower=-clip, upper=clip)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BUILD LSTM SEQUENCES
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    Build rolling window sequences for LSTM input.

    For each bar t (starting at t = seq_len):
        X[i] = features at bars [t - seq_len, t)   shape: [seq_len, n_features]
        y[i] = label at bar t                       shape: scalar

    Stride = 1 bar — maximises number of training sequences.

    Returns:
        X : np.ndarray  [n_samples, seq_len, n_features]
        y : np.ndarray  [n_samples]  — PyTorch class indices {0, 1, 2}
    """
    feature_vals = df[ALL_FEATURES].values   # [n_bars, n_features]
    label_vals   = df["label"].map(LABEL_MAP).values   # [n_bars]

    X, y = [], []
    for t in range(seq_len, len(df)):
        X.append(feature_vals[t - seq_len : t])   # [seq_len, n_features]
        y.append(label_vals[t])

    X = np.array(X, dtype=np.float32)   # [n_samples, seq_len, n_features]
    y = np.array(y, dtype=np.int64)     # [n_samples]

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE — combines all steps
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    df: pd.DataFrame,
    rolling_window: int  = ROLLING_WINDOW,
    winsor_clip:    float = WINSOR_CLIP,
    seq_len:        int   = SEQ_LEN,
    scaler_path:    str   = None,
):
    """
    Full feature engineering pipeline. Steps in correct order:

        1. Compute raw features
        2. Rolling Z-score  (DELTA_1, DELTA_3, BUY_RATIO)
        3. Drop warmup rows
        4. Temporal split
        5. Global Z-score   (fit on train, apply to all splits)
        6. Winsorise        (±3σ, per split)
        7. Build sequences

    Args:
        df           : raw Binance kline DataFrame with required columns
        rolling_window: bars for rolling Z-score
        winsor_clip  : clip threshold in σ units
        seq_len      : LSTM sequence length
        scaler_path  : optional path to save fitted scaler (.pkl)

    Returns:
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        scaler   (fitted StandardScaler — save alongside model weights)
    """

    print("── Step 1: Computing raw features ──────────────────────────────")
    df = compute_features(df)
    print(f"  Rows after feature computation: {len(df):,}")

    print("── Step 2: Rolling Z-score (DELTA_1, DELTA_3, BUY_RATIO) ───────")
    df = apply_rolling_zscore(df, window=rolling_window)
    print(f"  Rolling window: {rolling_window} bars")

    print("── Step 3: Dropping warmup rows ─────────────────────────────────")
    df = drop_warmup(df, window=rolling_window)
    print(f"  Rows after warmup drop: {len(df):,}")

    print("── Step 4: Temporal split ───────────────────────────────────────")
    train_df, val_df, test_df = temporal_split(df)

    print("── Step 5: Global Z-score (fit on train only) ───────────────────")
    scaler   = fit_global_scaler(train_df, save_path=scaler_path)
    train_df = apply_global_scaler(train_df, scaler)
    val_df   = apply_global_scaler(val_df,   scaler)
    test_df  = apply_global_scaler(test_df,  scaler)
    print(f"  Fitted on {GLOBAL_FEATURES}")

    print("── Step 6: Winsorising at ±{:.0f}σ ────────────────────────────────".format(winsor_clip))
    train_df = winsorise(train_df, clip=winsor_clip)
    val_df   = winsorise(val_df,   clip=winsor_clip)
    test_df  = winsorise(test_df,  clip=winsor_clip)

    print("── Step 7: Building LSTM sequences ─────────────────────────────")
    X_train, y_train = build_sequences(train_df, seq_len=seq_len)
    X_val,   y_val   = build_sequences(val_df,   seq_len=seq_len)
    X_test,  y_test  = build_sequences(test_df,  seq_len=seq_len)

    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}  y_test:  {y_test.shape}")

    # ── label distribution check ──────────────────────────────────────────────
    print("\n── Label distribution ───────────────────────────────────────────")
    for split_name, y_split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        total = len(y_split)
        counts = {cls: (y_split == cls).sum() for cls in [0, 1, 2]}
        print(f"  {split_name:5s} — "
              f"dn(0): {counts[0]:,} ({counts[0]/total:.1%})  "
              f"no(1): {counts[1]:,} ({counts[1]/total:.1%})  "
              f"up(2): {counts[2]:,} ({counts[2]/total:.1%})")

    print("\n── Pipeline complete ────────────────────────────────────────────")

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPER — apply same pipeline to a live rolling buffer
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Applies the same feature engineering to a live buffer of bars.

    At inference time you maintain a rolling buffer of the last
    max(SEQ_LEN + ROLLING_WINDOW) raw bars. This class handles:
      - Feature computation
      - Rolling Z-score (causal — only past bars used)
      - Global Z-score (using the scaler fitted on train)
      - Winsorisation
      - Sequence extraction (last SEQ_LEN bars)

    Usage:
        pipeline = InferencePipeline(scaler, seq_len=20, rolling_window=500)
        x = pipeline.transform(buffer_df)   # [1, seq_len, n_features]
        probs = model(torch.tensor(x))
    """

    def __init__(
        self,
        scaler:         StandardScaler,
        seq_len:        int   = SEQ_LEN,
        rolling_window: int   = ROLLING_WINDOW,
        winsor_clip:    float = WINSOR_CLIP,
    ):
        self.scaler         = scaler
        self.seq_len        = seq_len
        self.rolling_window = rolling_window
        self.winsor_clip    = winsor_clip
        self.min_bars       = rolling_window + seq_len

    def transform(self, buffer_df: pd.DataFrame) -> np.ndarray:
        """
        Args:
            buffer_df : last (rolling_window + seq_len) raw bars minimum
                        must have same columns as training df
        Returns:
            x : np.ndarray [1, seq_len, n_features] — ready for model input
        """
        assert len(buffer_df) >= self.min_bars, (
            f"Buffer too short. Need {self.min_bars} bars, got {len(buffer_df)}"
        )

        # compute features on full buffer (causal — no future data)
        df = compute_features(buffer_df)

        # rolling Z-score — uses the buffer's own history (causal)
        df = apply_rolling_zscore(df, window=self.rolling_window)

        # drop warmup NaNs — keep only the last seq_len rows
        df = df.dropna(subset=ALL_FEATURES).reset_index(drop=True)
        df = df.iloc[-self.seq_len:]

        assert len(df) == self.seq_len, (
            f"Sequence too short after warmup drop: {len(df)} < {self.seq_len}"
        )

        # global Z-score using train scaler
        df[GLOBAL_FEATURES] = self.scaler.transform(df[GLOBAL_FEATURES])

        # winsorise
        df[ALL_FEATURES] = df[ALL_FEATURES].clip(
            lower=-self.winsor_clip, upper=self.winsor_clip
        )

        # extract sequence
        x = df[ALL_FEATURES].values.astype(np.float32)   # [seq_len, n_features]
        x = x[np.newaxis, ...]                            # [1, seq_len, n_features]

        return x


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    n = 5000   # small synthetic dataset for smoke test

    # synthetic Binance kline dataframe
    close      = 50000 + np.cumsum(np.random.randn(n) * 10)
    high       = close + np.abs(np.random.randn(n) * 5)
    low        = close - np.abs(np.random.randn(n) * 5)
    volume     = np.abs(np.random.randn(n) * 1000 + 5000)
    buy_vol    = volume * np.random.uniform(0.3, 0.7, n)

    # synthetic ATR_42
    tr         = np.maximum(high - low,
                 np.maximum(np.abs(high - np.roll(close, 1)),
                            np.abs(low  - np.roll(close, 1))))
    atr_42     = pd.Series(tr).ewm(span=42, adjust=False).mean().values

    # synthetic labels {-1, 0, +1}
    labels     = np.random.choice([-1, 0, 1], size=n,
                                  p=[0.34, 0.33, 0.33])

    df_raw = pd.DataFrame({
        "open":          close,
        "high":          high,
        "low":           low,
        "close":         close,
        "volume":        volume,
        "taker_buy_vol": buy_vol,
        "atr_42":        atr_42,
        "label":         labels,
    })

    print(f"Raw dataframe shape: {df_raw.shape}\n")

    X_train, y_train, X_val, y_val, X_test, y_test, scaler = run_pipeline(
        df_raw,
        scaler_path="models/feature_scaler.pkl",
    )

    print("\n── Smoke test: InferencePipeline ───────────────────────────────")
    inference = InferencePipeline(scaler)
    buffer    = df_raw.iloc[-(ROLLING_WINDOW + SEQ_LEN + 10):]
    x_live    = inference.transform(buffer)
    print(f"  Live inference input shape: {x_live.shape}")
    print(f"  Value range: [{x_live.min():.3f}, {x_live.max():.3f}]  "
          f"(should be within ±{WINSOR_CLIP})")
    print("\nAll checks passed.")
