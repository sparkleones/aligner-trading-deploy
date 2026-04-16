"""
ML TRADE CLASSIFIER — Phase 6 Integration
==========================================
Trains XGBoost + Random Forest on V14 trade data to learn which setups
produce winners vs losers. Used as an additional confidence filter.

Features extracted from:
  1. Trade-level: action, entry_type, confidence, VIX, minute, lots
  2. Market-level: RSI, VWAP proximity, squeeze state, BB width, ATR%, EMA alignment
  3. Temporal: hour, day of week, month, is_expiry

Cross-validation: Time-series walk-forward (train H1 -> test H2, etc.)
"""

import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.daywise_analysis import add_all_indicators
from backtesting.multi_month_oos_test import download_range

DATA_DIR = project_root / "data" / "historical"
TRADE_CSV = DATA_DIR / "v9_hybrid_real_option_trades.csv"


def load_trades():
    """Load trade log from V14 backtest."""
    df = pd.read_csv(TRADE_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month_num"] = df["date"].dt.month
    df["hour_of_entry"] = df["entry_minute"] // 60
    df["is_put"] = (df["action"] == "BUY_PUT").astype(int)
    df["is_expiry"] = df["exit_reason"].str.contains("expiry", na=False).astype(int)
    df["is_winner"] = (df["pnl"] > 0).astype(int)
    df["strike_distance_pct"] = abs(df["entry_spot"] - df["strike"]) / df["entry_spot"] * 100
    df["premium_pct"] = df["entry_prem"] / df["entry_spot"] * 100
    return df


def enrich_with_market_features(trades_df):
    """
    For each trade, load the NIFTY 1-min data for that date
    and extract indicator values at the exact entry minute.
    """
    print("  Enriching trades with market features...")

    # Group trades by approximate date ranges to minimize data loading
    all_features = []
    cached_data = {}

    for idx, trade in trades_df.iterrows():
        trade_date = trade["date"].date()
        entry_min = int(trade["entry_minute"])

        # Cache key: month
        cache_key = f"{trade_date.year}-{trade_date.month:02d}"

        if cache_key not in cached_data:
            # Load data for this month (with 1-month warmup)
            warmup_start = (trade_date.replace(day=1) - dt.timedelta(days=35)).strftime("%Y-%m-%d")
            data_end = (trade_date.replace(day=28) + dt.timedelta(days=5)).strftime("%Y-%m-%d")

            try:
                nifty, vix_data = download_range(warmup_start, data_end)
                if nifty is not None and len(nifty) > 100:
                    nifty_ind = add_all_indicators(nifty.copy())
                    day_groups = {}
                    for d, group in nifty_ind.groupby(nifty_ind.index.date):
                        day_groups[d] = group

                    # Build daily data for VWAP, regime
                    daily_rows = []
                    for d in sorted(day_groups.keys()):
                        bars = day_groups[d]
                        daily_rows.append({
                            "Date": d, "Close": bars["close"].iloc[-1],
                            "High": bars["high"].max(), "Low": bars["low"].min(),
                        })
                    daily_df = pd.DataFrame(daily_rows).set_index("Date")
                    daily_df["SMA20"] = daily_df["Close"].rolling(20, min_periods=1).mean()
                    daily_df["SMA50"] = daily_df["Close"].rolling(50, min_periods=1).mean()

                    cached_data[cache_key] = {"day_groups": day_groups, "daily": daily_df}
                else:
                    cached_data[cache_key] = None
            except Exception as e:
                cached_data[cache_key] = None

        data = cached_data.get(cache_key)
        features = extract_bar_features(data, trade_date, entry_min)
        all_features.append(features)

    features_df = pd.DataFrame(all_features)
    return pd.concat([trades_df.reset_index(drop=True), features_df], axis=1)


def extract_bar_features(data, trade_date, entry_minute):
    """Extract indicator values at a specific minute of a specific day."""
    defaults = {
        "rsi_at_entry": 50.0, "bb_width_at_entry": 1.0,
        "atr_pct_at_entry": 0.5, "ema9_above_21": 0,
        "above_ema50": 0, "st_direction": 0,
        "macd_hist_at_entry": 0.0, "adx_at_entry": 20.0,
        "vwap_proximity": 0.0, "squeeze_at_entry": 0,
        "stoch_k_at_entry": 50.0, "above_sma20": 0,
        "above_sma50": 0, "price_vs_open_pct": 0.0,
    }

    if data is None:
        return defaults

    day_groups = data["day_groups"]
    daily = data["daily"]

    if trade_date not in day_groups:
        return defaults

    bars = day_groups[trade_date]
    if entry_minute >= len(bars):
        entry_minute = len(bars) - 1

    bar = bars.iloc[entry_minute]

    features = {}

    # Indicator values at entry
    features["rsi_at_entry"] = float(bar.get("rsi", 50))

    if "bb_upper" in bars.columns and "bb_lower" in bars.columns and "bb_mid" in bars.columns:
        bb_w = bar.get("bb_upper", 0) - bar.get("bb_lower", 0)
        bb_m = bar.get("bb_mid", 1)
        features["bb_width_at_entry"] = float(bb_w / bb_m * 100) if bb_m > 0 else 1.0
    else:
        features["bb_width_at_entry"] = 1.0

    features["atr_pct_at_entry"] = float(bar.get("atr_pct", 0.5))

    features["ema9_above_21"] = int(bar.get("ema9_above_ema21", False))
    features["above_ema50"] = int(bar.get("above_ema50", False))
    features["st_direction"] = int(bar.get("st_direction", 0))
    features["macd_hist_at_entry"] = float(bar.get("macd_hist", 0))
    features["adx_at_entry"] = float(bar.get("adx", 20))
    features["stoch_k_at_entry"] = float(bar.get("stoch_k", 50))

    # VWAP proximity
    close = float(bar["close"])
    highs = bars["high"].values[:entry_minute + 1]
    lows = bars["low"].values[:entry_minute + 1]
    closes = bars["close"].values[:entry_minute + 1]
    tp = (highs + lows + closes) / 3.0
    vwap = np.mean(tp) if len(tp) > 0 else close
    features["vwap_proximity"] = (close - vwap) / vwap * 100 if vwap > 0 else 0.0

    # Squeeze state
    if all(c in bars.columns for c in ["bb_upper", "bb_lower", "ema21", "atr"]):
        bb_up = float(bar["bb_upper"])
        bb_lo = float(bar["bb_lower"])
        ema21 = float(bar["ema21"])
        atr = float(bar["atr"])
        kc_up = ema21 + 1.5 * atr
        kc_lo = ema21 - 1.5 * atr
        features["squeeze_at_entry"] = int(bb_lo > kc_lo and bb_up < kc_up)
    else:
        features["squeeze_at_entry"] = 0

    # Daily context
    if trade_date in daily.index:
        d = daily.loc[trade_date]
        features["above_sma20"] = int(close > float(d.get("SMA20", close)))
        features["above_sma50"] = int(close > float(d.get("SMA50", close)))
    else:
        features["above_sma20"] = 0
        features["above_sma50"] = 0

    # Price vs day open
    day_open = float(bars["open"].iloc[0])
    features["price_vs_open_pct"] = (close - day_open) / day_open * 100 if day_open > 0 else 0

    return features


def build_feature_matrix(df):
    """Build the ML feature matrix from enriched trade data."""
    feature_cols = [
        # Trade-level
        "is_put", "confidence", "vix", "entry_minute", "lots",
        "hour_of_entry", "day_of_week", "month_num", "is_expiry",
        "strike_distance_pct", "premium_pct",
        # Market-level
        "rsi_at_entry", "bb_width_at_entry", "atr_pct_at_entry",
        "ema9_above_21", "above_ema50", "st_direction",
        "macd_hist_at_entry", "adx_at_entry", "vwap_proximity",
        "squeeze_at_entry", "stoch_k_at_entry",
        "above_sma20", "above_sma50", "price_vs_open_pct",
    ]

    # Encode entry_type
    le = LabelEncoder()
    df["entry_type_encoded"] = le.fit_transform(df["entry_type"].fillna("unknown"))
    feature_cols.append("entry_type_encoded")

    X = df[feature_cols].copy()
    X = X.fillna(0)

    # Target: 1 = winner (pnl > 0), 0 = loser
    y = df["is_winner"].values

    return X, y, feature_cols, le


def train_and_evaluate(X, y, trades_df):
    """Train ML models with time-series walk-forward validation."""
    print("\n" + "=" * 100)
    print("  ML TRADE CLASSIFIER — Walk-Forward Validation")
    print("=" * 100)

    n = len(X)
    print(f"\n  Total trades: {n}")
    print(f"  Winners: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Losers: {n - y.sum()} ({(1-y.mean())*100:.1f}%)")

    # ================================================================
    # 1. TIME-SERIES WALK-FORWARD (realistic — no future leakage)
    # ================================================================
    print(f"\n  {'='*80}")
    print(f"  Walk-Forward Splits (Train on past -> Predict future)")
    print(f"  {'='*80}")

    # Split by months: train on first N months, test on month N+1
    trades_df_sorted = trades_df.sort_values("date").reset_index(drop=True)
    months = trades_df_sorted["month"].unique()

    wf_results = {"xgb": [], "rf": [], "gb": []}
    wf_preds = np.full(n, -1)  # -1 = not predicted

    # Walk forward: train on months 1..k, test on month k+1
    for split_idx in range(3, len(months)):
        train_months = set(months[:split_idx])
        test_month = months[split_idx]

        train_mask = trades_df_sorted["month"].isin(train_months)
        test_mask = trades_df_sorted["month"] == test_month

        X_train = X.loc[train_mask]
        y_train = y[train_mask.values]
        X_test = X.loc[test_mask]
        y_test = y[test_mask.values]

        if len(X_train) < 10 or len(X_test) < 3:
            continue

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, reg_alpha=1, reg_lambda=2,
            random_state=42, verbosity=0,
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=5,
            random_state=42,
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)

        # Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)

        xgb_acc = accuracy_score(y_test, xgb_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        gb_acc = accuracy_score(y_test, gb_pred)

        wf_results["xgb"].append({"month": test_month, "acc": xgb_acc, "n": len(y_test)})
        wf_results["rf"].append({"month": test_month, "acc": rf_acc, "n": len(y_test)})
        wf_results["gb"].append({"month": test_month, "acc": gb_acc, "n": len(y_test)})

        # Store XGBoost predictions for P&L analysis
        test_indices = trades_df_sorted.index[test_mask]
        for i, ti in enumerate(test_indices):
            wf_preds[ti] = xgb_prob[i]

        print(f"    Train {','.join(train_months)} -> Test {test_month}: "
              f"XGB={xgb_acc:.1%} RF={rf_acc:.1%} GB={gb_acc:.1%} "
              f"({len(y_test)} trades, {y_test.sum()} wins)")

    # Overall walk-forward accuracy
    print(f"\n  Walk-Forward Accuracy (weighted average):")
    for model_name in ["xgb", "rf", "gb"]:
        results = wf_results[model_name]
        if results:
            total_correct = sum(r["acc"] * r["n"] for r in results)
            total_n = sum(r["n"] for r in results)
            avg_acc = total_correct / total_n if total_n > 0 else 0
            print(f"    {model_name.upper()}: {avg_acc:.1%} ({total_n} trades)")

    # ================================================================
    # 2. FULL MODEL TRAINING (for feature importance analysis)
    # ================================================================
    print(f"\n  {'='*80}")
    print(f"  Feature Importance Analysis (full dataset)")
    print(f"  {'='*80}")

    xgb_full = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=1, reg_lambda=2,
        random_state=42, verbosity=0,
    )
    xgb_full.fit(X, y)

    importances = xgb_full.feature_importances_
    feat_names = X.columns.tolist()
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  {'Feature':<25} {'Importance':>12} {'Bar':>30}")
    print(f"  {'-'*67}")
    for fname, imp in feat_imp:
        bar = "#" * int(imp * 100)
        print(f"  {fname:<25} {imp:>12.4f}  {bar}")

    # ================================================================
    # 3. PROFITABLE TRADE PATTERN DISCOVERY
    # ================================================================
    print(f"\n  {'='*80}")
    print(f"  High-Edge Trade Patterns (data-mined rules)")
    print(f"  {'='*80}")

    discover_patterns(trades_df)

    # ================================================================
    # 4. ML-FILTERED P&L ANALYSIS
    # ================================================================
    if np.any(wf_preds >= 0):
        print(f"\n  {'='*80}")
        print(f"  ML-Filtered P&L Impact (walk-forward predictions)")
        print(f"  {'='*80}")

        predicted_mask = wf_preds >= 0
        pred_trades = trades_df_sorted[predicted_mask].copy()
        pred_trades["ml_prob"] = wf_preds[predicted_mask]

        for threshold in [0.3, 0.4, 0.5]:
            take = pred_trades[pred_trades["ml_prob"] >= threshold]
            skip = pred_trades[pred_trades["ml_prob"] < threshold]
            if len(take) > 0 and len(skip) > 0:
                print(f"\n    Threshold {threshold:.0%} (take if ML win_prob >= {threshold:.0%}):")
                print(f"      TAKE: {len(take)} trades, WR={take['is_winner'].mean()*100:.1f}%, "
                      f"P&L=Rs{take['pnl'].sum():+,.0f}, Avg=Rs{take['pnl'].mean():+,.0f}")
                print(f"      SKIP: {len(skip)} trades, WR={skip['is_winner'].mean()*100:.1f}%, "
                      f"P&L=Rs{skip['pnl'].sum():+,.0f}, Avg=Rs{skip['pnl'].mean():+,.0f}")

    return xgb_full


def discover_patterns(df):
    """Find high-edge trade subsets from the data."""
    patterns = []

    # Pattern 1: By VIX range
    for vix_lo, vix_hi in [(10, 13), (13, 15), (15, 17), (17, 20), (20, 30)]:
        mask = (df["vix"] >= vix_lo) & (df["vix"] < vix_hi)
        subset = df[mask]
        if len(subset) >= 5:
            wr = subset["is_winner"].mean() * 100
            avg_pnl = subset["pnl"].mean()
            patterns.append({
                "pattern": f"VIX {vix_lo}-{vix_hi}",
                "trades": len(subset), "wr": wr, "avg_pnl": avg_pnl,
                "total_pnl": subset["pnl"].sum(),
            })

    # Pattern 2: By hour of entry
    for hour in range(7):
        mask = df["hour_of_entry"] == hour
        subset = df[mask]
        if len(subset) >= 5:
            wr = subset["is_winner"].mean() * 100
            patterns.append({
                "pattern": f"Hour {hour} ({9+hour}:15-{10+hour}:14)",
                "trades": len(subset), "wr": wr, "avg_pnl": subset["pnl"].mean(),
                "total_pnl": subset["pnl"].sum(),
            })

    # Pattern 3: By direction + VIX
    for action in ["BUY_PUT", "BUY_CALL"]:
        for vix_lo, vix_hi in [(10, 14), (14, 17), (17, 30)]:
            mask = (df["action"] == action) & (df["vix"] >= vix_lo) & (df["vix"] < vix_hi)
            subset = df[mask]
            if len(subset) >= 5:
                wr = subset["is_winner"].mean() * 100
                patterns.append({
                    "pattern": f"{action} + VIX {vix_lo}-{vix_hi}",
                    "trades": len(subset), "wr": wr, "avg_pnl": subset["pnl"].mean(),
                    "total_pnl": subset["pnl"].sum(),
                })

    # Pattern 4: By RSI range at entry
    if "rsi_at_entry" in df.columns:
        for rsi_lo, rsi_hi in [(20, 35), (35, 45), (45, 55), (55, 65), (65, 80)]:
            mask = (df["rsi_at_entry"] >= rsi_lo) & (df["rsi_at_entry"] < rsi_hi)
            subset = df[mask]
            if len(subset) >= 5:
                wr = subset["is_winner"].mean() * 100
                patterns.append({
                    "pattern": f"RSI {rsi_lo}-{rsi_hi}",
                    "trades": len(subset), "wr": wr, "avg_pnl": subset["pnl"].mean(),
                    "total_pnl": subset["pnl"].sum(),
                })

    # Pattern 5: VWAP proximity
    if "vwap_proximity" in df.columns:
        for desc, cond in [
            ("Close > VWAP+0.1%", df["vwap_proximity"] > 0.1),
            ("Close near VWAP (±0.1%)", df["vwap_proximity"].abs() <= 0.1),
            ("Close < VWAP-0.1%", df["vwap_proximity"] < -0.1),
        ]:
            subset = df[cond]
            if len(subset) >= 5:
                wr = subset["is_winner"].mean() * 100
                patterns.append({
                    "pattern": desc,
                    "trades": len(subset), "wr": wr, "avg_pnl": subset["pnl"].mean(),
                    "total_pnl": subset["pnl"].sum(),
                })

    # Pattern 6: Squeeze state
    if "squeeze_at_entry" in df.columns:
        for sq_val, sq_desc in [(0, "No Squeeze"), (1, "In Squeeze")]:
            mask = df["squeeze_at_entry"] == sq_val
            subset = df[mask]
            if len(subset) >= 3:
                wr = subset["is_winner"].mean() * 100
                patterns.append({
                    "pattern": sq_desc,
                    "trades": len(subset), "wr": wr, "avg_pnl": subset["pnl"].mean(),
                    "total_pnl": subset["pnl"].sum(),
                })

    # Sort by total P&L
    patterns.sort(key=lambda x: x["total_pnl"], reverse=True)

    print(f"\n  {'Pattern':<35} {'Trades':>7} {'WR':>7} {'Avg P&L':>12} {'Total P&L':>14}")
    print(f"  {'-'*75}")
    for p in patterns:
        marker = "[OK]" if p["wr"] >= 45 else "[??]" if p["wr"] >= 35 else "[XX]"
        print(f"  {p['pattern']:<35} {p['trades']:>5}t {p['wr']:>5.1f}% "
              f"Rs{p['avg_pnl']:>+10,.0f} Rs{p['total_pnl']:>+12,.0f}  {marker}")


def run_ml_pipeline():
    """Run the complete ML pipeline."""
    print("=" * 100)
    print("  PHASE 6: ML TRADE CLASSIFIER — Feature Engineering + Model Training")
    print("  Based on: 'Architecting a Python-Based Algo Trading System' research doc")
    print(f"  Data: {TRADE_CSV}")
    print("=" * 100)

    # Step 1: Load trades
    print("\n  Step 1: Loading trade data...")
    trades = load_trades()
    print(f"    Loaded {len(trades)} trades")

    # Step 2: Enrich with market features
    print("\n  Step 2: Enriching with market indicator features...")
    enriched = enrich_with_market_features(trades)
    print(f"    Enriched {len(enriched)} trades with {len([c for c in enriched.columns if c.endswith('_at_entry') or c.startswith('vwap') or c.startswith('squeeze') or c.startswith('above') or c.startswith('price_vs')])} market features")

    # Step 3: Build feature matrix
    print("\n  Step 3: Building feature matrix...")
    X, y, feature_cols, label_encoder = build_feature_matrix(enriched)
    print(f"    Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"    Features: {feature_cols}")

    # Step 4: Train and evaluate
    print("\n  Step 4: Training ML models...")
    model = train_and_evaluate(X, y, enriched)

    # Step 5: Save enriched data
    enriched_path = DATA_DIR / "v14_ml_enriched_trades.csv"
    enriched.to_csv(enriched_path, index=False)
    print(f"\n  Enriched trade data saved: {enriched_path}")

    # Step 6: Save trained model for live ML trade filter
    print("\n  Step 6: Saving model for live trading...")
    try:
        from orchestrator.ml_trade_filter import MLTradeFilter
        MLTradeFilter.save_model(model)
        print(f"    Model saved for live use (orchestrator.ml_trade_filter)")
    except Exception as e:
        print(f"    Warning: could not save model for live use: {e}")

    print("\n" + "=" * 100)
    print("  ML Pipeline Complete")
    print("=" * 100)

    return model


if __name__ == "__main__":
    run_ml_pipeline()
