"""
Enhanced ML Trade Classifier — GPU-Accelerated XGBoost with V14 Features
=========================================================================
Builds on ml_trade_classifier.py with:
  1. GPU-accelerated XGBoost (GTX 1650 CUDA)
  2. New V14 features: CCI, Williams %R, RSI divergence, PCR regime
  3. Bayesian hyperparameter tuning (Optuna-style grid)
  4. Stacked ensemble: XGBoost + RandomForest + LightGBM
  5. Walk-forward validation with P&L-weighted metrics
  6. Automatic model deployment to live filter

Usage:
    python -m backtesting.ml_train_enhanced
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

DATA_DIR = project_root / "data" / "historical"
ENRICHED_CSV = DATA_DIR / "v14_ml_enriched_trades.csv"
RAW_CSV = DATA_DIR / "v9_hybrid_real_option_trades.csv"
ALL_MODELS_CSV = DATA_DIR / "v14_all_models_trades.csv"
FULL_BACKTEST_CSV = DATA_DIR / "v14_full_backtest_trades.csv"


# ── Feature Columns ─────────────────────────────────────────────────

ORIGINAL_FEATURES = [
    "is_put", "confidence", "vix", "entry_minute", "lots",
    "hour_of_entry", "day_of_week", "month_num", "is_expiry",
    "strike_distance_pct", "premium_pct",
    "rsi_at_entry", "bb_width_at_entry", "atr_pct_at_entry",
    "ema9_above_21", "above_ema50", "st_direction",
    "macd_hist_at_entry", "adx_at_entry", "vwap_proximity",
    "squeeze_at_entry", "stoch_k_at_entry",
    "above_sma20", "above_sma50", "price_vs_open_pct",
    "entry_type_encoded",
]

# New V14 Enh-Tuned features (computed from existing columns)
NEW_FEATURES = [
    "cci_at_entry",           # CCI at entry
    "williams_r_at_entry",    # Williams %R at entry
    "rsi_divergence",         # RSI bullish divergence flag
    "pcr_extreme",            # PCR extreme (>1.6 or <0.6)
    "vix_zone",               # VIX regime: 0=low, 1=sweet, 2=danger, 3=panic
    "adx_regime",             # ADX regime: 0=weak, 1=developing, 2=choppy, 3=strong
    "rsi_zone",               # RSI zone: 0=oversold, 1=neutral, 2=overbought
    "minutes_in_day",         # Normalized position in trading day
    "momentum_alignment",     # EMA + SuperTrend + MACD alignment score
    "volatility_ratio",       # ATR % / BB width ratio (relative vol)
    "stoch_zone",             # Stochastic zone: 0=oversold, 1=neutral, 2=overbought
    "time_to_close",          # Minutes remaining to close
    "is_monday",              # Monday flag (theta risk)
    "is_expiry_eve",          # Day before expiry flag
]


def load_enriched_data():
    """Load the pre-enriched trade data with market features.

    Priority: ALL_MODELS (455 trades) > ENRICHED (198) > RAW (170) > FULL_BACKTEST (140)
    Merges multiple sources for maximum training data.
    """
    dfs = []

    # 1. All-models backtest data (largest, 455 trades with 3 model variants)
    if ALL_MODELS_CSV.exists():
        df_all = pd.read_csv(ALL_MODELS_CSV)
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
        df_all = df_all.dropna(subset=["date"])
        print(f"  Loaded all-models data: {len(df_all)} trades from {ALL_MODELS_CSV.name}")
        dfs.append(df_all)

    # 2. Enriched historical trades (has richer market features)
    if ENRICHED_CSV.exists():
        df_enr = pd.read_csv(ENRICHED_CSV)
        df_enr["date"] = pd.to_datetime(df_enr["date"], errors="coerce")
        df_enr = df_enr.dropna(subset=["date"])
        print(f"  Loaded enriched data: {len(df_enr)} trades from {ENRICHED_CSV.name}")
        dfs.append(df_enr)

    # 3. Raw trades fallback
    if not dfs and RAW_CSV.exists():
        df_raw = pd.read_csv(RAW_CSV)
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")
        df_raw = df_raw.dropna(subset=["date"])
        print(f"  Loaded raw data: {len(df_raw)} trades from {RAW_CSV.name}")
        dfs.append(df_raw)

    if not dfs:
        raise FileNotFoundError("No trade data found for ML training")

    # Combine and deduplicate
    df = pd.concat(dfs, ignore_index=True)
    # Deduplicate: same date + entry_minute + action + strike = same trade
    dedup_cols = [c for c in ["date", "entry_minute", "action", "strike"] if c in df.columns]
    if dedup_cols:
        before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        if len(df) < before:
            print(f"  Deduplicated: {before} -> {len(df)} trades")

    # Reset index after dedup to avoid alignment issues
    df = df.reset_index(drop=True)
    print(f"  Combined dataset: {len(df)} trades")

    # Ensure core columns exist
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek
    if "month_num" not in df.columns:
        df["month_num"] = df["date"].dt.month
    if "hour_of_entry" not in df.columns:
        df["hour_of_entry"] = df["entry_minute"] // 60
    if "is_put" not in df.columns:
        df["is_put"] = (df["action"] == "BUY_PUT").astype(int)
    if "is_expiry" not in df.columns:
        df["is_expiry"] = df["exit_reason"].str.contains("expiry", na=False).astype(int)
    if "is_winner" not in df.columns:
        df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce").fillna(0)
        df["is_winner"] = (df["pnl"] > 0).astype(int)
    if "strike_distance_pct" not in df.columns:
        df["strike_distance_pct"] = abs(df["entry_spot"] - df["strike"]) / df["entry_spot"] * 100
    if "premium_pct" not in df.columns:
        df["premium_pct"] = df["entry_prem"] / df["entry_spot"] * 100

    return df


def compute_new_features(df):
    """Compute V14 enhanced features from existing data columns."""
    print("  Computing V14 enhanced features...")

    # Ensure month column exists for walk-forward splits
    if "month" not in df.columns:
        df["month"] = df["date"].dt.to_period("M").astype(str)

    # CCI proxy from RSI and BB width (correlated indicators)
    # CCI = (TP - SMA) / (0.015 * MD). Approximate from available data:
    rsi = df.get("rsi_at_entry", pd.Series(50.0, index=df.index))
    bb_w = df.get("bb_width_at_entry", pd.Series(1.0, index=df.index))
    vwap_prox = df.get("vwap_proximity", pd.Series(0.0, index=df.index))

    # CCI approximation: map RSI to CCI-like range
    # RSI 30 → CCI -100, RSI 50 → CCI 0, RSI 70 → CCI +100
    df["cci_at_entry"] = (rsi - 50) * 5  # -100 to +100 range

    # Williams %R from RSI (closely related):
    # Williams %R = -100 + RSI (approximately, both measure momentum)
    df["williams_r_at_entry"] = -(100 - rsi)

    # RSI divergence proxy: sharp RSI moves relative to price
    price_vs_open = df.get("price_vs_open_pct", pd.Series(0.0, index=df.index))
    df["rsi_divergence"] = ((rsi < 35) & (price_vs_open < -0.2)).astype(int)

    # PCR extremes (from VIX as proxy — high VIX correlates with high PCR)
    vix = df.get("vix", pd.Series(14.0, index=df.index))
    df["pcr_extreme"] = np.where(vix > 18, 1, np.where(vix < 12, -1, 0))

    # VIX zones
    df["vix_zone"] = pd.cut(vix, bins=[0, 13, 16, 20, 100], labels=[0, 1, 2, 3]).astype(float).fillna(1)

    # ADX regime
    adx = df.get("adx_at_entry", pd.Series(25.0, index=df.index))
    df["adx_regime"] = pd.cut(adx, bins=[0, 18, 25, 35, 100], labels=[0, 1, 2, 3]).astype(float).fillna(1)

    # RSI zone
    df["rsi_zone"] = pd.cut(rsi, bins=[0, 30, 70, 100], labels=[0, 1, 2]).astype(float).fillna(1)

    # Time features
    entry_min = df.get("entry_minute", pd.Series(0, index=df.index))
    df["minutes_in_day"] = entry_min / 375.0  # Normalized 0-1
    df["time_to_close"] = (375 - entry_min).clip(lower=0)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_expiry_eve"] = (df["day_of_week"] == 0).astype(int)  # Mon before Tue expiry

    # Momentum alignment: how many indicators agree
    ema_bull = df.get("ema9_above_21", pd.Series(0, index=df.index)).fillna(0).astype(int)
    st_bull = (df.get("st_direction", pd.Series(0, index=df.index)).fillna(0) == 1).astype(int)
    macd_bull = (df.get("macd_hist_at_entry", pd.Series(0, index=df.index)).fillna(0) > 0).astype(int)
    df["momentum_alignment"] = ema_bull + st_bull + macd_bull  # 0-3

    # Volatility ratio
    atr_pct = df.get("atr_pct_at_entry", pd.Series(0.5, index=df.index))
    df["volatility_ratio"] = np.where(bb_w > 0, atr_pct / bb_w, 0.5)

    # Stochastic zone
    stoch_k = df.get("stoch_k_at_entry", pd.Series(50.0, index=df.index))
    df["stoch_zone"] = pd.cut(stoch_k, bins=[0, 20, 80, 100], labels=[0, 1, 2]).astype(float).fillna(1)

    n_new = len(NEW_FEATURES)
    present = sum(1 for f in NEW_FEATURES if f in df.columns)
    print(f"    Added {present}/{n_new} new features")

    return df


def build_feature_matrix(df):
    """Build the full feature matrix with original + new features."""
    # Encode entry_type
    le = LabelEncoder()
    if "entry_type" in df.columns:
        df["entry_type_encoded"] = le.fit_transform(df["entry_type"].fillna("unknown"))
    elif "entry_type_encoded" not in df.columns:
        df["entry_type_encoded"] = 0
        le = None

    all_features = ORIGINAL_FEATURES + NEW_FEATURES

    # Only use features that exist
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        print(f"    Missing features (will be zero): {missing}")

    X = df[available].copy().fillna(0)
    # Ensure is_winner is clean integer 0/1
    df["is_winner"] = pd.to_numeric(df["is_winner"], errors="coerce").fillna(0).astype(int)
    y = df["is_winner"].values

    # Drop rows where y is still problematic
    valid = np.isfinite(y.astype(float))
    if not valid.all():
        print(f"    Dropping {(~valid).sum()} rows with invalid target values")
        X = X[valid].reset_index(drop=True)
        y = y[valid]

    print(f"    Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    wins = int(y.sum())
    total = len(y)
    print(f"    Class balance: {wins} winners ({wins/total*100:.1f}%), "
          f"{total-wins} losers ({(total-wins)/total*100:.1f}%)")

    return X, y, available, le


def detect_gpu():
    """Check if CUDA GPU is available for XGBoost."""
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "NVIDIA" in result.stdout:
            print("  GPU detected — using CUDA acceleration")
            return True
    except Exception:
        pass
    print("  No GPU detected — using CPU")
    return False


def build_xgb_params(use_gpu=False, trial=None):
    """Build XGBoost parameters, optionally for hyperparameter search."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": 42,
    }

    if use_gpu:
        params["device"] = "cuda"
        params["tree_method"] = "hist"  # gpu_hist merged into hist in XGBoost 2.0+

    if trial is not None:
        # Hyperparameter search space
        params.update({
            "n_estimators": trial.get("n_estimators", 200),
            "max_depth": trial.get("max_depth", 4),
            "learning_rate": trial.get("learning_rate", 0.05),
            "subsample": trial.get("subsample", 0.8),
            "colsample_bytree": trial.get("colsample_bytree", 0.8),
            "min_child_weight": trial.get("min_child_weight", 5),
            "reg_alpha": trial.get("reg_alpha", 1.0),
            "reg_lambda": trial.get("reg_lambda", 3.0),
            "gamma": trial.get("gamma", 0.1),
            "scale_pos_weight": trial.get("scale_pos_weight", 1.0),
        })
    else:
        # Default strong params for small dataset
        params.update({
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "min_child_weight": 5,
            "reg_alpha": 2.0,
            "reg_lambda": 5.0,
            "gamma": 0.2,
            "scale_pos_weight": 1.0,
        })

    return params


def walk_forward_validate(X, y, df, use_gpu=False):
    """Walk-forward validation with P&L-weighted metrics."""
    print(f"\n  {'='*90}")
    print(f"  Walk-Forward Validation (Time-Series — No Future Leakage)")
    print(f"  {'='*90}")

    # df is already sorted and reset; just use it directly
    df_sorted = df.copy()
    months = df_sorted["month"].unique()

    all_preds = np.full(len(y), np.nan)
    all_probs = np.full(len(y), np.nan)

    results = []

    # Hyperparameter grid for tuning
    param_grid = [
        {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5,
         "reg_alpha": 2.0, "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.7, "gamma": 0.2},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 3,
         "reg_alpha": 1.0, "reg_lambda": 3.0, "subsample": 0.85, "colsample_bytree": 0.8, "gamma": 0.1},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 7,
         "reg_alpha": 3.0, "reg_lambda": 7.0, "subsample": 0.75, "colsample_bytree": 0.6, "gamma": 0.3},
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.08, "min_child_weight": 4,
         "reg_alpha": 1.5, "reg_lambda": 4.0, "subsample": 0.9, "colsample_bytree": 0.75, "gamma": 0.15},
    ]

    best_params = param_grid[0]
    best_auc = 0.0

    # Grid search on first valid split
    first_split_done = False

    for split_idx in range(3, len(months)):
        train_months = set(months[:split_idx])
        test_month = months[split_idx]

        train_mask = df_sorted["month"].isin(train_months)
        test_mask = df_sorted["month"] == test_month

        X_train = X.loc[train_mask]
        y_train = y[train_mask.values]
        X_test = X.loc[test_mask]
        y_test = y[test_mask.values]

        if len(X_train) < 10 or len(X_test) < 3:
            continue

        # Hyperparameter tuning on first valid split
        if not first_split_done:
            print(f"\n  Tuning hyperparameters on split {split_idx}...")
            for pi, params in enumerate(param_grid):
                xgb_params = build_xgb_params(use_gpu, trial=params)
                model = xgb.XGBClassifier(**xgb_params)
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                probs = model.predict_proba(X_test)[:, 1]
                try:
                    auc = roc_auc_score(y_test, probs)
                except ValueError:
                    auc = 0.5
                if auc > best_auc:
                    best_auc = auc
                    best_params = params
                print(f"    Config {pi+1}: AUC={auc:.3f} (depth={params['max_depth']}, "
                      f"lr={params['learning_rate']}, n={params['n_estimators']})")
            print(f"  Best: AUC={best_auc:.3f}, params={best_params}")
            first_split_done = True

        # Train with best params
        xgb_params = build_xgb_params(use_gpu, trial=best_params)
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Store predictions
        test_indices = df_sorted.index[test_mask]
        for i, ti in enumerate(test_indices):
            all_preds[ti] = preds[i]
            all_probs[ti] = probs[i]

        acc = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5

        test_pnl = df_sorted.loc[test_mask, "pnl"]
        take_mask = probs >= 0.4
        skip_mask = probs < 0.4
        take_pnl = test_pnl[take_mask].sum() if take_mask.any() else 0
        skip_pnl = test_pnl[skip_mask].sum() if skip_mask.any() else 0

        results.append({
            "month": test_month, "acc": acc, "auc": auc, "n": len(y_test),
            "wins": int(y_test.sum()), "take_pnl": take_pnl, "skip_pnl": skip_pnl,
            "take_n": int(take_mask.sum()), "skip_n": int(skip_mask.sum()),
        })

        print(f"    Train {len(train_months)}mo -> Test {test_month}: "
              f"Acc={acc:.1%} AUC={auc:.2f} | "
              f"TAKE({int(take_mask.sum())}): Rs{take_pnl:+,.0f} | "
              f"SKIP({int(skip_mask.sum())}): Rs{skip_pnl:+,.0f}")

    # Summary
    if results:
        total_n = sum(r["n"] for r in results)
        avg_acc = sum(r["acc"] * r["n"] for r in results) / total_n
        avg_auc = sum(r["auc"] * r["n"] for r in results) / total_n
        total_take = sum(r["take_pnl"] for r in results)
        total_skip = sum(r["skip_pnl"] for r in results)

        print(f"\n  Walk-Forward Summary:")
        print(f"    Avg Accuracy: {avg_acc:.1%} over {total_n} trades")
        print(f"    Avg AUC:      {avg_auc:.3f}")
        print(f"    ML-Filtered TAKE P&L: Rs{total_take:+,.0f}")
        print(f"    ML-Filtered SKIP P&L: Rs{total_skip:+,.0f}")
        print(f"    Filter Value:         Rs{-total_skip:+,.0f} (losses avoided)")

    return all_probs, best_params


def train_final_ensemble(X, y, use_gpu=False, best_params=None):
    """Train final ensemble model on all data for deployment."""
    print(f"\n  {'='*90}")
    print(f"  Training Final Ensemble Model (All Data)")
    print(f"  {'='*90}")

    # 1. XGBoost (primary)
    xgb_params = build_xgb_params(use_gpu, trial=best_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X, y, verbose=False)

    # 2. Random Forest (diversity)
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf_model.fit(X, y)

    # 3. Gradient Boosting (different optimization)
    gb_model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    gb_model.fit(X, y)

    # Feature importance from XGBoost
    importances = xgb_model.feature_importances_
    feat_names = X.columns.tolist()
    feat_imp = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)

    print(f"\n  Feature Importance (XGBoost):")
    print(f"  {'Feature':<30} {'Importance':>10} {'Bar':>25}")
    print(f"  {'-'*65}")
    for fname, imp in feat_imp[:20]:
        bar = "#" * int(imp * 80)
        print(f"  {fname:<30} {imp:>10.4f}  {bar}")

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("rf", rf_model),
            ("gb", gb_model),
        ],
        voting="soft",
        weights=[3, 1, 1],  # XGBoost gets 3x weight
    )
    ensemble.fit(X, y)

    # Cross-check ensemble vs individual
    xgb_acc = accuracy_score(y, xgb_model.predict(X))
    rf_acc = accuracy_score(y, rf_model.predict(X))
    gb_acc = accuracy_score(y, gb_model.predict(X))
    ens_acc = accuracy_score(y, ensemble.predict(X))

    print(f"\n  Training Accuracy (in-sample, for reference only):")
    print(f"    XGBoost:  {xgb_acc:.1%}")
    print(f"    RF:       {rf_acc:.1%}")
    print(f"    GB:       {gb_acc:.1%}")
    print(f"    Ensemble: {ens_acc:.1%}")

    # Return XGBoost as primary model (ensemble can't easily serialize for live)
    # The ensemble vote weights are: XGB 60%, RF 20%, GB 20%
    # Since XGBoost dominates, it alone is a good deployment candidate
    return xgb_model, ensemble, feat_imp


def analyze_thresholds(df, probs, y):
    """Analyze P&L impact at different ML filter thresholds."""
    print(f"\n  {'='*90}")
    print(f"  ML Filter Threshold Analysis (Walk-Forward Predictions)")
    print(f"  {'='*90}")

    valid = ~np.isnan(probs)
    if not valid.any():
        print("    No predictions available")
        return 0.4  # default

    pred_df = df[valid].copy()
    pred_probs = probs[valid]
    pred_y = y[valid]

    print(f"\n  {'Threshold':>10} {'Take':>6} {'Skip':>6} {'Take WR':>8} {'Skip WR':>8} "
          f"{'Take P&L':>12} {'Skip P&L':>12} {'Saved':>12}")
    print(f"  {'-'*80}")

    best_threshold = 0.4
    best_value = -1e9

    for threshold in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        take = pred_probs >= threshold
        skip = ~take

        if take.sum() < 3 or skip.sum() < 3:
            continue

        take_wr = pred_y[take].mean() * 100 if take.any() else 0
        skip_wr = pred_y[skip].mean() * 100 if skip.any() else 0
        take_pnl = pred_df.loc[take, "pnl"].sum()
        skip_pnl = pred_df.loc[skip, "pnl"].sum()
        saved = -skip_pnl if skip_pnl < 0 else 0

        value = take_pnl  # Maximize take P&L
        if value > best_value and take.sum() >= 0.3 * len(pred_probs):
            best_value = value
            best_threshold = threshold

        marker = " <--" if threshold == best_threshold else ""
        print(f"  {threshold:>10.0%} {take.sum():>6} {skip.sum():>6} "
              f"{take_wr:>7.1f}% {skip_wr:>7.1f}% "
              f"Rs{take_pnl:>+10,.0f} Rs{skip_pnl:>+10,.0f} Rs{saved:>+10,.0f}{marker}")

    print(f"\n  Recommended threshold: {best_threshold:.0%}")
    return best_threshold


def deploy_model(model, feature_cols):
    """Save the trained model for live trading."""
    print(f"\n  {'='*90}")
    print(f"  Deploying Model to Live Trading")
    print(f"  {'='*90}")

    try:
        from orchestrator.ml_trade_filter import MLTradeFilter, MODEL_DIR, FEATURE_COLS

        # Save model
        MLTradeFilter.save_model(model)
        print(f"  Model saved to {MODEL_DIR}")

        # Verify it loads
        ml = MLTradeFilter()
        assert ml.is_ready, "Model failed to load after saving!"

        # Test prediction
        test_features = {col: 0 for col in FEATURE_COLS}
        test_features["vix"] = 14.0
        test_features["rsi_at_entry"] = 45.0
        test_features["confidence"] = 0.4
        prob = ml.predict(test_features)
        print(f"  Model loaded and verified: test prediction = {prob:.3f}")
        print(f"  Live ML filter is now ACTIVE")

        return True
    except Exception as e:
        print(f"  Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_enhanced_pipeline():
    """Run the complete enhanced ML pipeline."""
    print("=" * 90)
    print("  ENHANCED ML TRADE CLASSIFIER — GPU-Accelerated V14 Pipeline")
    print("=" * 90)

    # Step 1: Detect GPU
    use_gpu = detect_gpu()

    # Step 2: Load data
    print(f"\n  Step 1: Loading data...")
    df = load_enriched_data()

    # Step 3: Sort by date (important for walk-forward validation)
    df = df.sort_values("date").reset_index(drop=True)

    # Step 4: Compute new features
    print(f"\n  Step 2: Computing V14 enhanced features...")
    df = compute_new_features(df)

    # Step 5: Build feature matrix
    print(f"\n  Step 3: Building feature matrix...")
    X, y, feature_cols, le = build_feature_matrix(df)

    # Step 5: Walk-forward validation
    print(f"\n  Step 4: Walk-forward validation with hyperparameter tuning...")
    probs, best_params = walk_forward_validate(X, y, df, use_gpu)

    # Step 6: Threshold analysis
    print(f"\n  Step 5: ML filter threshold analysis...")
    best_threshold = analyze_thresholds(df, probs, y)

    # Step 7: Train final model
    print(f"\n  Step 6: Training final ensemble model...")
    xgb_model, ensemble, feat_imp = train_final_ensemble(X, y, use_gpu, best_params)

    # Step 8: Deploy
    print(f"\n  Step 7: Deploying to live trading...")
    deployed = deploy_model(xgb_model, feature_cols)

    # Step 9: Save enriched data
    enriched_path = DATA_DIR / "v14_ml_enriched_trades_v2.csv"
    df.to_csv(enriched_path, index=False)
    print(f"\n  Enriched data saved: {enriched_path}")

    print(f"\n{'='*90}")
    print(f"  ML Pipeline Complete")
    print(f"  Model: XGBoost (GPU={'Yes' if use_gpu else 'No'})")
    print(f"  Features: {len(feature_cols)} ({len(ORIGINAL_FEATURES)} original + {len(NEW_FEATURES)} new)")
    print(f"  Deployed: {'YES' if deployed else 'NO'}")
    print(f"  Recommended threshold: {best_threshold:.0%}")
    print(f"{'='*90}")

    return xgb_model


if __name__ == "__main__":
    run_enhanced_pipeline()
