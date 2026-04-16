"""
learn_global_signals.py
-----------------------
Studies how GLOBAL MARKET SIGNALS predict NIFTY direction.

Downloads 6 months of data (Oct 2025 - Apr 2026) for major global indices,
commodities, and currencies.  Then computes correlations between overnight
global moves and next-day NIFTY performance, builds actionable trading rules,
and saves them to data/global_signal_rules.json.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_DATE = "2025-10-01"
END_DATE = "2026-04-05"

SYMBOLS = {
    "NIFTY": "^NSEI",
    "INDIA_VIX": "^INDIAVIX",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    "NIKKEI": "^N225",
    "HANG_SENG": "^HSI",
    "CRUDE_OIL": "CL=F",
    "GOLD": "GC=F",
    "USD_INR": "USDINR=X",
    "US_10Y": "^TNX",
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_JSON = DATA_DIR / "global_signal_rules.json"


# ---------------------------------------------------------------------------
# 1. Download data
# ---------------------------------------------------------------------------
def download_data() -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for each symbol using yfinance."""
    import yfinance as yf

    data: dict[str, pd.DataFrame] = {}
    for name, ticker in SYMBOLS.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if df is None or df.empty:
                print(f"  [SKIP] {name} ({ticker}): no data returned")
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data[name] = df
            print(f"  [OK]   {name} ({ticker}): {len(df)} rows  "
                  f"({df.index.min().date()} -> {df.index.max().date()})")
        except Exception as e:
            print(f"  [SKIP] {name} ({ticker}): {e}")
    return data


# ---------------------------------------------------------------------------
# 2. Build feature matrix
# ---------------------------------------------------------------------------
def build_features(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each NIFTY trading day, compute previous-day changes for global assets.

    Because US/EU markets close *after* Indian markets, the relevant US move
    for predicting NIFTY on day T is the US close on day T-1 (which happened
    after NIFTY closed on T-1, so it's fresh information for NIFTY on day T).
    """
    nifty = data.get("NIFTY")
    if nifty is None:
        raise RuntimeError("NIFTY data is required but not available.")

    # NIFTY daily returns
    nifty_ret = nifty["Close"].pct_change().rename("nifty_ret")
    nifty_ret = nifty_ret.dropna()

    # Helper: compute daily pct change for a symbol, then shift by 1 day
    # relative to NIFTY calendar (previous trading day's move)
    features = pd.DataFrame(index=nifty_ret.index)
    features["nifty_ret"] = nifty_ret

    def add_prev_day_feature(sym_name: str, col_name: str):
        if sym_name not in data:
            return
        df = data[sym_name]
        ret = df["Close"].pct_change().rename(col_name)
        ret = ret.dropna()
        # Reindex to NIFTY dates, forward-fill to handle holidays
        ret = ret.reindex(features.index, method="ffill")
        # Shift by 1 so we use *previous day's* move
        features[col_name] = ret.shift(1)

    add_prev_day_feature("SP500", "sp500_ret")
    add_prev_day_feature("NASDAQ", "nasdaq_ret")
    add_prev_day_feature("DOW", "dow_ret")
    add_prev_day_feature("NIKKEI", "nikkei_ret")
    add_prev_day_feature("HANG_SENG", "hangseng_ret")
    add_prev_day_feature("CRUDE_OIL", "oil_ret")
    add_prev_day_feature("GOLD", "gold_ret")
    add_prev_day_feature("US_10Y", "us10y_ret")

    # VIX level and change
    if "INDIA_VIX" in data:
        vix_close = data["INDIA_VIX"]["Close"].rename("india_vix")
        vix_close = vix_close.reindex(features.index, method="ffill")
        features["india_vix"] = vix_close
        features["india_vix_chg"] = vix_close.pct_change()

    # USD/INR change
    if "USD_INR" in data:
        fx = data["USD_INR"]["Close"].pct_change().rename("usdinr_ret")
        fx = fx.reindex(features.index, method="ffill")
        features["usdinr_ret"] = fx.shift(1)

    features = features.dropna(subset=["nifty_ret"])
    return features


# ---------------------------------------------------------------------------
# 3. Signal analysis helpers
# ---------------------------------------------------------------------------
def analyse_signal(features: pd.DataFrame, mask: pd.Series, label: str) -> dict:
    """Analyse NIFTY returns when a boolean signal mask is True."""
    subset = features.loc[mask, "nifty_ret"]
    if len(subset) < 3:
        return {"label": label, "count": len(subset), "skip": True}

    mean_ret = subset.mean()
    win_rate_call = (subset > 0).mean()  # fraction where NIFTY was up
    win_rate_put = (subset < 0).mean()

    best_action = "BUY_CALL" if mean_ret > 0 else "BUY_PUT"
    win_rate = win_rate_call if best_action == "BUY_CALL" else win_rate_put

    # Estimated P&L per trade (NIFTY lot = 25, assume ATM option delta~0.5)
    avg_points = mean_ret * 22000  # approximate NIFTY level
    pnl_per_trade = abs(avg_points) * 25 * 0.5  # lot_size * delta

    # t-test: is mean return significantly different from 0?
    t_stat, p_val = stats.ttest_1samp(subset, 0)

    return {
        "label": label,
        "count": int(len(subset)),
        "mean_nifty_ret_pct": round(mean_ret * 100, 3),
        "median_nifty_ret_pct": round(subset.median() * 100, 3),
        "std_pct": round(subset.std() * 100, 3),
        "win_rate_up": round(win_rate_call * 100, 1),
        "win_rate_down": round(win_rate_put * 100, 1),
        "best_action": best_action,
        "best_win_rate": round(win_rate * 100, 1),
        "avg_nifty_points": round(avg_points, 1),
        "est_pnl_per_trade": round(pnl_per_trade, 0),
        "t_stat": round(t_stat, 2),
        "p_value": round(p_val, 4),
        "significant": p_val < 0.10,
    }


def print_signal_table(results: list[dict], title: str):
    """Pretty-print a table of signal analysis results."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"{'Signal':<40} {'N':>4} {'MeanRet%':>9} {'WinUp%':>7} {'WinDn%':>7} "
          f"{'Action':<10} {'WinR%':>6} {'~Pts':>7} {'~PnL':>8} {'pVal':>7} {'Sig':>4}")
    print("-" * 100)
    for r in results:
        if r.get("skip"):
            print(f"{'  ' + r['label']:<40} {r['count']:>4}   (insufficient data)")
            continue
        sig_marker = " **" if r["significant"] else ""
        print(f"{'  ' + r['label']:<40} {r['count']:>4} {r['mean_nifty_ret_pct']:>8.3f}% "
              f"{r['win_rate_up']:>6.1f}% {r['win_rate_down']:>6.1f}% "
              f"{r['best_action']:<10} {r['best_win_rate']:>5.1f}% "
              f"{r['avg_nifty_points']:>6.1f} {r['est_pnl_per_trade']:>7.0f} "
              f"{r['p_value']:>7.4f}{sig_marker}")


# ---------------------------------------------------------------------------
# 4. Compute the global sentiment score
# ---------------------------------------------------------------------------
def compute_sentiment_score(row: pd.Series) -> float:
    """
    Global sentiment score from -5 to +5.
      S&P:  up >1% => +2, up >0.5% => +1, down >0.5% => -1, down >1% => -2
      Oil:  up >2% => -1, down >2% => +1
      Gold: up >1% => -0.5 (risk-off), down >1% => +0.5
    """
    score = 0.0

    sp = row.get("sp500_ret", 0) or 0
    if sp > 0.01:
        score += 2
    elif sp > 0.005:
        score += 1
    elif sp < -0.01:
        score -= 2
    elif sp < -0.005:
        score -= 1

    oil = row.get("oil_ret", 0) or 0
    if oil > 0.02:
        score -= 1
    elif oil < -0.02:
        score += 1

    gold = row.get("gold_ret", 0) or 0
    if gold > 0.01:
        score -= 0.5
    elif gold < -0.01:
        score += 0.5

    return np.clip(score, -5, 5)


# ---------------------------------------------------------------------------
# 5. Correlation matrix
# ---------------------------------------------------------------------------
def compute_correlations(features: pd.DataFrame) -> dict:
    """Pearson and rank correlations between global signals and NIFTY returns."""
    cols = [c for c in features.columns if c != "nifty_ret" and c != "india_vix"]
    corr_results = {}
    for col in cols:
        valid = features[["nifty_ret", col]].dropna()
        if len(valid) < 10:
            continue
        pearson_r, pearson_p = stats.pearsonr(valid["nifty_ret"], valid[col])
        spearman_r, spearman_p = stats.spearmanr(valid["nifty_ret"], valid[col])
        corr_results[col] = {
            "pearson_r": round(pearson_r, 4),
            "pearson_p": round(pearson_p, 4),
            "spearman_r": round(spearman_r, 4),
            "spearman_p": round(spearman_p, 4),
            "n": int(len(valid)),
        }
    return corr_results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 100)
    print("  GLOBAL SIGNALS -> NIFTY DIRECTION STUDY")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print("=" * 100)

    # ---- Download ----
    print("\n[1] Downloading data...")
    data = download_data()
    print(f"\n    Successfully loaded {len(data)} symbols.")

    if "NIFTY" not in data:
        print("FATAL: Could not download NIFTY data. Exiting.")
        sys.exit(1)

    # ---- Build features ----
    print("\n[2] Building feature matrix...")
    features = build_features(data)
    print(f"    Feature matrix: {features.shape[0]} rows x {features.shape[1]} cols")
    print(f"    Columns: {list(features.columns)}")

    # ---- S&P 500 signals ----
    sp_results = []
    if "sp500_ret" in features.columns:
        sp = features["sp500_ret"]
        sp_results = [
            analyse_signal(features, sp > 0.01, "S&P up > 1%"),
            analyse_signal(features, (sp > 0.005) & (sp <= 0.01), "S&P up 0.5%-1%"),
            analyse_signal(features, (sp > 0) & (sp <= 0.005), "S&P up 0-0.5%"),
            analyse_signal(features, (sp < 0) & (sp >= -0.005), "S&P down 0-0.5%"),
            analyse_signal(features, (sp < -0.005) & (sp >= -0.01), "S&P down 0.5%-1%"),
            analyse_signal(features, sp < -0.01, "S&P down > 1%"),
        ]
        print_signal_table(sp_results, "S&P 500 PREVIOUS-DAY SIGNAL -> NIFTY NEXT DAY")

    # ---- NASDAQ signals ----
    nq_results = []
    if "nasdaq_ret" in features.columns:
        nq = features["nasdaq_ret"]
        nq_results = [
            analyse_signal(features, nq > 0.01, "NASDAQ up > 1%"),
            analyse_signal(features, (nq > 0.005) & (nq <= 0.01), "NASDAQ up 0.5%-1%"),
            analyse_signal(features, nq < -0.005, "NASDAQ down > 0.5%"),
            analyse_signal(features, nq < -0.01, "NASDAQ down > 1%"),
        ]
        print_signal_table(nq_results, "NASDAQ PREVIOUS-DAY SIGNAL -> NIFTY NEXT DAY")

    # ---- OIL signals ----
    oil_results = []
    if "oil_ret" in features.columns:
        oil = features["oil_ret"]
        oil_results = [
            analyse_signal(features, oil > 0.02, "Oil spike > 2%"),
            analyse_signal(features, (oil > 0.01) & (oil <= 0.02), "Oil up 1%-2%"),
            analyse_signal(features, (oil > 0) & (oil <= 0.01), "Oil up 0-1%"),
            analyse_signal(features, (oil < 0) & (oil >= -0.01), "Oil down 0-1%"),
            analyse_signal(features, (oil < -0.01) & (oil >= -0.02), "Oil down 1%-2%"),
            analyse_signal(features, oil < -0.02, "Oil drop > 2%"),
        ]
        print_signal_table(oil_results, "CRUDE OIL PREVIOUS-DAY SIGNAL -> NIFTY NEXT DAY")

    # ---- GOLD signals ----
    gold_results = []
    if "gold_ret" in features.columns:
        g = features["gold_ret"]
        gold_results = [
            analyse_signal(features, g > 0.01, "Gold up > 1%"),
            analyse_signal(features, (g > 0) & (g <= 0.01), "Gold up 0-1%"),
            analyse_signal(features, (g < 0) & (g >= -0.01), "Gold down 0-1%"),
            analyse_signal(features, g < -0.01, "Gold down > 1%"),
        ]
        print_signal_table(gold_results, "GOLD PREVIOUS-DAY SIGNAL -> NIFTY NEXT DAY")

    # ---- VIX signals ----
    vix_results = []
    if "india_vix_chg" in features.columns:
        vc = features["india_vix_chg"]
        vix_results = [
            analyse_signal(features, vc > 0.10, "VIX spike > 10%"),
            analyse_signal(features, (vc > 0.05) & (vc <= 0.10), "VIX up 5%-10%"),
            analyse_signal(features, (vc > 0) & (vc <= 0.05), "VIX up 0-5%"),
            analyse_signal(features, (vc < 0) & (vc >= -0.05), "VIX down 0-5%"),
            analyse_signal(features, vc < -0.05, "VIX drop > 5%"),
        ]
        print_signal_table(vix_results, "INDIA VIX CHANGE -> NIFTY SAME DAY")

    # ---- USD/INR signals ----
    fx_results = []
    if "usdinr_ret" in features.columns:
        fx = features["usdinr_ret"]
        fx_results = [
            analyse_signal(features, fx > 0.005, "INR weakens > 0.5%"),
            analyse_signal(features, (fx > 0) & (fx <= 0.005), "INR weakens 0-0.5%"),
            analyse_signal(features, (fx < 0) & (fx >= -0.005), "INR strengthens 0-0.5%"),
            analyse_signal(features, fx < -0.005, "INR strengthens > 0.5%"),
        ]
        print_signal_table(fx_results, "USD/INR PREVIOUS-DAY SIGNAL -> NIFTY NEXT DAY")

    # ---- COMBINED signals ----
    combined_results = []
    if "sp500_ret" in features.columns and "oil_ret" in features.columns:
        sp = features["sp500_ret"]
        oil = features["oil_ret"]
        combined_results = [
            analyse_signal(features, (sp > 0.005) & (oil < -0.01),
                           "S&P up >0.5% + Oil down >1% (BULLISH)"),
            analyse_signal(features, (sp < -0.005) & (oil > 0.01),
                           "S&P down >0.5% + Oil up >1% (BEARISH)"),
            analyse_signal(features, (sp > 0.005) & (oil > 0.01),
                           "S&P up >0.5% + Oil up >1% (MIXED)"),
            analyse_signal(features, (sp < -0.005) & (oil < -0.01),
                           "S&P down >0.5% + Oil down >1% (RISK-OFF)"),
        ]

        # Extra combined with gold
        if "gold_ret" in features.columns:
            g = features["gold_ret"]
            combined_results.extend([
                analyse_signal(features, (sp > 0.005) & (g < 0),
                               "S&P up >0.5% + Gold down (RISK-ON)"),
                analyse_signal(features, (sp < -0.005) & (g > 0.01),
                               "S&P down >0.5% + Gold up >1% (FLIGHT)"),
            ])

        print_signal_table(combined_results,
                           "COMBINED GLOBAL SIGNALS -> NIFTY NEXT DAY")

    # ---- NIKKEI + HANG SENG (Asian peers) ----
    asia_results = []
    if "nikkei_ret" in features.columns:
        nk = features["nikkei_ret"]
        asia_results.extend([
            analyse_signal(features, nk > 0.01, "Nikkei up > 1%"),
            analyse_signal(features, nk < -0.01, "Nikkei down > 1%"),
        ])
    if "hangseng_ret" in features.columns:
        hs = features["hangseng_ret"]
        asia_results.extend([
            analyse_signal(features, hs > 0.01, "Hang Seng up > 1%"),
            analyse_signal(features, hs < -0.01, "Hang Seng down > 1%"),
        ])
    if asia_results:
        print_signal_table(asia_results, "ASIAN MARKETS SIGNAL -> NIFTY")

    # ---- US 10Y yield ----
    yield_results = []
    if "us10y_ret" in features.columns:
        y10 = features["us10y_ret"]
        yield_results = [
            analyse_signal(features, y10 > 0.02, "US 10Y yield up > 2%"),
            analyse_signal(features, y10 < -0.02, "US 10Y yield down > 2%"),
        ]
        print_signal_table(yield_results, "US 10Y YIELD SIGNAL -> NIFTY")

    # ---- Global Sentiment Score ----
    print(f"\n{'=' * 100}")
    print("  GLOBAL SENTIMENT SCORE ANALYSIS")
    print(f"{'=' * 100}")
    features["sentiment_score"] = features.apply(compute_sentiment_score, axis=1)
    score_results = []
    ss = features["sentiment_score"]
    score_results = [
        analyse_signal(features, ss >= 3, "Sentiment >= +3 (STRONGLY BULLISH)"),
        analyse_signal(features, (ss >= 2) & (ss < 3), "Sentiment +2 to +3 (BULLISH)"),
        analyse_signal(features, (ss >= 1) & (ss < 2), "Sentiment +1 to +2 (MILD BULL)"),
        analyse_signal(features, (ss > -1) & (ss < 1), "Sentiment -1 to +1 (NEUTRAL)"),
        analyse_signal(features, (ss <= -1) & (ss > -2), "Sentiment -1 to -2 (MILD BEAR)"),
        analyse_signal(features, (ss <= -2) & (ss > -3), "Sentiment -2 to -3 (BEARISH)"),
        analyse_signal(features, ss <= -3, "Sentiment <= -3 (STRONGLY BEARISH)"),
    ]
    print_signal_table(score_results, "GLOBAL SENTIMENT SCORE -> NIFTY")

    # Distribution
    print(f"\n  Score distribution:")
    for s_val in sorted(features["sentiment_score"].unique()):
        cnt = (features["sentiment_score"] == s_val).sum()
        avg_r = features.loc[features["sentiment_score"] == s_val, "nifty_ret"].mean()
        bar = "+" * max(1, int(cnt / 2)) if avg_r > 0 else "-" * max(1, int(cnt / 2))
        print(f"    Score {s_val:>5.1f}: {cnt:>3} days  avg_ret={avg_r * 100:>7.3f}%  {bar}")

    # ---- Correlations ----
    print(f"\n{'=' * 100}")
    print("  CORRELATION MATRIX: GLOBAL SIGNALS vs NIFTY RETURN")
    print(f"{'=' * 100}")
    corr_stats = compute_correlations(features)
    print(f"\n  {'Signal':<20} {'Pearson r':>10} {'p-val':>8} {'Spearman r':>11} {'p-val':>8} {'N':>5}")
    print(f"  {'-' * 65}")
    for col, vals in sorted(corr_stats.items(), key=lambda x: abs(x[1]["pearson_r"]),
                            reverse=True):
        sig = " **" if vals["pearson_p"] < 0.05 else (" *" if vals["pearson_p"] < 0.10 else "")
        print(f"  {col:<20} {vals['pearson_r']:>10.4f} {vals['pearson_p']:>8.4f} "
              f"{vals['spearman_r']:>11.4f} {vals['spearman_p']:>8.4f} {vals['n']:>5}{sig}")

    # ---- Build rules JSON ----
    print(f"\n{'=' * 100}")
    print("  BUILDING RULES JSON")
    print(f"{'=' * 100}")

    def extract_rules(results_list: list[dict]) -> dict:
        rules = {}
        for r in results_list:
            if r.get("skip"):
                continue
            key = r["label"].lower().replace(" ", "_").replace(">", "gt").replace("<", "lt")
            key = key.replace("%", "pct").replace("-", "_").replace("(", "").replace(")", "")
            key = key.replace("+", "plus").replace("__", "_")
            rules[key] = {
                "action": r["best_action"],
                "win_rate": r["best_win_rate"],
                "count": r["count"],
                "mean_ret_pct": r["mean_nifty_ret_pct"],
                "significant": r["significant"],
            }
        return rules

    rules_json = {
        "sp500_rules": extract_rules(sp_results),
        "nasdaq_rules": extract_rules(nq_results),
        "oil_rules": extract_rules(oil_results),
        "gold_rules": extract_rules(gold_results),
        "vix_rules": extract_rules(vix_results),
        "usdinr_rules": extract_rules(fx_results),
        "combined_rules": extract_rules(combined_results),
        "asian_market_rules": extract_rules(asia_results),
        "us10y_rules": extract_rules(yield_results),
        "sentiment_score_actions": extract_rules(score_results),
        "sentiment_score_logic": {
            "sp500_up_gt_1pct": "+2",
            "sp500_up_gt_0.5pct": "+1",
            "sp500_down_gt_0.5pct": "-1",
            "sp500_down_gt_1pct": "-2",
            "oil_up_gt_2pct": "-1",
            "oil_down_gt_2pct": "+1",
            "gold_up_gt_1pct": "-0.5 (risk-off)",
            "gold_down_gt_1pct": "+0.5",
        },
        "correlation_stats": corr_stats,
        "metadata": {
            "period": f"{START_DATE} to {END_DATE}",
            "nifty_trading_days": int(len(features)),
            "symbols_loaded": list(data.keys()),
            "generated_at": datetime.now().isoformat(),
        },
    }

    # ---- Key findings summary ----
    print("\n  KEY FINDINGS:")
    all_results = (sp_results + nq_results + oil_results + gold_results
                   + vix_results + fx_results + combined_results
                   + asia_results + yield_results + score_results)
    significant = [r for r in all_results if not r.get("skip") and r.get("significant")]
    significant.sort(key=lambda x: abs(x["mean_nifty_ret_pct"]), reverse=True)
    for i, r in enumerate(significant[:15], 1):
        direction = "NIFTY UP" if r["mean_nifty_ret_pct"] > 0 else "NIFTY DOWN"
        print(f"    {i:>2}. {r['label']:<45} -> {direction} "
              f"({r['mean_nifty_ret_pct']:+.3f}%, win={r['best_win_rate']:.0f}%, "
              f"p={r['p_value']:.3f})")

    # ---- Top actionable rules ----
    print("\n  TOP ACTIONABLE RULES (by win rate, min 5 occurrences):")
    actionable = [r for r in all_results
                  if not r.get("skip") and r["count"] >= 5 and r["best_win_rate"] >= 55]
    actionable.sort(key=lambda x: x["best_win_rate"], reverse=True)
    for i, r in enumerate(actionable[:15], 1):
        print(f"    {i:>2}. {r['best_action']:<10} when {r['label']:<40} "
              f"(win={r['best_win_rate']:.0f}%, N={r['count']}, "
              f"est_pnl=Rs.{r['est_pnl_per_trade']:,.0f})")

    rules_json["top_actionable_rules"] = [
        {
            "rank": i,
            "action": r["best_action"],
            "signal": r["label"],
            "win_rate": r["best_win_rate"],
            "count": r["count"],
            "mean_ret_pct": r["mean_nifty_ret_pct"],
            "est_pnl_per_trade": r["est_pnl_per_trade"],
        }
        for i, r in enumerate(actionable[:15], 1)
    ]

    # ---- Save ----
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(rules_json, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Rules saved to: {OUTPUT_JSON}")
    print(f"{'=' * 100}")


if __name__ == "__main__":
    main()
