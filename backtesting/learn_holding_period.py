"""
Optimal Holding Period Study for NIFTY Options Trades.

Downloads 6 months of NIFTY + India VIX data and simulates BUY_CALL / BUY_PUT
trades with different holding/exit strategies:
  1. Intraday close (same-day EOD)
  2. Hold 2 days (next-day close)
  3. Hold until expiry (DTE 2 -> 0)
  4. Partial: exit half at +0.5% underlying move, hold rest to EOD
  5. Scale out: exit 1/3 at +0.3%, 1/3 at +0.6%, hold 1/3 to EOD

Also studies:
  - Overnight gap risk vs momentum continuation
  - Trending vs rangebound holding behaviour
  - Theta decay at different VIX levels

Saves rules to data/holding_rules.json.
"""

import sys
import os
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── project root on path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yfinance as yf
import numpy as np

from backtesting.option_pricer import price_option
from config.constants import (
    INDEX_CONFIG,
    STT_RATES,
    SEBI_TURNOVER_FEE,
    NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY,
    GST_RATE,
    VIX_LOW,
    VIX_NORMAL_LOW,
    VIX_NORMAL_HIGH,
    VIX_HIGH,
)

# ── Parameters ──────────────────────────────────────────────────────────
LOT_SIZE = 65
CAPITAL = 200_000
RISK_FREE_RATE = 0.07
DTE_ENTRY = 2.0          # days to expiry at entry
STRIKE_INTERVAL = 50     # NIFTY strike interval
BROKERAGE_PER_ORDER = 20 # flat brokerage

# ── Transaction cost helper ─────────────────────────────────────────────

def compute_costs(premium_buy: float, premium_sell: float, lots: int = 1) -> float:
    """Total round-trip cost for one lot of options (buy + sell)."""
    qty = LOT_SIZE * lots
    buy_turnover = premium_buy * qty
    sell_turnover = premium_sell * qty

    # Brokerage (buy + sell)
    brokerage = BROKERAGE_PER_ORDER * 2

    # STT: on sell side only for options
    stt = sell_turnover * STT_RATES["options_sell"]

    # Exchange transaction charges (both sides)
    exchange = (buy_turnover + sell_turnover) * NSE_TRANSACTION_CHARGE

    # SEBI turnover fee (both sides)
    sebi = (buy_turnover + sell_turnover) * SEBI_TURNOVER_FEE

    # Stamp duty (buy side only)
    stamp = buy_turnover * STAMP_DUTY_BUY

    # GST on brokerage + exchange charges
    gst = (brokerage + exchange) * GST_RATE

    return brokerage + stt + exchange + sebi + stamp + gst


def atm_strike(spot: float) -> float:
    """Round to nearest NIFTY strike interval."""
    return round(spot / STRIKE_INTERVAL) * STRIKE_INTERVAL


# ── Data download ──────────────────────────────────────────────────────

def download_data():
    """Download 6 months of NIFTY and India VIX data."""
    print("Downloading NIFTY (^NSEI) data Oct 2025 - Apr 2026 ...")
    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-05",
                         progress=False, auto_adjust=True)
    print(f"  Got {len(nifty)} NIFTY rows")

    print("Downloading India VIX (^INDIAVIX) ...")
    vix = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-05",
                       progress=False, auto_adjust=True)
    print(f"  Got {len(vix)} VIX rows")

    # Flatten multi-level columns if needed
    if hasattr(nifty.columns, 'levels') and nifty.columns.nlevels > 1:
        nifty.columns = nifty.columns.get_level_values(0)
    if hasattr(vix.columns, 'levels') and vix.columns.nlevels > 1:
        vix.columns = vix.columns.get_level_values(0)

    # Merge on date
    nifty = nifty.rename(columns={"Close": "nifty_close", "Open": "nifty_open",
                                   "High": "nifty_high", "Low": "nifty_low"})
    vix = vix.rename(columns={"Close": "vix_close"})

    df = nifty[["nifty_open", "nifty_high", "nifty_low", "nifty_close"]].copy()
    df["vix"] = vix["vix_close"]
    df = df.dropna()

    # SMA50 for trend classification
    df["sma50"] = df["nifty_close"].rolling(50).mean()
    df = df.dropna()

    print(f"  Merged dataset: {len(df)} trading days")
    return df


# ── VIX regime classification ──────────────────────────────────────────

def vix_regime(v: float) -> str:
    if v < VIX_LOW:
        return "LOW"
    elif v < VIX_NORMAL_HIGH:
        return "NORMAL"
    else:
        return "HIGH"


def trend_regime(row) -> str:
    if row["nifty_close"] > row["sma50"] * 1.01:
        return "ABOVE_SMA50"
    elif row["nifty_close"] < row["sma50"] * 0.99:
        return "BELOW_SMA50"
    else:
        return "NEAR_SMA50"


# ── Simulate a single trade ────────────────────────────────────────────

def simulate_trade(spot_entry, spot_exit, vix_val, action, dte_entry, dte_exit):
    """
    Simulate a directional option trade and return P&L per lot.

    Parameters
    ----------
    spot_entry : float — underlying at entry
    spot_exit  : float — underlying at exit
    vix_val    : float — VIX at entry
    action     : str   — 'BUY_CALL' or 'BUY_PUT'
    dte_entry  : float — DTE at entry
    dte_exit   : float — DTE at exit
    """
    strike = atm_strike(spot_entry)
    opt_type = "CE" if action == "BUY_CALL" else "PE"

    entry_price = price_option(spot_entry, strike, dte_entry, vix_val, opt_type)
    exit_price = price_option(spot_exit, strike, dte_exit, vix_val, opt_type)

    prem_buy = entry_price["premium"]
    prem_sell = exit_price["premium"]

    pnl_per_unit = prem_sell - prem_buy
    pnl_per_lot = pnl_per_unit * LOT_SIZE
    costs = compute_costs(prem_buy, prem_sell)
    net_pnl = pnl_per_lot - costs

    return {
        "pnl": net_pnl,
        "prem_buy": prem_buy,
        "prem_sell": prem_sell,
        "costs": costs,
        "theta_entry": entry_price["theta"],
        "delta_entry": entry_price["delta"],
    }


# ── Strategy simulators ────────────────────────────────────────────────

def strategy_intraday(row, action):
    """Enter at open, exit at close same day. DTE 2.0 -> ~1.6 (0.4 day passes)."""
    return simulate_trade(
        row["nifty_open"], row["nifty_close"],
        row["vix"], action, DTE_ENTRY, DTE_ENTRY - 0.4
    )


def strategy_hold2(row, next_row, action):
    """Enter at open day 1, exit at close day 2. DTE 2.0 -> ~0.6."""
    return simulate_trade(
        row["nifty_open"], next_row["nifty_close"],
        row["vix"], action, DTE_ENTRY, DTE_ENTRY - 1.4
    )


def strategy_hold_expiry(row, next_row, action):
    """Enter at open, hold until expiry. DTE 2.0 -> 0."""
    # At expiry: intrinsic value only
    strike = atm_strike(row["nifty_open"])
    opt_type = "CE" if action == "BUY_CALL" else "PE"

    entry = price_option(row["nifty_open"], strike, DTE_ENTRY, row["vix"], opt_type)
    prem_buy = entry["premium"]

    # Expiry payoff (use next day close as proxy for expiry settlement)
    if opt_type == "CE":
        intrinsic = max(0, next_row["nifty_close"] - strike)
    else:
        intrinsic = max(0, strike - next_row["nifty_close"])

    pnl_per_lot = (intrinsic - prem_buy) * LOT_SIZE
    costs = compute_costs(prem_buy, max(intrinsic, 0.05))
    return {
        "pnl": pnl_per_lot - costs,
        "prem_buy": prem_buy,
        "prem_sell": intrinsic,
        "costs": costs,
        "theta_entry": entry["theta"],
        "delta_entry": entry["delta"],
    }


def strategy_partial(row, action):
    """
    Exit half at +0.5% favorable move in underlying, hold rest to EOD.
    DTE 2.0 entry.
    """
    spot_open = row["nifty_open"]
    spot_close = row["nifty_close"]
    high = row["nifty_high"]
    low = row["nifty_low"]

    # Check if favorable move happened intraday
    if action == "BUY_CALL":
        favorable_target = spot_open * 1.005
        favorable_hit = high >= favorable_target
        mid_exit_spot = favorable_target if favorable_hit else spot_close
    else:
        favorable_target = spot_open * 0.995
        favorable_hit = low <= favorable_target
        mid_exit_spot = favorable_target if favorable_hit else spot_close

    # Part 1: half lot exits at favorable move (or EOD if not hit)
    dte_mid = DTE_ENTRY - 0.2 if favorable_hit else DTE_ENTRY - 0.4
    t1 = simulate_trade(spot_open, mid_exit_spot, row["vix"], action, DTE_ENTRY, dte_mid)

    # Part 2: half lot exits at EOD
    t2 = simulate_trade(spot_open, spot_close, row["vix"], action, DTE_ENTRY, DTE_ENTRY - 0.4)

    # Weighted P&L (half each)
    half_qty = LOT_SIZE / 2
    pnl = ((t1["prem_sell"] - t1["prem_buy"]) * half_qty +
           (t2["prem_sell"] - t2["prem_buy"]) * half_qty)
    costs = compute_costs(t1["prem_buy"], t1["prem_sell"]) * 0.5 + \
            compute_costs(t2["prem_buy"], t2["prem_sell"]) * 0.5
    # Extra leg cost for partial exit
    costs += BROKERAGE_PER_ORDER

    return {
        "pnl": pnl - costs,
        "prem_buy": t1["prem_buy"],
        "prem_sell": (t1["prem_sell"] + t2["prem_sell"]) / 2,
        "costs": costs,
        "theta_entry": t1["theta_entry"],
        "delta_entry": t1["delta_entry"],
        "hit_target": favorable_hit,
    }


def strategy_scale_out(row, action):
    """
    Scale out: exit 1/3 at +0.3%, 1/3 at +0.6%, hold 1/3 to EOD.
    """
    spot_open = row["nifty_open"]
    spot_close = row["nifty_close"]
    high = row["nifty_high"]
    low = row["nifty_low"]

    if action == "BUY_CALL":
        t1_target = spot_open * 1.003
        t2_target = spot_open * 1.006
        t1_hit = high >= t1_target
        t2_hit = high >= t2_target
    else:
        t1_target = spot_open * 0.997
        t2_target = spot_open * 0.994
        t1_hit = low <= t1_target
        t2_hit = low <= t2_target

    third_qty = LOT_SIZE / 3

    # Tranche 1: 1/3 at +0.3% or EOD
    s1_exit = t1_target if t1_hit else spot_close
    dte_s1 = DTE_ENTRY - 0.15 if t1_hit else DTE_ENTRY - 0.4
    tr1 = simulate_trade(spot_open, s1_exit, row["vix"], action, DTE_ENTRY, dte_s1)

    # Tranche 2: 1/3 at +0.6% or EOD
    s2_exit = t2_target if t2_hit else spot_close
    dte_s2 = DTE_ENTRY - 0.25 if t2_hit else DTE_ENTRY - 0.4
    tr2 = simulate_trade(spot_open, s2_exit, row["vix"], action, DTE_ENTRY, dte_s2)

    # Tranche 3: 1/3 at EOD
    tr3 = simulate_trade(spot_open, spot_close, row["vix"], action, DTE_ENTRY, DTE_ENTRY - 0.4)

    pnl = ((tr1["prem_sell"] - tr1["prem_buy"]) * third_qty +
           (tr2["prem_sell"] - tr2["prem_buy"]) * third_qty +
           (tr3["prem_sell"] - tr3["prem_buy"]) * third_qty)

    costs = (compute_costs(tr1["prem_buy"], tr1["prem_sell"]) / 3 +
             compute_costs(tr2["prem_buy"], tr2["prem_sell"]) / 3 +
             compute_costs(tr3["prem_buy"], tr3["prem_sell"]) / 3)
    # Extra brokerage for 2 additional exit legs
    costs += BROKERAGE_PER_ORDER * 2

    return {
        "pnl": pnl - costs,
        "prem_buy": tr1["prem_buy"],
        "prem_sell": (tr1["prem_sell"] + tr2["prem_sell"] + tr3["prem_sell"]) / 3,
        "costs": costs,
        "theta_entry": tr1["theta_entry"],
        "delta_entry": tr1["delta_entry"],
        "hit_t1": t1_hit,
        "hit_t2": t2_hit,
    }


# ── Overnight gap analysis ─────────────────────────────────────────────

def analyse_overnight(df):
    """Study gap direction and size for overnight holds."""
    gaps = []
    for i in range(1, len(df)):
        prev_close = df.iloc[i - 1]["nifty_close"]
        curr_open = df.iloc[i]["nifty_open"]
        gap_pct = (curr_open - prev_close) / prev_close * 100
        curr_close = df.iloc[i]["nifty_close"]
        intraday_move = (curr_close - curr_open) / curr_open * 100
        continuation = (gap_pct > 0 and intraday_move > 0) or (gap_pct < 0 and intraday_move < 0)

        gaps.append({
            "gap_pct": gap_pct,
            "intraday_move_pct": intraday_move,
            "continuation": continuation,
            "vix": df.iloc[i]["vix"],
        })

    gap_arr = np.array([g["gap_pct"] for g in gaps])
    cont_arr = np.array([g["continuation"] for g in gaps])

    stats = {
        "avg_gap_pct": round(float(np.mean(np.abs(gap_arr))), 4),
        "max_gap_pct": round(float(np.max(np.abs(gap_arr))), 4),
        "gap_up_pct": round(float(np.mean(gap_arr > 0) * 100), 1),
        "gap_down_pct": round(float(np.mean(gap_arr < 0) * 100), 1),
        "continuation_rate_pct": round(float(np.mean(cont_arr) * 100), 1),
        "reversal_rate_pct": round(float((1 - np.mean(cont_arr)) * 100), 1),
        "avg_gap_up": round(float(np.mean(gap_arr[gap_arr > 0])), 4) if np.any(gap_arr > 0) else 0,
        "avg_gap_down": round(float(np.mean(gap_arr[gap_arr < 0])), 4) if np.any(gap_arr < 0) else 0,
    }

    # Gap analysis by VIX regime
    for regime in ["LOW", "NORMAL", "HIGH"]:
        mask = [vix_regime(g["vix"]) == regime for g in gaps]
        if sum(mask):
            regime_gaps = gap_arr[mask]
            regime_cont = cont_arr[mask]
            stats[f"{regime}_avg_gap"] = round(float(np.mean(np.abs(regime_gaps))), 4)
            stats[f"{regime}_continuation"] = round(float(np.mean(regime_cont) * 100), 1)
            stats[f"{regime}_count"] = int(sum(mask))

    return stats


# ── Theta decay study ──────────────────────────────────────────────────

def study_theta_decay(df):
    """Study how much premium is lost per day at different VIX levels."""
    results = {}
    for regime in ["LOW", "NORMAL", "HIGH"]:
        mask = df.apply(lambda r: vix_regime(r["vix"]) == regime, axis=1)
        subset = df[mask]
        if len(subset) == 0:
            continue

        thetas_ce = []
        thetas_pe = []
        daily_decay_pct_ce = []
        daily_decay_pct_pe = []

        for _, row in subset.iterrows():
            spot = row["nifty_open"]
            strike = atm_strike(spot)
            v = row["vix"]

            ce = price_option(spot, strike, DTE_ENTRY, v, "CE")
            pe = price_option(spot, strike, DTE_ENTRY, v, "PE")

            thetas_ce.append(ce["theta"])
            thetas_pe.append(pe["theta"])

            # Decay as % of premium
            if ce["premium"] > 0:
                daily_decay_pct_ce.append(abs(ce["theta"]) / ce["premium"] * 100)
            if pe["premium"] > 0:
                daily_decay_pct_pe.append(abs(pe["theta"]) / pe["premium"] * 100)

        results[regime] = {
            "avg_theta_CE": round(float(np.mean(thetas_ce)), 2),
            "avg_theta_PE": round(float(np.mean(thetas_pe)), 2),
            "avg_daily_decay_pct_CE": round(float(np.mean(daily_decay_pct_ce)), 2),
            "avg_daily_decay_pct_PE": round(float(np.mean(daily_decay_pct_pe)), 2),
            "count": len(subset),
        }

    return results


# ── Compute stats helper ───────────────────────────────────────────────

def compute_stats(pnls):
    """Compute summary stats for a list of P&L values."""
    if not pnls:
        return {"total": 0, "avg": 0, "win_rate": 0, "sharpe": 0, "count": 0,
                "max_win": 0, "max_loss": 0, "profit_factor": 0}

    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr < 0]
    avg = float(np.mean(arr))
    std = float(np.std(arr)) if len(arr) > 1 else 1.0

    # Annualised Sharpe (approx 250 trading days)
    sharpe = (avg / std) * math.sqrt(250) if std > 0 else 0

    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999

    return {
        "total": round(float(np.sum(arr)), 0),
        "avg": round(avg, 2),
        "win_rate": round(float(np.mean(arr > 0) * 100), 1),
        "sharpe": round(sharpe, 2),
        "count": len(arr),
        "max_win": round(float(np.max(arr)), 0) if len(arr) > 0 else 0,
        "max_loss": round(float(np.min(arr)), 0) if len(arr) > 0 else 0,
        "profit_factor": round(profit_factor, 2),
    }


# ── Main backtest loop ─────────────────────────────────────────────────

def run_backtest(df):
    """Run all holding period strategies across the dataset."""

    strategies = ["intraday", "hold2", "hold_expiry", "partial", "scale_out"]
    actions = ["BUY_CALL", "BUY_PUT"]

    # Store P&L per strategy/action/regime
    results = {s: {a: [] for a in actions} for s in strategies}
    results_by_vix = {s: {a: {r: [] for r in ["LOW", "NORMAL", "HIGH"]} for a in actions} for s in strategies}
    results_by_trend = {s: {a: {t: [] for t in ["ABOVE_SMA50", "BELOW_SMA50", "NEAR_SMA50"]} for a in actions} for s in strategies}

    # Overnight hold vs intraday comparison
    overnight_pnls = {a: [] for a in actions}
    intraday_pnls = {a: [] for a in actions}

    # Trending holds
    trending_long_hold = {"BUY_PUT": {"BELOW_SMA50": [], "ABOVE_SMA50": [], "NEAR_SMA50": []},
                          "BUY_CALL": {"BELOW_SMA50": [], "ABOVE_SMA50": [], "NEAR_SMA50": []}}
    rangebound_intraday = {"BUY_PUT": [], "BUY_CALL": []}

    n = len(df)
    print(f"\nRunning backtest on {n} days ...")

    for i in range(n - 1):  # need next row for multi-day
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        v = row["vix"]
        vr = vix_regime(v)
        tr = trend_regime(row)

        for action in actions:
            # 1. Intraday
            t_intra = strategy_intraday(row, action)
            results["intraday"][action].append(t_intra["pnl"])
            results_by_vix["intraday"][action][vr].append(t_intra["pnl"])
            results_by_trend["intraday"][action][tr].append(t_intra["pnl"])

            # 2. Hold 2 days
            t_h2 = strategy_hold2(row, next_row, action)
            results["hold2"][action].append(t_h2["pnl"])
            results_by_vix["hold2"][action][vr].append(t_h2["pnl"])
            results_by_trend["hold2"][action][tr].append(t_h2["pnl"])

            # 3. Hold to expiry
            t_exp = strategy_hold_expiry(row, next_row, action)
            results["hold_expiry"][action].append(t_exp["pnl"])
            results_by_vix["hold_expiry"][action][vr].append(t_exp["pnl"])
            results_by_trend["hold_expiry"][action][tr].append(t_exp["pnl"])

            # 4. Partial exit
            t_part = strategy_partial(row, action)
            results["partial"][action].append(t_part["pnl"])
            results_by_vix["partial"][action][vr].append(t_part["pnl"])
            results_by_trend["partial"][action][tr].append(t_part["pnl"])

            # 5. Scale out
            t_scale = strategy_scale_out(row, action)
            results["scale_out"][action].append(t_scale["pnl"])
            results_by_vix["scale_out"][action][vr].append(t_scale["pnl"])
            results_by_trend["scale_out"][action][tr].append(t_scale["pnl"])

            # Overnight comparison
            overnight_pnls[action].append(t_h2["pnl"])
            intraday_pnls[action].append(t_intra["pnl"])

            # Trending holds
            trending_long_hold[action][tr].append(t_h2["pnl"])
            if tr == "NEAR_SMA50":
                rangebound_intraday[action].append(t_intra["pnl"])

    return (results, results_by_vix, results_by_trend,
            overnight_pnls, intraday_pnls,
            trending_long_hold, rangebound_intraday)


# ── Print results ──────────────────────────────────────────────────────

def print_results(results, results_by_vix, results_by_trend,
                  overnight_pnls, intraday_pnls,
                  trending_long_hold, rangebound_intraday,
                  gap_stats, theta_stats):
    """Print comprehensive results tables."""

    strategies = ["intraday", "hold2", "hold_expiry", "partial", "scale_out"]
    strat_labels = {
        "intraday": "Intraday (EOD)",
        "hold2": "Hold 2 Days",
        "hold_expiry": "Hold to Expiry",
        "partial": "Partial (+0.5%)",
        "scale_out": "Scale Out (3 tranche)",
    }
    actions = ["BUY_CALL", "BUY_PUT"]

    print("\n" + "=" * 90)
    print("  OPTIMAL HOLDING PERIOD STUDY — NIFTY OPTIONS (6 months)")
    print("=" * 90)

    # ── Overall Results ──
    print(f"\n{'Strategy':<22} {'Action':<10} {'Total P&L':>10} {'Avg P&L':>9} {'WinRate':>8} {'Sharpe':>7} {'PF':>6} {'N':>5}")
    print("-" * 82)

    best_sharpe = {}
    for action in actions:
        best_s = None
        best_v = -999
        for s in strategies:
            stats = compute_stats(results[s][action])
            label = strat_labels[s]
            print(f"{label:<22} {action:<10} {stats['total']:>10,.0f} {stats['avg']:>9.1f} "
                  f"{stats['win_rate']:>7.1f}% {stats['sharpe']:>7.2f} {stats['profit_factor']:>6.2f} {stats['count']:>5}")
            if stats["sharpe"] > best_v:
                best_v = stats["sharpe"]
                best_s = s
        best_sharpe[action] = best_s
        print()

    # ── VIX Regime Breakdown ──
    print("\n" + "=" * 90)
    print("  HOLDING PERIOD BY VIX REGIME")
    print("=" * 90)

    for regime in ["LOW", "NORMAL", "HIGH"]:
        print(f"\n  VIX Regime: {regime}")
        print(f"  {'Strategy':<22} {'Action':<10} {'Total P&L':>10} {'Avg P&L':>9} {'WinRate':>8} {'Sharpe':>7} {'N':>5}")
        print("  " + "-" * 76)
        for action in actions:
            for s in strategies:
                pnls = results_by_vix[s][action][regime]
                if not pnls:
                    continue
                stats = compute_stats(pnls)
                label = strat_labels[s]
                print(f"  {label:<22} {action:<10} {stats['total']:>10,.0f} {stats['avg']:>9.1f} "
                      f"{stats['win_rate']:>7.1f}% {stats['sharpe']:>7.2f} {stats['count']:>5}")
            print()

    # ── Trend Regime Breakdown ──
    print("\n" + "=" * 90)
    print("  HOLDING PERIOD BY TREND REGIME")
    print("=" * 90)

    for tr in ["ABOVE_SMA50", "BELOW_SMA50", "NEAR_SMA50"]:
        print(f"\n  Trend: {tr}")
        print(f"  {'Strategy':<22} {'Action':<10} {'Total P&L':>10} {'Avg P&L':>9} {'WinRate':>8} {'Sharpe':>7} {'N':>5}")
        print("  " + "-" * 76)
        for action in actions:
            for s in strategies:
                pnls = results_by_trend[s][action][tr]
                if not pnls:
                    continue
                stats = compute_stats(pnls)
                label = strat_labels[s]
                print(f"  {label:<22} {action:<10} {stats['total']:>10,.0f} {stats['avg']:>9.1f} "
                      f"{stats['win_rate']:>7.1f}% {stats['sharpe']:>7.2f} {stats['count']:>5}")
            print()

    # ── Overnight Gap Analysis ──
    print("\n" + "=" * 90)
    print("  OVERNIGHT GAP ANALYSIS")
    print("=" * 90)
    print(f"\n  Average gap size:       {gap_stats['avg_gap_pct']:.4f}%")
    print(f"  Max gap size:           {gap_stats['max_gap_pct']:.4f}%")
    print(f"  Gap up frequency:       {gap_stats['gap_up_pct']:.1f}%")
    print(f"  Gap down frequency:     {gap_stats['gap_down_pct']:.1f}%")
    print(f"  Avg gap up:             {gap_stats['avg_gap_up']:.4f}%")
    print(f"  Avg gap down:           {gap_stats['avg_gap_down']:.4f}%")
    print(f"  Continuation rate:      {gap_stats['continuation_rate_pct']:.1f}%")
    print(f"  Reversal rate:          {gap_stats['reversal_rate_pct']:.1f}%")

    for regime in ["LOW", "NORMAL", "HIGH"]:
        key = f"{regime}_avg_gap"
        if key in gap_stats:
            print(f"\n  {regime} VIX: avg gap {gap_stats[key]:.4f}%, "
                  f"continuation {gap_stats[f'{regime}_continuation']:.1f}%, "
                  f"count {gap_stats[f'{regime}_count']}")

    # ── Overnight vs Intraday ──
    print("\n\n  OVERNIGHT HOLD vs INTRADAY (does holding overnight help?)")
    print("  " + "-" * 60)
    for action in actions:
        ov_stats = compute_stats(overnight_pnls[action])
        id_stats = compute_stats(intraday_pnls[action])
        verdict = "OVERNIGHT BETTER" if ov_stats["sharpe"] > id_stats["sharpe"] else "INTRADAY BETTER"
        print(f"  {action}:")
        print(f"    Intraday:  Sharpe={id_stats['sharpe']:+.2f}  Avg={id_stats['avg']:+.1f}  WR={id_stats['win_rate']:.1f}%")
        print(f"    Overnight: Sharpe={ov_stats['sharpe']:+.2f}  Avg={ov_stats['avg']:+.1f}  WR={ov_stats['win_rate']:.1f}%")
        print(f"    Verdict:   {verdict}")
        print()

    # ── Trending market: longer hold for puts? ──
    print("\n  TRENDING MARKET: Does holding longer help puts?")
    print("  " + "-" * 60)
    for tr in ["BELOW_SMA50", "ABOVE_SMA50"]:
        pnls = trending_long_hold["BUY_PUT"][tr]
        if pnls:
            stats = compute_stats(pnls)
            print(f"  BUY_PUT {tr}: 2-day hold  Sharpe={stats['sharpe']:+.2f}  Avg={stats['avg']:+.1f}  N={stats['count']}")
        intra_pnls = results_by_trend["intraday"]["BUY_PUT"][tr]
        if intra_pnls:
            stats = compute_stats(intra_pnls)
            print(f"  BUY_PUT {tr}: intraday    Sharpe={stats['sharpe']:+.2f}  Avg={stats['avg']:+.1f}  N={stats['count']}")
        print()

    # ── Rangebound: intraday better? ──
    print("\n  RANGEBOUND (NEAR_SMA50): Is intraday better?")
    print("  " + "-" * 60)
    for action in actions:
        intra = results_by_trend["intraday"][action]["NEAR_SMA50"]
        hold2 = results_by_trend["hold2"][action]["NEAR_SMA50"]
        if intra:
            s1 = compute_stats(intra)
            print(f"  {action} Intraday:  Sharpe={s1['sharpe']:+.2f}  Avg={s1['avg']:+.1f}  N={s1['count']}")
        if hold2:
            s2 = compute_stats(hold2)
            print(f"  {action} Hold 2-day: Sharpe={s2['sharpe']:+.2f}  Avg={s2['avg']:+.1f}  N={s2['count']}")
        print()

    # ── Theta Decay Impact ──
    print("\n" + "=" * 90)
    print("  THETA DECAY BY VIX REGIME (ATM options, DTE=2)")
    print("=" * 90)
    print(f"\n  {'Regime':<10} {'Theta CE':>10} {'Theta PE':>10} {'Decay%/day CE':>15} {'Decay%/day PE':>15} {'Days':>6}")
    print("  " + "-" * 70)
    for regime in ["LOW", "NORMAL", "HIGH"]:
        if regime in theta_stats:
            ts = theta_stats[regime]
            print(f"  {regime:<10} {ts['avg_theta_CE']:>10.2f} {ts['avg_theta_PE']:>10.2f} "
                  f"{ts['avg_daily_decay_pct_CE']:>14.2f}% {ts['avg_daily_decay_pct_PE']:>14.2f}% {ts['count']:>6}")

    return best_sharpe


# ── Build and save rules ───────────────────────────────────────────────

def build_rules(results, results_by_vix, results_by_trend,
                best_sharpe, gap_stats, theta_stats,
                overnight_pnls, intraday_pnls,
                trending_long_hold, rangebound_intraday):
    """Build the holding_rules.json output."""

    strategies = ["intraday", "hold2", "hold_expiry", "partial", "scale_out"]
    actions = ["BUY_CALL", "BUY_PUT"]

    # Best overall holding per action
    best_holding = {}
    for action in actions:
        best_holding[action] = best_sharpe.get(action, "intraday")

    # VIX-based holding rules
    vix_holding_rules = {}
    for regime in ["LOW", "NORMAL", "HIGH"]:
        regime_best = {}
        for action in actions:
            best_s = None
            best_v = -999
            for s in strategies:
                pnls = results_by_vix[s][action][regime]
                if not pnls:
                    continue
                stats = compute_stats(pnls)
                if stats["sharpe"] > best_v:
                    best_v = stats["sharpe"]
                    best_s = s
            regime_best[action] = {
                "strategy": best_s if best_s else "intraday",
                "sharpe": round(best_v, 2) if best_v > -999 else 0,
            }
        vix_holding_rules[regime] = regime_best

    # Scale out rules (extract hit rates and effectiveness)
    scale_stats = {}
    for action in actions:
        pnls = results[action] if action in results else results["scale_out"][action]
        stats = compute_stats(results["scale_out"][action])
        partial_stats = compute_stats(results["partial"][action])
        scale_stats[action] = {
            "scale_out_sharpe": stats["sharpe"],
            "scale_out_avg": stats["avg"],
            "partial_sharpe": partial_stats["sharpe"],
            "partial_avg": partial_stats["avg"],
            "recommended": "scale_out" if stats["sharpe"] > partial_stats["sharpe"] else "partial",
        }

    # Trend-based rules
    trend_holding = {}
    for tr in ["ABOVE_SMA50", "BELOW_SMA50", "NEAR_SMA50"]:
        trend_best = {}
        for action in actions:
            best_s = None
            best_v = -999
            for s in strategies:
                pnls = results_by_trend[s][action][tr]
                if not pnls:
                    continue
                stats = compute_stats(pnls)
                if stats["sharpe"] > best_v:
                    best_v = stats["sharpe"]
                    best_s = s
            trend_best[action] = {
                "strategy": best_s if best_s else "intraday",
                "sharpe": round(best_v, 2) if best_v > -999 else 0,
            }
        trend_holding[tr] = trend_best

    # Overnight verdict
    overnight_verdict = {}
    for action in actions:
        ov = compute_stats(overnight_pnls[action])
        intra = compute_stats(intraday_pnls[action])
        overnight_verdict[action] = {
            "overnight_sharpe": ov["sharpe"],
            "intraday_sharpe": intra["sharpe"],
            "recommendation": "overnight" if ov["sharpe"] > intra["sharpe"] else "intraday",
            "gap_risk_pct": gap_stats["avg_gap_pct"],
        }

    rules = {
        "best_holding": best_holding,
        "vix_holding_rules": vix_holding_rules,
        "scale_out_rules": scale_stats,
        "overnight_gap_stats": {
            "gap_analysis": gap_stats,
            "overnight_verdict": overnight_verdict,
        },
        "trend_holding_rules": trend_holding,
        "theta_decay_by_vix": theta_stats,
    }

    return rules


# ── Entry point ────────────────────────────────────────────────────────

def main():
    # Download data
    df = download_data()

    # Run backtests
    (results, results_by_vix, results_by_trend,
     overnight_pnls, intraday_pnls,
     trending_long_hold, rangebound_intraday) = run_backtest(df)

    # Overnight gap analysis
    gap_stats = analyse_overnight(df)

    # Theta decay study
    theta_stats = study_theta_decay(df)

    # Print results
    best_sharpe = print_results(
        results, results_by_vix, results_by_trend,
        overnight_pnls, intraday_pnls,
        trending_long_hold, rangebound_intraday,
        gap_stats, theta_stats
    )

    # Build and save rules
    rules = build_rules(
        results, results_by_vix, results_by_trend,
        best_sharpe, gap_stats, theta_stats,
        overnight_pnls, intraday_pnls,
        trending_long_hold, rangebound_intraday
    )

    out_path = PROJECT_ROOT / "data" / "holding_rules.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rules, f, indent=2)
    print(f"\nSaved holding rules to {out_path}")

    # Final summary
    print("\n" + "=" * 90)
    print("  RECOMMENDATIONS")
    print("=" * 90)
    for action in ["BUY_CALL", "BUY_PUT"]:
        print(f"\n  {action}: Best overall strategy = {best_sharpe.get(action, 'N/A')}")
        for regime in ["LOW", "NORMAL", "HIGH"]:
            vr = rules["vix_holding_rules"][regime][action]
            print(f"    VIX {regime}: {vr['strategy']} (Sharpe={vr['sharpe']:+.2f})")

    print()


if __name__ == "__main__":
    main()
