"""
Strike Selection Study for NIFTY Options.

Downloads 6 months of NIFTY + India VIX data, then simulates BUY_CALL and BUY_PUT
trades at different strike distances (ITM, ATM, OTM) to find optimal strike selection
rules per VIX regime.

Entry at open, exit at close, DTE=2, lot size=65, capital=200000.
"""

import sys
import os
import json
import math
from collections import defaultdict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import numpy as np

from backtesting.option_pricer import price_option
from config.constants import (
    STT_RATES, SEBI_TURNOVER_FEE, NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY, GST_RATE,
)

# ── Configuration ─────────────────────────────────────────────────────────
LOT_SIZE = 65
CAPITAL = 200_000
DTE = 2.0
RISK_FREE_RATE = 0.07
STRIKE_INTERVAL = 50

# Strike offsets to study (in terms of strike intervals)
# Positive = OTM for calls / ITM for puts
STRIKE_LABELS = {
    -1: "ITM_1",    # 1 strike ITM
     0: "ATM",      # At the money
     1: "OTM_1",    # 1 strike OTM
     2: "OTM_2",    # 2 strikes OTM
     3: "OTM_3",    # 3 strikes OTM
}

# VIX regime boundaries
VIX_LOW_MAX = 12.0
VIX_NORMAL_MAX = 20.0
# HIGH = anything above 20

# ── Transaction cost calculator ───────────────────────────────────────────

def calculate_costs(premium_buy: float, premium_sell: float, lot_size: int) -> float:
    """
    Calculate total transaction costs for a round-trip options trade.
    Buy side: stamp duty
    Sell side: STT on premium
    Both sides: exchange charges, SEBI fee, GST on charges
    """
    buy_turnover = premium_buy * lot_size
    sell_turnover = premium_sell * lot_size

    # STT on sell side only
    stt = sell_turnover * STT_RATES["options_sell"]

    # Exchange transaction charges (both sides)
    exchange_charges = (buy_turnover + sell_turnover) * NSE_TRANSACTION_CHARGE

    # SEBI turnover fee (both sides)
    sebi_fee = (buy_turnover + sell_turnover) * SEBI_TURNOVER_FEE

    # Stamp duty on buy side only
    stamp = buy_turnover * STAMP_DUTY_BUY

    # GST on exchange charges + SEBI fee (no brokerage for zero-brokerage brokers)
    gst = (exchange_charges + sebi_fee) * GST_RATE

    return stt + exchange_charges + sebi_fee + stamp + gst


# ── ATM strike helper ────────────────────────────────────────────────────

def round_to_strike(spot: float, interval: int = 50) -> float:
    """Round spot to nearest strike interval."""
    return round(spot / interval) * interval


# ── VIX regime classifier ────────────────────────────────────────────────

def vix_regime(vix_val: float) -> str:
    if vix_val < VIX_LOW_MAX:
        return "LOW"
    elif vix_val <= VIX_NORMAL_MAX:
        return "NORMAL"
    else:
        return "HIGH"


# ── Download market data ─────────────────────────────────────────────────

def download_data():
    """Download NIFTY and India VIX data for Oct 2025 - Apr 2026."""
    print("Downloading NIFTY (^NSEI) data...")
    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-05", progress=False)
    print(f"  Got {len(nifty)} trading days")

    print("Downloading India VIX (^INDIAVIX) data...")
    vix = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-05", progress=False)
    print(f"  Got {len(vix)} trading days")

    # Handle multi-level columns from yfinance
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Merge on date
    nifty = nifty[["Open", "High", "Low", "Close"]].copy()
    nifty.columns = ["nifty_open", "nifty_high", "nifty_low", "nifty_close"]

    vix_close = vix[["Close"]].copy()
    vix_close.columns = ["vix"]

    data = nifty.join(vix_close, how="inner")
    data.dropna(inplace=True)
    print(f"  Merged dataset: {len(data)} days")

    return data


# ── Simulate a single trade ──────────────────────────────────────────────

def simulate_trade(
    spot_open: float,
    spot_close: float,
    vix_val: float,
    action: str,
    strike_offset: int,
) -> dict:
    """
    Simulate a single-day options trade.

    Parameters
    ----------
    spot_open : entry spot (open price)
    spot_close : exit spot (close price)
    vix_val : India VIX for the day
    action : 'BUY_CALL' or 'BUY_PUT'
    strike_offset : offset in multiples of STRIKE_INTERVAL from ATM
                    For calls: +ve = OTM, -ve = ITM
                    For puts:  +ve = OTM (lower strike), -ve = ITM (higher strike)

    Returns
    -------
    dict with trade details
    """
    atm_strike = round_to_strike(spot_open)

    if action == "BUY_CALL":
        option_type = "CE"
        strike = atm_strike + strike_offset * STRIKE_INTERVAL
    else:  # BUY_PUT
        option_type = "PE"
        strike = atm_strike - strike_offset * STRIKE_INTERVAL

    # Price at entry (open)
    entry_pricing = price_option(
        spot=spot_open,
        strike=strike,
        dte_days=DTE,
        vix=vix_val,
        option_type=option_type,
        r=RISK_FREE_RATE,
        apply_skew=True,
    )
    entry_premium = entry_pricing["premium"]

    # Price at exit (close) — DTE decreases by ~0.3 (intraday, ~6.5 hours of a 365-day year)
    exit_dte = max(DTE - 1.0 / 365.0 * 0.3 * 365, 0.5)  # rough intraday decay
    # Actually, within the same day open->close is about 6.25 hours = 6.25/24 = 0.26 days
    exit_dte = DTE - 0.26

    exit_pricing = price_option(
        spot=spot_close,
        strike=strike,
        dte_days=max(exit_dte, 0.1),
        vix=vix_val,
        option_type=option_type,
        r=RISK_FREE_RATE,
        apply_skew=True,
    )
    exit_premium = exit_pricing["premium"]

    # P&L
    premium_pnl = (exit_premium - entry_premium) * LOT_SIZE
    costs = calculate_costs(entry_premium, exit_premium, LOT_SIZE)
    net_pnl = premium_pnl - costs

    # Capital used (premium paid)
    capital_used = entry_premium * LOT_SIZE

    # Risk/reward metrics
    max_loss = capital_used + costs  # can lose entire premium
    pct_return = (net_pnl / capital_used * 100) if capital_used > 0 else 0

    return {
        "action": action,
        "strike_label": STRIKE_LABELS[strike_offset],
        "strike_offset": strike_offset,
        "strike": strike,
        "atm_strike": atm_strike,
        "option_type": option_type,
        "entry_premium": entry_premium,
        "exit_premium": exit_premium,
        "entry_delta": entry_pricing["delta"],
        "entry_iv": entry_pricing["iv"],
        "entry_theta": entry_pricing["theta"],
        "premium_pnl": round(premium_pnl, 2),
        "costs": round(costs, 2),
        "net_pnl": round(net_pnl, 2),
        "capital_used": round(capital_used, 2),
        "pct_return": round(pct_return, 2),
        "max_loss": round(max_loss, 2),
        "vix": vix_val,
        "vix_regime": vix_regime(vix_val),
        "spot_open": spot_open,
        "spot_close": spot_close,
        "spot_move_pct": round((spot_close - spot_open) / spot_open * 100, 4),
    }


# ── Run full simulation ──────────────────────────────────────────────────

def run_simulation(data: pd.DataFrame) -> list:
    """Run all strike simulations across all days."""
    all_trades = []
    total_days = len(data)

    for i, (date, row) in enumerate(data.iterrows()):
        if (i + 1) % 20 == 0:
            print(f"  Processing day {i+1}/{total_days}...")

        spot_open = float(row["nifty_open"])
        spot_close = float(row["nifty_close"])
        vix_val = float(row["vix"])

        # Skip if VIX data is unreasonable
        if vix_val < 5 or vix_val > 80:
            continue

        for action in ["BUY_CALL", "BUY_PUT"]:
            for offset in STRIKE_LABELS.keys():
                trade = simulate_trade(spot_open, spot_close, vix_val, action, offset)
                trade["date"] = str(date.date()) if hasattr(date, 'date') else str(date)
                all_trades.append(trade)

    return all_trades


# ── Analysis functions ────────────────────────────────────────────────────

def analyze_results(trades: list) -> dict:
    """Comprehensive analysis of strike selection results."""

    df = pd.DataFrame(trades)

    results = {}

    # ── 1. Overall best strike per action ─────────────────────────────
    print("\n" + "=" * 80)
    print("OVERALL STRIKE SELECTION RESULTS")
    print("=" * 80)

    overall_best = {}
    for action in ["BUY_CALL", "BUY_PUT"]:
        action_df = df[df["action"] == action]
        print(f"\n--- {action} ---")
        print(f"{'Strike':<10} {'Trades':<8} {'Win%':<8} {'Avg PnL':<12} {'Total PnL':<14} "
              f"{'Avg Delta':<10} {'Avg IV%':<9} {'Avg Cost':<10} {'Avg Prem':<10} {'Sharpe':<8}")
        print("-" * 105)

        best_pnl = -float("inf")
        best_label = None

        for label in ["ITM_1", "ATM", "OTM_1", "OTM_2", "OTM_3"]:
            subset = action_df[action_df["strike_label"] == label]
            if len(subset) == 0:
                continue

            wins = len(subset[subset["net_pnl"] > 0])
            win_rate = wins / len(subset) * 100
            avg_pnl = subset["net_pnl"].mean()
            total_pnl = subset["net_pnl"].sum()
            avg_delta = subset["entry_delta"].mean()
            avg_iv = subset["entry_iv"].mean()
            avg_cost = subset["costs"].mean()
            avg_prem = subset["entry_premium"].mean()

            # Sharpe-like ratio (mean/std of returns)
            returns = subset["pct_return"]
            sharpe = (returns.mean() / returns.std()) if returns.std() > 0 else 0

            print(f"{label:<10} {len(subset):<8} {win_rate:<8.1f} {avg_pnl:<12.1f} {total_pnl:<14.0f} "
                  f"{abs(avg_delta):<10.3f} {avg_iv:<9.1f} {avg_cost:<10.1f} {avg_prem:<10.1f} {sharpe:<8.3f}")

            if avg_pnl > best_pnl:
                best_pnl = avg_pnl
                best_label = label

        overall_best[action] = best_label
        print(f"\n  >> Best strike for {action}: {best_label} (avg PnL = {best_pnl:.1f})")

    # ── 2. Strike selection by VIX regime ─────────────────────────────
    print("\n\n" + "=" * 80)
    print("STRIKE SELECTION BY VIX REGIME")
    print("=" * 80)

    vix_strike_rules = {}

    for regime in ["LOW", "NORMAL", "HIGH"]:
        regime_df = df[df["vix_regime"] == regime]
        if len(regime_df) == 0:
            print(f"\n  No data for VIX regime: {regime}")
            continue

        vix_range = f"VIX {regime_df['vix'].min():.1f} - {regime_df['vix'].max():.1f}"
        n_days = len(regime_df) // (len(STRIKE_LABELS) * 2)  # approximate unique days
        print(f"\n{'='*60}")
        print(f"VIX REGIME: {regime} ({vix_range}, ~{n_days} days)")
        print(f"{'='*60}")

        vix_strike_rules[regime] = {}

        for action in ["BUY_CALL", "BUY_PUT"]:
            action_df = regime_df[regime_df["action"] == action]
            print(f"\n  --- {action} in {regime} VIX ---")
            print(f"  {'Strike':<10} {'Win%':<8} {'Avg PnL':<12} {'Total PnL':<14} "
                  f"{'Avg Delta':<10} {'Risk/Rew':<10} {'Avg Prem':<10}")
            print(f"  {'-'*84}")

            best_pnl = -float("inf")
            best_label = None
            strike_metrics = {}

            for label in ["ITM_1", "ATM", "OTM_1", "OTM_2", "OTM_3"]:
                subset = action_df[action_df["strike_label"] == label]
                if len(subset) == 0:
                    continue

                wins = len(subset[subset["net_pnl"] > 0])
                win_rate = wins / len(subset) * 100
                avg_pnl = subset["net_pnl"].mean()
                total_pnl = subset["net_pnl"].sum()
                avg_delta = subset["entry_delta"].mean()
                avg_prem = subset["entry_premium"].mean()

                # Risk/reward: average win / average loss
                avg_win = subset[subset["net_pnl"] > 0]["net_pnl"].mean() if wins > 0 else 0
                losses = len(subset[subset["net_pnl"] <= 0])
                avg_loss = abs(subset[subset["net_pnl"] <= 0]["net_pnl"].mean()) if losses > 0 else 1
                rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

                strike_metrics[label] = {
                    "win_rate": round(win_rate, 1),
                    "avg_pnl": round(avg_pnl, 1),
                    "total_pnl": round(total_pnl, 0),
                    "avg_delta": round(abs(avg_delta), 3),
                    "risk_reward": round(rr_ratio, 2),
                    "avg_premium": round(avg_prem, 1),
                    "trades": len(subset),
                }

                print(f"  {label:<10} {win_rate:<8.1f} {avg_pnl:<12.1f} {total_pnl:<14.0f} "
                      f"{abs(avg_delta):<10.3f} {rr_ratio:<10.2f} {avg_prem:<10.1f}")

                if avg_pnl > best_pnl:
                    best_pnl = avg_pnl
                    best_label = label

            vix_strike_rules[regime][action] = {
                "best_strike": best_label,
                "best_avg_pnl": round(best_pnl, 1),
                "all_strikes": strike_metrics,
            }
            print(f"\n    >> Best for {action} in {regime} VIX: {best_label} (avg PnL = {best_pnl:.1f})")

    # ── 3. Risk/Reward analysis ───────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("RISK/REWARD ANALYSIS BY STRIKE")
    print("=" * 80)

    risk_reward = {}

    for action in ["BUY_CALL", "BUY_PUT"]:
        action_df = df[df["action"] == action]
        print(f"\n--- {action} ---")
        print(f"{'Strike':<10} {'Avg Prem':<10} {'Max Win':<12} {'Max Loss':<12} "
              f"{'Win/Loss':<10} {'Leverage':<10} {'Breakeven%':<12}")
        print("-" * 76)

        risk_reward[action] = {}

        for label in ["ITM_1", "ATM", "OTM_1", "OTM_2", "OTM_3"]:
            subset = action_df[action_df["strike_label"] == label]
            if len(subset) == 0:
                continue

            avg_prem = subset["entry_premium"].mean()
            max_win = subset["net_pnl"].max()
            max_loss = subset["net_pnl"].min()
            avg_capital = subset["capital_used"].mean()

            # Best single-day return
            best_return = subset["pct_return"].max()
            worst_return = subset["pct_return"].min()

            # Win/loss ratio
            wins = subset[subset["net_pnl"] > 0]
            losses = subset[subset["net_pnl"] <= 0]
            avg_win = wins["net_pnl"].mean() if len(wins) > 0 else 0
            avg_loss = abs(losses["net_pnl"].mean()) if len(losses) > 0 else 1
            wl_ratio = avg_win / avg_loss if avg_loss > 0 else 0

            # Leverage (notional / premium)
            avg_spot = subset["spot_open"].mean()
            leverage = (avg_spot * LOT_SIZE) / avg_capital if avg_capital > 0 else 0

            # Breakeven move needed (approximate)
            avg_theta_cost = abs(subset["entry_theta"].mean()) * LOT_SIZE
            breakeven_move = (avg_prem * LOT_SIZE + avg_theta_cost) / (abs(subset["entry_delta"].mean()) * LOT_SIZE) if abs(subset["entry_delta"].mean()) > 0 else 0
            breakeven_pct = breakeven_move / avg_spot * 100 if avg_spot > 0 else 0

            risk_reward[action][label] = {
                "avg_premium": round(avg_prem, 1),
                "avg_capital_per_lot": round(avg_capital, 0),
                "max_win": round(max_win, 0),
                "max_loss": round(max_loss, 0),
                "win_loss_ratio": round(wl_ratio, 2),
                "leverage": round(leverage, 1),
                "best_return_pct": round(best_return, 1),
                "worst_return_pct": round(worst_return, 1),
                "breakeven_move_pct": round(breakeven_pct, 3),
            }

            print(f"{label:<10} {avg_prem:<10.1f} {max_win:<12.0f} {max_loss:<12.0f} "
                  f"{wl_ratio:<10.2f} {leverage:<10.1f} {breakeven_pct:<12.3f}")

    # ── 4. Key Insights ──────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    # High VIX OTM analysis
    if "HIGH" in vix_strike_rules:
        high_vix = vix_strike_rules["HIGH"]
        for action in ["BUY_CALL", "BUY_PUT"]:
            if action in high_vix:
                all_strikes = high_vix[action].get("all_strikes", {})
                otm_pnls = {k: v["avg_pnl"] for k, v in all_strikes.items() if "OTM" in k}
                atm_pnl = all_strikes.get("ATM", {}).get("avg_pnl", 0)
                itm_pnl = all_strikes.get("ITM_1", {}).get("avg_pnl", 0)

                if otm_pnls:
                    best_otm = max(otm_pnls, key=otm_pnls.get)
                    otm_better = otm_pnls[best_otm] > atm_pnl
                    print(f"\n  HIGH VIX + {action}:")
                    print(f"    OTM options better than ATM? {'YES' if otm_better else 'NO'}")
                    print(f"    Best OTM: {best_otm} (avg PnL={otm_pnls[best_otm]:.1f}) vs ATM (avg PnL={atm_pnl:.1f})")
                    if otm_pnls:
                        avg_otm_prem = all_strikes.get(best_otm, {}).get("avg_premium", 0)
                        atm_prem = all_strikes.get("ATM", {}).get("avg_premium", 0)
                        if atm_prem > 0:
                            print(f"    Cost advantage: OTM prem={avg_otm_prem:.0f} vs ATM prem={atm_prem:.0f} "
                                  f"({(1 - avg_otm_prem/atm_prem)*100:.0f}% cheaper)")

    # Low VIX ITM/ATM analysis
    if "LOW" in vix_strike_rules:
        low_vix = vix_strike_rules["LOW"]
        for action in ["BUY_CALL", "BUY_PUT"]:
            if action in low_vix:
                all_strikes = low_vix[action].get("all_strikes", {})
                itm_pnl = all_strikes.get("ITM_1", {}).get("avg_pnl", -999)
                atm_pnl = all_strikes.get("ATM", {}).get("avg_pnl", -999)
                best_deep = max(itm_pnl, atm_pnl)
                otm_pnls = [v["avg_pnl"] for k, v in all_strikes.items() if "OTM" in k]
                best_otm = max(otm_pnls) if otm_pnls else -999

                print(f"\n  LOW VIX + {action}:")
                print(f"    ITM/ATM better than OTM? {'YES' if best_deep > best_otm else 'NO'}")
                print(f"    ITM_1 avg PnL={itm_pnl:.1f}, ATM avg PnL={atm_pnl:.1f}, Best OTM avg PnL={best_otm:.1f}")
                itm_delta = all_strikes.get("ITM_1", {}).get("avg_delta", 0)
                atm_delta = all_strikes.get("ATM", {}).get("avg_delta", 0)
                print(f"    Delta advantage: ITM delta={itm_delta:.3f}, ATM delta={atm_delta:.3f}")

    # ── 5. Build final rules ──────────────────────────────────────────
    # Format strike labels for output
    def format_strike_label(label, action):
        if label == "ATM":
            return "ATM"
        elif label == "ITM_1":
            return "ATM-50" if action == "BUY_CALL" else "ATM+50"
        elif label == "OTM_1":
            return "ATM+50" if action == "BUY_CALL" else "ATM-50"
        elif label == "OTM_2":
            return "ATM+100" if action == "BUY_CALL" else "ATM-100"
        elif label == "OTM_3":
            return "ATM+150" if action == "BUY_CALL" else "ATM-150"
        return label

    best_strike = {}
    for action in ["BUY_CALL", "BUY_PUT"]:
        best_strike[action] = format_strike_label(overall_best.get(action, "ATM"), action)

    vix_rules_formatted = {}
    for regime in ["LOW", "NORMAL", "HIGH"]:
        vix_rules_formatted[regime] = {}
        if regime in vix_strike_rules:
            for action in ["BUY_CALL", "BUY_PUT"]:
                if action in vix_strike_rules[regime]:
                    info = vix_strike_rules[regime][action]
                    vix_rules_formatted[regime][action] = {
                        "best_strike": format_strike_label(info["best_strike"], action),
                        "best_avg_pnl": info["best_avg_pnl"],
                        "all_strikes": {
                            format_strike_label(k, action): {
                                "win_rate": v["win_rate"],
                                "avg_pnl": v["avg_pnl"],
                                "risk_reward": v["risk_reward"],
                                "avg_premium": v["avg_premium"],
                                "avg_delta": v["avg_delta"],
                            }
                            for k, v in info.get("all_strikes", {}).items()
                        },
                    }

    # Risk/reward formatted
    rr_formatted = {}
    for action in ["BUY_CALL", "BUY_PUT"]:
        rr_formatted[action] = {}
        if action in risk_reward:
            for label, metrics in risk_reward[action].items():
                fmt_label = format_strike_label(label, action)
                rr_formatted[action][fmt_label] = metrics

    output = {
        "best_strike": best_strike,
        "vix_strike_rules": vix_rules_formatted,
        "risk_reward": rr_formatted,
        "study_params": {
            "lot_size": LOT_SIZE,
            "capital": CAPITAL,
            "dte": DTE,
            "period": "Oct 2025 - Apr 2026",
            "total_trades": len(trades),
            "total_days": len(trades) // (len(STRIKE_LABELS) * 2),
        },
    }

    return output


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("NIFTY OPTIONS — OPTIMAL STRIKE SELECTION STUDY")
    print("Period: Oct 2025 - Apr 2026 | DTE=2 | Lot=65 | Capital=200K")
    print("=" * 80)

    # Download data
    data = download_data()

    if len(data) < 10:
        print("ERROR: Insufficient data downloaded. Check network/tickers.")
        return

    # Show VIX distribution
    print(f"\nVIX Distribution:")
    print(f"  Min: {data['vix'].min():.1f}")
    print(f"  Max: {data['vix'].max():.1f}")
    print(f"  Mean: {data['vix'].mean():.1f}")
    print(f"  Median: {data['vix'].median():.1f}")
    low_days = len(data[data["vix"] < VIX_LOW_MAX])
    normal_days = len(data[(data["vix"] >= VIX_LOW_MAX) & (data["vix"] <= VIX_NORMAL_MAX)])
    high_days = len(data[data["vix"] > VIX_NORMAL_MAX])
    print(f"  LOW (<{VIX_LOW_MAX}): {low_days} days")
    print(f"  NORMAL ({VIX_LOW_MAX}-{VIX_NORMAL_MAX}): {normal_days} days")
    print(f"  HIGH (>{VIX_NORMAL_MAX}): {high_days} days")

    # Run simulation
    print("\nRunning strike selection simulations...")
    trades = run_simulation(data)
    print(f"  Total simulated trades: {len(trades)}")

    # Analyze
    results = analyze_results(trades)

    # Save results
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "strike_rules.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n  Overall best strike for BUY_CALL: {results['best_strike']['BUY_CALL']}")
    print(f"  Overall best strike for BUY_PUT:  {results['best_strike']['BUY_PUT']}")

    print(f"\n  By VIX regime:")
    for regime in ["LOW", "NORMAL", "HIGH"]:
        if regime in results["vix_strike_rules"]:
            for action in ["BUY_CALL", "BUY_PUT"]:
                if action in results["vix_strike_rules"][regime]:
                    best = results["vix_strike_rules"][regime][action]["best_strike"]
                    pnl = results["vix_strike_rules"][regime][action]["best_avg_pnl"]
                    print(f"    {regime:>6} VIX + {action}: {best} (avg PnL = {pnl:.1f})")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
