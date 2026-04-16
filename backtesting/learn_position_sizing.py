"""
Position Sizing Study for NIFTY Options Trading.

Downloads 6 months of NIFTY (^NSEI) and India VIX (^INDIAVIX) data,
then tests 10 different position sizing strategies on a composite
BUY_CALL/BUY_PUT directional strategy.

Evaluates each sizing strategy on:
  - Final equity, Return %, Max drawdown %
  - Sharpe ratio, Largest single-day loss
  - Risk of ruin proxy (times equity < 150,000)

Outputs a comparison table and saves best rules to data/sizing_rules.json.
"""

import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtesting.option_pricer import price_option
from config.constants import (
    INDEX_CONFIG,
    STT_RATES,
    SEBI_TURNOVER_FEE,
    NSE_TRANSACTION_CHARGE,
    STAMP_DUTY_BUY,
    GST_RATE,
    VIX_LOW,
    VIX_NORMAL_HIGH,
    VIX_HIGH,
    VIX_EXTREME,
)

# ── Constants ─────────────────────────────────────────────────────────────
LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
FREEZE_QTY = INDEX_CONFIG["NIFTY"]["freeze_qty"]  # 1800
INITIAL_CAPITAL = 200_000.0
RISK_FREE_RATE = 0.07
DTE_DAYS = 2.0  # Weekly options, ~2 DTE at entry
STRIKE_OFFSET_PTS = 50  # ATM-50 for slight OTM

# Realistic lot cap: margin for 1 NIFTY lot ~ Rs 40-60K depending on VIX.
# With Rs 200K starting capital, max ~4 lots. As equity grows, allow proportional
# increase but cap at freeze limit / lot_size = 27 lots (1800/65).
MAX_LOTS_ABSOLUTE = FREEZE_QTY // LOT_SIZE  # 27 lots — NSE freeze limit
MIN_MARGIN_PER_LOT = 35_000  # Rs 35K margin per lot (conservative)


def cap_qty(lots: int, equity: float) -> int:
    """Cap lots based on margin requirement and freeze limit."""
    # Can't exceed freeze qty
    lots = min(lots, MAX_LOTS_ABSOLUTE)
    # Can't exceed what margin allows
    margin_lots = max(1, int(equity / MIN_MARGIN_PER_LOT))
    lots = min(lots, margin_lots)
    lots = max(1, lots)
    return lots * LOT_SIZE


# ── Download market data ─────────────────────────────────────────────────

def download_data() -> pd.DataFrame:
    """Download NIFTY and India VIX data for Oct 2025 - Apr 2026."""
    import yfinance as yf

    print("Downloading NIFTY (^NSEI) data...")
    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-06", progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)

    print("Downloading India VIX (^INDIAVIX) data...")
    vix = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-06", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    # Build combined dataframe
    df = pd.DataFrame(index=nifty.index)
    df["close"] = nifty["Close"]
    df["open"] = nifty["Open"]
    df["high"] = nifty["High"]
    df["low"] = nifty["Low"]
    df["vix"] = vix["Close"].reindex(df.index, method="ffill")

    df.dropna(subset=["close"], inplace=True)
    df["vix"] = df["vix"].ffill()
    df["vix"] = df["vix"].fillna(14.0)  # fallback

    # Daily change
    df["daily_change"] = df["close"] - df["open"]
    df["daily_change_pct"] = df["daily_change"] / df["open"] * 100

    print(f"  Got {len(df)} trading days: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  NIFTY range: {df['close'].min():.0f} - {df['close'].max():.0f}")
    print(f"  VIX range: {df['vix'].min():.1f} - {df['vix'].max():.1f}")
    return df


# ── Transaction cost calculator ───────────────────────────────────────────

def calc_transaction_costs(premium: float, qty: int, is_buy: bool) -> float:
    """
    Calculate realistic transaction costs for an options trade.
    Returns total cost in Rs.
    """
    turnover = premium * qty

    # STT: only on sell side for options
    stt = turnover * STT_RATES["options_sell"] if not is_buy else 0.0

    # Exchange transaction charges
    exchange = turnover * NSE_TRANSACTION_CHARGE

    # SEBI turnover fee
    sebi = turnover * SEBI_TURNOVER_FEE

    # Stamp duty: only on buy side
    stamp = turnover * STAMP_DUTY_BUY if is_buy else 0.0

    # Brokerage: Rs 20 flat per order (discount broker)
    brokerage = 20.0

    # GST on brokerage + exchange charges
    gst = (brokerage + exchange) * GST_RATE

    return stt + exchange + sebi + stamp + brokerage + gst


# ── Simulate a single trade P&L ──────────────────────────────────────────

def simulate_trade_pnl(
    action: str,
    spot_open: float,
    spot_close: float,
    vix: float,
    qty: int,
) -> float:
    """
    Simulate P&L for a single-day option trade using Black-Scholes.

    action: 'BUY_CALL' or 'BUY_PUT'
    Returns net P&L in Rs after transaction costs.
    """
    option_type = "CE" if action == "BUY_CALL" else "PE"

    # Strike: ATM rounded to nearest 50
    strike = round(spot_open / 50) * 50

    # Entry price (at open)
    entry_pricing = price_option(
        spot=spot_open,
        strike=strike,
        dte_days=DTE_DAYS,
        vix=vix,
        option_type=option_type,
    )
    entry_premium = entry_pricing["premium"]

    # Exit price (at close, 1 day less DTE)
    exit_pricing = price_option(
        spot=spot_close,
        strike=strike,
        dte_days=max(DTE_DAYS - 1.0, 0.5),
        vix=vix,
        option_type=option_type,
    )
    exit_premium = exit_pricing["premium"]

    # Gross P&L
    pnl_per_unit = exit_premium - entry_premium
    gross_pnl = pnl_per_unit * qty

    # Transaction costs (entry buy + exit sell)
    cost_entry = calc_transaction_costs(entry_premium, qty, is_buy=True)
    cost_exit = calc_transaction_costs(exit_premium, qty, is_buy=False)

    net_pnl = gross_pnl - cost_entry - cost_exit
    return net_pnl


# ── Determine correct action based on daily move ─────────────────────────

def determine_action(daily_change: float) -> str:
    """Composite strategy: BUY_CALL in uptrend, BUY_PUT in downtrend."""
    return "BUY_CALL" if daily_change >= 0 else "BUY_PUT"


# ── Position sizing strategies ────────────────────────────────────────────

class PositionSizer:
    """Implements 10 different position sizing strategies."""

    def __init__(self, name: str, initial_capital: float = INITIAL_CAPITAL):
        self.name = name
        self.equity = initial_capital
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.equity_curve = []
        self.daily_pnl = []
        self.win_count = 0
        self.loss_count = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.total_wins_amount = 0.0
        self.total_losses_amount = 0.0
        self.below_150k_count = 0
        self.max_drawdown_pct = 0.0
        self.largest_loss = 0.0

    def _record(self, pnl: float):
        """Record trade result and update statistics."""
        self.equity += pnl
        self.equity_curve.append(self.equity)
        self.daily_pnl.append(pnl)

        if pnl >= 0:
            self.win_count += 1
            self.total_wins_amount += pnl
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.loss_count += 1
            self.total_losses_amount += abs(pnl)
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if pnl < self.largest_loss:
                self.largest_loss = pnl

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        dd = (self.peak_equity - self.equity) / self.peak_equity * 100
        if dd > self.max_drawdown_pct:
            self.max_drawdown_pct = dd

        if self.equity < 150_000:
            self.below_150k_count += 1

    @property
    def win_rate(self) -> float:
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.5

    @property
    def avg_win(self) -> float:
        return self.total_wins_amount / self.win_count if self.win_count > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        return self.total_losses_amount / self.loss_count if self.loss_count > 0 else 1.0

    def get_results(self) -> dict:
        """Calculate performance metrics."""
        returns = pd.Series(self.daily_pnl)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * math.sqrt(252)
        else:
            sharpe = 0.0

        return {
            "strategy": self.name,
            "final_equity": round(self.equity, 0),
            "return_pct": round((self.equity - self.initial_capital) / self.initial_capital * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "largest_loss": round(self.largest_loss, 0),
            "below_150k_count": self.below_150k_count,
            "win_rate": round(self.win_rate * 100, 1),
            "total_trades": self.win_count + self.loss_count,
        }


def get_qty_fixed_lots(lots: int, equity: float) -> int:
    """Fixed lot sizing with margin cap."""
    return cap_qty(lots, equity)


def get_qty_fixed_fractional(equity: float, risk_pct: float, entry_premium: float) -> int:
    """
    Fixed fractional: risk X% of equity per trade.
    Position size = (equity * risk_pct) / (entry_premium * lot_size) lots.
    """
    if entry_premium <= 0:
        return cap_qty(1, equity)
    risk_amount = equity * risk_pct
    max_lots = risk_amount / (entry_premium * LOT_SIZE)
    lots = max(1, int(max_lots))
    return cap_qty(lots, equity)


def get_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly criterion: f* = (p*b - q) / b
    where p=win_rate, q=1-p, b=avg_win/avg_loss
    """
    if avg_loss <= 0 or avg_win <= 0:
        return 0.05  # default conservative
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return max(0.01, min(kelly, 0.50))  # clamp 1%-50%


def get_qty_kelly(equity: float, kelly_f: float, entry_premium: float) -> int:
    """Kelly criterion sizing."""
    if entry_premium <= 0:
        return cap_qty(1, equity)
    risk_amount = equity * kelly_f
    max_lots = risk_amount / (entry_premium * LOT_SIZE)
    lots = max(1, int(max_lots))
    return cap_qty(lots, equity)


def get_qty_vix_adaptive(equity: float, vix: float, base_risk_pct: float = 0.08) -> int:
    """
    VIX-adaptive: more lots when VIX is low (cheaper options),
    fewer lots when VIX is high (expensive options, risky).
    """
    if vix < VIX_LOW:
        multiplier = 2.0   # Low VIX: options cheap, 2x
    elif vix < 15.0:
        multiplier = 1.5   # Normal-low: 1.5x
    elif vix < VIX_NORMAL_HIGH:
        multiplier = 1.0   # Normal: 1x
    elif vix < VIX_HIGH:
        multiplier = 0.7   # High: reduce
    elif vix < VIX_EXTREME:
        multiplier = 0.5   # Very high: half
    else:
        multiplier = 0.3   # Extreme: minimal

    risk_amount = equity * base_risk_pct * multiplier
    lots = max(1, int(risk_amount / (100 * LOT_SIZE)))  # approx premium ~100
    return cap_qty(lots, equity)


def get_qty_streak(
    base_lots: int,
    consecutive_wins: int,
    consecutive_losses: int,
) -> int:
    """
    Streak-based (anti-martingale): increase after wins, decrease after losses.
    After 2+ wins: add 1 lot per win streak (max 4 lots)
    After 2+ losses: reduce to 1 lot
    """
    if consecutive_losses >= 2:
        lots = 1
    elif consecutive_wins >= 4:
        lots = base_lots + 3
    elif consecutive_wins >= 3:
        lots = base_lots + 2
    elif consecutive_wins >= 2:
        lots = base_lots + 1
    else:
        lots = base_lots

    lots = max(1, min(lots, 5))  # cap at 5 lots
    return cap_qty(lots, 999_999_999)  # streak uses fixed lots, only freeze cap


def get_qty_drawdown_based(
    equity: float,
    peak_equity: float,
    base_risk_pct: float = 0.08,
) -> int:
    """
    Drawdown-based: reduce size in drawdown >5%, increase at new highs.
    """
    dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0

    if equity >= peak_equity:
        # At new equity high: increase
        multiplier = 1.3
    elif dd_pct > 10:
        # Deep drawdown: minimal
        multiplier = 0.4
    elif dd_pct > 5:
        # Moderate drawdown: reduce
        multiplier = 0.6
    else:
        multiplier = 1.0

    risk_amount = equity * base_risk_pct * multiplier
    lots = max(1, int(risk_amount / (100 * LOT_SIZE)))  # approx premium ~100
    return cap_qty(lots, equity)


# ── Main simulation ──────────────────────────────────────────────────────

def run_simulation(df: pd.DataFrame) -> list[dict]:
    """Run all 10 position sizing strategies on the data."""

    # Initialize sizers
    strategies = {
        "1_fixed_1lot": PositionSizer("Fixed 1 Lot (65 qty)"),
        "2_fixed_2lots": PositionSizer("Fixed 2 Lots (130 qty)"),
        "3_frac_5pct": PositionSizer("Fixed Frac 5%"),
        "4_frac_8pct": PositionSizer("Fixed Frac 8%"),
        "5_frac_12pct": PositionSizer("Fixed Frac 12%"),
        "6_kelly": PositionSizer("Full Kelly"),
        "7_half_kelly": PositionSizer("Half Kelly"),
        "8_vix_adaptive": PositionSizer("VIX-Adaptive"),
        "9_streak": PositionSizer("Streak-Based"),
        "10_drawdown": PositionSizer("Drawdown-Based"),
    }

    # Pre-calculate entry premiums for qty calculation
    # We need at least a few trades to establish kelly params
    # Use running estimates starting from conservative defaults

    print(f"\nSimulating {len(df)} days across 10 strategies...\n")

    for i, (date, row) in enumerate(df.iterrows()):
        spot_open = row["open"]
        spot_close = row["close"]
        vix = row["vix"]
        daily_change = row["daily_change"]

        # Determine the correct action (perfect foresight composite)
        action = determine_action(daily_change)
        option_type = "CE" if action == "BUY_CALL" else "PE"

        # Calculate entry premium for qty sizing
        strike = round(spot_open / 50) * 50
        entry_info = price_option(spot_open, strike, DTE_DAYS, vix, option_type)
        entry_premium = entry_info["premium"]

        # Strategy 1: Fixed 1 lot
        qty = get_qty_fixed_lots(1, strategies["1_fixed_1lot"].equity)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        strategies["1_fixed_1lot"]._record(pnl)

        # Strategy 2: Fixed 2 lots
        qty = get_qty_fixed_lots(2, strategies["2_fixed_2lots"].equity)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        strategies["2_fixed_2lots"]._record(pnl)

        # Strategy 3: Fixed fractional 5%
        s = strategies["3_frac_5pct"]
        qty = get_qty_fixed_fractional(s.equity, 0.05, entry_premium)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 4: Fixed fractional 8%
        s = strategies["4_frac_8pct"]
        qty = get_qty_fixed_fractional(s.equity, 0.08, entry_premium)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 5: Fixed fractional 12%
        s = strategies["5_frac_12pct"]
        qty = get_qty_fixed_fractional(s.equity, 0.12, entry_premium)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 6: Full Kelly
        s = strategies["6_kelly"]
        kelly_f = get_kelly_fraction(s.win_rate, s.avg_win, s.avg_loss)
        qty = get_qty_kelly(s.equity, kelly_f, entry_premium)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 7: Half Kelly
        s = strategies["7_half_kelly"]
        kelly_f_half = get_kelly_fraction(s.win_rate, s.avg_win, s.avg_loss) / 2.0
        qty = get_qty_kelly(s.equity, kelly_f_half, entry_premium)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 8: VIX-Adaptive
        s = strategies["8_vix_adaptive"]
        qty = get_qty_vix_adaptive(s.equity, vix)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 9: Streak-based
        s = strategies["9_streak"]
        qty = get_qty_streak(2, s.consecutive_wins, s.consecutive_losses)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

        # Strategy 10: Drawdown-based
        s = strategies["10_drawdown"]
        qty = get_qty_drawdown_based(s.equity, s.peak_equity)
        pnl = simulate_trade_pnl(action, spot_open, spot_close, vix, qty)
        s._record(pnl)

    # Collect results
    results = []
    for key in sorted(strategies.keys()):
        results.append(strategies[key].get_results())

    return results, strategies


# ── Print comparison table ────────────────────────────────────────────────

def print_results_table(results: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 120)
    print("POSITION SIZING STRATEGY COMPARISON")
    print("=" * 120)

    header = (
        f"{'#':<3} {'Strategy':<22} {'Final Equity':>14} {'Return %':>10} "
        f"{'Max DD %':>9} {'Sharpe':>8} {'Largest Loss':>14} "
        f"{'<150K':>6} {'Win%':>6} {'Trades':>7}"
    )
    print(header)
    print("-" * 120)

    for i, r in enumerate(results, 1):
        row = (
            f"{i:<3} {r['strategy']:<22} "
            f"{r['final_equity']:>14,.0f} "
            f"{r['return_pct']:>10.1f} "
            f"{r['max_drawdown_pct']:>9.2f} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['largest_loss']:>14,.0f} "
            f"{r['below_150k_count']:>6} "
            f"{r['win_rate']:>6.1f} "
            f"{r['total_trades']:>7}"
        )
        print(row)

    print("-" * 120)


def find_best_strategy(results: list[dict]) -> dict:
    """Find best strategy: max return with max drawdown < 15%."""
    # Filter those with max_dd < 15%
    eligible = [r for r in results if r["max_drawdown_pct"] < 15.0]

    if eligible:
        best = max(eligible, key=lambda x: x["return_pct"])
        print(f"\n>>> BEST STRATEGY (max return, DD < 15%): {best['strategy']}")
        print(f"    Return: {best['return_pct']:.1f}% | Max DD: {best['max_drawdown_pct']:.2f}% | Sharpe: {best['sharpe_ratio']:.2f}")
    else:
        # If none meet DD constraint, pick lowest DD
        print("\n>>> WARNING: No strategy kept drawdown under 15%!")
        best = min(results, key=lambda x: x["max_drawdown_pct"])
        print(f"    Least risky: {best['strategy']} with DD {best['max_drawdown_pct']:.2f}%")

    return best


# ── Save results ──────────────────────────────────────────────────────────

def save_results(results: list[dict], best: dict, strategies: dict):
    """Save sizing rules to data/sizing_rules.json."""
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(exist_ok=True)

    # Get kelly params from the half-kelly sizer
    hk = strategies["7_half_kelly"]
    kelly_f_full = get_kelly_fraction(hk.win_rate, hk.avg_win, hk.avg_loss)

    # VIX adaptive rules
    vix_adaptive_rules = {
        "vix_below_12": {"multiplier": 2.0, "description": "Low VIX: options cheap, double position"},
        "vix_12_to_15": {"multiplier": 1.5, "description": "Normal-low: 1.5x position"},
        "vix_15_to_20": {"multiplier": 1.0, "description": "Normal: standard position"},
        "vix_20_to_25": {"multiplier": 0.7, "description": "High: reduce 30%"},
        "vix_25_to_30": {"multiplier": 0.5, "description": "Very high: half position"},
        "vix_above_30": {"multiplier": 0.3, "description": "Extreme: minimal position"},
    }

    # Build comparison dict
    comparison = {}
    for r in results:
        key = r["strategy"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        comparison[key] = {
            "final_equity": r["final_equity"],
            "return_pct": r["return_pct"],
            "max_drawdown_pct": r["max_drawdown_pct"],
            "sharpe_ratio": r["sharpe_ratio"],
            "largest_loss": r["largest_loss"],
            "below_150k_count": r["below_150k_count"],
        }

    # Determine best params
    best_name = best["strategy"]
    best_params = {"initial_capital": INITIAL_CAPITAL, "lot_size": LOT_SIZE}

    if "Frac" in best_name:
        if "5%" in best_name:
            best_params["risk_pct"] = 0.05
        elif "8%" in best_name:
            best_params["risk_pct"] = 0.08
        elif "12%" in best_name:
            best_params["risk_pct"] = 0.12
        best_params["method"] = "fixed_fractional"
    elif "Kelly" in best_name:
        best_params["method"] = "half_kelly" if "Half" in best_name else "full_kelly"
        best_params["kelly_fraction"] = kelly_f_full if "Full" in best_name else kelly_f_full / 2
    elif "VIX" in best_name:
        best_params["method"] = "vix_adaptive"
        best_params["base_risk_pct"] = 0.08
        best_params["vix_rules"] = vix_adaptive_rules
    elif "Streak" in best_name:
        best_params["method"] = "streak_anti_martingale"
        best_params["base_lots"] = 2
        best_params["max_lots"] = 5
    elif "Drawdown" in best_name:
        best_params["method"] = "drawdown_based"
        best_params["base_risk_pct"] = 0.08
        best_params["dd_threshold_reduce"] = 5.0
        best_params["dd_threshold_minimal"] = 10.0
    elif "1 Lot" in best_name:
        best_params["method"] = "fixed_lots"
        best_params["lots"] = 1
    elif "2 Lot" in best_name:
        best_params["method"] = "fixed_lots"
        best_params["lots"] = 2

    output = {
        "best_strategy": best["strategy"],
        "best_params": best_params,
        "best_metrics": {
            "return_pct": best["return_pct"],
            "max_drawdown_pct": best["max_drawdown_pct"],
            "sharpe_ratio": best["sharpe_ratio"],
        },
        "comparison": comparison,
        "kelly_fraction": round(kelly_f_full, 4),
        "half_kelly_fraction": round(kelly_f_full / 2, 4),
        "vix_adaptive_rules": vix_adaptive_rules,
        "simulation_params": {
            "initial_capital": INITIAL_CAPITAL,
            "lot_size": LOT_SIZE,
            "dte_days": DTE_DAYS,
            "risk_free_rate": RISK_FREE_RATE,
            "period": "2025-10-01 to 2026-04-06",
        },
    }

    out_path = data_dir / "sizing_rules.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved sizing rules to {out_path}")
    return output


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("NIFTY OPTIONS POSITION SIZING STUDY")
    print("=" * 80)
    print(f"Capital: Rs {INITIAL_CAPITAL:,.0f} | Lot size: {LOT_SIZE} | DTE: {DTE_DAYS} days")
    print(f"Period: Oct 2025 - Apr 2026")
    print()

    # 1. Download data
    df = download_data()

    # 2. Run simulation
    results, strategies = run_simulation(df)

    # 3. Print comparison
    print_results_table(results)

    # 4. Find best strategy
    best = find_best_strategy(results)

    # 5. Print Kelly stats
    hk = strategies["7_half_kelly"]
    kelly_f = get_kelly_fraction(hk.win_rate, hk.avg_win, hk.avg_loss)
    print(f"\nKelly Statistics:")
    print(f"  Win rate: {hk.win_rate:.1%}")
    print(f"  Avg win: Rs {hk.avg_win:,.0f}")
    print(f"  Avg loss: Rs {hk.avg_loss:,.0f}")
    print(f"  Full Kelly f*: {kelly_f:.4f} ({kelly_f*100:.1f}%)")
    print(f"  Half Kelly f*/2: {kelly_f/2:.4f} ({kelly_f/2*100:.1f}%)")

    # 6. Print VIX adaptive rules
    print(f"\nVIX-Adaptive Rules:")
    print(f"  VIX < 12:    2.0x lots (cheap options, go bigger)")
    print(f"  VIX 12-15:   1.5x lots")
    print(f"  VIX 15-20:   1.0x lots (standard)")
    print(f"  VIX 20-25:   0.7x lots (reduce)")
    print(f"  VIX 25-30:   0.5x lots (defensive)")
    print(f"  VIX > 30:    0.3x lots (minimal)")

    # 7. Save results
    output = save_results(results, best, strategies)

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)

    return output


if __name__ == "__main__":
    main()
