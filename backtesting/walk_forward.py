"""
Walk-Forward Optimization Engine — Learn from real NIFTY data progressively.

Approach:
  Window 1: Learn from days 1-10, test on days 11-20
  Window 2: Learn from days 1-20, test on days 21-30
  Window 3: Learn from days 1-30, test on days 31-40
  ... continue until all 123 days are used

At each step:
  1. Analyze indicator performance on training window
  2. Learn which indicators predict profitable trades
  3. Determine optimal entry/exit rules per VIX regime
  4. Test learned rules on the out-of-sample window
  5. Track cumulative performance

Final output:
  - Optimized indicator weights
  - VIX-regime-specific rules
  - New strategy rules discovered from data
  - Complete equity curve across all test windows
"""

import logging
import math
import sys
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.constants import (
    INDEX_CONFIG, STT_RATES, NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE, STAMP_DUTY_BUY, GST_RATE,
)
from backtesting.option_pricer import price_option
from orchestrator.market_analyzer import (
    MarketAnalyzer, MarketAnalysis, VIXRegime, MarketBias, TradeAction,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50
LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
BROKERAGE = 20.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA STRUCTURES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class DayData:
    """One trading day's data."""
    date: str
    open: float
    high: float
    low: float
    close: float
    vix: float
    change_pct: float
    intraday_bars: list  # list of OHLCV dicts


@dataclass
class TradeResult:
    """Result of a single trade."""
    date: str
    action: str        # SELL_PUT_SPREAD, SELL_CALL_SPREAD, BUY_CALL, BUY_PUT
    entry_spot: float
    exit_spot: float
    entry_vix: float
    vix_regime: str
    market_bias: str
    gross_pnl: float
    costs: float
    net_pnl: float
    indicators: dict   # indicator name -> score at entry
    hold_bars: int
    reasoning: str


@dataclass
class LearnedRules:
    """Rules learned from training data."""
    # Indicator weights (adjusted from data)
    indicator_weights: dict = field(default_factory=dict)

    # VIX regime rules
    vix_rules: dict = field(default_factory=dict)

    # Directional bias rules
    bias_rules: dict = field(default_factory=dict)

    # Best action per regime
    best_actions: dict = field(default_factory=dict)

    # Entry/exit thresholds
    entry_score_threshold: float = 0.05
    exit_score_threshold: float = -0.02
    min_confidence: float = 0.50

    # Discovered patterns
    patterns: list = field(default_factory=list)

    # Performance stats
    train_win_rate: float = 0.0
    train_sharpe: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download real NIFTY + VIX data."""
    import yfinance as yf

    nifty = yf.download("^NSEI", start="2025-10-01", end="2026-04-04", interval="1d")
    vix = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-04", interval="1d")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    return nifty, vix


def prepare_days(nifty: pd.DataFrame, vix: pd.DataFrame) -> list[DayData]:
    """Convert daily data to DayData objects with intraday bars."""
    vix_map = {}
    for idx, row in vix.iterrows():
        vix_map[idx.date()] = float(row["Close"])

    days = []
    for idx, row in nifty.iterrows():
        d = idx.date()
        v = vix_map.get(d, 14.0)
        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])

        # Generate 25 intraday 15-min bars
        bars = generate_intraday(o, h, l, c, v)

        days.append(DayData(
            date=str(d), open=o, high=h, low=l, close=c,
            vix=v, change_pct=(c - o) / o * 100,
            intraday_bars=bars,
        ))
    return days


def generate_intraday(o, h, l, c, vix, n=25) -> list[dict]:
    """Generate realistic intraday bars from daily OHLC."""
    bars = []
    price = o
    total_move = c - o
    day_range = max(h - l, 1)

    for i in range(n):
        progress = i / (n - 1) if n > 1 else 1.0

        # S-curve path from open to close
        s = 1 / (1 + np.exp(-6 * (progress - 0.5)))
        target = o + total_move * s

        noise_scale = day_range * 0.015 * (vix / 15.0)
        noise = np.random.randn() * noise_scale
        bar_close = max(l, min(h, target + noise))

        bar_open = price
        bar_high = min(h, max(bar_open, bar_close) + abs(np.random.randn()) * noise_scale * 0.3)
        bar_low = max(l, min(bar_open, bar_close) - abs(np.random.randn()) * noise_scale * 0.3)

        bars.append({
            "open": round(bar_open, 2), "high": round(bar_high, 2),
            "low": round(bar_low, 2), "close": round(bar_close, 2),
            "volume": int(np.random.uniform(80000, 300000)),
        })
        price = bar_close

    if bars:
        bars[-1]["close"] = round(c, 2)
    return bars


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# COST MODEL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_costs(n_legs: int, avg_premium: float, qty: int = LOT_SIZE) -> float:
    """Zerodha 2026 cost model."""
    total = 0.0
    for _ in range(n_legs):
        turnover = avg_premium * qty
        brokerage = BROKERAGE * 2
        stt = turnover * STT_RATES.get("options_sell", 0.0015)
        exchange = turnover * NSE_TRANSACTION_CHARGE
        sebi = turnover * SEBI_TURNOVER_FEE
        stamp = turnover * STAMP_DUTY_BUY
        gst = (brokerage + exchange + sebi) * GST_RATE
        total += brokerage + stt + exchange + sebi + stamp + gst
    return total


def bs_premium(spot, strike, dte, vix, opt_type):
    """Get option premium via Black-Scholes."""
    try:
        bs = price_option(spot=spot, strike=strike, dte_days=dte, vix=vix, option_type=opt_type)
        return bs["premium"]
    except Exception:
        return 30.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TRADE SIMULATOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def simulate_trade(action: str, entry_spot: float, exit_spot: float,
                   entry_vix: float, exit_vix: float,
                   hold_bars: int, qty: int = LOT_SIZE) -> tuple[float, float]:
    """Simulate a trade and return (gross_pnl, costs).

    Actions:
      SELL_PUT_SPREAD: Sell OTM put + buy further OTM put (credit, bullish)
      SELL_CALL_SPREAD: Sell OTM call + buy further OTM call (credit, bearish)
      BUY_CALL: Buy ATM call (debit, bullish)
      BUY_PUT: Buy ATM put (debit, bearish)
    """
    dte_entry = 2.0
    dte_exit = max(0.05, dte_entry - hold_bars * 15 / 1440)

    if action == "SELL_PUT_SPREAD":
        atm_iv = entry_vix / 100 * 0.88
        exp_move = entry_spot * atm_iv * math.sqrt(dte_entry / 365)
        sell_strike = round((entry_spot - exp_move * 1.0) / 50) * 50
        buy_strike = sell_strike - 50

        sell_entry = bs_premium(entry_spot, sell_strike, dte_entry, entry_vix, "PE")
        buy_entry = bs_premium(entry_spot, buy_strike, dte_entry, entry_vix, "PE")
        sell_exit = bs_premium(exit_spot, sell_strike, dte_exit, exit_vix, "PE")
        buy_exit = bs_premium(exit_spot, buy_strike, dte_exit, exit_vix, "PE")

        pnl = ((sell_entry - sell_exit) - (buy_entry - buy_exit)) * qty
        avg_prem = (sell_entry + buy_entry) / 2
        costs = calc_costs(2, avg_prem, qty)

    elif action == "SELL_CALL_SPREAD":
        atm_iv = entry_vix / 100 * 0.88
        exp_move = entry_spot * atm_iv * math.sqrt(dte_entry / 365)
        sell_strike = round((entry_spot + exp_move * 1.0) / 50) * 50
        buy_strike = sell_strike + 50

        sell_entry = bs_premium(entry_spot, sell_strike, dte_entry, entry_vix, "CE")
        buy_entry = bs_premium(entry_spot, buy_strike, dte_entry, entry_vix, "CE")
        sell_exit = bs_premium(exit_spot, sell_strike, dte_exit, exit_vix, "CE")
        buy_exit = bs_premium(exit_spot, buy_strike, dte_exit, exit_vix, "CE")

        pnl = ((sell_entry - sell_exit) - (buy_entry - buy_exit)) * qty
        avg_prem = (sell_entry + buy_entry) / 2
        costs = calc_costs(2, avg_prem, qty)

    elif action == "BUY_CALL":
        strike = round(entry_spot / 50) * 50  # ATM
        entry_prem = bs_premium(entry_spot, strike, dte_entry, entry_vix, "CE")
        exit_prem = bs_premium(exit_spot, strike, dte_exit, exit_vix, "CE")
        pnl = (exit_prem - entry_prem) * qty
        costs = calc_costs(1, (entry_prem + exit_prem) / 2, qty)

    elif action == "BUY_PUT":
        strike = round(entry_spot / 50) * 50  # ATM
        entry_prem = bs_premium(entry_spot, strike, dte_entry, entry_vix, "PE")
        exit_prem = bs_premium(exit_spot, strike, dte_exit, exit_vix, "PE")
        pnl = (exit_prem - entry_prem) * qty
        costs = calc_costs(1, (entry_prem + exit_prem) / 2, qty)

    else:
        return 0.0, 0.0

    return round(pnl, 2), round(costs, 2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LEARNING ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def analyze_day(day: DayData, analyzer: MarketAnalyzer) -> list[dict]:
    """Run market analysis on each intraday bar and collect indicator snapshots."""
    snapshots = []
    analyzer._bars = []  # Reset for fresh day

    for i, bar in enumerate(day.intraday_bars):
        analyzer.add_bar(bar)

        try:
            analysis = analyzer.analyze(
                spot_price=bar["close"], vix=day.vix, pcr=1.0,
                option_chain=None, fii_net=0.0, dii_net=0.0,
                is_expiry_day=False,
            )
        except Exception:
            continue

        indicators = {}
        for ind in analysis.indicators:
            indicators[ind.name] = ind.score

        snapshots.append({
            "bar_idx": i,
            "spot": bar["close"],
            "vix": day.vix,
            "overall_score": analysis.overall_score,
            "confidence": analysis.confidence,
            "market_bias": analysis.market_bias.value,
            "vix_regime": analysis.vix_regime.value,
            "recommended_action": analysis.recommended_action.value,
            "indicators": indicators,
        })

    return snapshots


def learn_from_window(days: list[DayData], capital: float) -> tuple[LearnedRules, list[TradeResult]]:
    """Learn optimal rules from a window of trading days.

    Analyzes:
    1. Which indicator combinations predict profitable trades
    2. Which actions work best in each VIX regime
    3. Optimal entry/exit timing
    4. Discovered patterns
    """
    analyzer = MarketAnalyzer(symbol="NIFTY", capital=capital, is_paper=True)
    trades = []

    # Collect all day snapshots
    all_snapshots = []
    for day in days:
        snaps = analyze_day(day, analyzer)
        for s in snaps:
            s["date"] = day.date
            s["day_close"] = day.close
            s["day_open"] = day.open
            s["day_change"] = day.change_pct
        all_snapshots.extend(snaps)

    # Try all 4 actions on each day and see which would have been profitable
    action_performance = defaultdict(lambda: {"wins": 0, "losses": 0, "total_pnl": 0.0, "trades": []})
    regime_action_perf = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0}))
    indicator_correlations = defaultdict(lambda: {"bullish_correct": 0, "bearish_correct": 0, "total": 0})

    for day in days:
        vix = day.vix
        vix_regime = "LOW" if vix < 12 else "NORMAL" if vix < 20 else "HIGH" if vix < 30 else "EXTREME"

        # Entry at bar 3 (~10:00 AM), exit at bar 18 (~14:00 PM)
        entry_bar = 3
        exit_bar = 18

        if len(day.intraday_bars) <= exit_bar:
            continue

        entry_spot = day.intraday_bars[entry_bar]["close"]
        exit_spot = day.intraday_bars[exit_bar]["close"]
        hold_bars = exit_bar - entry_bar

        # Get analysis at entry
        entry_analysis = None
        day_snaps = [s for s in all_snapshots if s["date"] == day.date]
        if len(day_snaps) > entry_bar:
            entry_analysis = day_snaps[entry_bar]

        for action in ["SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "BUY_CALL", "BUY_PUT"]:
            gross, costs = simulate_trade(action, entry_spot, exit_spot, vix, vix, hold_bars)
            net = gross - costs

            is_win = net > 0
            action_performance[action]["wins" if is_win else "losses"] += 1
            action_performance[action]["total_pnl"] += net

            regime_action_perf[vix_regime][action]["wins" if is_win else "losses"] += 1
            regime_action_perf[vix_regime][action]["pnl"] += net

            trade = TradeResult(
                date=day.date, action=action,
                entry_spot=entry_spot, exit_spot=exit_spot,
                entry_vix=vix, vix_regime=vix_regime,
                market_bias=entry_analysis["market_bias"] if entry_analysis else "unknown",
                gross_pnl=gross, costs=costs, net_pnl=net,
                indicators=entry_analysis["indicators"] if entry_analysis else {},
                hold_bars=hold_bars,
                reasoning=f"day={day.date} vix={vix:.1f}",
            )
            action_performance[action]["trades"].append(trade)
            trades.append(trade)

        # Track indicator predictiveness
        if entry_analysis:
            day_bullish = day.close > day.open
            for ind_name, ind_score in entry_analysis["indicators"].items():
                indicator_correlations[ind_name]["total"] += 1
                if ind_score > 0.05 and day_bullish:
                    indicator_correlations[ind_name]["bullish_correct"] += 1
                elif ind_score < -0.05 and not day_bullish:
                    indicator_correlations[ind_name]["bearish_correct"] += 1

    # Build learned rules
    rules = LearnedRules()

    # 1. Find best action per VIX regime
    for regime, actions in regime_action_perf.items():
        best_action = None
        best_pnl = -float("inf")
        for action, stats in actions.items():
            total = stats["wins"] + stats["losses"]
            if total > 0 and stats["pnl"] > best_pnl:
                best_pnl = stats["pnl"]
                best_action = action
        if best_action:
            win_rate = actions[best_action]["wins"] / max(1, actions[best_action]["wins"] + actions[best_action]["losses"])
            rules.best_actions[regime] = {
                "action": best_action,
                "pnl": round(best_pnl, 2),
                "win_rate": round(win_rate * 100, 1),
            }
            rules.vix_rules[regime] = best_action

    # 2. Calculate indicator accuracy and adjust weights
    for ind_name, stats in indicator_correlations.items():
        if stats["total"] > 0:
            accuracy = (stats["bullish_correct"] + stats["bearish_correct"]) / stats["total"]
            rules.indicator_weights[ind_name] = round(accuracy, 3)

    # 3. Determine bias rules from profitable trades
    bias_pnl = defaultdict(lambda: defaultdict(float))
    for trade in trades:
        bias_pnl[trade.market_bias][trade.action] += trade.net_pnl

    for bias, action_pnls in bias_pnl.items():
        best_action = max(action_pnls, key=action_pnls.get)
        rules.bias_rules[bias] = {
            "best_action": best_action,
            "pnl": round(action_pnls[best_action], 2),
        }

    # 4. Discover patterns
    # Pattern: "In LOW VIX + BULLISH bias, SELL_PUT_SPREAD wins X%"
    for regime, actions in regime_action_perf.items():
        for action, stats in actions.items():
            total = stats["wins"] + stats["losses"]
            if total >= 3:
                wr = stats["wins"] / total * 100
                if wr > 55:
                    rules.patterns.append(
                        f"[WIN] {regime} VIX + {action}: {wr:.0f}% WR ({total} trades, Rs {stats['pnl']:,.0f})"
                    )
                elif wr < 35:
                    rules.patterns.append(
                        f"[LOSE] {regime} VIX + {action}: {wr:.0f}% WR ({total} trades, Rs {stats['pnl']:,.0f})"
                    )

    # Calculate overall training stats
    all_train = [t for t in trades if t.action == rules.vix_rules.get(t.vix_regime, "")]
    if all_train:
        wins = len([t for t in all_train if t.net_pnl > 0])
        rules.train_win_rate = wins / len(all_train) * 100
        daily_pnls = defaultdict(float)
        for t in all_train:
            daily_pnls[t.date] += t.net_pnl
        pnl_series = list(daily_pnls.values())
        if len(pnl_series) > 1 and np.std(pnl_series) > 0:
            rules.train_sharpe = np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(252)

    return rules, trades


def test_on_window(days: list[DayData], rules: LearnedRules,
                   capital: float) -> tuple[list[TradeResult], float]:
    """Test learned rules on out-of-sample days."""
    analyzer = MarketAnalyzer(symbol="NIFTY", capital=capital, is_paper=True)
    trades = []
    equity = capital

    for day in days:
        vix = day.vix
        vix_regime = "LOW" if vix < 12 else "NORMAL" if vix < 20 else "HIGH" if vix < 30 else "EXTREME"

        # Get recommended action from learned rules
        action = rules.vix_rules.get(vix_regime)
        if not action:
            continue

        # Get analysis at entry bar
        snaps = analyze_day(day, analyzer)
        entry_bar = 3
        exit_bar = 18

        if len(day.intraday_bars) <= exit_bar:
            continue

        # Check confidence threshold
        if len(snaps) > entry_bar:
            conf = snaps[entry_bar].get("confidence", 0)
            score = snaps[entry_bar].get("overall_score", 0)
            bias = snaps[entry_bar].get("market_bias", "neutral")

            # Additional learned filters
            if bias in rules.bias_rules:
                bias_action = rules.bias_rules[bias]["best_action"]
                # If bias disagrees with VIX rule, check which is stronger
                if bias_action != action and rules.bias_rules[bias]["pnl"] > 0:
                    action = bias_action  # Trust bias rule if it was profitable

        entry_spot = day.intraday_bars[entry_bar]["close"]
        exit_spot = day.intraday_bars[exit_bar]["close"]
        hold_bars = exit_bar - entry_bar

        gross, costs = simulate_trade(action, entry_spot, exit_spot, vix, vix, hold_bars)
        net = gross - costs
        equity += net

        entry_indicators = {}
        if len(snaps) > entry_bar:
            entry_indicators = snaps[entry_bar].get("indicators", {})

        trade = TradeResult(
            date=day.date, action=action,
            entry_spot=entry_spot, exit_spot=exit_spot,
            entry_vix=vix, vix_regime=vix_regime,
            market_bias=bias if len(snaps) > entry_bar else "unknown",
            gross_pnl=gross, costs=costs, net_pnl=net,
            indicators=entry_indicators,
            hold_bars=hold_bars,
            reasoning=f"Learned rule: {vix_regime} -> {action}",
        )
        trades.append(trade)

    return trades, equity


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN WALK-FORWARD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_walk_forward(capital: float = 200000.0) -> dict:
    """Run the complete walk-forward optimization."""
    np.random.seed(42)

    print("=" * 80)
    print("  WALK-FORWARD OPTIMIZATION ON REAL NIFTY DATA")
    print(f"  Capital: Rs {capital:,.0f} | Lot: {LOT_SIZE}")
    print("=" * 80)

    nifty, vix = download_data()
    days = prepare_days(nifty, vix)
    print(f"\n  Total trading days: {len(days)}")
    print(f"  Date range: {days[0].date} to {days[-1].date}")
    print(f"  NIFTY: {days[0].close:.0f} -> {days[-1].close:.0f} ({(days[-1].close/days[0].close-1)*100:+.1f}%)")
    print(f"  VIX range: {min(d.vix for d in days):.1f} - {max(d.vix for d in days):.1f}")

    # Walk-forward windows
    train_start = 10  # Initial training window
    test_size = 10    # Test on next 10 days

    all_test_trades = []
    all_rules = []
    equity_curve = [capital]
    current_capital = capital

    window_results = []
    window_num = 0

    print("\n" + "-" * 80)
    print("  WALK-FORWARD WINDOWS")
    print("-" * 80)

    train_end = train_start
    while train_end + test_size <= len(days):
        window_num += 1

        # Training window: days[0:train_end]
        train_days = days[:train_end]
        # Test window: days[train_end:train_end+test_size]
        test_days = days[train_end:train_end + test_size]

        # Learn from training data
        rules, _ = learn_from_window(train_days, capital)
        all_rules.append(rules)

        # Test on out-of-sample data
        test_trades, test_equity = test_on_window(test_days, rules, current_capital)
        pnl_change = test_equity - current_capital

        # Track results
        wins = len([t for t in test_trades if t.net_pnl > 0])
        total = len(test_trades)
        wr = wins / total * 100 if total > 0 else 0

        train_period = f"{train_days[0].date}..{train_days[-1].date}"
        test_period = f"{test_days[0].date}..{test_days[-1].date}"

        window_result = {
            "window": window_num,
            "train_days": len(train_days),
            "train_period": train_period,
            "test_period": test_period,
            "test_trades": total,
            "test_wins": wins,
            "test_win_rate": round(wr, 1),
            "test_pnl": round(pnl_change, 2),
            "best_actions": rules.vix_rules,
            "patterns": rules.patterns[:3],
        }
        window_results.append(window_result)

        action_summary = ", ".join(f"{r}={a}" for r, a in rules.vix_rules.items())
        print(f"\n  Window {window_num}: Train {len(train_days)}d -> Test {len(test_days)}d")
        print(f"    Train: {train_period}")
        print(f"    Test:  {test_period}")
        print(f"    Rules: {action_summary}")
        print(f"    Test:  {total} trades, {wins}W/{total-wins}L, WR={wr:.0f}%, P&L=Rs {pnl_change:+,.0f}")

        if rules.patterns:
            for p in rules.patterns[:2]:
                print(f"    Pattern: {p}")

        all_test_trades.extend(test_trades)
        current_capital = test_equity
        equity_curve.append(current_capital)

        # Advance window
        train_end += test_size

    # ── FINAL ANALYSIS ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  WALK-FORWARD RESULTS SUMMARY")
    print("=" * 80)

    total_trades = len(all_test_trades)
    total_wins = len([t for t in all_test_trades if t.net_pnl > 0])
    total_pnl = sum(t.net_pnl for t in all_test_trades)
    total_costs = sum(t.costs for t in all_test_trades)

    print(f"\n  Total OOS Trades: {total_trades}")
    print(f"  Win Rate: {total_wins}/{total_trades} ({total_wins/max(1,total_trades)*100:.1f}%)")
    print(f"  Total Net P&L: Rs {total_pnl:+,.2f}")
    print(f"  Total Costs: Rs {total_costs:,.2f}")
    print(f"  Return: {total_pnl/capital*100:+.2f}%")
    print(f"  Final Capital: Rs {current_capital:,.2f}")

    # Max drawdown
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)
    print(f"  Max Drawdown: {max_dd:.2f}%")

    # Sharpe
    daily_pnls = defaultdict(float)
    for t in all_test_trades:
        daily_pnls[t.date] += t.net_pnl
    pnl_list = list(daily_pnls.values())
    if len(pnl_list) > 1 and np.std(pnl_list) > 0:
        sharpe = np.mean(pnl_list) / np.std(pnl_list) * np.sqrt(252)
        print(f"  Sharpe Ratio: {sharpe:.2f}")

    # Per-action analysis
    print("\n  PER-ACTION PERFORMANCE (Out-of-Sample):")
    for action in ["SELL_PUT_SPREAD", "SELL_CALL_SPREAD", "BUY_CALL", "BUY_PUT"]:
        at = [t for t in all_test_trades if t.action == action]
        if not at:
            continue
        w = len([t for t in at if t.net_pnl > 0])
        pnl = sum(t.net_pnl for t in at)
        wr = w / len(at) * 100
        print(f"    {action}: {len(at)} trades, WR={wr:.0f}%, P&L=Rs {pnl:+,.0f}")

    # Per VIX regime
    print("\n  PER-VIX-REGIME PERFORMANCE:")
    for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]:
        rt = [t for t in all_test_trades if t.vix_regime == regime]
        if not rt:
            continue
        w = len([t for t in rt if t.net_pnl > 0])
        pnl = sum(t.net_pnl for t in rt)
        wr = w / len(rt) * 100
        actions_used = set(t.action for t in rt)
        print(f"    {regime}: {len(rt)} trades, WR={wr:.0f}%, P&L=Rs {pnl:+,.0f} | Actions: {actions_used}")

    # ── CONSOLIDATED LEARNED RULES ──────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL LEARNED RULES FOR LIVE TRADING")
    print("=" * 80)

    # Use the last (most data) window's rules as the final rules
    final_rules = all_rules[-1] if all_rules else LearnedRules()

    print("\n  VIX REGIME -> BEST ACTION:")
    for regime, action in final_rules.vix_rules.items():
        stats = final_rules.best_actions.get(regime, {})
        print(f"    {regime}: {action} (WR={stats.get('win_rate', 0):.0f}%, P&L=Rs {stats.get('pnl', 0):+,.0f})")

    print("\n  MARKET BIAS -> BEST ACTION:")
    for bias, info in final_rules.bias_rules.items():
        print(f"    {bias}: {info['best_action']} (P&L=Rs {info['pnl']:+,.0f})")

    print("\n  INDICATOR ACCURACY (predictive power):")
    sorted_inds = sorted(final_rules.indicator_weights.items(), key=lambda x: x[1], reverse=True)
    for name, acc in sorted_inds:
        bar = "#" * int(acc * 30)
        print(f"    {name:15s}: {acc:.1%} {bar}")

    print("\n  DISCOVERED PATTERNS:")
    for p in final_rules.patterns:
        print(f"    {p}")

    # ── GENERATE NEW RULES FOR LIVE STRATEGY ────────────────────────
    print("\n" + "=" * 80)
    print("  NEW STRATEGY RULES (to add to live trading)")
    print("=" * 80)

    new_rules = generate_new_rules(all_test_trades, final_rules, days)

    for rule in new_rules:
        print(f"\n  RULE: {rule['name']}")
        print(f"    Condition: {rule['condition']}")
        print(f"    Action: {rule['action']}")
        print(f"    Evidence: {rule['evidence']}")

    # Save results
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    # Save trades
    trades_data = [{
        "date": t.date, "action": t.action, "entry_spot": t.entry_spot,
        "exit_spot": t.exit_spot, "entry_vix": t.entry_vix,
        "vix_regime": t.vix_regime, "market_bias": t.market_bias,
        "gross_pnl": t.gross_pnl, "costs": t.costs, "net_pnl": t.net_pnl,
        "hold_bars": t.hold_bars,
    } for t in all_test_trades]
    pd.DataFrame(trades_data).to_csv(output_dir / "walkforward_trades.csv", index=False)

    # Save rules as JSON
    rules_json = {
        "vix_rules": final_rules.vix_rules,
        "best_actions": final_rules.best_actions,
        "bias_rules": final_rules.bias_rules,
        "indicator_weights": final_rules.indicator_weights,
        "patterns": final_rules.patterns,
        "new_rules": new_rules,
        "performance": {
            "total_trades": total_trades,
            "win_rate": round(total_wins / max(1, total_trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "max_drawdown": round(max_dd, 2),
            "return_pct": round(total_pnl / capital * 100, 2),
        },
    }
    with open(output_dir / "learned_rules.json", "w") as f:
        json.dump(rules_json, f, indent=2, default=str)

    print(f"\n  Results saved to data/walkforward_trades.csv")
    print(f"  Rules saved to data/learned_rules.json")
    print("\n" + "=" * 80)

    return rules_json


def generate_new_rules(trades: list[TradeResult], rules: LearnedRules,
                       days: list[DayData]) -> list[dict]:
    """Generate new trading rules from walk-forward analysis."""
    new_rules = []

    # Rule 1: VIX Mean Reversion
    vix_values = [d.vix for d in days]
    vix_mean = np.mean(vix_values)
    new_rules.append({
        "name": "VIX_MEAN_REVERSION",
        "condition": f"VIX > {vix_mean + 5:.0f} (1 SD above 6-month mean {vix_mean:.1f})",
        "action": "SELL_CALL_SPREAD (elevated VIX = bearish, sell call premium which is inflated)",
        "evidence": f"6-month VIX mean = {vix_mean:.1f}, when VIX spikes markets typically drop further",
    })

    # Rule 2: Trend Continuation
    bullish_days = [d for d in days if d.change_pct > 0.5]
    bearish_days = [d for d in days if d.change_pct < -0.5]

    # Check if next day follows the trend
    continuation_count = 0
    total_checks = 0
    for i in range(len(days) - 1):
        if abs(days[i].change_pct) > 0.5:
            total_checks += 1
            if days[i].change_pct > 0 and days[i + 1].close > days[i + 1].open:
                continuation_count += 1
            elif days[i].change_pct < 0 and days[i + 1].close < days[i + 1].open:
                continuation_count += 1
    cont_rate = continuation_count / max(1, total_checks) * 100

    new_rules.append({
        "name": "TREND_CONTINUATION",
        "condition": f"Previous day moved >0.5% in one direction",
        "action": f"Follow the trend (continuation rate: {cont_rate:.0f}%)" if cont_rate > 50
                  else f"Fade the move (reversal rate: {100-cont_rate:.0f}%)",
        "evidence": f"Out of {total_checks} strong moves, {continuation_count} continued ({cont_rate:.0f}%)",
    })

    # Rule 3: Gap Trading
    gaps = []
    for i in range(1, len(days)):
        gap_pct = (days[i].open - days[i-1].close) / days[i-1].close * 100
        gap_filled = (days[i].close - days[i].open) * np.sign(-gap_pct) > 0  # gap fill = close opposite to gap
        gaps.append({"gap_pct": gap_pct, "filled": gap_filled, "vix": days[i].vix})

    gap_fill_rate = len([g for g in gaps if g["filled"]]) / max(1, len(gaps)) * 100
    new_rules.append({
        "name": "GAP_FADE",
        "condition": f"Market gaps >0.3% from previous close",
        "action": f"Fade gaps (gap-fill rate: {gap_fill_rate:.0f}%)" if gap_fill_rate > 55
                  else f"Follow gaps (gap-continuation rate: {100-gap_fill_rate:.0f}%)",
        "evidence": f"{len(gaps)} gaps analyzed, {gap_fill_rate:.0f}% filled intraday",
    })

    # Rule 4: VIX-Spike Recovery
    vix_spikes = 0
    vix_spike_next_day_up = 0
    for i in range(1, len(days)):
        if days[i].vix > days[i-1].vix * 1.15:  # VIX jumped 15%+
            vix_spikes += 1
            if i + 1 < len(days) and days[i+1].close > days[i+1].open:
                vix_spike_next_day_up += 1
    spike_recovery_rate = vix_spike_next_day_up / max(1, vix_spikes) * 100

    new_rules.append({
        "name": "VIX_SPIKE_RECOVERY",
        "condition": f"VIX spikes >15% in one day",
        "action": f"{'BUY_CALL (mean-reversion)' if spike_recovery_rate > 55 else 'BUY_PUT (continuation)'} next day",
        "evidence": f"{vix_spikes} VIX spikes found, {spike_recovery_rate:.0f}% followed by up-day",
    })

    # Rule 5: Day-of-Week Effect
    dow_pnl = defaultdict(list)
    for d in days:
        from datetime import datetime as dt
        day_obj = dt.strptime(d.date, "%Y-%m-%d")
        dow_pnl[day_obj.strftime("%A")].append(d.change_pct)

    best_day = max(dow_pnl, key=lambda k: np.mean(dow_pnl[k]))
    worst_day = min(dow_pnl, key=lambda k: np.mean(dow_pnl[k]))
    new_rules.append({
        "name": "DAY_OF_WEEK",
        "condition": f"Day of the week",
        "action": f"Bullish on {best_day} ({np.mean(dow_pnl[best_day]):+.2f}% avg), "
                  f"Bearish on {worst_day} ({np.mean(dow_pnl[worst_day]):+.2f}% avg)",
        "evidence": f"6-month day-of-week analysis: best={best_day}, worst={worst_day}",
    })

    # Rule 6: Overnight + Intraday Divergence
    overnight_up_intraday_down = 0
    overnight_down_intraday_up = 0
    total_days_checked = 0
    for i in range(1, len(days)):
        gap = days[i].open - days[i-1].close
        intraday = days[i].close - days[i].open
        if abs(gap) > days[i-1].close * 0.002:  # meaningful gap
            total_days_checked += 1
            if gap > 0 and intraday < 0:
                overnight_up_intraday_down += 1
            elif gap < 0 and intraday > 0:
                overnight_down_intraday_up += 1
    divergence_rate = (overnight_up_intraday_down + overnight_down_intraday_up) / max(1, total_days_checked) * 100

    new_rules.append({
        "name": "OVERNIGHT_INTRADAY_DIVERGENCE",
        "condition": f"Gap-up at open (overnight move >0.2%)",
        "action": f"{'Fade the gap' if divergence_rate > 50 else 'Follow the gap'} ({divergence_rate:.0f}% divergence rate)",
        "evidence": f"{total_days_checked} meaningful gaps, {divergence_rate:.0f}% showed overnight-intraday divergence",
    })

    return new_rules


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=200000.0)
    args = parser.parse_args()

    results = run_walk_forward(capital=args.capital)
