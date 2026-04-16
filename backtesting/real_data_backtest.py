"""
Real-Data Backtester — Tests all 3 strategy agents on actual NIFTY data.

Downloads 6 months of real NIFTY + India VIX data via yfinance,
converts daily bars to simulated intraday bars (realistic intraday patterns),
runs all 3 agents (iron_condor, bull_put_spread, ddqn_agent),
and produces a comprehensive analysis of what works and when.

Output:
  - Per-strategy P&L, win rate, drawdown
  - Performance by VIX regime (LOW/NORMAL/HIGH/EXTREME)
  - Performance by market phase (rally, crash, range-bound)
  - Best/worst days and conditions
  - Optimal parameters learned from the data
"""

import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.constants import (
    INDEX_CONFIG, STT_RATES, NSE_TRANSACTION_CHARGE,
    SEBI_TURNOVER_FEE, STAMP_DUTY_BUY, GST_RATE,
)
from backtesting.option_pricer import price_option
from orchestrator.market_analyzer import MarketAnalyzer, MarketAnalysis, VIXRegime, MarketBias
from orchestrator.strategy_agents.iron_condor_agent import IronCondorLiveAgent
from orchestrator.strategy_agents.bull_put_spread_agent import BullPutSpreadLiveAgent
from orchestrator.strategy_agents.ddqn_live_agent import DDQNLiveAgent
from orchestrator.trade_signal import TradeSignal

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

STRIKE_INTERVAL = 50
BROKERAGE_PER_ORDER = 20.0


@dataclass
class TradeRecord:
    """Records a completed trade for analysis."""
    strategy: str
    action: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    entry_vix: float
    exit_vix: float
    vix_regime: str
    market_bias: str
    gross_pnl: float
    costs: float
    net_pnl: float
    legs_count: int
    hold_bars: int
    date: str
    reasoning: str


def download_real_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download 6 months of real NIFTY + VIX data from Yahoo Finance."""
    import yfinance as yf

    print("=" * 70)
    print("  DOWNLOADING REAL MARKET DATA (Oct 2025 - Apr 2026)")
    print("=" * 70)

    # Daily NIFTY data for 6 months
    nifty_daily = yf.download("^NSEI", start="2025-10-01", end="2026-04-04", interval="1d")
    if isinstance(nifty_daily.columns, pd.MultiIndex):
        nifty_daily.columns = nifty_daily.columns.get_level_values(0)

    # India VIX daily
    vix_daily = yf.download("^INDIAVIX", start="2025-10-01", end="2026-04-04", interval="1d")
    if isinstance(vix_daily.columns, pd.MultiIndex):
        vix_daily.columns = vix_daily.columns.get_level_values(0)

    print(f"\n  NIFTY: {len(nifty_daily)} trading days")
    print(f"  Range: {nifty_daily.index[0].date()} to {nifty_daily.index[-1].date()}")
    print(f"  Price: {nifty_daily['Close'].iloc[0]:.0f} -> {nifty_daily['Close'].iloc[-1]:.0f}")
    print(f"  High: {nifty_daily['High'].max():.0f}  Low: {nifty_daily['Low'].min():.0f}")
    print(f"\n  VIX: {len(vix_daily)} days")
    print(f"  VIX Range: {vix_daily['Close'].min():.1f} - {vix_daily['Close'].max():.1f}")
    print(f"  VIX Mean: {vix_daily['Close'].mean():.1f}")

    return nifty_daily, vix_daily


def daily_to_intraday(daily_row: pd.Series, vix: float, bars_per_day: int = 25) -> list[dict]:
    """Convert a single daily OHLCV bar into realistic intraday 15-min bars.

    Models actual NSE intraday behavior:
    - Opening gap and initial volatility (first 30 min)
    - Mid-morning trend establishment
    - Lunch lull (low volatility)
    - Afternoon push
    - Closing rush

    Returns list of 25 bars (9:15 to 15:15 in 15-min intervals).
    """
    day_open = float(daily_row["Open"])
    day_high = float(daily_row["High"])
    day_low = float(daily_row["Low"])
    day_close = float(daily_row["Close"])
    day_range = day_high - day_low

    bars = []
    price = day_open

    # Determine if day was bullish or bearish
    is_bullish = day_close > day_open
    total_move = day_close - day_open

    for i in range(bars_per_day):
        progress = i / (bars_per_day - 1) if bars_per_day > 1 else 1.0

        # Target price follows a realistic intraday path
        if i < 2:
            # Opening 30 min: sharp move toward day's direction
            target = day_open + total_move * 0.3 * (i + 1) / 2
        elif i < 5:
            # 9:45-10:15: continuation
            target = day_open + total_move * (0.3 + 0.2 * (i - 2) / 3)
        elif i < 10:
            # 10:15-11:45: main trend
            target = day_open + total_move * (0.5 + 0.15 * (i - 5) / 5)
        elif i < 15:
            # 11:45-13:00: lunch lull (slight pullback)
            target = day_open + total_move * (0.65 - 0.05 * (i - 10) / 5)
        elif i < 20:
            # 13:00-14:15: afternoon push
            target = day_open + total_move * (0.6 + 0.2 * (i - 15) / 5)
        else:
            # 14:15-15:15: closing rush toward day close
            target = day_open + total_move * (0.8 + 0.2 * (i - 20) / (bars_per_day - 21))

        # Add noise proportional to VIX
        noise_scale = day_range * 0.02 * (vix / 15.0)
        noise = np.random.randn() * noise_scale

        bar_close = target + noise
        # Constrain within day's range
        bar_close = max(day_low, min(day_high, bar_close))

        bar_open = price
        bar_high = max(bar_open, bar_close) + abs(np.random.randn()) * noise_scale * 0.5
        bar_low = min(bar_open, bar_close) - abs(np.random.randn()) * noise_scale * 0.5
        bar_high = min(bar_high, day_high)
        bar_low = max(bar_low, day_low)

        bars.append({
            "open": round(bar_open, 2),
            "high": round(bar_high, 2),
            "low": round(bar_low, 2),
            "close": round(bar_close, 2),
            "volume": int(np.random.uniform(50000, 300000)),
        })

        price = bar_close

    # Force last bar to close at actual day close
    if bars:
        bars[-1]["close"] = round(day_close, 2)

    return bars


def calculate_costs(legs_count: int, premium_estimate: float, qty: int) -> float:
    """Calculate Zerodha transaction costs for a trade (entry + exit).

    For OTM credit spreads:
    - Premiums are typically Rs 20-80 for 1-2 SD OTM strikes
    - Turnover = premium * qty (not spot * qty!)
    - Each leg has entry + exit = 2 orders = Rs 40 brokerage
    """
    total = 0.0
    for _ in range(legs_count):
        turnover = premium_estimate * qty
        brokerage = BROKERAGE_PER_ORDER * 2  # entry + exit
        stt = turnover * STT_RATES.get("options_sell", 0.0015)  # sell side
        exchange = turnover * NSE_TRANSACTION_CHARGE
        sebi = turnover * SEBI_TURNOVER_FEE
        stamp = turnover * STAMP_DUTY_BUY
        gst = (brokerage + exchange + sebi) * GST_RATE
        total += brokerage + stt + exchange + sebi + stamp + gst
    return total


def estimate_spread_pnl(
    signal: TradeSignal,
    spot_entry: float,
    spot_exit: float,
    vix_entry: float,
    vix_exit: float,
    bars_held: int,
    lot_size: int = 65,
) -> float:
    """Estimate P&L for a credit spread trade using Black-Scholes pricing.

    Prices each leg at entry and exit using BS model, accounting for:
    - Spot price movement (delta P&L)
    - Time decay (theta P&L)
    - VIX change (vega P&L)
    """
    if not signal.legs:
        return 0.0

    total_pnl = 0.0
    dte_entry = 2.0  # Assume 2 DTE at entry
    dte_exit = max(0.05, dte_entry - bars_held * (15 / (60 * 24)))  # Decay DTE

    for leg in signal.legs:
        opt_type = leg.option_type or ("CE" if "CE" in leg.symbol else "PE")

        # Price at entry
        entry_bs = price_option(
            spot=spot_entry, strike=leg.strike,
            dte_days=dte_entry, vix=vix_entry, option_type=opt_type,
        )
        entry_premium = entry_bs["premium"]

        # Price at exit
        exit_bs = price_option(
            spot=spot_exit, strike=leg.strike,
            dte_days=dte_exit, vix=vix_exit, option_type=opt_type,
        )
        exit_premium = exit_bs["premium"]

        # P&L depends on side
        if leg.side == "SELL":
            # Sold at entry, buy back at exit
            leg_pnl = (entry_premium - exit_premium) * leg.qty
        else:
            # Bought at entry, sell at exit
            leg_pnl = (exit_premium - entry_premium) * leg.qty

        total_pnl += leg_pnl

    return total_pnl


def run_backtest(capital: float = 25000.0) -> dict:
    """Run comprehensive backtest on real 6-month data."""
    np.random.seed(42)  # Reproducible intraday noise

    nifty_daily, vix_daily = download_real_data()
    lot_size = INDEX_CONFIG["NIFTY"]["lot_size"]

    # Merge VIX with NIFTY dates
    vix_map = {}
    for idx, row in vix_daily.iterrows():
        vix_map[idx.date()] = float(row["Close"])

    # Initialize agents
    agents = {
        "iron_condor": IronCondorLiveAgent(capital=capital, lot_size=lot_size),
        "bull_put_spread": BullPutSpreadLiveAgent(capital=capital, lot_size=lot_size),
        "ddqn_agent": DDQNLiveAgent(capital=capital, lot_size=lot_size),
    }

    # Market analyzer
    analyzer = MarketAnalyzer(symbol="NIFTY", capital=capital, is_paper=True)

    # Track results
    all_trades: list[TradeRecord] = []
    strategy_equity = {name: [capital] for name in agents}
    strategy_capital = {name: capital for name in agents}
    open_signals: dict[str, tuple] = {}  # strategy -> (signal, entry_bar, entry_spot, entry_vix)
    daily_results = []

    # Cooldowns: prevent re-entry within N bars after a close (realistic MetaAgent behavior)
    last_trade_bar: dict[str, int] = {name: -100 for name in agents}
    COOLDOWN_BARS = 8  # ~2 hours in 15-min bars (same as 45s cooldown in live at 1s/bar)
    MAX_TRADES_PER_DAY = 2  # max entries per strategy per day
    MIN_HOLD_BARS = 4  # minimum hold ~1 hour in 15-min bars (theta needs time)

    print("\n" + "=" * 70)
    print("  BACKTESTING 3 STRATEGIES ON REAL NIFTY DATA")
    print(f"  Capital: Rs {capital:,.0f} | Lot size: {lot_size}")
    print("=" * 70)

    global_bar_idx = 0

    for day_idx, (day_date, day_row) in enumerate(nifty_daily.iterrows()):
        day_d = day_date.date()
        vix = vix_map.get(day_d, 14.0)

        # Generate intraday bars for this day
        intraday_bars = daily_to_intraday(day_row, vix)

        # Reset agents for new day
        for agent in agents.values():
            agent._bars = []

        # Reset analyzer daily state and daily trade counters
        day_pnl = {name: 0.0 for name in agents}
        day_trade_count = {name: 0 for name in agents}

        for bar_i, bar in enumerate(intraday_bars):
            spot = bar["close"]

            # Feed bar to analyzer
            analyzer.add_bar(bar)

            # Run market analysis
            try:
                analysis = analyzer.analyze(
                    spot_price=spot, vix=vix, pcr=1.0,
                    option_chain=None, fii_net=0.0, dii_net=0.0,
                    is_expiry_day=False,
                )
            except Exception:
                analysis = None

            for name, agent in agents.items():
                # Check for exit on open positions
                if name in open_signals:
                    sig_data = open_signals[name]
                    entry_signal, entry_bar, entry_spot, entry_vix = sig_data
                    bars_held = global_bar_idx - entry_bar

                    # Try to generate a CLOSE signal
                    try:
                        close_signal = agent.generate_signal(
                            bar, bar_i, option_chain=None,
                            market_analysis=analysis,
                        )
                    except Exception:
                        close_signal = None

                    # Exit logic: respect minimum hold time for theta capture
                    should_close = False
                    if bars_held >= MIN_HOLD_BARS:
                        # Only consider closing after minimum hold
                        if close_signal and close_signal.action == "CLOSE":
                            should_close = True
                        elif bars_held > 16:  # ~4 hours: take theta and run
                            should_close = True
                    # Always close near end of day
                    if bar_i >= len(intraday_bars) - 2:
                        should_close = True

                    if should_close:
                        # Calculate P&L using BS pricing
                        gross_pnl = estimate_spread_pnl(
                            entry_signal, entry_spot, spot,
                            entry_vix, vix, bars_held, lot_size,
                        )

                        # Calculate costs using realistic OTM premium
                        # For 1-2 SD OTM options: Rs 15-60 premium range
                        # Use BS pricing for more accurate estimate
                        avg_premium = 0.0
                        for leg in entry_signal.legs:
                            opt_type = leg.option_type or "PE"
                            try:
                                bs = price_option(spot=spot, strike=leg.strike,
                                                  dte_days=max(0.1, 2.0 - bars_held * 15 / 1440),
                                                  vix=vix, option_type=opt_type)
                                avg_premium += bs["premium"]
                            except Exception:
                                avg_premium += 30.0  # fallback Rs 30 OTM premium
                        avg_premium = avg_premium / max(1, len(entry_signal.legs))
                        costs = calculate_costs(
                            len(entry_signal.legs), avg_premium,
                            lot_size,
                        )

                        net_pnl = gross_pnl - costs

                        # Determine VIX regime
                        vix_regime_str = "UNKNOWN"
                        bias_str = "UNKNOWN"
                        if analysis:
                            vix_regime_str = analysis.vix_regime.value
                            bias_str = analysis.market_bias.value

                        trade = TradeRecord(
                            strategy=name,
                            action=entry_signal.action,
                            entry_bar=entry_bar,
                            exit_bar=global_bar_idx,
                            entry_price=entry_spot,
                            exit_price=spot,
                            entry_vix=entry_vix,
                            exit_vix=vix,
                            vix_regime=vix_regime_str,
                            market_bias=bias_str,
                            gross_pnl=round(gross_pnl, 2),
                            costs=round(costs, 2),
                            net_pnl=round(net_pnl, 2),
                            legs_count=len(entry_signal.legs),
                            hold_bars=bars_held,
                            date=str(day_d),
                            reasoning=entry_signal.reasoning[:100],
                        )
                        all_trades.append(trade)
                        day_pnl[name] += net_pnl

                        strategy_capital[name] += net_pnl
                        strategy_equity[name].append(strategy_capital[name])

                        del open_signals[name]
                        agent.set_position(False)
                        last_trade_bar[name] = global_bar_idx
                        day_trade_count[name] += 1
                        continue

                else:
                    # Cooldown check: don't re-enter too soon after closing
                    bars_since_last = global_bar_idx - last_trade_bar[name]
                    if bars_since_last < COOLDOWN_BARS:
                        continue

                    # Daily trade limit
                    if day_trade_count[name] >= MAX_TRADES_PER_DAY:
                        continue

                    # Don't enter in last 4 bars of day (gamma risk)
                    if bar_i >= len(intraday_bars) - 4:
                        continue

                    # Try to generate entry signal
                    try:
                        signal = agent.generate_signal(
                            bar, bar_i, option_chain=None,
                            market_analysis=analysis,
                        )
                    except Exception:
                        signal = None

                    if signal and signal.action != "CLOSE" and signal.legs:
                        open_signals[name] = (signal, global_bar_idx, spot, vix)
                        agent.set_position(True)

            global_bar_idx += 1

        # End of day: force close any remaining positions
        for name in list(open_signals.keys()):
            sig_data = open_signals[name]
            entry_signal, entry_bar, entry_spot, entry_vix = sig_data
            bars_held = global_bar_idx - entry_bar
            spot = float(day_row["Close"])

            gross_pnl = estimate_spread_pnl(
                entry_signal, entry_spot, spot,
                entry_vix, vix, bars_held, lot_size,
            )
            # Realistic premium for cost calculation
            avg_premium = 0.0
            for leg in entry_signal.legs:
                opt_type = leg.option_type or "PE"
                try:
                    bs = price_option(spot=spot, strike=leg.strike,
                                      dte_days=0.1, vix=vix, option_type=opt_type)
                    avg_premium += bs["premium"]
                except Exception:
                    avg_premium += 20.0
            avg_premium = avg_premium / max(1, len(entry_signal.legs))
            costs = calculate_costs(len(entry_signal.legs), avg_premium, lot_size)
            net_pnl = gross_pnl - costs

            trade = TradeRecord(
                strategy=name, action=entry_signal.action,
                entry_bar=entry_bar, exit_bar=global_bar_idx,
                entry_price=entry_spot, exit_price=spot,
                entry_vix=entry_vix, exit_vix=vix,
                vix_regime="EOD_CLOSE", market_bias="EOD_CLOSE",
                gross_pnl=round(gross_pnl, 2), costs=round(costs, 2),
                net_pnl=round(net_pnl, 2), legs_count=len(entry_signal.legs),
                hold_bars=bars_held, date=str(day_d),
                reasoning="EOD forced close",
            )
            all_trades.append(trade)
            day_pnl[name] += net_pnl
            strategy_capital[name] += net_pnl
            strategy_equity[name].append(strategy_capital[name])

            del open_signals[name]
            agents[name].set_position(False)

        daily_results.append({
            "date": str(day_d),
            "nifty_close": float(day_row["Close"]),
            "vix": vix,
            "nifty_change_pct": (float(day_row["Close"]) - float(day_row["Open"])) / float(day_row["Open"]) * 100,
            **{f"{name}_pnl": round(day_pnl[name], 2) for name in agents},
        })

        # Progress
        if (day_idx + 1) % 20 == 0:
            print(f"  Day {day_idx+1}/{len(nifty_daily)} | NIFTY={float(day_row['Close']):.0f} VIX={vix:.1f} | "
                  f"IC={strategy_capital['iron_condor']:.0f} BP={strategy_capital['bull_put_spread']:.0f} "
                  f"DQ={strategy_capital['ddqn_agent']:.0f}")

    return analyze_results(all_trades, daily_results, strategy_equity, strategy_capital, capital)


def analyze_results(
    trades: list[TradeRecord],
    daily_results: list[dict],
    equity_curves: dict,
    final_capital: dict,
    initial_capital: float,
) -> dict:
    """Comprehensive analysis of backtest results."""
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS — REAL NIFTY DATA (6 MONTHS)")
    print("=" * 70)

    # ── Overall Summary ──
    total_trades = len(trades)
    total_net_pnl = sum(t.net_pnl for t in trades)
    total_costs = sum(t.costs for t in trades)

    print(f"\n  Total Trades: {total_trades}")
    print(f"  Total Net P&L: Rs {total_net_pnl:,.2f}")
    print(f"  Total Costs: Rs {total_costs:,.2f}")

    # ── Per-Strategy Analysis ──
    print("\n" + "-" * 70)
    print("  PER-STRATEGY PERFORMANCE")
    print("-" * 70)

    strategy_stats = {}
    for name in ["iron_condor", "bull_put_spread", "ddqn_agent"]:
        strat_trades = [t for t in trades if t.strategy == name]
        if not strat_trades:
            print(f"\n  {name.upper()}: No trades executed")
            strategy_stats[name] = {"trades": 0}
            continue

        wins = [t for t in strat_trades if t.net_pnl > 0]
        losses = [t for t in strat_trades if t.net_pnl <= 0]
        win_rate = len(wins) / len(strat_trades) * 100 if strat_trades else 0

        total_pnl = sum(t.net_pnl for t in strat_trades)
        avg_win = np.mean([t.net_pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.net_pnl for t in losses]) if losses else 0
        total_cost = sum(t.costs for t in strat_trades)

        # Max drawdown
        equity = equity_curves[name]
        peak = equity[0]
        max_dd = 0
        for val in equity:
            if val > peak:
                peak = val
            dd = (peak - val) / peak * 100
            if dd > max_dd:
                max_dd = dd

        # Sharpe (daily)
        daily_pnls = [d[f"{name}_pnl"] for d in daily_results]
        if np.std(daily_pnls) > 0:
            sharpe = np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252)
        else:
            sharpe = 0

        ret_pct = (final_capital[name] - initial_capital) / initial_capital * 100

        strategy_stats[name] = {
            "trades": len(strat_trades),
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_drawdown": max_dd,
            "sharpe": sharpe,
            "return_pct": ret_pct,
            "total_costs": total_cost,
        }

        print(f"\n  {name.upper()}")
        print(f"    Trades: {len(strat_trades)} | Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"    Win Rate: {win_rate:.1f}%")
        print(f"    Total P&L: Rs {total_pnl:,.2f} ({ret_pct:+.2f}%)")
        print(f"    Avg Win: Rs {avg_win:,.2f} | Avg Loss: Rs {avg_loss:,.2f}")
        print(f"    Max Drawdown: {max_dd:.2f}%")
        print(f"    Sharpe Ratio: {sharpe:.2f}")
        print(f"    Total Costs: Rs {total_cost:,.2f}")

    # ── Performance by VIX Regime ──
    print("\n" + "-" * 70)
    print("  PERFORMANCE BY VIX REGIME")
    print("-" * 70)

    for regime in ["LOW", "NORMAL", "HIGH", "EXTREME"]:
        regime_trades = [t for t in trades if t.vix_regime == regime]
        if not regime_trades:
            continue
        regime_pnl = sum(t.net_pnl for t in regime_trades)
        regime_wins = len([t for t in regime_trades if t.net_pnl > 0])
        regime_wr = regime_wins / len(regime_trades) * 100 if regime_trades else 0

        print(f"\n  VIX {regime}:")
        print(f"    Trades: {len(regime_trades)} | Win Rate: {regime_wr:.1f}% | P&L: Rs {regime_pnl:,.2f}")

        # Per strategy in this regime
        for name in ["iron_condor", "bull_put_spread", "ddqn_agent"]:
            st = [t for t in regime_trades if t.strategy == name]
            if st:
                pnl = sum(t.net_pnl for t in st)
                wr = len([t for t in st if t.net_pnl > 0]) / len(st) * 100
                print(f"      {name}: {len(st)} trades, WR={wr:.0f}%, P&L=Rs {pnl:,.0f}")

    # ── Performance by Market Bias ──
    print("\n" + "-" * 70)
    print("  PERFORMANCE BY MARKET BIAS")
    print("-" * 70)

    for bias in ["STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH"]:
        bias_trades = [t for t in trades if t.market_bias == bias]
        if not bias_trades:
            continue
        bias_pnl = sum(t.net_pnl for t in bias_trades)
        bias_wr = len([t for t in bias_trades if t.net_pnl > 0]) / len(bias_trades) * 100

        print(f"  {bias}: {len(bias_trades)} trades | WR: {bias_wr:.0f}% | P&L: Rs {bias_pnl:,.0f}")

    # ── Market Phase Analysis ──
    print("\n" + "-" * 70)
    print("  MARKET PHASE ANALYSIS")
    print("-" * 70)

    df_daily = pd.DataFrame(daily_results)

    # Identify market phases
    if len(df_daily) > 20:
        # Oct-Nov 2025: Rally phase (NIFTY ~24000-26000)
        rally_days = df_daily[df_daily["nifty_close"] > 25000]
        crash_days = df_daily[(df_daily["nifty_close"] < 23500) & (df_daily["vix"] > 18)]
        range_days = df_daily[(df_daily["nifty_close"] >= 23500) & (df_daily["nifty_close"] <= 25000)]

        for phase, phase_df, label in [
            ("RALLY", rally_days, "NIFTY > 25000"),
            ("CRASH", crash_days, "NIFTY < 23500 + VIX > 18"),
            ("RANGE", range_days, "23500-25000"),
        ]:
            if phase_df.empty:
                continue
            phase_pnl = {name: phase_df[f"{name}_pnl"].sum() for name in ["iron_condor", "bull_put_spread", "ddqn_agent"]}
            print(f"\n  {phase} ({label}) - {len(phase_df)} days:")
            for name in ["iron_condor", "bull_put_spread", "ddqn_agent"]:
                print(f"    {name}: Rs {phase_pnl[name]:,.0f}")

    # ── VIX Distribution ──
    print("\n" + "-" * 70)
    print("  VIX DISTRIBUTION OVER 6 MONTHS")
    print("-" * 70)
    if "vix" in df_daily.columns:
        vix_data = df_daily["vix"].dropna()
        low_days = len(vix_data[vix_data < 12])
        normal_days = len(vix_data[(vix_data >= 12) & (vix_data < 20)])
        high_days = len(vix_data[(vix_data >= 20) & (vix_data < 30)])
        extreme_days = len(vix_data[vix_data >= 30])
        print(f"  LOW (<12):      {low_days} days ({low_days/len(vix_data)*100:.0f}%)")
        print(f"  NORMAL (12-20): {normal_days} days ({normal_days/len(vix_data)*100:.0f}%)")
        print(f"  HIGH (20-30):   {high_days} days ({high_days/len(vix_data)*100:.0f}%)")
        print(f"  EXTREME (>30):  {extreme_days} days ({extreme_days/len(vix_data)*100:.0f}%)")

    # ── Key Learnings ──
    print("\n" + "=" * 70)
    print("  KEY LEARNINGS & STRATEGY RECOMMENDATIONS")
    print("=" * 70)

    best_strategy = max(strategy_stats, key=lambda k: strategy_stats[k].get("total_pnl", -999999))
    worst_strategy = min(strategy_stats, key=lambda k: strategy_stats[k].get("total_pnl", 999999))

    print(f"\n  BEST STRATEGY: {best_strategy.upper()}")
    if strategy_stats[best_strategy].get("trades", 0) > 0:
        print(f"    Return: {strategy_stats[best_strategy]['return_pct']:+.2f}%")
        print(f"    Win Rate: {strategy_stats[best_strategy]['win_rate']:.1f}%")
        print(f"    Sharpe: {strategy_stats[best_strategy]['sharpe']:.2f}")

    print(f"\n  WORST STRATEGY: {worst_strategy.upper()}")
    if strategy_stats[worst_strategy].get("trades", 0) > 0:
        print(f"    Return: {strategy_stats[worst_strategy]['return_pct']:+.2f}%")

    # Print actionable recommendations
    print("\n  RECOMMENDATIONS FOR LIVE TRADING:")

    if df_daily["vix"].iloc[-5:].mean() > 20:
        print("  [!] Current VIX is HIGH (>20). Recommendations:")
        print("      - REDUCE credit selling aggressiveness")
        print("      - WIDEN strike distances (1.3+ SD OTM)")
        print("      - PREFER bull_put_spread over iron_condor (2 legs vs 4)")
        print("      - USE tighter stop losses (1.5x credit, not 2x)")
        print("      - AVOID iron condors until VIX drops below 18")
    else:
        print("  [*] VIX is in normal range. Full strategy deployment OK.")

    total_combined = sum(strategy_stats[s].get("total_pnl", 0) for s in strategy_stats)
    print(f"\n  COMBINED P&L (all 3 strategies): Rs {total_combined:,.2f}")
    print(f"  COMBINED RETURN: {total_combined/initial_capital*100:+.2f}%")

    # ── Save detailed results ──
    output_dir = project_root / "data"
    output_dir.mkdir(exist_ok=True)

    trades_df = pd.DataFrame([vars(t) for t in trades])
    if not trades_df.empty:
        trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
        print(f"\n  Trade details saved to: data/backtest_trades.csv")

    daily_df = pd.DataFrame(daily_results)
    daily_df.to_csv(output_dir / "backtest_daily.csv", index=False)
    print(f"  Daily results saved to: data/backtest_daily.csv")

    print("\n" + "=" * 70)

    return {
        "strategy_stats": strategy_stats,
        "total_trades": len(trades),
        "daily_results": daily_results,
        "trades": trades,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Backtest on real NIFTY data")
    parser.add_argument("--capital", type=float, default=25000.0, help="Starting capital")
    args = parser.parse_args()

    results = run_backtest(capital=args.capital)
