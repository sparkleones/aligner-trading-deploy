"""
15-Day Paper Trading Test — Full Ensemble Strategy (6 agents).

Runs the complete orchestrator + paper trading broker for 15 consecutive
simulated trading days. Each day generates 375 synthetic 1-min bars with
realistic intraday patterns (opening gap, lunch lull, closing rush).

Capital: Rs 200,000 | Strategy: learned_rules (Full Ensemble)
Tests: Composite scoring, VIX-adaptive sizing/strikes, S/R-based exits,
       timing gates, overnight holding.

Output: Daily P&L, equity curve, trade log, performance metrics.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.paper_trading import PaperTradingBroker
from orchestrator.live_orchestrator import LiveTradingOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-5s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("paper_15day")

# Suppress noisy loggers
logging.getLogger("orchestrator.live_orchestrator").setLevel(logging.WARNING)
logging.getLogger("orchestrator.market_analyzer").setLevel(logging.WARNING)
logging.getLogger("orchestrator.meta_agent").setLevel(logging.WARNING)
logging.getLogger("risk_management.risk_manager").setLevel(logging.WARNING)

CAPITAL = 200_000
NUM_DAYS = 15
STRATEGY = "learned_rules"  # Full ensemble agent


async def run_one_day(day_num: int, starting_capital: float) -> dict:
    """Run one trading day simulation. Returns day results."""
    events = []

    async def event_callback(event_type: str, data: dict):
        """Capture events from orchestrator."""
        if event_type in ("live_order", "live_trade_close"):
            events.append({"type": event_type, **data})

    # Create fresh broker with current capital
    broker = PaperTradingBroker(
        initial_capital=starting_capital,
        brokerage_per_order=20.0,
        latency_ms=5.0,
    )

    # Create orchestrator with ONLY the learned_rules agent
    orchestrator = LiveTradingOrchestrator(
        broker=broker,
        capital=starting_capital,
        strategies=[STRATEGY],
        callback=event_callback,
        symbol="NIFTY",
    )

    logger.info(f"DAY {day_num}: Starting | Capital=Rs {starting_capital:,.0f}")

    try:
        result = await asyncio.wait_for(orchestrator.run(), timeout=600)
    except asyncio.TimeoutError:
        logger.warning(f"DAY {day_num}: Timeout after 600s")
        result = {}
    except Exception as e:
        logger.error(f"DAY {day_num}: Error: {e}")
        result = {}

    # Collect results
    final_capital = broker.capital
    day_pnl = final_capital - starting_capital
    trades = broker.trade_log if hasattr(broker, 'trade_log') else []
    positions = broker.positions if hasattr(broker, 'positions') else {}
    orders = broker.order_history if hasattr(broker, 'order_history') else []

    # Count wins/losses from events
    trade_count = len([e for e in events if e["type"] == "live_order"])
    wins = 0
    losses = 0
    trade_pnls = []

    for e in events:
        if e["type"] == "live_trade_close":
            pnl = e.get("realized_pnl", 0)
            trade_pnls.append(pnl)
            if pnl > 0:
                wins += 1
            else:
                losses += 1

    # Also check broker's trade log
    if hasattr(broker, '_trade_log'):
        for t in broker._trade_log:
            realized = t.get("realized_pnl", 0)
            if realized != 0:
                trade_pnls.append(realized)
                if realized > 0:
                    wins += 1
                else:
                    losses += 1

    day_result = {
        "day": day_num,
        "starting_capital": round(starting_capital, 0),
        "final_capital": round(final_capital, 0),
        "day_pnl": round(day_pnl, 0),
        "day_return_pct": round(day_pnl / starting_capital * 100, 2),
        "orders": len(orders),
        "trades_closed": wins + losses,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / max(1, wins + losses) * 100, 1),
        "trade_pnls": [round(p, 0) for p in trade_pnls],
        "events_captured": len(events),
    }

    status = "WIN" if day_pnl > 0 else "LOSS" if day_pnl < 0 else "FLAT"
    logger.info(
        f"DAY {day_num}: {status} | P&L=Rs {day_pnl:+,.0f} "
        f"({day_pnl/starting_capital*100:+.2f}%) | "
        f"Orders={len(orders)} | Trades={wins+losses} (W{wins}/L{losses}) | "
        f"Capital=Rs {final_capital:,.0f}"
    )

    return day_result


async def run_15_days():
    """Run 15 consecutive trading days."""
    print("=" * 80)
    print("  15-DAY PAPER TRADING TEST — FULL ENSEMBLE (6 AGENTS)")
    print(f"  Capital: Rs {CAPITAL:,} | Strategy: {STRATEGY}")
    print(f"  Period: {NUM_DAYS} simulated trading days")
    print("=" * 80)

    equity = CAPITAL
    equity_curve = [CAPITAL]
    all_days = []
    peak_equity = CAPITAL
    max_dd = 0
    consecutive_wins = 0
    max_consecutive_wins = 0
    consecutive_losses = 0
    max_consecutive_losses = 0

    for day in range(1, NUM_DAYS + 1):
        print(f"\n{'-'*80}")
        print(f"  TRADING DAY {day}/{NUM_DAYS}")
        print(f"{'-'*80}")

        day_result = await run_one_day(day, equity)
        all_days.append(day_result)

        equity = day_result["final_capital"]
        equity_curve.append(equity)

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (peak_equity - equity) / peak_equity * 100
        if dd > max_dd:
            max_dd = dd

        # Track streaks
        if day_result["day_pnl"] > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            if consecutive_wins > max_consecutive_wins:
                max_consecutive_wins = consecutive_wins
        elif day_result["day_pnl"] < 0:
            consecutive_losses += 1
            consecutive_wins = 0
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses
        else:
            consecutive_wins = 0
            consecutive_losses = 0

    # ── FINAL REPORT ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  15-DAY PAPER TRADING RESULTS")
    print("=" * 80)

    total_pnl = equity - CAPITAL
    win_days = len([d for d in all_days if d["day_pnl"] > 0])
    loss_days = len([d for d in all_days if d["day_pnl"] < 0])
    flat_days = len([d for d in all_days if d["day_pnl"] == 0])
    total_trades = sum(d["trades_closed"] for d in all_days)
    total_wins = sum(d["wins"] for d in all_days)
    total_losses = sum(d["losses"] for d in all_days)

    daily_pnls = [d["day_pnl"] for d in all_days]
    import numpy as np
    avg_daily = np.mean(daily_pnls) if daily_pnls else 0
    std_daily = np.std(daily_pnls) if daily_pnls else 1
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    best_day = max(daily_pnls) if daily_pnls else 0
    worst_day = min(daily_pnls) if daily_pnls else 0

    print(f"\n  EQUITY:")
    print(f"    Starting Capital:     Rs {CAPITAL:>12,.0f}")
    print(f"    Final Capital:        Rs {equity:>12,.0f}")
    print(f"    Total P&L:            Rs {total_pnl:>+12,.0f}")
    print(f"    Return:               {total_pnl/CAPITAL*100:>+11.2f}%")
    print(f"    Max Drawdown:         {max_dd:>11.2f}%")
    print(f"    Annualized Sharpe:    {sharpe:>11.2f}")

    print(f"\n  DAILY STATS:")
    print(f"    Win Days:             {win_days}/{NUM_DAYS} ({win_days/NUM_DAYS*100:.0f}%)")
    print(f"    Loss Days:            {loss_days}/{NUM_DAYS}")
    print(f"    Flat Days:            {flat_days}/{NUM_DAYS}")
    print(f"    Avg Daily P&L:        Rs {avg_daily:>+10,.0f}")
    print(f"    Best Day:             Rs {best_day:>+10,.0f}")
    print(f"    Worst Day:            Rs {worst_day:>+10,.0f}")
    print(f"    Max Consecutive Wins: {max_consecutive_wins}")
    print(f"    Max Consecutive Loss: {max_consecutive_losses}")

    print(f"\n  TRADES:")
    print(f"    Total Trades:         {total_trades}")
    print(f"    Win Trades:           {total_wins}")
    print(f"    Loss Trades:          {total_losses}")
    print(f"    Trade Win Rate:       {total_wins/max(1,total_trades)*100:.1f}%")

    print(f"\n  DAILY BREAKDOWN:")
    print(f"    {'Day':>4s} {'P&L':>12s} {'Return':>8s} {'Orders':>7s} {'Trades':>7s} {'WR':>6s} {'Capital':>14s}")
    print(f"    {'-'*4} {'-'*12} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*14}")
    for d in all_days:
        wr = f"{d['win_rate']:.0f}%" if d['trades_closed'] > 0 else "--"
        marker = " +" if d["day_pnl"] > 0 else " -" if d["day_pnl"] < 0 else "  "
        print(f"    {d['day']:>4d} Rs {d['day_pnl']:>+10,.0f} {d['day_return_pct']:>+7.2f}% "
              f"{d['orders']:>7d} {d['trades_closed']:>7d} {wr:>6s} "
              f"Rs {d['final_capital']:>10,.0f}{marker}")

    print(f"\n  EQUITY CURVE:")
    for i, eq in enumerate(equity_curve):
        bar_len = max(0, int((eq - CAPITAL) / max(1, abs(total_pnl)) * 40))
        bar = "+" * bar_len if eq >= CAPITAL else "-" * min(40, abs(bar_len))
        print(f"    Day {i:>2d}: Rs {eq:>12,.0f} {bar}")

    # ── Save results ──────────────────────────────────────────────
    output = {
        "test_date": datetime.now().isoformat(),
        "capital": CAPITAL,
        "strategy": STRATEGY,
        "num_days": NUM_DAYS,
        "final_equity": round(equity, 0),
        "total_pnl": round(total_pnl, 0),
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "max_dd_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "win_days": win_days,
        "loss_days": loss_days,
        "total_trades": total_trades,
        "trade_win_rate": round(total_wins / max(1, total_trades) * 100, 1),
        "daily_results": all_days,
        "equity_curve": [round(e, 0) for e in equity_curve],
    }

    output_path = project_root / "data" / "paper_trading_15day_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # ── VERDICT ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    if total_pnl > 0 and win_days >= loss_days and max_dd < 30:
        print("  VERDICT: PASS — Strategy is profitable in paper trading")
        print(f"  Ready for live trading with Rs {CAPITAL:,} capital")
    elif total_pnl > 0:
        print("  VERDICT: CAUTION — Profitable but with concerns")
        if max_dd >= 30:
            print(f"  Warning: Max drawdown {max_dd:.1f}% is high")
        if win_days < loss_days:
            print(f"  Warning: More loss days ({loss_days}) than win days ({win_days})")
    else:
        print("  VERDICT: FAIL — Strategy lost money in paper trading")
        print("  Do NOT deploy to live trading without fixes")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(run_15_days())
