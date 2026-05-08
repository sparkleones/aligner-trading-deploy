"""
Last-1-year backtest of deployed Option B config.

Window: 2025-04-07 -> 2026-04-06 (365 days, ~244 trading days)
Config: V15_CONFIG (avoid=[0,2], vix_floor=12, vix_ceil=25)

Reports: full-year + post-Sep (the regime that matters), monthly P&L,
drawdown, losing streaks, R:R distribution. Compares to 21-month context.

Usage:
    python -m backtesting.backtest_last_year
"""
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.v14_unified_backtest import load_period_data, run_backtest
from scoring.config import V17_CONFIG

CAPITAL = 2_00_000
POST_SEP = dt.date(2025, 9, 1)


def to_date(d):
    return dt.date.fromisoformat(d) if isinstance(d, str) else d


def metrics(trades):
    if not trades:
        return {"n": 0, "pnl": 0.0, "wr": 0.0, "pf": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0, "max_loss": 0.0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    tw = sum(wins)
    tl = -sum(losses)
    pf = tw / tl if tl > 0 else float("inf")
    return {
        "n": len(trades), "pnl": sum(pnls),
        "wr": len(wins) / len(trades) * 100, "pf": pf,
        "avg_win": tw / max(1, len(wins)),
        "avg_loss": -tl / max(1, len(losses)),
        "max_win": max(pnls), "max_loss": min(pnls),
    }


def equity_drawdown(trades, capital):
    daily = defaultdict(float)
    for t in trades:
        daily[to_date(t["date"])] += t["pnl"]
    eq = capital
    peak = capital
    max_dd = 0.0
    max_dd_pct = 0.0
    max_dd_date = None
    for d in sorted(daily.keys()):
        eq += daily[d]
        peak = max(peak, eq)
        dd = eq - peak
        dd_pct = (dd / peak * 100) if peak > 0 else 0
        if dd < max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
            max_dd_date = d
    return max_dd, max_dd_pct, max_dd_date, eq


def losing_streaks(trades):
    if not trades:
        return [], []
    sorted_t = sorted(trades, key=lambda t: to_date(t["date"]))
    cur, runs = 0, []
    for t in sorted_t:
        if t["pnl"] <= 0:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    daily = defaultdict(float)
    for t in sorted_t:
        daily[to_date(t["date"])] += t["pnl"]
    day_runs = []
    cur_d = 0
    for d in sorted(daily.keys()):
        if daily[d] < 0:
            cur_d += 1
        else:
            if cur_d > 0:
                day_runs.append(cur_d)
            cur_d = 0
    if cur_d > 0:
        day_runs.append(cur_d)
    return sorted(runs, reverse=True)[:5], sorted(day_runs, reverse=True)[:5]


def monthly(trades):
    bm = defaultdict(list)
    for t in trades:
        bm[to_date(t["date"]).strftime("%Y-%m")].append(t)
    rows = []
    for m in sorted(bm.keys()):
        ts = bm[m]
        wins = [t for t in ts if t["pnl"] > 0]
        tw = sum(t["pnl"] for t in wins)
        tl = -sum(t["pnl"] for t in ts if t["pnl"] <= 0)
        pf = tw / tl if tl > 0 else float("inf")
        rows.append({
            "month": m, "n": len(ts), "pnl": sum(t["pnl"] for t in ts),
            "wr": len(wins) / len(ts) * 100, "pf": pf,
        })
    return rows


def split(trades, cutover):
    pre, post = [], []
    for t in trades:
        d = to_date(t.get("date"))
        (pre if d < cutover else post).append(t)
    return pre, post


def main():
    # Last 1 year
    end = dt.date(2026, 4, 6)
    start = end - dt.timedelta(days=365)
    print(f"Loading period data {start} -> {end} (last 1 year) ...", flush=True)
    preloaded = load_period_data(start_date=start, end_date=end, quiet=True)
    print(f"  {len(preloaded[2])} trading days\n", flush=True)

    # Option B config
    cfg = dict(V17_CONFIG)
    cfg["avoid_days"] = [0, 2]
    cfg["use_v17_regime_gate"] = False
    cfg["use_v17_monwed_gate"] = False
    cfg["vix_floor"] = 12
    cfg["vix_ceil"] = 25

    print("Running V17_PROD_ONLY (Option B) on last 1 year...", flush=True)
    trades, _ = run_backtest(start_date=start, end_date=end,
                             cfg_override=cfg, quiet=True, preloaded=preloaded)
    print(f"  {len(trades)} trades generated\n", flush=True)

    full = metrics(trades)
    pre, post = split(trades, POST_SEP)
    pre_m = metrics(pre)
    post_m = metrics(post)
    max_dd, max_dd_pct, max_dd_date, final_eq = equity_drawdown(trades, CAPITAL)
    trade_runs, day_runs = losing_streaks(trades)
    m_rows = monthly(trades)
    months_neg = sum(1 for m in m_rows if m["pnl"] < 0)
    months_pos = sum(1 for m in m_rows if m["pnl"] > 0)

    print("=" * 90)
    print(f"  LAST 1 YEAR  ({start} -> {end})")
    print(f"  Config: avoid=[0,2], vix_floor=12, vix_ceil=25 (deployed Option B)")
    print("=" * 90)
    print()
    print(f"  Trades:       {full['n']}")
    print(f"  P&L:          Rs {full['pnl']:+,.0f}")
    print(f"  Return:       {(CAPITAL+full['pnl'])/CAPITAL:.2f}x")
    print(f"  Win Rate:     {full['wr']:.1f}%")
    print(f"  Profit Factor:{full['pf']:.2f}")
    print(f"  Avg Win:      Rs {full['avg_win']:+,.0f}")
    print(f"  Avg Loss:     Rs {full['avg_loss']:+,.0f}")
    print(f"  R:R:          {full['avg_win']/abs(full['avg_loss']) if full['avg_loss'] else 0:.2f}")
    print(f"  Best Day:     Rs {full['max_win']:+,.0f}")
    print(f"  Worst Day:    Rs {full['max_loss']:+,.0f}")
    print()
    print(f"  Max Drawdown: Rs {max_dd:+,.0f} ({max_dd_pct:+.1f}%) on {max_dd_date}")
    print(f"  Final Equity: Rs {final_eq:,.0f}")
    print()
    print(f"  Losing streaks (trade): {trade_runs}")
    print(f"  Losing streaks (day):   {day_runs}")
    print()
    print(f"  Months: {months_pos} positive, {months_neg} negative, {len(m_rows)-months_pos-months_neg} flat")
    print()

    print(f"  Monthly P&L:")
    for m in m_rows:
        bar = "+" * min(20, max(0, int(m["pnl"]/100000))) if m["pnl"] > 0 else "-" * min(20, max(0, int(-m["pnl"]/100000)))
        print(f"    {m['month']}  n={m['n']:>3d}  Rs {m['pnl']:>+12,.0f}  WR={m['wr']:>5.1f}%  PF={m['pf']:>5.2f}  {bar}")
    print()

    print("=" * 90)
    print("  PRE-Sep-2025 (Thursday expiry regime, ~5 months)")
    print("=" * 90)
    print(f"  n={pre_m['n']}  PnL=Rs {pre_m['pnl']:+,.0f}  WR={pre_m['wr']:.1f}%  PF={pre_m['pf']:.2f}")
    print()
    print("=" * 90)
    print("  POST-Sep-2025 (Tuesday expiry regime, ~7 months — current)")
    print("=" * 90)
    print(f"  n={post_m['n']}  PnL=Rs {post_m['pnl']:+,.0f}  WR={post_m['wr']:.1f}%  PF={post_m['pf']:.2f}")
    print()

    print("=" * 90)
    print("  CONTEXT: 21-month full window (Jul-2024 -> Apr-2026)")
    print("=" * 90)
    print(f"  21-month deployed:  29.65x / Rs +57,29,880  WR 42.3%  PF 1.94    "
          f"post-Sep: Rs +15,54,471  PF 1.87")
    print()

    # Annualized return (last 1 year)
    yrs = (end - start).days / 365.0
    if full["pnl"] > 0:
        cagr = ((CAPITAL + full["pnl"]) / CAPITAL) ** (1/yrs) - 1
        print(f"  Annualized return (last 1Y window): {cagr*100:.1f}%")
        print(f"  Capital deployed: Rs {CAPITAL:,.0f}")
        print(f"  Profit per Rs 1L of capital: Rs {full['pnl'] / (CAPITAL/100000):,.0f}")


if __name__ == "__main__":
    main()
