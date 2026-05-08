"""
Monthly stability + drawdown analysis for the V17_PROD_ONLY trades CSV.

Reads data/historical/v14_unified_5min_trades.csv (most recent backtest output)
and reports:
  - Monthly P&L + WR + PF
  - Equity curve peak / max drawdown
  - Longest losing streak (by trade, by day)
  - Distribution of pnl per entry_type
  - Pre-cutover vs post-cutover split

Usage:
    python -m backtesting.stability_analysis
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import datetime as dt

CAPITAL = 2_00_000
CUTOVER = dt.date(2025, 9, 1)


def pf(series):
    wins = series[series > 0].sum()
    losses = -series[series < 0].sum()
    return wins / losses if losses > 0 else float("inf")


def main():
    csv_path = Path("data/historical/v14_unified_5min_trades.csv")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M")
    df["regime"] = df["date"].apply(lambda d: "post-Sep" if d >= CUTOVER else "pre-Sep")

    print("=" * 80)
    print(f"STABILITY ANALYSIS — {len(df)} trades, {df['date'].min()} to {df['date'].max()}")
    print("=" * 80)

    # ── Monthly ───────────────────────────────────────────────
    print("\n[Monthly P&L / WR / PF]")
    m = df.groupby("month").agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        wins=("pnl", lambda s: (s > 0).sum()),
        pf=("pnl", pf),
    )
    m["wr"] = (m["wins"] / m["n"] * 100).round(1)
    # Cumulative
    m["cum_pnl"] = m["pnl"].cumsum()
    m["cum_x"] = (CAPITAL + m["cum_pnl"]) / CAPITAL
    print(m[["n", "pnl", "wr", "pf", "cum_pnl", "cum_x"]].round(2).to_string())

    # ── Equity curve peak / drawdown ──────────────────────────
    print("\n[Equity curve: peak, drawdown]")
    eq = df.sort_values("date").reset_index(drop=True)
    eq["equity"] = CAPITAL + eq["pnl"].cumsum()
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = eq["equity"] - eq["peak"]
    eq["dd_pct"] = eq["dd"] / eq["peak"] * 100
    max_dd = eq["dd"].min()
    max_dd_pct = eq["dd_pct"].min()
    max_dd_idx = eq["dd"].idxmin()
    peak_before = eq.loc[:max_dd_idx, "peak"].iloc[-1]
    date_of_max_dd = eq.loc[max_dd_idx, "date"]
    print(f"  Peak equity:   Rs {eq['peak'].max():>12,.0f}  ({eq.loc[eq['peak'].idxmax(), 'date']})")
    print(f"  Max drawdown:  Rs {max_dd:>12,.0f}  ({max_dd_pct:+.1f}%)  on {date_of_max_dd}")
    print(f"    (from peak of Rs {peak_before:>12,.0f})")
    print(f"  Final equity:  Rs {eq['equity'].iloc[-1]:>12,.0f}")

    # ── Losing streaks ───────────────────────────────────────
    print("\n[Consecutive losing trades]")
    eq["is_loss"] = eq["pnl"] <= 0
    eq["loss_group"] = (~eq["is_loss"]).cumsum()
    streaks = eq[eq["is_loss"]].groupby("loss_group").size()
    if len(streaks):
        top5 = streaks.nlargest(5)
        print(f"  Longest losing streaks (trades): {top5.tolist()}")

    # Losing DAYS streak
    daily = eq.groupby("date")["pnl"].sum().reset_index()
    daily["is_loss"] = daily["pnl"] < 0
    daily["grp"] = (~daily["is_loss"]).cumsum()
    day_streaks = daily[daily["is_loss"]].groupby("grp").size()
    if len(day_streaks):
        print(f"  Longest losing day streaks:      {day_streaks.nlargest(5).tolist()}")

    # ── Entry type breakdown ─────────────────────────────────
    print("\n[By entry_type]")
    et = df.groupby("entry_type").agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda s: (s > 0).mean() * 100),
        pf=("pnl", pf),
        avg=("pnl", "mean"),
    ).round(2)
    print(et.to_string())

    # ── Exit reason breakdown ────────────────────────────────
    print("\n[By exit_reason]")
    er = df.groupby("exit_reason").agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda s: (s > 0).mean() * 100),
        pf=("pnl", pf),
        avg=("pnl", "mean"),
    ).round(2).sort_values("pnl", ascending=False)
    print(er.to_string())

    # ── Pre / Post cutover ───────────────────────────────────
    print("\n[Pre- vs Post-Sep-2025 cutover]")
    rg = df.groupby("regime").agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda s: (s > 0).mean() * 100),
        pf=("pnl", pf),
        avg_win=("pnl", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        avg_loss=("pnl", lambda s: s[s <= 0].mean() if (s <= 0).any() else 0),
    ).round(2)
    print(rg.to_string())

    # ── Product breakdown (MIS vs NRML/BTST) ─────────────────
    print("\n[By product]")
    pr = df.groupby("product").agg(
        n=("pnl", "size"),
        pnl=("pnl", "sum"),
        wr=("pnl", lambda s: (s > 0).mean() * 100),
        pf=("pnl", pf),
        avg=("pnl", "mean"),
    ).round(2)
    print(pr.to_string())

    # ── Worst individual days ────────────────────────────────
    print("\n[Worst 10 days]")
    worst = daily.nsmallest(10, "pnl")[["date", "pnl"]]
    print(worst.to_string(index=False))

    print("\n[Best 10 days]")
    best = daily.nlargest(10, "pnl")[["date", "pnl"]]
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
