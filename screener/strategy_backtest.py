"""
Pluggable backtest engine that runs ANY strategy (from screener.strategies)
on ANY universe. Reuses the same monthly hold-to-rotate structure that
worked best in the original backtest sweep.

Outputs a comparative DataFrame: strategy x cap-tier -> (CAGR, Sharpe, DD).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .strategies import BaseStrategy
from .universe_extended import get_cap_tier


@dataclass
class StrategyBacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2026-04-30"
    initial_capital: float = 100_000.0
    n_picks: int = 2
    rebalance_freq: str = "BMS"  # business month start (Monthly)
    slippage_bps: float = 10.0
    brokerage_bps: float = 3.0
    stt_sell_bps: float = 10.0
    time_stop_bars: int = 60
    min_score: float = -1e9   # by default no min, each strategy decides via NaN
    max_per_sector: int = 1
    min_bars_required: int = 252


def _rank_at(
    strategy: BaseStrategy,
    history: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    cfg: StrategyBacktestConfig,
) -> list[tuple[str, float]]:
    """Score every stock using `strategy` at date `asof`. Return sorted list."""
    rows = []
    for sym, df in history.items():
        slice_ = df.loc[:asof]
        if len(slice_) < cfg.min_bars_required:
            continue
        score = strategy.score(sym, slice_, asof=asof)
        if score is None or pd.isna(score) or not np.isfinite(score):
            continue
        if score < cfg.min_score:
            continue
        rows.append((sym, float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _pick_top(
    ranked: list[tuple[str, float]],
    n: int,
    max_per_sector: int,
    sector_fn,
) -> list[str]:
    picks = []
    sec_count: dict[str, int] = {}
    for sym, _ in ranked:
        sec = sector_fn(sym)
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append(sym)
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def run_strategy_backtest(
    strategy: BaseStrategy,
    history: dict[str, pd.DataFrame],
    sector_fn,
    cfg: Optional[StrategyBacktestConfig] = None,
) -> dict:
    """Run a strategy-agnostic backtest. Returns equity curve + stats."""
    cfg = cfg or StrategyBacktestConfig()

    # Build calendar
    all_dates = pd.Index([])
    for df in history.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    trading_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
    if len(trading_dates) == 0:
        return {"error": "no trading dates", "stats": {}}

    rebal_index = pd.date_range(start, end, freq=cfg.rebalance_freq)
    rebal_dates = []
    for d in rebal_index:
        idx = trading_dates.searchsorted(d, side="left")
        if idx < len(trading_dates):
            rebal_dates.append(trading_dates[idx])
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique()

    cash = cfg.initial_capital
    positions: dict[str, dict] = {}
    trades = []
    equity_curve = []

    cost_buy = 1.0 + (cfg.slippage_bps + cfg.brokerage_bps) / 10_000.0
    cost_sell = 1.0 - (cfg.slippage_bps + cfg.brokerage_bps + cfg.stt_sell_bps) / 10_000.0

    for d in trading_dates:
        # time-stop exit check
        to_close = []
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is None or d not in df.index:
                continue
            pos["bars_held"] += 1
            if pos["bars_held"] >= cfg.time_stop_bars:
                to_close.append((sym, float(df.loc[d, "Close"]), "time_stop"))
        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            sell = pos["qty"] * px * cost_sell
            buy = pos["qty"] * pos["entry"] * cost_buy
            cash += sell
            trades.append({
                "symbol": sym, "entry": pos["entry"], "exit": px,
                "qty": pos["qty"], "pnl_net": sell - buy,
                "pnl_pct": (px / pos["entry"]) - 1.0,
                "bars_held": pos["bars_held"], "exit_reason": reason,
            })

        # Mark-to-market
        pv = cash
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is not None and d in df.index:
                pv += pos["qty"] * float(df.loc[d, "Close"])
        equity_curve.append({"date": d, "equity": pv})

        # Rebalance
        if d in rebal_dates:
            ranked = _rank_at(strategy, history, d, cfg)
            if not ranked:
                continue
            target_syms = _pick_top(ranked, cfg.n_picks, cfg.max_per_sector, sector_fn)
            if not target_syms:
                continue
            target_set = set(target_syms)
            current_set = set(positions.keys())

            # Sell positions no longer in target
            for sym in list(current_set - target_set):
                df = history.get(sym)
                if df is None or d not in df.index:
                    continue
                px = float(df.loc[d, "Close"])
                pos = positions.pop(sym)
                sell = pos["qty"] * px * cost_sell
                buy = pos["qty"] * pos["entry"] * cost_buy
                cash += sell
                trades.append({
                    "symbol": sym, "entry": pos["entry"], "exit": px,
                    "qty": pos["qty"], "pnl_net": sell - buy,
                    "pnl_pct": (px / pos["entry"]) - 1.0,
                    "bars_held": pos["bars_held"], "exit_reason": "rebalance",
                })

            # Open new
            new_syms = [s for s in target_syms if s not in positions]
            open_slots = cfg.n_picks - len(positions)
            new_syms = new_syms[:open_slots]
            if new_syms:
                budget = (cash * 0.95) / len(new_syms)
                for sym in new_syms:
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    px = float(df.loc[d, "Close"])
                    if px <= 0:
                        continue
                    eff = px * cost_buy
                    qty = int(budget / eff)
                    if qty < 1:
                        continue
                    cost = qty * eff
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry": px, "qty": qty, "bars_held": 0,
                    }

    # Close remaining at end
    last_d = trading_dates[-1]
    for sym, pos in list(positions.items()):
        df = history.get(sym)
        if df is None or last_d not in df.index:
            continue
        px = float(df.loc[last_d, "Close"])
        sell = pos["qty"] * px * cost_sell
        buy = pos["qty"] * pos["entry"] * cost_buy
        cash += sell
        trades.append({
            "symbol": sym, "entry": pos["entry"], "exit": px,
            "qty": pos["qty"], "pnl_net": sell - buy,
            "pnl_pct": (px / pos["entry"]) - 1.0,
            "bars_held": pos["bars_held"], "exit_reason": "end",
        })
        positions.pop(sym)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)

    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "stats": _compute_stats(eq_df, trades_df, cfg.initial_capital),
    }


def _compute_stats(eq_df, trades_df, init_cap):
    if eq_df.empty:
        return {}
    eq = eq_df["equity"]
    final = float(eq.iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / init_cap) ** (1.0 / years) - 1.0
    daily = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    dd = (eq / eq.cummax() - 1.0).min()
    calmar = (cagr / abs(dd)) if dd < 0 else float("inf")
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl_net"] > 0]
        n = len(trades_df)
        win_rate = len(wins) / n if n else 0
        gp = float(wins["pnl_net"].sum()) if len(wins) else 0
        gl = float(trades_df[trades_df["pnl_net"] <= 0]["pnl_net"].sum())
        pf = (gp / abs(gl)) if gl != 0 else float("inf")
    else:
        n = win_rate = pf = 0
    return {
        "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": float(dd),
        "calmar": calmar, "n_trades": int(n), "win_rate_pct": win_rate,
        "profit_factor": pf, "final_equity": final, "years": round(years, 2),
    }
