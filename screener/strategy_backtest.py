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
    stcg_tax_pct: float = 0.0     # 15% on profitable trades if enabled
    time_stop_bars: int = 60
    min_score: float = -1e9   # by default no min, each strategy decides via NaN
    max_per_sector: int = 1
    min_bars_required: int = 252

    # ── ANTI-LOOK-AHEAD: how many bars to delay scoring relative to execution ──
    # 0 = original buggy behavior (scoring and execution use same bar)
    # 1 = score on day T-1, execute on day T (the honest setting)
    score_lag_bars: int = 1

    # ── RISK MANAGEMENT (matches what live_picks_v2 actually does) ──
    apply_stop_loss: bool = True
    min_stop_pct: float = 0.04   # 4% floor (matches levels.py)
    max_stop_pct: float = 0.08   # 8% ceiling (matches levels.py)
    atr_multiplier: float = 2.0  # SL = max(min_stop_pct, atr_mult * ATR/price)
    reward_risk_ratio: float = 3.0  # target = entry + 3 * stop_dist


def _rank_at(
    strategy: BaseStrategy,
    history: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    cfg: StrategyBacktestConfig,
) -> list[tuple[str, float]]:
    """Score every stock using `strategy` as of (asof - score_lag_bars).

    Anti-look-ahead: with default score_lag_bars=1, scoring uses data
    only through the PRIOR trading day. The pick is then executed at
    asof's close. This matches reality: a screener run on Monday morning
    can only use Friday's data; the order fills at Monday's close.
    """
    rows = []
    for sym, df in history.items():
        # Find the bar `score_lag_bars` before asof (or use asof itself if lag=0)
        if cfg.score_lag_bars <= 0:
            slice_end = asof
        else:
            # df.index sorted ascending; find position of asof and step back
            try:
                pos = df.index.searchsorted(asof, side="right") - 1
                pos = max(0, pos - cfg.score_lag_bars)
                slice_end = df.index[pos]
            except Exception:
                slice_end = asof
        slice_ = df.loc[:slice_end]
        if len(slice_) < cfg.min_bars_required:
            continue
        score = strategy.score(sym, slice_, asof=slice_end)
        if score is None or pd.isna(score) or not np.isfinite(score):
            continue
        if score < cfg.min_score:
            continue
        rows.append((sym, float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _atr_pct(history: pd.DataFrame, asof: pd.Timestamp, period: int = 20) -> float:
    """ATR as % of close at `asof`. Default 4% if insufficient data."""
    df = history.loc[:asof]
    if len(df) < period + 1:
        return 0.04
    h = df["High"]
    l = df["Low"]
    c = df["Close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    px = c.iloc[-1]
    if px <= 0 or pd.isna(atr):
        return 0.04
    return float(atr / px)


def _compute_levels(history: pd.DataFrame, asof: pd.Timestamp, cfg: StrategyBacktestConfig) -> tuple[float, float, float]:
    """Return (entry_px, stop_loss, target) at `asof` matching live_picks_v2 rules."""
    entry = float(history.loc[asof, "Close"])
    atr_p = _atr_pct(history, asof)
    raw_stop = max(cfg.atr_multiplier * atr_p, cfg.min_stop_pct)
    stop_pct = min(raw_stop, cfg.max_stop_pct)
    sl = entry * (1.0 - stop_pct)
    target = entry + cfg.reward_risk_ratio * (entry - sl)
    return entry, sl, target


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
        # Exit checks: SL, target, time-stop. Checked intraday using
        # the day's High/Low. SL/target priority is conservative — if
        # both could fire on the same day, the SL fires first.
        to_close = []
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is None or d not in df.index:
                continue
            pos["bars_held"] += 1
            high = float(df.loc[d, "High"])
            low = float(df.loc[d, "Low"])
            close = float(df.loc[d, "Close"])
            sl = pos.get("stop_loss")
            tgt = pos.get("target")
            # Stop-loss hit intraday (fills AT the SL price)
            if cfg.apply_stop_loss and sl is not None and low <= sl:
                to_close.append((sym, sl, "stop_loss"))
                continue
            # Target hit intraday (fills AT the target price)
            if tgt is not None and high >= tgt:
                to_close.append((sym, tgt, "target"))
                continue
            # Time stop
            if pos["bars_held"] >= cfg.time_stop_bars:
                to_close.append((sym, close, "time_stop"))
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
                    # Compute entry + SL + target using the SAME rules as
                    # live_picks_v2 — this is the critical fix.
                    entry, sl, tgt = _compute_levels(df, d, cfg)
                    if entry <= 0:
                        continue
                    eff = entry * cost_buy
                    qty = int(budget / eff)
                    if qty < 1:
                        continue
                    cost = qty * eff
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry": entry, "stop_loss": sl, "target": tgt,
                        "qty": qty, "bars_held": 0,
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
