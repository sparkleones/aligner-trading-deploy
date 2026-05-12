"""
Small-Cap MF-style backtest engine.

Designed to mimic how a small-cap mutual fund actually operates:
  - Annual rebalance (not monthly/quarterly)
  - Concentrated 10-15 stock portfolio
  - No price-based stop-loss
  - Stage-2 break exit only (close below 30-week MA = quality broken)
  - Higher slippage (small-cap bid/ask = 25-40bps)
  - SIP simulation: optional monthly capital addition
  - STCG 15% on profits held < 1 year, LTCG 12.5% on > 1 year
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .strategies import BaseStrategy


@dataclass
class SmallCapMFConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2026-04-30"
    initial_capital: float = 100_000.0
    monthly_sip: float = 0.0    # set > 0 to simulate monthly SIP
    n_picks: int = 12           # concentrated 10-15 portfolio (top SC MFs avg ~50 but top-10 = 30-40%)
    rebalance_freq: str = "AS"  # annual start (Jan 1)
    slippage_bps: float = 25.0  # higher for small-caps
    brokerage_bps: float = 3.0
    stt_sell_bps: float = 10.0
    score_lag_bars: int = 1
    stage2_break_ma_bars: int = 150  # 30-week MA
    portfolio_dd_pause_pct: float = 0.30  # less aggressive than large-cap (SC MFs hold through 30% DD)
    max_per_sector: int = 3
    min_bars_required: int = 252
    apply_tax: bool = True
    stcg_tax_pct: float = 0.15
    ltcg_tax_pct: float = 0.125


def _rank_at(strategy, history, asof, cfg):
    rows = []
    for sym, df in history.items():
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
        rows.append((sym, float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _pick_top(ranked, n, max_per_sector, sector_fn):
    picks = []
    sec_count = {}
    for sym, _ in ranked:
        sec = sector_fn(sym)
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append((sym, sec))
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def run_smallcap_backtest(strategy, history, sector_fn, cfg=None):
    cfg = cfg or SmallCapMFConfig()
    all_dates = pd.Index([])
    for df in history.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    trading_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
    if len(trading_dates) == 0:
        return {"error": "no dates", "stats": {}}

    rebal_index = pd.date_range(start, end, freq=cfg.rebalance_freq)
    rebal_dates = []
    for d in rebal_index:
        idx = trading_dates.searchsorted(d, side="left")
        if idx < len(trading_dates):
            rebal_dates.append(trading_dates[idx])
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique()

    # SIP dates: 1st trading day of each month
    sip_dates = []
    if cfg.monthly_sip > 0:
        sip_index = pd.date_range(start, end, freq="MS")
        for d in sip_index:
            idx = trading_dates.searchsorted(d, side="left")
            if idx < len(trading_dates):
                sip_dates.append(trading_dates[idx])
        sip_dates = pd.DatetimeIndex(sip_dates).unique()

    cash = cfg.initial_capital
    total_invested = cfg.initial_capital
    positions = {}
    trades = []
    equity_curve = []
    peak_equity = cfg.initial_capital
    drawdown_paused = False
    cost_buy = 1.0 + (cfg.slippage_bps + cfg.brokerage_bps) / 10_000.0
    cost_sell = 1.0 - (cfg.slippage_bps + cfg.brokerage_bps + cfg.stt_sell_bps) / 10_000.0

    def _apply_tax(gross_pnl, bars_held):
        if not cfg.apply_tax or gross_pnl <= 0:
            return 0
        if bars_held < 252:
            return gross_pnl * cfg.stcg_tax_pct
        return gross_pnl * cfg.ltcg_tax_pct

    for d in trading_dates:
        # SIP injection on the 1st trading day of each month
        if cfg.monthly_sip > 0 and d in sip_dates:
            cash += cfg.monthly_sip
            total_invested += cfg.monthly_sip

        # Stage-2 break exits (long-term MA broken = quality concern)
        to_close = []
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is None or d not in df.index:
                continue
            pos["bars_held"] += 1
            c = df["Close"]
            ma = c.rolling(cfg.stage2_break_ma_bars).mean()
            if d in ma.index:
                ma_now = ma.loc[d]
                close_now = float(c.loc[d])
                if not pd.isna(ma_now) and close_now < ma_now:
                    to_close.append((sym, close_now, "stage2_break"))
        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            sell = pos["qty"] * px * cost_sell
            buy = pos["qty"] * pos["entry"] * cost_buy
            gross = sell - buy
            tax = _apply_tax(gross, pos["bars_held"])
            cash += sell - tax
            trades.append({
                "symbol": sym, "entry": pos["entry"], "exit": px,
                "qty": pos["qty"], "pnl_net": gross - tax,
                "pnl_pct": (px / pos["entry"]) - 1.0,
                "bars_held": pos["bars_held"], "exit_reason": reason,
            })

        # Mark-to-market
        pv = cash
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is not None and d in df.index:
                pv += pos["qty"] * float(df.loc[d, "Close"])
        equity_curve.append({"date": d, "equity": pv, "invested": total_invested})
        peak_equity = max(peak_equity, pv)
        cur_dd = pv / peak_equity - 1.0
        drawdown_paused = (cur_dd <= -cfg.portfolio_dd_pause_pct)

        # Rebalance
        if d in rebal_dates and not drawdown_paused:
            ranked = _rank_at(strategy, history, d, cfg)
            if not ranked:
                continue
            picks = _pick_top(ranked, cfg.n_picks, cfg.max_per_sector, sector_fn)
            target = set(s for s, _ in picks)
            current = set(positions.keys())
            # Sell what's no longer in target
            for sym in list(current - target):
                df = history.get(sym)
                if df is None or d not in df.index:
                    continue
                px = float(df.loc[d, "Close"])
                pos = positions.pop(sym)
                sell = pos["qty"] * px * cost_sell
                buy = pos["qty"] * pos["entry"] * cost_buy
                gross = sell - buy
                tax = _apply_tax(gross, pos["bars_held"])
                cash += sell - tax
                trades.append({
                    "symbol": sym, "entry": pos["entry"], "exit": px,
                    "qty": pos["qty"], "pnl_net": gross - tax,
                    "pnl_pct": (px / pos["entry"]) - 1.0,
                    "bars_held": pos["bars_held"], "exit_reason": "rebalance",
                })
            # Buy new picks
            new_syms = [s for s, _ in picks if s not in positions]
            open_slots = cfg.n_picks - len(positions)
            new_syms = new_syms[:open_slots]
            if new_syms:
                budget = (cash * 0.95) / max(1, len(new_syms))
                for sym in new_syms:
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    entry = float(df.loc[d, "Close"])
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
                    positions[sym] = {"entry": entry, "qty": qty, "bars_held": 0}

    # Close remaining at end
    last_d = trading_dates[-1]
    for sym, pos in list(positions.items()):
        df = history.get(sym)
        if df is None or last_d not in df.index:
            continue
        px = float(df.loc[last_d, "Close"])
        sell = pos["qty"] * px * cost_sell
        buy = pos["qty"] * pos["entry"] * cost_buy
        gross = sell - buy
        tax = _apply_tax(gross, pos["bars_held"])
        cash += sell - tax
        trades.append({
            "symbol": sym, "entry": pos["entry"], "exit": px,
            "qty": pos["qty"], "pnl_net": gross - tax,
            "pnl_pct": (px / pos["entry"]) - 1.0,
            "bars_held": pos["bars_held"], "exit_reason": "end",
        })
        positions.pop(sym)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "total_invested": total_invested,
        "stats": _stats(eq_df, trades_df, cfg.initial_capital, total_invested,
                          monthly_sip=cfg.monthly_sip),
    }


def _stats(eq, trades_df, init_cap, total_invested, monthly_sip=0):
    if eq.empty:
        return {}
    final = float(eq["equity"].iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)

    # For SIP, we need XIRR not just CAGR. Approximation: use IRR on cashflows
    if monthly_sip > 0:
        # Cashflows: initial -init_cap, monthly -sip, final +final
        cashflows = [-init_cap]
        n_months = int(years * 12)
        cashflows.extend([-monthly_sip] * n_months)
        cashflows.append(final)
        try:
            from scipy import optimize as _opt
            # Solve for monthly IRR
            def npv(r):
                total = 0
                for i, cf in enumerate(cashflows):
                    total += cf / ((1 + r) ** i)
                return total
            try:
                r_monthly = _opt.brentq(npv, -0.5, 0.5)
                xirr = (1 + r_monthly) ** 12 - 1
            except Exception:
                xirr = (final / total_invested) ** (1.0 / years) - 1.0
        except Exception:
            xirr = (final / total_invested) ** (1.0 / years) - 1.0
        cagr = xirr
    else:
        cagr = (final / init_cap) ** (1.0 / years) - 1.0

    daily = eq["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    dd = (eq["equity"] / eq["equity"].cummax() - 1.0).min()
    calmar = (cagr / abs(dd)) if dd < 0 else float("inf")
    n_trades = len(trades_df)
    win_rate = float((trades_df["pnl_net"] > 0).mean()) if n_trades else 0
    return {
        "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": float(dd),
        "calmar": calmar, "n_trades": n_trades, "win_rate_pct": win_rate,
        "final_equity": final, "total_invested": total_invested,
        "years": round(years, 2),
        "absolute_profit": final - total_invested,
        "absolute_return_pct": (final / total_invested - 1.0) if total_invested > 0 else 0,
    }
