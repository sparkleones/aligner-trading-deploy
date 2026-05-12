"""
Regime-adaptive backtest engine. Detects regime at every rebalance
and allocates between momentum / low-vol / mean-reversion / cash.

Same anti-look-ahead + hold-until-deterioration framework as
backtest_mf_style, but with the regime allocator on top.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .strategies.regime_adaptive import (
    Regime, RegimeSnapshot, ALLOCATION,
    classify_regime, compute_breadth,
    MomentumWrapper,
)


@dataclass
class RegimeAdaptiveConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2026-04-30"
    initial_capital: float = 100_000.0
    n_picks: int = 5
    rebalance_freq: str = "QS"
    slippage_bps: float = 10.0
    brokerage_bps: float = 3.0
    stt_sell_bps: float = 10.0
    score_lag_bars: int = 1
    stage2_break_ma_bars: int = 150
    portfolio_dd_pause_pct: float = 0.20
    max_per_sector: int = 1
    min_bars_required: int = 252
    apply_stcg_tax: bool = True
    stcg_tax_pct: float = 0.15


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


def _pick_top(ranked, n, max_per_sector, sector_fn,
               existing_sectors: Optional[set] = None):
    picks = []
    sec_count = dict(existing_sectors or {})
    for sym, _ in ranked:
        sec = sector_fn(sym)
        if isinstance(sec_count, set):
            sec_count = {s: 1 for s in sec_count}
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append((sym, sec))
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def run_regime_backtest(
    history: dict[str, pd.DataFrame],
    nifty_history: pd.DataFrame,
    vix_history: pd.DataFrame | None,
    sector_fn,
    cfg: Optional[RegimeAdaptiveConfig] = None,
) -> dict:
    cfg = cfg or RegimeAdaptiveConfig()

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

    cash = cfg.initial_capital
    positions = {}   # sym -> dict
    trades = []
    equity_curve = []
    regime_history = []
    peak_equity = cfg.initial_capital
    drawdown_paused = False
    cost_buy = 1.0 + (cfg.slippage_bps + cfg.brokerage_bps) / 10_000.0
    cost_sell = 1.0 - (cfg.slippage_bps + cfg.brokerage_bps + cfg.stt_sell_bps) / 10_000.0

    mom_strat = MomentumWrapper()

    for d in trading_dates:
        # Stage-2 break exits (skip NIFTY ETF — it doesn't get stopped out;
        # it's held until rebalance changes the allocation)
        to_close = []
        for sym, pos in positions.items():
            if sym == "__NIFTY_ETF__":
                pos["bars_held"] += 1
                continue
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
            tax = (gross * cfg.stcg_tax_pct) if (cfg.apply_stcg_tax and gross > 0
                                                  and pos["bars_held"] < 252) else 0
            cash += sell - tax
            trades.append({
                "symbol": sym, "entry": pos["entry"], "exit": px,
                "qty": pos["qty"], "pnl_net": gross - tax,
                "pnl_pct": (px / pos["entry"]) - 1.0,
                "bars_held": pos["bars_held"], "exit_reason": reason,
                "sub_strategy": pos.get("sub_strategy", "?"),
                "regime": pos.get("regime", "?"),
            })

        # Mark-to-market
        pv = cash
        for sym, pos in positions.items():
            if sym == "__NIFTY_ETF__":
                if nifty_history is not None and d in nifty_history.index:
                    pv += pos["qty"] * float(nifty_history.loc[d, "Close"])
                continue
            df = history.get(sym)
            if df is not None and d in df.index:
                pv += pos["qty"] * float(df.loc[d, "Close"])
        equity_curve.append({"date": d, "equity": pv})
        peak_equity = max(peak_equity, pv)
        cur_dd = pv / peak_equity - 1.0
        drawdown_paused = (cur_dd <= -cfg.portfolio_dd_pause_pct)

        # Rebalance
        if d in rebal_dates:
            # Classify regime first
            breadth = compute_breadth(history, d)
            snap = classify_regime(nifty_history, vix_history, breadth, d)
            regime_history.append({"date": d, **snap.__dict__})

            alloc = ALLOCATION[snap.regime].copy()

            if drawdown_paused:
                # Override: full cash on portfolio drawdown
                alloc = {"momentum": 0, "low_vol": 0, "mean_rev": 0, "cash": 1.0}

            # Sell everything not consistent with new allocation
            # (simplification: sell ALL at rebalance, then redeploy)
            for sym in list(positions.keys()):
                if sym == "__NIFTY_ETF__":
                    if nifty_history is None or d not in nifty_history.index:
                        continue
                    px = float(nifty_history.loc[d, "Close"])
                else:
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    px = float(df.loc[d, "Close"])
                pos = positions.pop(sym)
                sell = pos["qty"] * px * cost_sell
                buy = pos["qty"] * pos["entry"] * cost_buy
                gross = sell - buy
                tax = (gross * cfg.stcg_tax_pct) if (cfg.apply_stcg_tax and gross > 0
                                                      and pos["bars_held"] < 252) else 0
                cash += sell - tax
                trades.append({
                    "symbol": sym, "entry": pos["entry"], "exit": px,
                    "qty": pos["qty"], "pnl_net": gross - tax,
                    "pnl_pct": (px / pos["entry"]) - 1.0,
                    "bars_held": pos["bars_held"], "exit_reason": "rebalance",
                    "sub_strategy": pos.get("sub_strategy", "?"),
                    "regime": pos.get("regime", "?"),
                })

            # Now redeploy. Each sub-strategy gets a fraction of capital
            # and picks its own top stocks.
            deployable = cash * 0.97  # small cushion

            def buy_picks(strategy, label, fraction):
                nonlocal cash
                budget_total = deployable * fraction
                if budget_total <= 0:
                    return
                ranked = _rank_at(strategy, history, d, cfg)
                if not ranked:
                    return
                # n_picks scales with allocation fraction; min 1
                n_per_strat = max(1, int(round(cfg.n_picks * fraction)))
                picks = _pick_top(ranked, n_per_strat, cfg.max_per_sector, sector_fn,
                                    existing_sectors=set())
                if not picks:
                    return
                per_pick = budget_total / max(1, len(picks))
                for sym, sec in picks:
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    entry = float(df.loc[d, "Close"])
                    if entry <= 0:
                        continue
                    eff = entry * cost_buy
                    qty = int(per_pick / eff)
                    if qty < 1:
                        continue
                    cost = qty * eff
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry": entry, "qty": qty, "bars_held": 0,
                        "sub_strategy": label, "regime": snap.regime.value,
                    }

            buy_picks(mom_strat, "momentum", alloc.get("momentum", 0))

            # NIFTY ETF allocation — buy a synthetic NIFTYBEES position
            # using nifty_history's close as the price. Saves us from
            # needing the actual NIFTYBEES ticker.
            nifty_frac = alloc.get("nifty_etf", 0)
            if nifty_frac > 0 and nifty_history is not None and d in nifty_history.index:
                budget = deployable * nifty_frac
                entry = float(nifty_history.loc[d, "Close"])
                if entry > 0:
                    eff = entry * cost_buy
                    qty = int(budget / eff)
                    if qty >= 1:
                        cost = qty * eff
                        if cost <= cash:
                            cash -= cost
                            positions["__NIFTY_ETF__"] = {
                                "entry": entry, "qty": qty, "bars_held": 0,
                                "sub_strategy": "nifty_etf",
                                "regime": snap.regime.value,
                            }

    # Close remaining at end
    last_d = trading_dates[-1]
    for sym, pos in list(positions.items()):
        if sym == "__NIFTY_ETF__":
            if nifty_history is None or last_d not in nifty_history.index:
                continue
            px = float(nifty_history.loc[last_d, "Close"])
        else:
            df = history.get(sym)
            if df is None or last_d not in df.index:
                continue
            px = float(df.loc[last_d, "Close"])
        sell = pos["qty"] * px * cost_sell
        buy = pos["qty"] * pos["entry"] * cost_buy
        gross = sell - buy
        tax = (gross * cfg.stcg_tax_pct) if (cfg.apply_stcg_tax and gross > 0
                                              and pos["bars_held"] < 252) else 0
        cash += sell - tax
        trades.append({
            "symbol": sym, "entry": pos["entry"], "exit": px,
            "qty": pos["qty"], "pnl_net": gross - tax,
            "pnl_pct": (px / pos["entry"]) - 1.0,
            "bars_held": pos["bars_held"], "exit_reason": "end",
            "sub_strategy": pos.get("sub_strategy", "?"),
            "regime": pos.get("regime", "?"),
        })
        positions.pop(sym)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "regime_history": regime_history,
        "stats": _stats(eq_df, trades_df, cfg.initial_capital),
    }


def _stats(eq, trades_df, init):
    if eq.empty:
        return {}
    final = float(eq["equity"].iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)
    cagr = (final / init) ** (1.0 / years) - 1.0
    daily = eq["equity"].pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    dd = (eq["equity"] / eq["equity"].cummax() - 1.0).min()
    calmar = (cagr / abs(dd)) if dd < 0 else float("inf")
    n_trades = len(trades_df)
    win_rate = float((trades_df["pnl_net"] > 0).mean()) if n_trades else 0
    return {
        "cagr_pct": cagr, "sharpe": sharpe, "max_dd_pct": float(dd),
        "calmar": calmar, "n_trades": n_trades, "win_rate_pct": win_rate,
        "final_equity": final, "years": round(years, 2),
    }
