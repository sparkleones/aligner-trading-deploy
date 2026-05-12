"""
QGV-MF Backtest Engine.

Designed to mirror how top Indian small/mid-cap MFs actually operate:

  1. CONVICTION-WEIGHTED portfolio (NOT equal weight)
     Top-scoring stock gets max_weight (e.g. 15%), bottom gets min_weight (6%).

  2. DYNAMIC HOLD (NO fixed exit time)
     Exit only on:
       a. Stage 2 break (close below 30-week MA)
       b. Composite score falls into bottom 50% of universe
       c. Sector rotation: sector falls to bottom-3 by 6mo momentum
       d. Annual rebalance forced exit (rare — only if better opportunities)

  3. SECTOR-WEIGHTED ALLOCATION
     Top 3 sectors by 6mo momentum get 60% of capital, next 3 get 30%,
     remainder gets 10%.

  4. DIP-BUYING
     If an existing position drops 10% from entry AND still in top 30% of
     universe by composite score, ADD 25% to position. Max 2 add-ons.

  5. ANNUAL REBALANCE
     Full universe re-rank in January. Continuous monitoring of quality
     and stage-2 break otherwise.

  6. TAX
     STCG 15% on profits < 365 days, LTCG 12.5% otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class QGVMFConfig:
    start_date: str = "2022-01-01"
    end_date: str = "2026-04-30"
    initial_capital: float = 100_000.0
    monthly_sip: float = 0.0
    n_picks: int = 12
    annual_rebal_freq: str = "YS"
    quality_review_freq: str = "MS"   # monthly check for quality breaks
    max_weight_per_stock: float = 0.15
    min_weight_per_stock: float = 0.06
    # Sector allocation tiers
    sector_top_n: int = 3
    sector_top_weight: float = 0.60
    sector_mid_n: int = 3
    sector_mid_weight: float = 0.30
    sector_other_weight: float = 0.10
    max_per_sector: int = 4
    # Dip-buying
    dip_threshold: float = 0.10   # 10% drop from entry
    dip_add_pct: float = 0.25     # add 25% to position
    max_dip_addons: int = 2
    # Costs
    slippage_bps: float = 25.0    # higher for SC/MC
    brokerage_bps: float = 3.0
    stt_sell_bps: float = 10.0
    score_lag_bars: int = 1
    stage2_break_ma_bars: int = 150
    score_drop_exit_pct: float = 0.50  # exit if falls to bottom 50% by score
    # Tax
    apply_tax: bool = True
    stcg_tax_pct: float = 0.15
    ltcg_tax_pct: float = 0.125


def _rank_at(strategy, history, asof, cfg, sector_fn=None):
    """Score every stock as of (asof - score_lag_bars)."""
    rows = []
    for sym, df in history.items():
        try:
            pos = df.index.searchsorted(asof, side="right") - 1
            pos = max(0, pos - cfg.score_lag_bars)
            slice_end = df.index[pos]
        except Exception:
            slice_end = asof
        slice_ = df.loc[:slice_end]
        if len(slice_) < 750:  # QGV needs 3y
            continue
        score = strategy.score(sym, slice_, asof=slice_end)
        if score is None or pd.isna(score) or not np.isfinite(score):
            continue
        rows.append((sym, float(score), sector_fn(sym) if sector_fn else "OTHER"))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def _sector_momentum(history, asof, sector_fn) -> dict[str, float]:
    """Compute 6-month median return per sector as of `asof`."""
    by_sector: dict[str, list[float]] = {}
    for sym, df in history.items():
        try:
            pos = df.index.searchsorted(asof, side="right") - 1
            slice_end = df.index[pos]
        except Exception:
            continue
        slice_ = df.loc[:slice_end]
        if len(slice_) < 130:
            continue
        c = slice_["Close"]
        ret_6m = (c.iloc[-1] / c.iloc[-126] - 1.0) if c.iloc[-126] > 0 else 0
        sec = sector_fn(sym)
        by_sector.setdefault(sec, []).append(ret_6m)
    return {sec: float(np.median(rets)) for sec, rets in by_sector.items() if rets}


def _allocate_by_sector(picks, sector_momenta, cfg):
    """
    Assign a target weight to each pick based on sector momentum tier.
    Returns dict {symbol: weight}.
    """
    # Rank sectors by momentum
    ranked_secs = sorted(sector_momenta.items(), key=lambda kv: kv[1], reverse=True)
    top_secs = set(s for s, _ in ranked_secs[:cfg.sector_top_n])
    mid_secs = set(s for s, _ in ranked_secs[cfg.sector_top_n:cfg.sector_top_n + cfg.sector_mid_n])

    # Group picks by sector tier
    sec_to_picks: dict[str, list] = {"top": [], "mid": [], "other": []}
    for p in picks:
        sym, score, sec = p
        if sec in top_secs:
            sec_to_picks["top"].append(p)
        elif sec in mid_secs:
            sec_to_picks["mid"].append(p)
        else:
            sec_to_picks["other"].append(p)

    # Allocate capital per tier, conviction-weighted within tier
    weights = {}

    def _allocate_tier(tier_picks, tier_capital_frac):
        if not tier_picks or tier_capital_frac <= 0:
            return
        # Conviction weight: higher score = higher weight (within range)
        scores = np.array([p[1] for p in tier_picks])
        # Normalize: top score gets max, bottom gets min
        if len(scores) == 1:
            normalized = np.array([1.0])
        else:
            normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        # Map to [min_weight, max_weight] proportionally
        per_pick = cfg.min_weight_per_stock + normalized * (
            cfg.max_weight_per_stock - cfg.min_weight_per_stock
        )
        # Scale so total sums to tier_capital_frac
        scale = tier_capital_frac / per_pick.sum()
        per_pick *= scale
        for p, w in zip(tier_picks, per_pick):
            weights[p[0]] = float(w)

    _allocate_tier(sec_to_picks["top"], cfg.sector_top_weight)
    _allocate_tier(sec_to_picks["mid"], cfg.sector_mid_weight)
    _allocate_tier(sec_to_picks["other"], cfg.sector_other_weight)

    return weights


def _pick_top_with_sector_cap(ranked, n, max_per_sector):
    picks = []
    sec_count = {}
    for sym, score, sec in ranked:
        if sec_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append((sym, score, sec))
        sec_count[sec] = sec_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return picks


def run_qgv_backtest(strategy, history, sector_fn, cfg=None):
    cfg = cfg or QGVMFConfig()
    all_dates = pd.Index([])
    for df in history.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    trading_dates = all_dates[(all_dates >= start) & (all_dates <= end)]
    if len(trading_dates) == 0:
        return {"error": "no dates", "stats": {}}

    # Rebalance dates
    rebal_index = pd.date_range(start, end, freq=cfg.annual_rebal_freq)
    rebal_dates = []
    for d in rebal_index:
        idx = trading_dates.searchsorted(d, side="left")
        if idx < len(trading_dates):
            rebal_dates.append(trading_dates[idx])
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique()

    # Quality-review dates (monthly)
    review_index = pd.date_range(start, end, freq=cfg.quality_review_freq)
    review_dates = []
    for d in review_index:
        idx = trading_dates.searchsorted(d, side="left")
        if idx < len(trading_dates):
            review_dates.append(trading_dates[idx])
    review_dates = pd.DatetimeIndex(review_dates).unique()

    # SIP dates (monthly)
    sip_dates = pd.DatetimeIndex([])
    if cfg.monthly_sip > 0:
        sip_idx = pd.date_range(start, end, freq="MS")
        sd = []
        for d in sip_idx:
            idx = trading_dates.searchsorted(d, side="left")
            if idx < len(trading_dates):
                sd.append(trading_dates[idx])
        sip_dates = pd.DatetimeIndex(sd).unique()

    cash = cfg.initial_capital
    total_invested = cfg.initial_capital
    positions = {}      # sym -> {entry, qty, sector, bars_held, dip_addons, peak_score}
    trades = []
    equity_curve = []
    peak_equity = cfg.initial_capital
    cost_buy = 1.0 + (cfg.slippage_bps + cfg.brokerage_bps) / 10_000.0
    cost_sell = 1.0 - (cfg.slippage_bps + cfg.brokerage_bps + cfg.stt_sell_bps) / 10_000.0

    def _tax(gross, bars_held):
        if not cfg.apply_tax or gross <= 0:
            return 0
        return gross * (cfg.stcg_tax_pct if bars_held < 252 else cfg.ltcg_tax_pct)

    def _exit_position(sym, px, reason, d):
        nonlocal cash
        pos = positions.pop(sym)
        sell = pos["qty"] * px * cost_sell
        buy = pos["qty"] * pos["entry"] * cost_buy
        gross = sell - buy
        tax = _tax(gross, pos["bars_held"])
        cash += sell - tax
        trades.append({
            "symbol": sym, "entry": pos["entry"], "exit": px,
            "qty": pos["qty"], "pnl_net": gross - tax,
            "pnl_pct": (px / pos["entry"]) - 1.0,
            "bars_held": pos["bars_held"], "exit_reason": reason,
            "sector": pos.get("sector", "?"),
            "dip_addons": pos.get("dip_addons", 0),
        })

    for d in trading_dates:
        # SIP
        if d in sip_dates:
            cash += cfg.monthly_sip
            total_invested += cfg.monthly_sip

        # Stage-2 break exits + bar counter
        to_exit = []
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
                    to_exit.append((sym, close_now, "stage2_break"))
        for sym, px, reason in to_exit:
            _exit_position(sym, px, reason, d)

        # Monthly quality review
        if d in review_dates and positions:
            ranked = _rank_at(strategy, history, d, cfg, sector_fn=sector_fn)
            if ranked:
                ranked_syms = [r[0] for r in ranked]
                bottom_50_pct = ranked_syms[int(len(ranked_syms) * 0.5):]
                # Exit positions in bottom 50% AND have at least 60 bars hold
                for sym, pos in list(positions.items()):
                    if sym in bottom_50_pct and pos["bars_held"] >= 60:
                        df = history.get(sym)
                        if df is None or d not in df.index:
                            continue
                        _exit_position(sym, float(df.loc[d, "Close"]), "score_drop", d)

                # Dip-buying check
                for sym, pos in list(positions.items()):
                    if pos.get("dip_addons", 0) >= cfg.max_dip_addons:
                        continue
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    close_now = float(df.loc[d, "Close"])
                    drop = (close_now / pos["entry"]) - 1.0
                    if drop > -cfg.dip_threshold:
                        continue
                    # Position is down >= dip_threshold. Is it still in top 30%?
                    top_30_pct = ranked_syms[:int(len(ranked_syms) * 0.3)]
                    if sym not in top_30_pct:
                        continue
                    # Add 25% to position
                    add_value = pos["qty"] * pos["entry"] * cfg.dip_add_pct
                    if add_value < 100:  # too small
                        continue
                    eff = close_now * cost_buy
                    add_qty = int(add_value / eff)
                    if add_qty < 1 or add_qty * eff > cash:
                        continue
                    cash -= add_qty * eff
                    # New blended entry price (cost-weighted average)
                    new_total = pos["qty"] + add_qty
                    new_entry = (pos["qty"] * pos["entry"] + add_qty * close_now) / new_total
                    pos["qty"] = new_total
                    pos["entry"] = new_entry
                    pos["dip_addons"] = pos.get("dip_addons", 0) + 1

        # Mark-to-market
        pv = cash
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is not None and d in df.index:
                pv += pos["qty"] * float(df.loc[d, "Close"])
        equity_curve.append({"date": d, "equity": pv, "invested": total_invested})
        peak_equity = max(peak_equity, pv)

        # Annual rebalance
        if d in rebal_dates:
            ranked = _rank_at(strategy, history, d, cfg, sector_fn=sector_fn)
            if not ranked:
                continue
            picks = _pick_top_with_sector_cap(ranked, cfg.n_picks, cfg.max_per_sector)
            if not picks:
                continue

            target_set = set(s for s, _, _ in picks)
            current_set = set(positions.keys())

            # Sell what's NO LONGER in target
            for sym in list(current_set - target_set):
                df = history.get(sym)
                if df is None or d not in df.index:
                    continue
                _exit_position(sym, float(df.loc[d, "Close"]), "rebalance", d)

            # Compute conviction + sector-tier weights for new positions
            sec_mom = _sector_momentum(history, d, sector_fn)
            weights = _allocate_by_sector(picks, sec_mom, cfg)

            # Compute total portfolio value (cash + held)
            pv_now = cash
            for sym, pos in positions.items():
                df = history.get(sym)
                if df is not None and d in df.index:
                    pv_now += pos["qty"] * float(df.loc[d, "Close"])

            # For each target stock, ensure we hold approximately its target weight
            for sym, score, sec in picks:
                df = history.get(sym)
                if df is None or d not in df.index:
                    continue
                entry = float(df.loc[d, "Close"])
                if entry <= 0:
                    continue
                target_value = pv_now * weights.get(sym, 0)
                eff = entry * cost_buy
                target_qty = int(target_value / eff)
                if target_qty < 1:
                    continue
                if sym in positions:
                    # Adjust existing — top-up if under-weighted by > 20%
                    pos = positions[sym]
                    if pos["qty"] < target_qty * 0.8:
                        add_qty = target_qty - pos["qty"]
                        cost = add_qty * eff
                        if cost <= cash:
                            cash -= cost
                            # Blended cost basis
                            new_total = pos["qty"] + add_qty
                            new_entry = (pos["qty"] * pos["entry"] + add_qty * entry) / new_total
                            pos["qty"] = new_total
                            pos["entry"] = new_entry
                else:
                    cost = target_qty * eff
                    if cost > cash:
                        target_qty = int(cash / eff)
                        cost = target_qty * eff
                    if target_qty < 1 or cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = {
                        "entry": entry, "qty": target_qty,
                        "sector": sec, "bars_held": 0, "dip_addons": 0,
                    }

    # End-of-window close
    last_d = trading_dates[-1]
    for sym in list(positions.keys()):
        df = history.get(sym)
        if df is None or last_d not in df.index:
            continue
        _exit_position(sym, float(df.loc[last_d, "Close"]), "end", last_d)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame(trades)
    return {
        "equity_curve": eq_df, "trades": trades_df,
        "total_invested": total_invested,
        "stats": _stats(eq_df, trades_df, cfg.initial_capital, total_invested,
                          monthly_sip=cfg.monthly_sip),
    }


def _stats(eq, trades_df, init_cap, total_invested, monthly_sip=0):
    if eq.empty:
        return {}
    final = float(eq["equity"].iloc[-1])
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-6)

    # XIRR for SIP scenarios
    if monthly_sip > 0:
        cashflows = [-init_cap]
        n_months = int(years * 12)
        cashflows.extend([-monthly_sip] * n_months)
        cashflows.append(final)
        try:
            from scipy import optimize as _opt
            def npv(r):
                return sum(cf / ((1 + r) ** i) for i, cf in enumerate(cashflows))
            try:
                r_m = _opt.brentq(npv, -0.5, 0.5)
                xirr = (1 + r_m) ** 12 - 1
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
        "absolute_profit": final - total_invested,
        "absolute_return_pct": (final / total_invested - 1.0) if total_invested > 0 else 0,
        "years": round(years, 2),
    }
