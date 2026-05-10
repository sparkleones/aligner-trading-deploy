"""
Stock screener backtest engine.

Walk-forward simulation:
    - At each rebalance date (weekly Mondays), compute factor scores using
      data available UP TO and INCLUDING that day (no look-ahead).
    - Pick top N stocks (default 2).
    - Allocate equal weight, hold until exit rule fires.
    - Exit rules (per position): stop-loss, target, time-stop (20 bars),
      or replaced by a better-ranked stock on a future rebalance.
    - Apply slippage + brokerage on each trade.

Outputs portfolio equity curve and trade-by-trade log.

Survivorship bias: We use a static universe today. To validate that
backtest stats survive the bias, the user should re-run later with a
point-in-time universe (Phase 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import numpy as np
import pandas as pd

from .factors import compute_all_factors
from .ranker import RankerConfig, _zscore
from .universe import get_sector


@dataclass
class BacktestConfig:
    start_date: str = "2023-01-01"
    end_date: str = "2026-04-30"
    initial_capital: float = 100_000.0
    n_picks: int = 2                    # 2 concurrent positions
    rebalance_freq: str = "W-MON"       # weekly Mondays
    slippage_bps: float = 10.0          # 0.10% slippage per leg
    brokerage_bps: float = 3.0          # ~0.03% all-in (Zerodha equity)
    stt_sell_bps: float = 10.0          # 0.10% STT on sell
    # Exit rule params (mirror trade_plan defaults)
    min_stop_pct: float = 0.04
    max_stop_pct: float = 0.08
    reward_risk_ratio: float = 3.0
    time_stop_bars: int = 20
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum_12_1": 0.35,
        "trend":         0.25,
        "reversal_1m":  -0.15,
        "low_vol":       0.15,
        "gap_risk":      0.10,
    })
    min_liquidity_log10_inr: float = 9.0
    max_atr_pct: float = 0.06
    require_above_200dma: bool = True
    max_per_sector: int = 1


@dataclass
class Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_px: float
    stop_loss: float
    target: float
    qty: int
    bars_held: int = 0
    notes: str = ""


@dataclass
class TradeRecord:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    qty: int
    bars_held: int
    pnl_gross: float
    pnl_net: float
    pnl_pct: float
    exit_reason: str


def _compute_factors_at(df: pd.DataFrame, asof: pd.Timestamp) -> Optional[dict]:
    """Compute factors using only data up to and including `asof`."""
    slice_ = df.loc[:asof]
    if len(slice_) < 252:
        return None
    return compute_all_factors(slice_)


def _rank_at(
    history: dict[str, pd.DataFrame],
    asof: pd.Timestamp,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Compute ranking as of a date (no look-ahead). Returns sorted DataFrame."""
    rows = []
    for sym, df in history.items():
        f = _compute_factors_at(df, asof)
        if f is None:
            continue
        if math.isnan(f["liquidity"]) or f["liquidity"] < cfg.min_liquidity_log10_inr:
            continue
        if math.isnan(f["atr_pct"]) or f["atr_pct"] > cfg.max_atr_pct:
            continue
        slice_ = df.loc[:asof]
        if cfg.require_above_200dma:
            ma200 = slice_["Close"].rolling(200).mean().iloc[-1]
            if pd.isna(ma200) or slice_["Close"].iloc[-1] < ma200:
                continue
        rows.append({"symbol": sym, "sector": get_sector(sym), **f})

    if not rows:
        return pd.DataFrame()

    df_r = pd.DataFrame(rows).set_index("symbol")
    composite = pd.Series(0.0, index=df_r.index)
    for fac, w in cfg.factor_weights.items():
        if fac in df_r.columns:
            z = _zscore(df_r[fac])
            df_r[f"z_{fac}"] = z
            composite = composite + w * z.fillna(0.0)
    df_r["composite"] = composite
    return df_r.sort_values("composite", ascending=False).reset_index()


def _pick_with_sector_cap(
    ranked: pd.DataFrame,
    n: int,
    max_per_sector: int,
    current_symbols: set[str],
) -> pd.DataFrame:
    """Pick top N respecting sector cap, allow keeping existing positions."""
    picks = []
    sector_count: dict[str, int] = {}
    for _, row in ranked.iterrows():
        if row["composite"] <= 0:
            break
        sec = row["sector"]
        if sector_count.get(sec, 0) >= max_per_sector:
            continue
        picks.append(row)
        sector_count[sec] = sector_count.get(sec, 0) + 1
        if len(picks) >= n:
            break
    return pd.DataFrame(picks).reset_index(drop=True)


def run_backtest(
    history: dict[str, pd.DataFrame],
    cfg: BacktestConfig | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run walk-forward backtest. Returns dict with equity curve, trades,
    and stats.
    """
    cfg = cfg or BacktestConfig()

    # Build trading calendar (union of all stock dates) within window
    all_dates = pd.Index([])
    for df in history.values():
        all_dates = all_dates.union(df.index)
    all_dates = all_dates.sort_values()
    start = pd.Timestamp(cfg.start_date)
    end = pd.Timestamp(cfg.end_date)
    trading_dates = all_dates[(all_dates >= start) & (all_dates <= end)]

    if len(trading_dates) == 0:
        return {"error": "no trading dates in window"}

    # Rebalance dates: weekly Mondays within the window
    rebal_index = pd.date_range(start, end, freq=cfg.rebalance_freq)
    # Snap each rebalance date to the next available trading date
    rebal_dates = []
    for d in rebal_index:
        idx = trading_dates.searchsorted(d, side="left")
        if idx < len(trading_dates):
            rebal_dates.append(trading_dates[idx])
    rebal_dates = pd.DatetimeIndex(rebal_dates).unique()

    cash = cfg.initial_capital
    positions: dict[str, Position] = {}
    trades: list[TradeRecord] = []
    equity_curve = []

    cost_factor_buy = 1.0 + (cfg.slippage_bps + cfg.brokerage_bps) / 10_000.0
    cost_factor_sell = 1.0 - (cfg.slippage_bps + cfg.brokerage_bps + cfg.stt_sell_bps) / 10_000.0

    for d in trading_dates:
        # --- Mark-to-market open positions ---
        portfolio_value = cash
        to_close: list[tuple[str, float, str]] = []  # (sym, exit_px, reason)

        for sym, pos in positions.items():
            df = history.get(sym)
            if df is None or d not in df.index:
                continue
            row = df.loc[d]
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])
            pos.bars_held += 1

            # Intraday stop check (use low for stops, high for targets)
            if low <= pos.stop_loss:
                to_close.append((sym, pos.stop_loss, "stop"))
                continue
            if high >= pos.target:
                to_close.append((sym, pos.target, "target"))
                continue
            if pos.bars_held >= cfg.time_stop_bars:
                to_close.append((sym, close, "time_stop"))
                continue

            portfolio_value += pos.qty * close

        # Process exits (sell at decided exit px)
        for sym, exit_px, reason in to_close:
            pos = positions.pop(sym)
            gross_pnl = (exit_px - pos.entry_px) * pos.qty
            # Apply transaction costs (already paid on buy; pay on sell)
            sell_proceeds = pos.qty * exit_px * cost_factor_sell
            buy_cost = pos.qty * pos.entry_px * cost_factor_buy
            net_pnl = sell_proceeds - buy_cost
            cash += sell_proceeds
            trades.append(TradeRecord(
                symbol=sym,
                entry_date=pos.entry_date,
                exit_date=d,
                entry_px=pos.entry_px,
                exit_px=exit_px,
                qty=pos.qty,
                bars_held=pos.bars_held,
                pnl_gross=gross_pnl,
                pnl_net=net_pnl,
                pnl_pct=(exit_px / pos.entry_px) - 1.0,
                exit_reason=reason,
            ))

        # --- Recompute portfolio value for equity curve ---
        pv = cash
        for sym, pos in positions.items():
            df = history.get(sym)
            if df is not None and d in df.index:
                pv += pos.qty * float(df.loc[d, "Close"])

        equity_curve.append({"date": d, "equity": pv, "n_positions": len(positions)})

        # --- Rebalance check ---
        if d in rebal_dates:
            ranked = _rank_at(history, d, cfg)
            if ranked.empty:
                continue
            picks = _pick_with_sector_cap(
                ranked, cfg.n_picks, cfg.max_per_sector, set(positions.keys())
            )

            # If no qualifying picks, hold cash this period — skip rebalance
            if picks.empty or "symbol" not in picks.columns:
                continue

            target_set = set(picks["symbol"].tolist())
            current_set = set(positions.keys())

            # Sell positions no longer in target set (and not already exited via stops)
            for sym in list(current_set - target_set):
                if sym not in positions:
                    continue
                df = history.get(sym)
                if df is None or d not in df.index:
                    continue
                close_px = float(df.loc[d, "Close"])
                pos = positions.pop(sym)
                sell_proceeds = pos.qty * close_px * cost_factor_sell
                buy_cost = pos.qty * pos.entry_px * cost_factor_buy
                net_pnl = sell_proceeds - buy_cost
                cash += sell_proceeds
                trades.append(TradeRecord(
                    symbol=sym, entry_date=pos.entry_date, exit_date=d,
                    entry_px=pos.entry_px, exit_px=close_px, qty=pos.qty,
                    bars_held=pos.bars_held, pnl_gross=(close_px - pos.entry_px) * pos.qty,
                    pnl_net=net_pnl, pnl_pct=(close_px / pos.entry_px) - 1.0,
                    exit_reason="rebalance",
                ))

            # Open new positions (those in target_set but not yet held)
            open_slots = cfg.n_picks - len(positions)
            new_symbols = [s for s in picks["symbol"] if s not in positions][:open_slots]
            if new_symbols:
                # Equal-weight remaining cash across new slots
                budget_per_slot = cash / max(1, len(new_symbols) + 1)  # keep 1 slot of cushion
                budget_per_slot = min(budget_per_slot, cash * 0.45)    # at most 45% per name
                for sym in new_symbols:
                    df = history.get(sym)
                    if df is None or d not in df.index:
                        continue
                    entry_px = float(df.loc[d, "Close"])
                    if entry_px <= 0:
                        continue
                    # Compute SL / target using factors at this date
                    pick_row = picks[picks["symbol"] == sym].iloc[0]
                    atr_pct_v = float(pick_row.get("atr_pct", 0.03))
                    stop_pct = max(2.0 * atr_pct_v, cfg.min_stop_pct)
                    stop_pct = min(stop_pct, cfg.max_stop_pct)
                    sl = entry_px * (1.0 - stop_pct)
                    target = entry_px * (1.0 + stop_pct * cfg.reward_risk_ratio)

                    eff_entry = entry_px * cost_factor_buy
                    qty = int(budget_per_slot / eff_entry)
                    if qty < 1:
                        continue
                    cost = qty * eff_entry
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[sym] = Position(
                        symbol=sym,
                        entry_date=d,
                        entry_px=entry_px,
                        stop_loss=sl,
                        target=target,
                        qty=qty,
                    )

    # Close any remaining positions at last date
    last_date = trading_dates[-1]
    for sym, pos in list(positions.items()):
        df = history.get(sym)
        if df is None or last_date not in df.index:
            continue
        close_px = float(df.loc[last_date, "Close"])
        sell_proceeds = pos.qty * close_px * cost_factor_sell
        buy_cost = pos.qty * pos.entry_px * cost_factor_buy
        net_pnl = sell_proceeds - buy_cost
        cash += sell_proceeds
        trades.append(TradeRecord(
            symbol=sym, entry_date=pos.entry_date, exit_date=last_date,
            entry_px=pos.entry_px, exit_px=close_px, qty=pos.qty,
            bars_held=pos.bars_held, pnl_gross=(close_px - pos.entry_px) * pos.qty,
            pnl_net=net_pnl, pnl_pct=(close_px / pos.entry_px) - 1.0,
            exit_reason="end_of_window",
        ))
        positions.pop(sym)

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trades])

    return {
        "equity_curve": eq_df,
        "trades": trades_df,
        "final_equity": float(cash),
        "stats": compute_stats(eq_df, trades_df, cfg.initial_capital),
    }


def compute_stats(eq_df: pd.DataFrame, trades_df: pd.DataFrame, init_cap: float) -> dict:
    """Compute summary statistics."""
    if eq_df.empty:
        return {}

    eq = eq_df["equity"]
    final = float(eq.iloc[-1])
    total_return = final / init_cap - 1.0

    n_days = (eq.index[-1] - eq.index[0]).days
    years = max(n_days / 365.25, 1e-6)
    cagr = (final / init_cap) ** (1.0 / years) - 1.0

    daily_ret = eq.pct_change().dropna()
    sharpe = float(np.sqrt(252) * daily_ret.mean() / daily_ret.std()) if daily_ret.std() > 0 else 0.0

    running_max = eq.cummax()
    dd = (eq / running_max) - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("inf")

    if not trades_df.empty:
        wins = trades_df[trades_df["pnl_net"] > 0]
        losses = trades_df[trades_df["pnl_net"] <= 0]
        n_trades = len(trades_df)
        win_rate = len(wins) / n_trades if n_trades else 0.0
        avg_win = float(wins["pnl_pct"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl_pct"].mean()) if len(losses) else 0.0
        avg_hold = float(trades_df["bars_held"].mean())
        gross_profit = float(wins["pnl_net"].sum()) if len(wins) else 0.0
        gross_loss = float(losses["pnl_net"].sum()) if len(losses) else 0.0
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()
    else:
        n_trades = win_rate = avg_win = avg_loss = avg_hold = profit_factor = 0.0
        exit_reasons = {}

    return {
        "initial_capital": init_cap,
        "final_equity": final,
        "total_return_pct": total_return,
        "cagr_pct": cagr,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "calmar": calmar,
        "n_trades": int(n_trades) if n_trades else 0,
        "win_rate_pct": win_rate,
        "avg_win_pct": avg_win,
        "avg_loss_pct": avg_loss,
        "avg_hold_bars": avg_hold,
        "profit_factor": profit_factor,
        "exit_reasons": exit_reasons,
        "n_days": n_days,
        "years": round(years, 2),
    }
