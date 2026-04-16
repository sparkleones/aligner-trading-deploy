"""
Model Comparison — Tests ALL model variants on multiple date ranges.

Runs 5 model configurations:
  1. BASELINE:     8-rule scoring, no filters, CALL+PUT, all days
  2. MULTI_TF:     9-rule (+ Multi-TF alignment), no filters
  3. V2_OPTIMIZED: 9-rule + loss-reduction filters (PUT-only, Tue/Thu/Fri, no 0.6x conf, VIX>=10)
  4. PUT_ONLY:     8-rule, PUT-only (no other filters)
  5. DAY_FILTER:   8-rule, day filter only (Tue/Thu/Fri)

Tests each on:
  - Full 6-month period (Oct 2025 – Apr 2026)
  - April 2025 out-of-sample
  - Last 3 months (Jan – Apr 2026)

Also runs the 3 individual strategy agents (iron_condor, bull_put_spread, ddqn)
via the real_data_backtest framework for comparison.
"""

import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtesting.option_pricer import price_option
from backtesting.paper_trading import PaperTradingBroker
from config.constants import INDEX_CONFIG

LOT_SIZE = INDEX_CONFIG["NIFTY"]["lot_size"]  # 65
BROKERAGE = 20.0
STRIKE_INTERVAL = 50
CAPITAL = 200_000
TOTAL_BARS = 25


# ===========================================================================
# DATA DOWNLOAD (reused from paper_trading_real_data.py)
# ===========================================================================

def download_data(start, end):
    """Download real NIFTY + VIX data from Yahoo Finance."""
    import yfinance as yf

    # Need extra warmup for SMA50
    warmup_start = pd.Timestamp(start) - pd.Timedelta(days=90)
    warmup_str = warmup_start.strftime("%Y-%m-%d")

    nifty = yf.download("^NSEI", start=warmup_str, end=end, interval="1d", progress=False)
    vix_data = yf.download("^INDIAVIX", start=warmup_str, end=end, interval="1d", progress=False)

    if nifty.empty:
        raise ValueError(f"Failed to download NIFTY data for {start} to {end}")

    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = nifty.columns.get_level_values(0)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)

    nifty["VIX"] = vix_data["Close"] if not vix_data.empty else 14.0
    nifty["VIX"] = nifty["VIX"].ffill().bfill().fillna(14.0)
    nifty["PrevVIX"] = nifty["VIX"].shift(1).fillna(nifty["VIX"].iloc[0])
    nifty["Change%"] = nifty["Close"].pct_change() * 100
    nifty["PrevChange%"] = nifty["Change%"].shift(1).fillna(0)
    nifty["DOW"] = nifty.index.day_name()
    nifty["SMA50"] = nifty["Close"].rolling(50).mean()
    nifty["SMA20"] = nifty["Close"].rolling(20).mean()
    nifty["AboveSMA50"] = nifty["Close"] > nifty["SMA50"]
    nifty["AboveSMA20"] = nifty["Close"] > nifty["SMA20"]
    nifty["EMA9"] = nifty["Close"].ewm(span=9).mean()
    nifty["EMA21"] = nifty["Close"].ewm(span=21).mean()
    nifty["WeeklySMA"] = nifty["Close"].rolling(5).mean().rolling(4).mean()

    delta = nifty["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    nifty["RSI"] = 100 - (100 / (1 + gain / loss.replace(0, 0.001)))
    nifty["PrevHigh"] = nifty["High"].shift(1)
    nifty["PrevLow"] = nifty["Low"].shift(1)
    nifty["VIXSpike"] = nifty["VIX"] > nifty["PrevVIX"] * 1.15

    def _vix_regime(v):
        if v < 10: return "VERY_LOW"
        elif v < 13: return "LOW"
        elif v < 17: return "NORMAL_LOW"
        elif v < 20: return "NORMAL_HIGH"
        elif v < 25: return "HIGH"
        else: return "VERY_HIGH"
    nifty["VIXRegime"] = nifty["VIX"].apply(_vix_regime)

    nifty["ATH"] = nifty["High"].expanding().max()
    nifty["DrawdownFromATH"] = (nifty["ATH"] - nifty["Close"]) / nifty["ATH"] * 100
    nifty["PrevRSI"] = nifty["RSI"].shift(1)
    nifty["PrevClose"] = nifty["Close"].shift(1)
    nifty["DayOfMonth"] = nifty.index.day
    nifty["IsFirstWeek"] = nifty["DayOfMonth"] <= 7
    nifty["Mom3d"] = nifty["Close"].pct_change(3) * 100

    # Expiry: Tuesday for NIFTY (SEBI Nov 2025), Thursday before that
    nifty["IsExpiry"] = nifty.index.map(
        lambda d: d.strftime("%A") == ("Tuesday" if d >= pd.Timestamp("2025-11-01") else "Thursday")
    )

    dte_values = []
    for idx in nifty.index:
        current_dow = idx.weekday()
        # Tuesday expiry after Nov 2025, Thursday before
        if idx >= pd.Timestamp("2025-11-01"):
            target = 1  # Tuesday
        else:
            target = 3  # Thursday
        if current_dow <= target:
            dte = target - current_dow
        else:
            dte = 7 - current_dow + target
        dte_values.append(max(dte, 0.5))
    nifty["DTE"] = dte_values

    valid_start = nifty["SMA50"].first_valid_index()
    if valid_start is not None:
        nifty = nifty.loc[valid_start:]

    # Trim to requested range
    nifty = nifty.loc[start:end]

    return nifty


# ===========================================================================
# INTRADAY PATH + S/R + OPTION PRICING (reused)
# ===========================================================================

def generate_intraday_path(open_p, high, low, close, n_bars=TOTAL_BARS):
    path = [open_p]
    np.random.seed(int(abs(open_p * 100)) % 2**31)
    up = close > open_p
    if up:
        lb = max(1, int(n_bars * 0.15))
        hb = max(lb + 2, int(n_bars * 0.7))
        for i in range(1, n_bars):
            if i <= lb:
                t = i / lb; target = open_p + (low - open_p) * t
            elif i <= hb:
                t = (i - lb) / (hb - lb); target = low + (high - low) * t
            else:
                t = (i - hb) / (n_bars - hb); target = high + (close - high) * t
            path.append(target + target * 0.0002 * np.random.randn())
    else:
        hb = max(1, int(n_bars * 0.15))
        lb = max(hb + 2, int(n_bars * 0.7))
        for i in range(1, n_bars):
            if i <= hb:
                t = i / hb; target = open_p + (high - open_p) * t
            elif i <= lb:
                t = (i - hb) / (lb - hb); target = high + (low - high) * t
            else:
                t = (i - lb) / (n_bars - lb); target = low + (close - low) * t
            path.append(target + target * 0.0002 * np.random.randn())
    return path


def sr_multi_method(spot, prev_high, prev_low, sma20, sma50):
    support_cands = []
    resist_cands = []

    for level in range(int(spot // 500) * 500 - 1500, int(spot // 500) * 500 + 2000, 500):
        if level < spot: support_cands.append((level, 3.0))
        elif level > spot: resist_cands.append((level, 3.0))
    for level in range(int(spot // 100) * 100 - 500, int(spot // 100) * 100 + 600, 100):
        if level % 500 != 0:
            if level < spot: support_cands.append((level, 1.5))
            elif level > spot: resist_cands.append((level, 1.5))

    if prev_low is not None and prev_high is not None:
        pdl = round(prev_low / 50) * 50
        pdh = round(prev_high / 50) * 50
        if pdl < spot: support_cands.append((pdl, 2.5))
        if pdh > spot: resist_cands.append((pdh, 2.5))

    if sma20 and not np.isnan(sma20):
        if sma20 < spot: support_cands.append((round(sma20 / 50) * 50, 1.5))
        elif sma20 > spot: resist_cands.append((round(sma20 / 50) * 50, 1.5))
    if sma50 and not np.isnan(sma50):
        if sma50 < spot: support_cands.append((round(sma50 / 50) * 50, 1.5))
        elif sma50 > spot: resist_cands.append((round(sma50 / 50) * 50, 1.5))

    if support_cands:
        support_cands.sort(key=lambda x: (spot - x[0], -x[1]))
        support = support_cands[0][0]
    else:
        support = round((spot * 0.99) / 50) * 50
    if resist_cands:
        resist_cands.sort(key=lambda x: (x[0] - spot, -x[1]))
        resistance = resist_cands[0][0]
    else:
        resistance = round((spot * 1.01) / 50) * 50
    return support, resistance


def bs_premium(spot, strike, dte, vix, opt_type):
    try:
        return price_option(spot=spot, strike=strike, dte_days=dte,
                            vix=vix, option_type=opt_type)["premium"]
    except Exception:
        return 30.0


# ===========================================================================
# MODEL CONFIGURATIONS
# ===========================================================================

def compute_scores_baseline(vix, above_sma50, above_sma20, rsi, dow,
                            prev_change, vix_spike, spot, support, resistance,
                            ema9=None, ema21=None, weekly_sma=None):
    """8-rule v1 baseline scoring (no Multi-TF)."""
    scores = {"BUY_CALL": 0.0, "BUY_PUT": 0.0}

    # Rule 1: VIX 4-level
    if vix < 12: scores["BUY_CALL"] += 3.0
    elif vix < 17: scores["BUY_PUT"] += 3.0
    elif vix < 25: scores["BUY_PUT"] += 3.5
    else: scores["BUY_PUT"] += 4.0

    # Rule 2: SMA50
    if not above_sma50: scores["BUY_PUT"] += 2.0
    else: scores["BUY_CALL"] += 2.0

    # Rule 3: SMA20
    if not above_sma20: scores["BUY_PUT"] += 1.0
    else: scores["BUY_CALL"] += 1.0

    # Rule 4: RSI
    if rsi < 30: scores["BUY_PUT"] += 1.5
    elif rsi > 70: scores["BUY_PUT"] += 1.5

    # Rule 5: DOW
    dow_rules = {"Monday": "BUY_PUT", "Tuesday": "BUY_PUT",
                 "Wednesday": "BUY_CALL", "Thursday": "BUY_PUT",
                 "Friday": "BUY_CALL"}
    dow_action = dow_rules.get(dow)
    if dow_action: scores[dow_action] += 0.5

    # Rule 6: VIX spike
    if vix_spike: scores["BUY_CALL"] += 2.0

    # Rule 7: Prev momentum
    if prev_change < -1.0: scores["BUY_CALL"] += 1.0
    elif prev_change > 1.0: scores["BUY_PUT"] += 1.0

    # Rule 8: S/R proximity
    if support and spot:
        d = (spot - support) / spot * 100
        if 0 < d < 1.0: scores["BUY_CALL"] += 1.0
        elif d < 0: scores["BUY_PUT"] += 1.0
    if resistance and spot:
        d = (resistance - spot) / spot * 100
        if 0 < d < 1.0: scores["BUY_PUT"] += 1.0
        elif d < 0: scores["BUY_CALL"] += 1.0

    return scores


def compute_scores_multitf(vix, above_sma50, above_sma20, rsi, dow,
                           prev_change, vix_spike, spot, support, resistance,
                           ema9=None, ema21=None, weekly_sma=None):
    """9-rule scoring (baseline + Multi-TF alignment)."""
    scores = compute_scores_baseline(vix, above_sma50, above_sma20, rsi, dow,
                                     prev_change, vix_spike, spot, support, resistance)
    # Rule 9: Multi-TF alignment
    if ema9 is not None and ema21 is not None and weekly_sma is not None:
        alignment = 0
        if spot > weekly_sma: alignment += 1
        if ema9 > ema21: alignment += 1
        best = max(scores, key=scores.get)
        if best == "BUY_CALL" and alignment == 0:
            scores["BUY_CALL"] *= 0.5
        elif best == "BUY_PUT" and alignment == 2:
            scores["BUY_PUT"] *= 0.5
    return scores


# ── Model configurations ──

MODELS = {
    "1_BASELINE": {
        "desc": "8-rule v1, no filters, CALL+PUT, all days",
        "scoring": "baseline",
        "put_only": False,
        "skip_days": set(),
        "conf_death_zone": False,
        "min_vix": 0,
        "put_exit": "sr_trail_combo",
    },
    "2_FULL_ENSEMBLE": {
        "desc": "D_FULL_ENSEMBLE: 9-rule, trail_pct PUT exit, overnight hold (+647% original)",
        "scoring": "multitf",
        "put_only": False,
        "skip_days": set(),
        "conf_death_zone": False,
        "min_vix": 0,
        "put_exit": "trail_pct",  # The key difference: trail_pct for PUT (Sharpe 5.22)
    },
    "3_MULTI_TF": {
        "desc": "9-rule (+ Multi-TF), sr_trail_combo both, no filters",
        "scoring": "multitf",
        "put_only": False,
        "skip_days": set(),
        "conf_death_zone": False,
        "min_vix": 0,
        "put_exit": "sr_trail_combo",
    },
    "4_PUT_ONLY": {
        "desc": "9-rule, PUT-only (no other filters)",
        "scoring": "multitf",
        "put_only": True,
        "skip_days": set(),
        "conf_death_zone": False,
        "min_vix": 0,
        "put_exit": "sr_trail_combo",
    },
    "5_DAY_FILTER": {
        "desc": "9-rule, Tue/Thu/Fri only (no other filters)",
        "scoring": "multitf",
        "put_only": False,
        "skip_days": {"Monday", "Wednesday"},
        "conf_death_zone": False,
        "min_vix": 0,
        "put_exit": "sr_trail_combo",
    },
    "6_V2_OPTIMIZED": {
        "desc": "9-rule + ALL filters (PUT-only, Tue/Thu/Fri, no 0.6x, VIX>=10)",
        "scoring": "multitf",
        "put_only": True,
        "skip_days": {"Monday", "Wednesday"},
        "conf_death_zone": True,
        "min_vix": 10.0,
        "put_exit": "sr_trail_combo",
    },
    "7_V2+TRAIL_PCT": {
        "desc": "V2 filters + trail_pct PUT exit (best of both)",
        "scoring": "multitf",
        "put_only": True,
        "skip_days": {"Monday", "Wednesday"},
        "conf_death_zone": True,
        "min_vix": 10.0,
        "put_exit": "trail_pct",
    },
}


# ===========================================================================
# SIMULATION ENGINE
# ===========================================================================

def simulate_model(nifty, model_config):
    """Run one model configuration on the given data.

    Returns dict with: return_pct, max_dd, sharpe, profit_factor,
    num_trades, win_rate, total_pnl, monthly, trades list.
    """
    scoring_fn = (compute_scores_multitf if model_config["scoring"] == "multitf"
                  else compute_scores_baseline)
    put_only = model_config["put_only"]
    skip_days = model_config["skip_days"]
    conf_death = model_config["conf_death_zone"]
    min_vix = model_config["min_vix"]
    put_exit = model_config.get("put_exit", "sr_trail_combo")

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    trades = []
    monthly_pnl = defaultdict(float)
    close_prices = nifty["Close"].values.tolist()

    for i in range(len(nifty)):
        row = nifty.iloc[i]
        date_str = str(nifty.index[i].date())
        month_key = date_str[:7]

        entry_spot = float(row["Open"])
        day_high = float(row["High"])
        day_low = float(row["Low"])
        day_close = float(row["Close"])
        vix = float(row["VIX"]) if pd.notna(row["VIX"]) else 14.0
        dow = str(row["DOW"])
        above_sma50 = bool(row["AboveSMA50"]) if pd.notna(row.get("AboveSMA50")) else True
        above_sma20 = bool(row["AboveSMA20"]) if pd.notna(row.get("AboveSMA20")) else True
        rsi = float(row["RSI"]) if pd.notna(row.get("RSI")) else 50
        prev_change = float(row["PrevChange%"]) if pd.notna(row.get("PrevChange%")) else 0
        vix_spike = bool(row["VIXSpike"]) if pd.notna(row.get("VIXSpike")) else False
        sma20 = float(row["SMA20"]) if pd.notna(row.get("SMA20")) else None
        sma50 = float(row["SMA50"]) if pd.notna(row.get("SMA50")) else None
        prev_high = float(row["PrevHigh"]) if pd.notna(row.get("PrevHigh")) else day_high
        prev_low = float(row["PrevLow"]) if pd.notna(row.get("PrevLow")) else day_low
        is_expiry = bool(row.get("IsExpiry", False))
        dte_market = float(row.get("DTE", 2.0))
        ema9 = float(row["EMA9"]) if pd.notna(row.get("EMA9")) else None
        ema21 = float(row["EMA21"]) if pd.notna(row.get("EMA21")) else None
        weekly_sma = float(row["WeeklySMA"]) if pd.notna(row.get("WeeklySMA")) else None

        support, resistance = sr_multi_method(entry_spot, prev_high, prev_low, sma20, sma50)

        scores = scoring_fn(
            vix, above_sma50, above_sma20, rsi, dow, prev_change, vix_spike,
            entry_spot, support, resistance,
            ema9=ema9, ema21=ema21, weekly_sma=weekly_sma,
        )
        best_action = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_action] / total_score if total_score > 0 else 0

        # ── Filters ──
        if confidence < 0.25:
            continue
        if skip_days and dow in skip_days:
            continue
        if conf_death and 0.60 <= confidence < 0.70:
            continue
        if put_only and best_action == "BUY_CALL":
            continue
        if min_vix > 0 and vix < min_vix:
            continue

        # Timing gate
        entry_bar = 3 if best_action == "BUY_PUT" else 5
        if entry_bar < 2 or entry_bar > 13:
            continue

        # Strike + sizing
        if best_action == "BUY_CALL":
            offset = -50 if vix < 12 else (100 if vix < 20 else 150)
        else:
            offset = 0 if vix < 12 else 50
        strike = round(entry_spot / 50) * 50 + offset
        opt_type = "CE" if best_action == "BUY_CALL" else "PE"

        # Position sizing (fixed capital base)
        if vix < 12: vix_mult = 2.0
        elif vix < 15: vix_mult = 1.5
        elif vix < 20: vix_mult = 1.0
        elif vix < 25: vix_mult = 0.7
        elif vix < 30: vix_mult = 0.5
        else: vix_mult = 0.3
        base_lots = max(1, int(CAPITAL * 0.08 / (50 * LOT_SIZE)))
        num_lots = min(5, max(1, int(base_lots * vix_mult)))
        qty = min(num_lots * LOT_SIZE, 1800)

        # Intraday simulation
        np.random.seed(int(abs(entry_spot * 100)) % 2**31 + i)
        path = generate_intraday_path(entry_spot, day_high, day_low, day_close)

        dte_entry = max(0.5, dte_market)
        entry_prem = bs_premium(path[entry_bar], strike, dte_entry, vix, opt_type)

        # Determine exit strategy for this action
        # CALL always uses sr_trail_combo; PUT depends on config
        if best_action == "BUY_PUT":
            exit_strat = put_exit
        else:
            exit_strat = "sr_trail_combo"

        best_fav = path[entry_bar]
        exit_bar = TOTAL_BARS - 1
        exit_spot = day_close
        exit_reason = "eod_close"
        sr_target_hit = False
        trail_dist = entry_spot * 0.003

        for bi in range(entry_bar + 1, TOTAL_BARS):
            bar_spot = path[bi]

            if best_action == "BUY_CALL" and bar_spot > best_fav:
                best_fav = bar_spot
            elif best_action == "BUY_PUT" and bar_spot < best_fav:
                best_fav = bar_spot

            # Expiry handling
            if is_expiry:
                bar_prem = bs_premium(bar_spot, strike, max(0.05, dte_entry - bi * 15 / 1440), vix, opt_type)
                cpnl = (bar_prem - entry_prem) * qty
                if cpnl <= 0 and bi >= 20:
                    exit_bar, exit_spot, exit_reason = bi, bar_spot, "expiry_stop_loser"
                    break
                elif bi >= 24:
                    exit_bar, exit_spot, exit_reason = bi, bar_spot, "expiry_late_exit"
                    break

            # ── EXIT: trail_pct (simple trailing, best for BUY_PUT Sharpe 5.22) ──
            if exit_strat == "trail_pct":
                if bi > entry_bar + 3:  # min 3 bars before trail activates
                    if best_action == "BUY_PUT":
                        if bar_spot > best_fav + trail_dist:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "trail_pct"
                            break
                    else:
                        if bar_spot < best_fav - trail_dist:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "trail_pct"
                            break

            # ── EXIT: sr_trail_combo (S/R stop before target, trail after) ──
            elif exit_strat == "sr_trail_combo":
                if not sr_target_hit:
                    if best_action == "BUY_CALL":
                        if resistance and bar_spot >= resistance:
                            sr_target_hit = True; best_fav = bar_spot
                        if support and bar_spot < support:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "sr_stop"
                            break
                    else:
                        if support and bar_spot <= support:
                            sr_target_hit = True; best_fav = bar_spot
                        if resistance and bar_spot > resistance:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "sr_stop"
                            break
                else:
                    if best_action == "BUY_CALL":
                        if bar_spot < best_fav - trail_dist:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "sr_combo_trail"
                            break
                    else:
                        if bar_spot > best_fav + trail_dist:
                            exit_bar, exit_spot, exit_reason = bi, bar_spot, "sr_combo_trail"
                            break

        exit_dte = max(0.05, dte_entry - exit_bar * 15 / 1440)
        exit_prem = bs_premium(exit_spot, strike, exit_dte, vix, opt_type)
        intraday_pnl = (exit_prem - entry_prem) * qty - 80  # approx costs

        # Overnight for profitable PUTs
        overnight_pnl = 0
        if (best_action == "BUY_PUT" and intraday_pnl > 0
                and not is_expiry and i + 1 < len(nifty)):
            next_row = nifty.iloc[i + 1]
            next_open = float(next_row["Open"])
            gap = (next_open - day_close) / day_close * 100
            if gap < 0:
                overnight_pnl = (day_close - next_open) * qty * 0.5 - 50
                overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)
            else:
                overnight_pnl = -abs(gap) * qty * 0.3
                overnight_pnl = max(overnight_pnl, -intraday_pnl * 0.5)

        total_pnl = intraday_pnl + overnight_pnl
        equity += total_pnl
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd: max_dd = dd

        monthly_pnl[month_key] += total_pnl
        trades.append({
            "date": date_str, "dow": dow, "action": best_action,
            "confidence": round(confidence, 2), "vix": round(vix, 1),
            "pnl": round(total_pnl, 0), "exit_reason": exit_reason,
        })

    # Stats
    total_pnl = equity - CAPITAL
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] < 0]
    pnls = [t["pnl"] for t in trades]
    avg = np.mean(pnls) if pnls else 0
    std = np.std(pnls) if pnls else 1
    sharpe = (avg / std) * np.sqrt(252) if std > 0 else 0
    gross_w = sum(t["pnl"] for t in wins)
    gross_l = abs(sum(t["pnl"] for t in losses))
    pf = gross_w / max(1, gross_l)
    wr = len(wins) / max(1, len(trades)) * 100

    return {
        "return_pct": round(total_pnl / CAPITAL * 100, 2),
        "total_pnl": round(total_pnl, 0),
        "max_dd": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "profit_factor": round(pf, 2),
        "num_trades": len(trades),
        "win_rate": round(wr, 1),
        "wins": len(wins),
        "losses": len(losses),
        "avg_pnl": round(avg, 0),
        "best_trade": round(max(pnls), 0) if pnls else 0,
        "worst_trade": round(min(pnls), 0) if pnls else 0,
        "monthly": dict(monthly_pnl),
        "all_months_positive": all(v > 0 for v in monthly_pnl.values()) if monthly_pnl else False,
        "trades": trades,
    }


# ===========================================================================
# MAIN — RUN ALL MODELS ON ALL PERIODS
# ===========================================================================

def run_comparison():
    """Run all 5 model configurations on multiple date ranges."""
    print("=" * 90)
    print("  MODEL COMPARISON -- ALL VARIANTS ON MULTIPLE DATE RANGES")
    print("  Testing 5 model configurations on 3 date periods")
    print("=" * 90)

    # Define test periods
    periods = {
        "Apr 2025 (OOS)": ("2025-04-01", "2025-05-01"),
        "Oct25-Apr26 (6M)": ("2025-10-01", "2026-04-06"),
        "Jan-Apr 2026 (3M)": ("2026-01-01", "2026-04-06"),
    }

    all_results = {}

    for period_name, (start, end) in periods.items():
        print(f"\n{'='*90}")
        print(f"  PERIOD: {period_name} ({start} to {end})")
        print(f"{'='*90}")

        try:
            nifty = download_data(start, end)
            print(f"  Data: {len(nifty)} trading days | "
                  f"NIFTY {nifty['Close'].min():.0f}-{nifty['Close'].max():.0f} | "
                  f"VIX {nifty['VIX'].min():.1f}-{nifty['VIX'].max():.1f}")
        except Exception as e:
            print(f"  ERROR downloading data: {e}")
            continue

        period_results = {}
        for model_name, config in MODELS.items():
            result = simulate_model(nifty, config)
            period_results[model_name] = result

        all_results[period_name] = period_results

        # Print comparison table for this period
        print(f"\n  {'Model':<22s} | {'Return':>8s} | {'MaxDD':>6s} | {'Sharpe':>6s} | "
              f"{'PF':>5s} | {'Trades':>6s} | {'WR':>5s} | {'P&L':>12s} | {'AllM+':>5s}")
        print(f"  {'-'*22}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*5}-+-{'-'*6}-+-{'-'*5}-+-{'-'*12}-+-{'-'*5}")

        for model_name, r in period_results.items():
            allm = "YES" if r["all_months_positive"] else "NO"
            print(f"  {model_name:<22s} | {r['return_pct']:>+7.1f}% | "
                  f"{r['max_dd']:>5.1f}% | {r['sharpe']:>6.2f} | "
                  f"{r['profit_factor']:>5.2f} | {r['num_trades']:>6d} | "
                  f"{r['win_rate']:>4.0f}% | Rs {r['total_pnl']:>+10,.0f} | {allm:>5s}")

        # Monthly breakdown for best model
        best_model = max(period_results.items(),
                         key=lambda x: x[1]["sharpe"] * x[1]["profit_factor"])
        print(f"\n  BEST: {best_model[0]} (Sharpe*PF = "
              f"{best_model[1]['sharpe'] * best_model[1]['profit_factor']:.1f})")
        if best_model[1]["monthly"]:
            print(f"  Monthly P&L:")
            for m, p in sorted(best_model[1]["monthly"].items()):
                bar = ("+" if p > 0 else "-") * min(30, max(1, int(abs(p) / 10000)))
                print(f"    {m}: Rs {p:>+10,.0f}  {bar}")

    # ── OVERALL COMPARISON ACROSS ALL PERIODS ──
    print(f"\n\n{'='*90}")
    print(f"  OVERALL RANKING (averaged across all periods)")
    print(f"{'='*90}")

    model_scores = defaultdict(lambda: {"returns": [], "sharpes": [], "pfs": [], "dds": [], "wrs": []})
    for period_name, period_results in all_results.items():
        for model_name, r in period_results.items():
            model_scores[model_name]["returns"].append(r["return_pct"])
            model_scores[model_name]["sharpes"].append(r["sharpe"])
            model_scores[model_name]["pfs"].append(r["profit_factor"])
            model_scores[model_name]["dds"].append(r["max_dd"])
            model_scores[model_name]["wrs"].append(r["win_rate"])

    print(f"\n  {'Model':<22s} | {'AvgReturn':>10s} | {'AvgSharpe':>9s} | "
          f"{'AvgPF':>6s} | {'AvgDD':>6s} | {'AvgWR':>5s} | {'Score':>8s}")
    print(f"  {'-'*22}-+-{'-'*10}-+-{'-'*9}-+-"
          f"{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}")

    ranked = []
    for model_name, scores in sorted(model_scores.items()):
        avg_ret = np.mean(scores["returns"])
        avg_sh = np.mean(scores["sharpes"])
        avg_pf = np.mean(scores["pfs"])
        avg_dd = np.mean(scores["dds"])
        avg_wr = np.mean(scores["wrs"])
        # Combined score: Sharpe * PF, penalize high DD
        combo = avg_sh * avg_pf * (1.0 if avg_dd < 25 else 0.7)
        ranked.append((model_name, avg_ret, avg_sh, avg_pf, avg_dd, avg_wr, combo))

    ranked.sort(key=lambda x: -x[6])
    for r in ranked:
        medal = ">>>" if r == ranked[0] else "   "
        print(f"{medal}{r[0]:<22s} | {r[1]:>+9.1f}% | {r[2]:>9.2f} | "
              f"{r[3]:>6.2f} | {r[4]:>5.1f}% | {r[5]:>4.0f}% | {r[6]:>8.1f}")

    print(f"\n  WINNER: {ranked[0][0]}")
    print(f"  Combined Score (Sharpe x PF): {ranked[0][6]:.1f}")
    print(f"  Avg Return: {ranked[0][1]:+.1f}% | Avg Sharpe: {ranked[0][2]:.2f} | "
          f"Avg Max DD: {ranked[0][4]:.1f}%")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "periods": {},
    }
    for period_name, period_results in all_results.items():
        output["periods"][period_name] = {
            model_name: {k: v for k, v in r.items() if k != "trades"}
            for model_name, r in period_results.items()
        }
    output["ranking"] = [
        {"model": r[0], "avg_return": r[1], "avg_sharpe": r[2],
         "avg_pf": r[3], "avg_dd": r[4], "avg_wr": r[5], "score": r[6]}
        for r in ranked
    ]

    out_path = project_root / "data" / "model_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 90)

    return output


if __name__ == "__main__":
    run_comparison()
