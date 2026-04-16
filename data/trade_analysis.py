"""
Comprehensive Trade-by-Trade Analysis of V3 Paper Trading Results
Research only - does not modify any production files
"""
import json
from collections import defaultdict
from itertools import combinations

def load_data():
    with open("C:/Users/ssura/Desktop/Aligner/Trading/data/paper_trading_realdata_results.json") as f:
        return json.load(f)

def fmt(val):
    """Format rupee values"""
    if val >= 0:
        return f"Rs +{val:,.0f}"
    return f"Rs {val:,.0f}"

def pct(num, den):
    return f"{100*num/den:.1f}%" if den > 0 else "N/A"

def section(title):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

def main():
    data = load_data()
    trades = data["trades"]
    daily = data["daily_results"]

    N = len(trades)
    wins = [t for t in trades if t["total_pnl"] > 0]
    losses = [t for t in trades if t["total_pnl"] <= 0]
    total_pnl = sum(t["total_pnl"] for t in trades)

    print("=" * 80)
    print("  COMPREHENSIVE TRADE-BY-TRADE ANALYSIS")
    print(f"  {N} trades | {len(wins)} wins | {len(losses)} losses | Total P&L: {fmt(total_pnl)}")
    print("=" * 80)

    # =========================================================================
    # 1. GROUP BY ENTRY TYPE
    # =========================================================================
    section("1. ENTRY TYPE ANALYSIS")
    by_entry = defaultdict(list)
    for t in trades:
        by_entry[t["entry_type"]].append(t)

    rows = []
    for etype, tlist in sorted(by_entry.items(), key=lambda x: -sum(t["total_pnl"] for t in x[1])):
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        med_pnl = sorted([t["total_pnl"] for t in tlist])[len(tlist)//2]
        best = max(t["total_pnl"] for t in tlist)
        worst = min(t["total_pnl"] for t in tlist)
        rows.append((etype, len(tlist), w, pnl, avg, med_pnl, best, worst))

    print(f"  {'Entry Type':<25} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12} {'Median':>12} {'Best':>12} {'Worst':>12}")
    print(f"  {'-'*25} {'---':>4} {'-----':>6} {'-'*14} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    for etype, n, w, pnl, avg, med, best, worst in rows:
        print(f"  {etype:<25} {n:>4} {pct(w,n):>6} {fmt(pnl):>14} {fmt(avg):>12} {fmt(med):>12} {fmt(best):>12} {fmt(worst):>12}")

    # =========================================================================
    # 2. GROUP BY EXIT REASON
    # =========================================================================
    section("2. EXIT REASON ANALYSIS")
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t["exit_reason"]].append(t)

    print(f"  {'Exit Reason':<22} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12} {'Avg Bars':>9} {'Best':>12} {'Worst':>12}")
    print(f"  {'-'*22} {'---':>4} {'-----':>6} {'-'*14} {'-'*12} {'-'*9} {'-'*12} {'-'*12}")
    for reason, tlist in sorted(by_exit.items(), key=lambda x: -sum(t["total_pnl"] for t in x[1])):
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        avg_bars = sum(t["exit_bar"] - t["entry_bar"] for t in tlist) / len(tlist)
        best = max(t["total_pnl"] for t in tlist)
        worst = min(t["total_pnl"] for t in tlist)
        print(f"  {reason:<22} {len(tlist):>4} {pct(w,len(tlist)):>6} {fmt(pnl):>14} {fmt(avg):>12} {avg_bars:>9.1f} {fmt(best):>12} {fmt(worst):>12}")

    # =========================================================================
    # 3. DAY OF WEEK ANALYSIS
    # =========================================================================
    section("3. DAY OF WEEK ANALYSIS")
    by_dow = defaultdict(list)
    for t in trades:
        by_dow[t["dow"]].append(t)

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    print(f"  {'Day':<12} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12} {'Best':>12} {'Worst':>12}")
    print(f"  {'-'*12} {'---':>4} {'-----':>6} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")
    for dow in dow_order:
        tlist = by_dow.get(dow, [])
        if not tlist:
            continue
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        best = max(t["total_pnl"] for t in tlist)
        worst = min(t["total_pnl"] for t in tlist)
        print(f"  {dow:<12} {len(tlist):>4} {pct(w,len(tlist)):>6} {fmt(pnl):>14} {fmt(avg):>12} {fmt(best):>12} {fmt(worst):>12}")

    # Day of week by entry type
    print(f"\n  --- Day of Week by Entry Type (Avg P&L) ---")
    entry_types_main = ["composite", "sr_bounce_support", "sr_bounce_resistance", "gap_entry"]
    header = f"  {'Day':<12}" + "".join(f"{et:>22}" for et in entry_types_main)
    print(header)
    print(f"  {'-'*12}" + "".join(f"{'-'*22}" for _ in entry_types_main))
    for dow in dow_order:
        row = f"  {dow:<12}"
        for et in entry_types_main:
            subset = [t for t in trades if t["dow"] == dow and t["entry_type"] == et]
            if subset:
                avg = sum(t["total_pnl"] for t in subset) / len(subset)
                row += f"{fmt(avg) + f' ({len(subset)})':>22}"
            else:
                row += f"{'---':>22}"
        print(row)

    # =========================================================================
    # 4. TIME OF ENTRY (BAR INDEX) ANALYSIS
    # =========================================================================
    section("4. ENTRY BAR ANALYSIS (which entry bars are most profitable)")
    by_ebar = defaultdict(list)
    for t in trades:
        by_ebar[t["entry_bar"]].append(t)

    print(f"  {'Bar':>4} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12} {'Best':>12} {'Worst':>12}")
    print(f"  {'---':>4} {'---':>4} {'-----':>6} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")
    for bar in sorted(by_ebar.keys()):
        tlist = by_ebar[bar]
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        best = max(t["total_pnl"] for t in tlist)
        worst = min(t["total_pnl"] for t in tlist)
        print(f"  {bar:>4} {len(tlist):>4} {pct(w,len(tlist)):>6} {fmt(pnl):>14} {fmt(avg):>12} {fmt(best):>12} {fmt(worst):>12}")

    # =========================================================================
    # 5. EXIT BAR ANALYSIS
    # =========================================================================
    section("5. EXIT BAR ANALYSIS (which exit bars)")
    by_xbar = defaultdict(list)
    for t in trades:
        by_xbar[t["exit_bar"]].append(t)

    print(f"  {'Bar':>4} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12}")
    print(f"  {'---':>4} {'---':>4} {'-----':>6} {'-'*14} {'-'*12}")
    for bar in sorted(by_xbar.keys()):
        tlist = by_xbar[bar]
        if len(tlist) < 2:
            continue
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        print(f"  {bar:>4} {len(tlist):>4} {pct(w,len(tlist)):>6} {fmt(pnl):>14} {fmt(avg):>12}")

    # =========================================================================
    # 6. ACTION ANALYSIS (BUY_CALL vs BUY_PUT)
    # =========================================================================
    section("6. ACTION ANALYSIS (BUY_CALL vs BUY_PUT)")
    by_action = defaultdict(list)
    for t in trades:
        by_action[t["action"]].append(t)

    for action in ["BUY_CALL", "BUY_PUT"]:
        tlist = by_action.get(action, [])
        if not tlist:
            continue
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        print(f"  {action}: {len(tlist)} trades | WR={pct(w,len(tlist))} | Total={fmt(pnl)} | Avg={fmt(avg)}")

        # Breakdown by entry type within action
        by_et = defaultdict(list)
        for t in tlist:
            by_et[t["entry_type"]].append(t)
        for et, elist in sorted(by_et.items(), key=lambda x: -sum(t["total_pnl"] for t in x[1])):
            epnl = sum(t["total_pnl"] for t in elist)
            ew = sum(1 for t in elist if t["total_pnl"] > 0)
            print(f"    {et:<25} {len(elist):>3}x WR={pct(ew,len(elist)):>6} Avg={fmt(epnl/len(elist)):>12}")

    # =========================================================================
    # 7. VIX REGIME ANALYSIS
    # =========================================================================
    section("7. VIX REGIME ANALYSIS")
    by_vix = defaultdict(list)
    for t in trades:
        by_vix[t["vix_regime"]].append(t)

    print(f"  {'VIX Regime':<12} {'#':>4} {'WR':>6} {'Total P&L':>14} {'Avg P&L':>12} {'Best':>12} {'Worst':>12}")
    print(f"  {'-'*12} {'---':>4} {'-----':>6} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")
    for regime in ["LOW", "MEDIUM", "HIGH", "EXTREME"]:
        tlist = by_vix.get(regime, [])
        if not tlist:
            continue
        pnl = sum(t["total_pnl"] for t in tlist)
        w = sum(1 for t in tlist if t["total_pnl"] > 0)
        avg = pnl / len(tlist)
        best = max(t["total_pnl"] for t in tlist)
        worst = min(t["total_pnl"] for t in tlist)
        print(f"  {regime:<12} {len(tlist):>4} {pct(w,len(tlist)):>6} {fmt(pnl):>14} {fmt(avg):>12} {fmt(best):>12} {fmt(worst):>12}")

    # Granular VIX buckets
    print(f"\n  --- Granular VIX Buckets ---")
    vix_buckets = [(0,10), (10,12), (12,14), (14,16), (16,18), (18,20), (20,25), (25,30)]
    print(f"  {'VIX Range':<12} {'#':>4} {'WR':>6} {'Avg P&L':>12}")
    print(f"  {'-'*12} {'---':>4} {'-----':>6} {'-'*12}")
    for lo, hi in vix_buckets:
        subset = [t for t in trades if lo <= t["vix"] < hi]
        if not subset:
            continue
        pnl = sum(t["total_pnl"] for t in subset)
        w = sum(1 for t in subset if t["total_pnl"] > 0)
        avg = pnl / len(subset)
        print(f"  {f'{lo}-{hi}':<12} {len(subset):>4} {pct(w,len(subset)):>6} {fmt(avg):>12}")

    # =========================================================================
    # 8. TOP 10 BEST AND WORST TRADES
    # =========================================================================
    section("8. TOP 10 BEST TRADES (Highest P&L)")
    sorted_best = sorted(trades, key=lambda t: -t["total_pnl"])[:10]
    print(f"  {'#':>3} {'Date':<12} {'Day':<10} {'Entry Type':<22} {'Action':<10} {'P&L':>14} {'Exit Reason':<18} {'VIX':>5} {'Entry':>5} {'Exit':>5}")
    print(f"  {'---':>3} {'-'*12} {'-'*10} {'-'*22} {'-'*10} {'-'*14} {'-'*18} {'-----':>5} {'-----':>5} {'-----':>5}")
    for i, t in enumerate(sorted_best, 1):
        print(f"  {i:>3} {t['date']:<12} {t['dow']:<10} {t['entry_type']:<22} {t['action']:<10} {fmt(t['total_pnl']):>14} {t['exit_reason']:<18} {t['vix']:>5.1f} {t['entry_bar']:>5} {t['exit_bar']:>5}")

    section("9. TOP 10 WORST TRADES (Lowest P&L)")
    sorted_worst = sorted(trades, key=lambda t: t["total_pnl"])[:10]
    print(f"  {'#':>3} {'Date':<12} {'Day':<10} {'Entry Type':<22} {'Action':<10} {'P&L':>14} {'Exit Reason':<18} {'VIX':>5} {'Entry':>5} {'Exit':>5}")
    print(f"  {'---':>3} {'-'*12} {'-'*10} {'-'*22} {'-'*10} {'-'*14} {'-'*18} {'-----':>5} {'-----':>5} {'-----':>5}")
    for i, t in enumerate(sorted_worst, 1):
        print(f"  {i:>3} {t['date']:<12} {t['dow']:<10} {t['entry_type']:<22} {t['action']:<10} {fmt(t['total_pnl']):>14} {t['exit_reason']:<18} {t['vix']:>5.1f} {t['entry_bar']:>5} {t['exit_bar']:>5}")

    # =========================================================================
    # 10. BTST (OVERNIGHT) ANALYSIS
    # =========================================================================
    section("10. BTST (OVERNIGHT HELD) ANALYSIS")
    btst = [t for t in trades if t.get("overnight_held")]
    intra = [t for t in trades if not t.get("overnight_held")]

    print(f"  BTST trades: {len(btst)}")
    print(f"  Intraday trades: {len(intra)}")
    if btst:
        btst_pnl = sum(t["total_pnl"] for t in btst)
        btst_w = sum(1 for t in btst if t["total_pnl"] > 0)
        btst_avg = btst_pnl / len(btst)
        btst_intra_component = sum(t["intraday_pnl"] for t in btst)
        btst_ovn_component = sum(t["overnight_pnl"] for t in btst)
        print(f"  BTST Total P&L: {fmt(btst_pnl)} | Avg: {fmt(btst_avg)} | WR: {pct(btst_w, len(btst))}")
        print(f"  BTST Intraday Component: {fmt(btst_intra_component)}")
        print(f"  BTST Overnight Component: {fmt(btst_ovn_component)}")

        print(f"\n  --- BTST Trade Details ---")
        print(f"  {'Date':<12} {'Entry Type':<22} {'Action':<10} {'Intra P&L':>12} {'O/N P&L':>12} {'Total':>12} {'Exit':>18}")
        print(f"  {'-'*12} {'-'*22} {'-'*10} {'-'*12} {'-'*12} {'-'*12} {'-'*18}")
        for t in sorted(btst, key=lambda x: -x["total_pnl"]):
            print(f"  {t['date']:<12} {t['entry_type']:<22} {t['action']:<10} {fmt(t['intraday_pnl']):>12} {fmt(t['overnight_pnl']):>12} {fmt(t['total_pnl']):>12} {t['exit_reason']:<18}")

    intra_pnl = sum(t["total_pnl"] for t in intra)
    intra_w = sum(1 for t in intra if t["total_pnl"] > 0)
    intra_avg = intra_pnl / len(intra) if intra else 0
    print(f"\n  Intraday Total P&L: {fmt(intra_pnl)} | Avg: {fmt(intra_avg)} | WR: {pct(intra_w, len(intra))}")

    # =========================================================================
    # 11. ZERO-TO-HERO ANALYSIS
    # =========================================================================
    section("11. ZERO-TO-HERO ANALYSIS")
    zh = [t for t in trades if t.get("is_zero_hero")]
    if zh:
        zh_pnl = sum(t["total_pnl"] for t in zh)
        zh_w = sum(1 for t in zh if t["total_pnl"] > 0)
        print(f"  Zero-Hero Trades: {len(zh)}")
        print(f"  Success Rate: {pct(zh_w, len(zh))}")
        print(f"  Total P&L: {fmt(zh_pnl)} | Avg: {fmt(zh_pnl/len(zh))}")
        for t in zh:
            print(f"    {t['date']} {t['dow']} | {t['entry_type']} | {t['action']} | Strike={t['strike']} | Entry Prem={t['entry_prem']} | Exit Prem={t['exit_prem']} | P&L={fmt(t['total_pnl'])} | VIX={t['vix']} | Exit: {t['exit_reason']}")
    else:
        print("  No zero-hero trades found.")

    # =========================================================================
    # 12. CONCURRENT POSITION ANALYSIS
    # =========================================================================
    section("12. CONCURRENT POSITION ANALYSIS")
    # Group trades by date
    by_date = defaultdict(list)
    for t in trades:
        by_date[t["date"]].append(t)

    concurrent_stats = defaultdict(list)
    for date, tlist in by_date.items():
        n = len(tlist)
        day_pnl = sum(t["total_pnl"] for t in tlist)
        concurrent_stats[n].append(day_pnl)

    print(f"  {'# Trades/Day':>14} {'Days':>6} {'Avg Day P&L':>14} {'Total P&L':>14} {'Best Day':>12} {'Worst Day':>12}")
    print(f"  {'-'*14} {'-'*6} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for n in sorted(concurrent_stats.keys()):
        pnls = concurrent_stats[n]
        total = sum(pnls)
        avg = total / len(pnls)
        best = max(pnls)
        worst = min(pnls)
        print(f"  {n:>14} {len(pnls):>6} {fmt(avg):>14} {fmt(total):>14} {fmt(best):>12} {fmt(worst):>12}")

    # Overlap analysis: when 2+ trades overlap in bar ranges
    print(f"\n  --- Bar-Level Overlap Analysis ---")
    overlap_count = 0
    overlap_pnl_total = 0
    no_overlap_pnl_total = 0
    no_overlap_count = 0
    for date, tlist in by_date.items():
        if len(tlist) < 2:
            no_overlap_count += len(tlist)
            no_overlap_pnl_total += sum(t["total_pnl"] for t in tlist)
            continue
        for i, t1 in enumerate(tlist):
            has_overlap = False
            for j, t2 in enumerate(tlist):
                if i >= j:
                    continue
                # Check if bars overlap
                if t1["entry_bar"] <= t2["exit_bar"] and t2["entry_bar"] <= t1["exit_bar"]:
                    has_overlap = True
            if has_overlap:
                overlap_count += 1
                overlap_pnl_total += t1["total_pnl"]
            else:
                no_overlap_count += 1
                no_overlap_pnl_total += t1["total_pnl"]

    print(f"  Trades with bar overlap:    {overlap_count:>4} | Avg P&L: {fmt(overlap_pnl_total/max(overlap_count,1))}")
    print(f"  Trades without bar overlap: {no_overlap_count:>4} | Avg P&L: {fmt(no_overlap_pnl_total/max(no_overlap_count,1))}")

    # =========================================================================
    # 13. ENTRY TYPE COMBINATIONS PER DAY
    # =========================================================================
    section("13. ENTRY TYPE COMBINATIONS PER DAY")
    combo_stats = defaultdict(list)
    for date, tlist in by_date.items():
        types = tuple(sorted(set(t["entry_type"] for t in tlist)))
        day_pnl = sum(t["total_pnl"] for t in tlist)
        combo_stats[types].append(day_pnl)

    print(f"  {'Combination':<60} {'Days':>5} {'Avg P&L':>12} {'Total P&L':>14} {'WR':>6}")
    print(f"  {'-'*60} {'-'*5} {'-'*12} {'-'*14} {'-'*6}")
    for combo, pnls in sorted(combo_stats.items(), key=lambda x: -sum(x[1])/len(x[1])):
        total = sum(pnls)
        avg = total / len(pnls)
        w = sum(1 for p in pnls if p > 0)
        combo_str = " + ".join(combo)
        print(f"  {combo_str:<60} {len(pnls):>5} {fmt(avg):>12} {fmt(total):>14} {pct(w,len(pnls)):>6}")

    # =========================================================================
    # 14. LOSS ANALYSIS
    # =========================================================================
    section("14. LOSS ANALYSIS")
    total_loss = sum(t["total_pnl"] for t in losses)

    print(f"  Total Losses: {len(losses)} trades | Total Loss Amount: {fmt(total_loss)}")

    # Losses by entry type
    print(f"\n  --- Losses by Entry Type ---")
    loss_by_entry = defaultdict(list)
    for t in losses:
        loss_by_entry[t["entry_type"]].append(t)

    print(f"  {'Entry Type':<25} {'# Losses':>9} {'% of Losses':>12} {'Loss Amount':>14} {'% of Loss $':>12} {'Avg Loss':>12}")
    print(f"  {'-'*25} {'-'*9} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
    for etype, elist in sorted(loss_by_entry.items(), key=lambda x: sum(t["total_pnl"] for t in x[1])):
        epnl = sum(t["total_pnl"] for t in elist)
        print(f"  {etype:<25} {len(elist):>9} {pct(len(elist),len(losses)):>12} {fmt(epnl):>14} {pct(abs(epnl),abs(total_loss)):>12} {fmt(epnl/len(elist)):>12}")

    # Losses by exit reason
    print(f"\n  --- Losses by Exit Reason ---")
    loss_by_exit = defaultdict(list)
    for t in losses:
        loss_by_exit[t["exit_reason"]].append(t)

    print(f"  {'Exit Reason':<22} {'# Losses':>9} {'% of Losses':>12} {'Loss Amount':>14} {'% of Loss $':>12} {'Avg Loss':>12}")
    print(f"  {'-'*22} {'-'*9} {'-'*12} {'-'*14} {'-'*12} {'-'*12}")
    for reason, elist in sorted(loss_by_exit.items(), key=lambda x: sum(t["total_pnl"] for t in x[1])):
        epnl = sum(t["total_pnl"] for t in elist)
        print(f"  {reason:<22} {len(elist):>9} {pct(len(elist),len(losses)):>12} {fmt(epnl):>14} {pct(abs(epnl),abs(total_loss)):>12} {fmt(epnl/len(elist)):>12}")

    # Losses by action
    print(f"\n  --- Losses by Action ---")
    loss_by_action = defaultdict(list)
    for t in losses:
        loss_by_action[t["action"]].append(t)
    for action, elist in loss_by_action.items():
        epnl = sum(t["total_pnl"] for t in elist)
        print(f"  {action}: {len(elist)} losses ({pct(len(elist),len(losses))}) | Loss: {fmt(epnl)} ({pct(abs(epnl),abs(total_loss))} of total loss)")

    # =========================================================================
    # 15. OPTIMAL PARAMETER ANALYSIS
    # =========================================================================
    section("15. OPTIMAL PARAMETER ANALYSIS")

    # 15a. Optimal MAX_CONCURRENT
    print(f"\n  --- Simulating MAX_CONCURRENT = 2, 3, 4 ---")
    for max_conc in [2, 3, 4]:
        sim_pnl = 0
        sim_trades = 0
        sim_wins = 0
        for date, tlist in sorted(by_date.items()):
            # Take the top max_conc trades by confidence (simulate selection)
            day_trades = sorted(tlist, key=lambda t: -t.get("confidence", 0))[:max_conc]
            for t in day_trades:
                sim_pnl += t["total_pnl"]
                sim_trades += 1
                if t["total_pnl"] > 0:
                    sim_wins += 1
        sim_avg = sim_pnl / max(sim_trades, 1)
        print(f"  MAX_CONCURRENT={max_conc}: {sim_trades} trades | P&L={fmt(sim_pnl)} | Avg={fmt(sim_avg)} | WR={pct(sim_wins, sim_trades)}")

    # 15b. Optimal COOLDOWN_BARS
    print(f"\n  --- Simulating COOLDOWN_BARS = 0, 2, 4, 6, 8 ---")
    for cooldown in [0, 2, 4, 6, 8]:
        sim_pnl = 0
        sim_trades = 0
        sim_wins = 0
        for date, tlist in sorted(by_date.items()):
            # Sort by entry_bar
            sorted_trades = sorted(tlist, key=lambda t: t["entry_bar"])
            accepted = []
            for t in sorted_trades:
                can_enter = True
                for prev in accepted:
                    if t["entry_bar"] - prev["entry_bar"] < cooldown:
                        can_enter = False
                        break
                if can_enter:
                    accepted.append(t)
            for t in accepted:
                sim_pnl += t["total_pnl"]
                sim_trades += 1
                if t["total_pnl"] > 0:
                    sim_wins += 1
        sim_avg = sim_pnl / max(sim_trades, 1)
        print(f"  COOLDOWN_BARS={cooldown}: {sim_trades} trades | P&L={fmt(sim_pnl)} | Avg={fmt(sim_avg)} | WR={pct(sim_wins, sim_trades)}")

    # 15c. Optimal max bars held
    print(f"\n  --- Bars Held Distribution ---")
    bars_held = [(t["exit_bar"] - t["entry_bar"], t["total_pnl"]) for t in trades]
    bucket_stats = defaultdict(list)
    for bh, pnl in bars_held:
        bucket = (bh // 3) * 3  # Group in buckets of 3
        bucket_stats[bucket].append(pnl)

    print(f"  {'Bars Held':>10} {'#':>4} {'WR':>6} {'Avg P&L':>12} {'Total P&L':>14}")
    print(f"  {'-'*10} {'---':>4} {'-----':>6} {'-'*12} {'-'*14}")
    for bucket in sorted(bucket_stats.keys()):
        pnls = bucket_stats[bucket]
        total = sum(pnls)
        avg = total / len(pnls)
        w = sum(1 for p in pnls if p > 0)
        label = f"{bucket}-{bucket+2}"
        print(f"  {label:>10} {len(pnls):>4} {pct(w,len(pnls)):>6} {fmt(avg):>12} {fmt(total):>14}")

    # =========================================================================
    # 16. EXPIRY DAY ANALYSIS
    # =========================================================================
    section("16. EXPIRY DAY vs NON-EXPIRY DAY")
    expiry_trades = [t for t in trades if t.get("is_expiry")]
    non_expiry = [t for t in trades if not t.get("is_expiry")]

    for label, subset in [("Expiry Day", expiry_trades), ("Non-Expiry", non_expiry)]:
        if not subset:
            continue
        pnl = sum(t["total_pnl"] for t in subset)
        w = sum(1 for t in subset if t["total_pnl"] > 0)
        avg = pnl / len(subset)
        print(f"  {label}: {len(subset)} trades | WR={pct(w,len(subset))} | Total={fmt(pnl)} | Avg={fmt(avg)}")

    # =========================================================================
    # 17. CONFIDENCE ANALYSIS
    # =========================================================================
    section("17. CONFIDENCE LEVEL ANALYSIS")
    conf_buckets = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f"  {'Confidence':>12} {'#':>4} {'WR':>6} {'Avg P&L':>12} {'Total P&L':>14}")
    print(f"  {'-'*12} {'---':>4} {'-----':>6} {'-'*12} {'-'*14}")
    for lo, hi in conf_buckets:
        subset = [t for t in trades if lo <= t.get("confidence", 0) < hi]
        if not subset:
            continue
        pnl = sum(t["total_pnl"] for t in subset)
        w = sum(1 for t in subset if t["total_pnl"] > 0)
        avg = pnl / len(subset)
        print(f"  {f'{lo:.1f}-{hi:.1f}':<12} {len(subset):>4} {pct(w,len(subset)):>6} {fmt(avg):>12} {fmt(pnl):>14}")

    # =========================================================================
    # 18. STREAK ANALYSIS
    # =========================================================================
    section("18. WIN/LOSS STREAK ANALYSIS")
    # Streak by date
    dates_sorted = sorted(by_date.keys())
    current_streak = 0
    max_win_streak = 0
    max_loss_streak = 0
    streaks_win = []
    streaks_loss = []

    for date in dates_sorted:
        day_pnl = sum(t["total_pnl"] for t in by_date[date])
        if day_pnl > 0:
            if current_streak > 0:
                current_streak += 1
            else:
                if current_streak < 0:
                    streaks_loss.append(abs(current_streak))
                current_streak = 1
        elif day_pnl < 0:
            if current_streak < 0:
                current_streak -= 1
            else:
                if current_streak > 0:
                    streaks_win.append(current_streak)
                current_streak = -1
        # day_pnl == 0: don't break streak

    if current_streak > 0:
        streaks_win.append(current_streak)
    elif current_streak < 0:
        streaks_loss.append(abs(current_streak))

    max_win_streak = max(streaks_win) if streaks_win else 0
    max_loss_streak = max(streaks_loss) if streaks_loss else 0
    avg_win_streak = sum(streaks_win) / len(streaks_win) if streaks_win else 0
    avg_loss_streak = sum(streaks_loss) / len(streaks_loss) if streaks_loss else 0

    print(f"  Max Winning Streak (days): {max_win_streak}")
    print(f"  Max Losing Streak (days):  {max_loss_streak}")
    print(f"  Avg Winning Streak:        {avg_win_streak:.1f} days")
    print(f"  Avg Losing Streak:         {avg_loss_streak:.1f} days")
    print(f"  Win Streaks: {streaks_win}")
    print(f"  Loss Streaks: {streaks_loss}")

    # =========================================================================
    # 19. RISK-REWARD ANALYSIS
    # =========================================================================
    section("19. RISK-REWARD ANALYSIS")
    avg_win = sum(t["total_pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["total_pnl"] for t in losses) / len(losses) if losses else 0

    print(f"  Average Win:  {fmt(avg_win)}")
    print(f"  Average Loss: {fmt(avg_loss)}")
    print(f"  Risk-Reward Ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "  Risk-Reward: N/A")
    print(f"  Win Rate: {pct(len(wins), N)}")
    print(f"  Expectancy per trade: {fmt(total_pnl / N)}")

    # By entry type
    print(f"\n  --- Risk-Reward by Entry Type ---")
    for etype, tlist in sorted(by_entry.items(), key=lambda x: -sum(t["total_pnl"] for t in x[1])):
        ew = [t for t in tlist if t["total_pnl"] > 0]
        el = [t for t in tlist if t["total_pnl"] <= 0]
        avg_w = sum(t["total_pnl"] for t in ew) / len(ew) if ew else 0
        avg_l = sum(t["total_pnl"] for t in el) / len(el) if el else 0
        rr = abs(avg_w / avg_l) if avg_l != 0 else float('inf')
        print(f"  {etype:<25} AvgWin={fmt(avg_w):>12} AvgLoss={fmt(avg_l):>12} RR={rr:.2f}x WR={pct(len(ew),len(tlist))}")

    # =========================================================================
    # 20. SUMMARY & RECOMMENDATIONS
    # =========================================================================
    section("20. SUMMARY & KEY FINDINGS")

    # Best entry type by avg P&L
    best_entry = max(by_entry.items(), key=lambda x: sum(t["total_pnl"] for t in x[1])/len(x[1]))
    worst_entry = min(by_entry.items(), key=lambda x: sum(t["total_pnl"] for t in x[1])/len(x[1]))

    # Best day
    best_dow = max([(dow, sum(t["total_pnl"] for t in tlist)/len(tlist)) for dow, tlist in by_dow.items()], key=lambda x: x[1])
    worst_dow = min([(dow, sum(t["total_pnl"] for t in tlist)/len(tlist)) for dow, tlist in by_dow.items()], key=lambda x: x[1])

    # Best entry bar
    best_bar = max([(bar, sum(t["total_pnl"] for t in tlist)/len(tlist)) for bar, tlist in by_ebar.items() if len(tlist) >= 3], key=lambda x: x[1])
    worst_bar = min([(bar, sum(t["total_pnl"] for t in tlist)/len(tlist)) for bar, tlist in by_ebar.items() if len(tlist) >= 3], key=lambda x: x[1])

    # Best VIX regime
    best_vix = max([(reg, sum(t["total_pnl"] for t in tlist)/len(tlist)) for reg, tlist in by_vix.items()], key=lambda x: x[1])

    # Biggest loss contributor
    biggest_loss_entry = min(loss_by_entry.items(), key=lambda x: sum(t["total_pnl"] for t in x[1]))
    biggest_loss_exit = min(loss_by_exit.items(), key=lambda x: sum(t["total_pnl"] for t in x[1]))

    print(f"""
  KEY FINDINGS:

  1. BEST ENTRY TYPE (by avg P&L): {best_entry[0]} ({fmt(sum(t['total_pnl'] for t in best_entry[1])/len(best_entry[1]))} avg)
  2. WORST ENTRY TYPE (by avg P&L): {worst_entry[0]} ({fmt(sum(t['total_pnl'] for t in worst_entry[1])/len(worst_entry[1]))} avg)
  3. BEST DAY OF WEEK: {best_dow[0]} ({fmt(best_dow[1])} avg per trade)
  4. WORST DAY OF WEEK: {worst_dow[0]} ({fmt(worst_dow[1])} avg per trade)
  5. BEST ENTRY BAR (>3 trades): Bar {best_bar[0]} ({fmt(best_bar[1])} avg)
  6. WORST ENTRY BAR (>3 trades): Bar {worst_bar[0]} ({fmt(worst_bar[1])} avg)
  7. BEST VIX REGIME: {best_vix[0]} ({fmt(best_vix[1])} avg)
  8. BIGGEST LOSS CONTRIBUTOR (entry): {biggest_loss_entry[0]} ({fmt(sum(t['total_pnl'] for t in biggest_loss_entry[1]))})
  9. BIGGEST LOSS CONTRIBUTOR (exit): {biggest_loss_exit[0]} ({fmt(sum(t['total_pnl'] for t in biggest_loss_exit[1]))})
  10. BUY_PUT outperforms BUY_CALL: PUT avg={fmt(sum(t['total_pnl'] for t in by_action['BUY_PUT'])/len(by_action['BUY_PUT']))}, CALL avg={fmt(sum(t['total_pnl'] for t in by_action['BUY_CALL'])/len(by_action['BUY_CALL']))}
  11. BTST trades: {len(btst)} total, avg={fmt(sum(t['total_pnl'] for t in btst)/len(btst)) if btst else 'N/A'}
  12. Zero-Hero: {len(zh)} trades, {'all lost' if all(t['total_pnl']<=0 for t in zh) else 'mixed'}
  13. Profit Factor: {data.get('profit_factor', 'N/A')}
  14. Max Drawdown: {data.get('max_dd_pct', 'N/A')}%
  15. Sharpe Ratio: {data.get('sharpe', 'N/A')}
""")

    print("=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
