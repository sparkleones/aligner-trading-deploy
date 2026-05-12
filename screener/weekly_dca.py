"""
Weekly DCA runner.

Runs every Monday at 09:30 IST:
  1. Loads (or creates) the DCA plan
  2. Evaluates current market triggers
  3. Computes this week's tranche size
  4. Pulls fresh screener picks via live_picks_v2
  5. Sizes individual positions for this week's tranche
  6. Sends Telegram with full plan + picks + levels
  7. Records the week in plan history

Usage:
    python -m screener.weekly_dca                     # send for real
    python -m screener.weekly_dca --dry-run           # print only
    python -m screener.weekly_dca --init --capital 100000 --tranches 4
    python -m screener.weekly_dca --status            # show current plan
    python -m screener.weekly_dca --reset             # wipe plan and start over
"""
from __future__ import annotations

import argparse
import os
import sys
import re
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifications import telegram_notifier as tg
from .dca_state import load_plan, new_plan, reset_plan, save_plan, pause_plan, record_week, DCAPlan
from .dca_triggers import evaluate as evaluate_triggers
from .live_picks_v2 import generate as generate_picks


def _format_inr(n: float) -> str:
    return f"Rs {n:,.0f}"


def format_message(plan: DCAPlan, snap, tranche: float, picks_payload: dict) -> str:
    lines = []
    lines.append("<b>WEEKLY DCA UPDATE</b>")
    lines.append(f"<i>{datetime.now().strftime('%a, %d %b %Y')}</i>")
    lines.append("")
    lines.append(f"<b>Plan status:</b> {plan.status}")
    lines.append(f"Week {plan.weeks_completed + 1} of {plan.base_tranches} (after this run)")
    lines.append(f"Total capital: {_format_inr(plan.total_capital)}")
    lines.append(f"Deployed so far: {_format_inr(plan.deployed_so_far)} ({plan.deployed_so_far/plan.total_capital*100:.0f}%)")
    lines.append("")

    # Market snapshot
    lines.append("<b>MARKET</b>")
    lines.append(f"NIFTY: {snap.nifty_close:,.0f}  |  50DMA: {snap.nifty_50dma:,.0f}  |  200DMA: {snap.nifty_200dma:,.0f}")
    lines.append(f"VIX: {snap.vix:.2f}  |  RSI(d): {snap.rsi_daily:.1f}  |  Breadth: {snap.pct_above_200dma_breadth:.0f}%")
    lines.append("")

    # Trigger summary
    if snap.accelerate_fired:
        lines.append("<b>+ ACCELERATE triggers fired:</b>")
        for t in snap.accelerate_fired:
            lines.append(f"  • {t}")
    else:
        lines.append("No accelerate triggers fired.")
    if snap.stop_fired:
        lines.append("<b>- STOP triggers fired:</b>")
        for t in snap.stop_fired:
            lines.append(f"  • {t}")
    lines.append("")

    # Decision
    lines.append(f"<b>DECISION:</b> {snap.recommended_action} (multiplier {snap.tranche_multiplier:.1f}x)")
    lines.append(f"<b>This week's tranche:</b> {_format_inr(tranche)}")
    if snap.notes:
        lines.append(f"<i>{'; '.join(snap.notes)}</i>")
    lines.append("")

    # Picks
    if tranche > 0 and picks_payload and not picks_payload.get("error"):
        picks = picks_payload.get("picks", [])
        if picks:
            lines.append("<b>DEPLOY TO THESE PICKS:</b>")
            # Reallocate this week's tranche proportionally across picks
            base_caps = [p["capital_deployed"] for p in picks]
            base_total = sum(base_caps)
            for p, base in zip(picks, base_caps):
                if base_total > 0:
                    this_week_deploy = (base / base_total) * tranche
                else:
                    this_week_deploy = tranche / len(picks)
                qty = int(this_week_deploy / max(1, p["entry"]))
                lines.append(f"  <b>{p['symbol']}</b> ({p['cap_tier']}/{p['sector']})")
                lines.append(f"     Buy {qty} @ ~Rs {p['entry']:.1f} = Rs {qty * p['entry']:,.0f}")
                lines.append(f"     SL Rs {p['stop_loss']:.1f}  Tgt Rs {p['target']:.1f}  Hold {p['hold_days']}d")
            lines.append("")
    elif tranche == 0:
        lines.append("<b>No new deployment this week.</b>")
        lines.append("")

    # Next-week criteria
    # NOTE: HTML entities (&lt; / &gt;) needed because Telegram HTML mode
    # would otherwise treat '< 16' as a malformed tag and reject the message.
    lines.append("<b>NEXT WEEK CHECKLIST</b> (criteria to deploy more aggressively):")
    lines.append(f"  [{'X' if snap.nifty_close > snap.nifty_50dma else ' '}] NIFTY above 50-DMA ({snap.nifty_50dma:.0f})")
    lines.append(f"  [{'X' if snap.rsi_daily > 50 else ' '}] RSI(d) above 50 (currently {snap.rsi_daily:.1f})")
    lines.append(f"  [{'X' if snap.vix and snap.vix < 16 else ' '}] VIX under 16 (currently {snap.vix:.1f})")
    lines.append(f"  [{'X' if snap.pct_above_200dma_breadth > 55 else ' '}] Breadth above 55% (currently {snap.pct_above_200dma_breadth:.0f}%)")
    lines.append(f"  [{'X' if snap.nifty_50dma > snap.nifty_200dma else ' '}] Golden cross (50-DMA above 200-DMA)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=100_000,
                        help="Total DCA capital (used only if --init)")
    parser.add_argument("--tranches", type=int, default=4,
                        help="Number of base tranches (used only if --init)")
    parser.add_argument("--init", action="store_true",
                        help="Initialize a new plan (will reset existing)")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the plan file")
    parser.add_argument("--status", action="store_true",
                        help="Show current plan status and exit")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from PAUSED state")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print Telegram message instead of sending; "
                             "do NOT update plan state")
    parser.add_argument("--no-ai", action="store_true",
                        help="Skip AI agent (faster)")
    args = parser.parse_args()

    if args.reset:
        reset_plan()
        print("[OK] Plan wiped.")
        return 0

    if args.init:
        reset_plan()
        plan = new_plan(total_capital=args.capital, base_tranches=args.tranches)
        print(f"[OK] New plan: capital Rs {args.capital:,.0f}, {args.tranches} tranches")
        print(f"[OK] Plan ID: {plan.plan_id}")
        return 0

    plan = load_plan()
    if plan is None:
        print("[error] No plan exists. Run with --init first:")
        print("  python -m screener.weekly_dca --init --capital 100000 --tranches 4")
        return 1

    if args.status:
        print(f"Plan ID:           {plan.plan_id}")
        print(f"Started:           {plan.started_at}")
        print(f"Status:            {plan.status}")
        print(f"Total capital:     Rs {plan.total_capital:,.0f}")
        print(f"Deployed so far:   Rs {plan.deployed_so_far:,.0f}  ({plan.deployed_so_far/plan.total_capital*100:.1f}%)")
        print(f"Weeks completed:   {plan.weeks_completed} of {plan.base_tranches}")
        print(f"Remaining:         Rs {plan.remaining_capital():,.0f}")
        if plan.pause_reason:
            print(f"Pause reason:      {plan.pause_reason}")
        if plan.history:
            print(f"\nHistory:")
            for h in plan.history[-5:]:
                print(f"  Week {h['week']:>2d}  {h['date']}  tranche Rs {h['tranche_inr']:,.0f}  "
                      f"triggers={len(h['triggers_fired'])}  verdict={h['market_verdict']}")
        return 0

    if args.resume:
        if plan.status == "PAUSED":
            plan.status = "ACTIVE"
            plan.pause_reason = None
            save_plan(plan)
            print("[OK] Plan resumed.")
        else:
            print(f"[info] Plan is {plan.status}, nothing to resume.")
        return 0

    # ── Main weekly flow ──
    print(f"DCA plan: {plan.plan_id}  status={plan.status}")
    print(f"Deployed: Rs {plan.deployed_so_far:,.0f} of Rs {plan.total_capital:,.0f}")

    if plan.status == "DONE":
        print("[info] Plan already complete — no further deployment.")
        return 0

    if plan.status == "STOPPED":
        print("[info] Plan stopped by user. Use --resume or --reset.")
        return 0

    print("Evaluating triggers...")
    snap = evaluate_triggers()

    # Compute tranche
    base = plan.base_tranche_size()
    tranche = base * snap.tranche_multiplier
    # If plan was paused but stop triggers cleared, allow auto-resume
    if plan.status == "PAUSED" and not snap.stop_fired:
        plan.status = "ACTIVE"
        plan.pause_reason = None
    # If new stop trigger fires, pause
    if snap.stop_fired:
        tranche = 0.0
        if plan.status != "PAUSED":
            pause_plan(plan, "; ".join(snap.stop_fired))

    # Cap tranche to remaining
    tranche = min(tranche, plan.remaining_capital())

    # Fetch picks for the message (only if we'll deploy)
    picks_payload = {}
    if tranche > 0:
        print("Fetching fresh picks (capital scaled to tranche)...")
        picks_payload = generate_picks(
            capital=tranche,
            n_large=2, n_mid=1,
            enable_ai=not args.no_ai,
            force_refresh=True,
        )

    msg = format_message(plan, snap, tranche, picks_payload)

    if args.dry_run:
        print("=" * 78)
        plain = re.sub(r"<[^>]+>", "", msg)
        print(plain)
        print("=" * 78)
        return 0

    # Real send
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required in env.")
        print("Falling back to dry-run output:")
        plain = re.sub(r"<[^>]+>", "", msg)
        print(plain)
        return 1
    tg.configure(bot_token, chat_id)
    tg.notify(msg)
    print("[OK] Sent Telegram alert.")

    # Record the week
    if tranche > 0:
        record_week(
            plan=plan,
            tranche=tranche,
            triggers_fired=snap.accelerate_fired + snap.stop_fired,
            market_score=0,  # could be populated by analyzer
            market_verdict=snap.recommended_action,
            picks=picks_payload.get("picks", []),
            notes="; ".join(snap.notes),
        )
        print(f"[OK] Recorded week {plan.weeks_completed}: Rs {tranche:,.0f} tranche.")
    else:
        # No tranche this week — still record the check
        record_week(
            plan=plan, tranche=0.0,
            triggers_fired=snap.stop_fired,
            market_score=0, market_verdict=snap.recommended_action,
            picks=[], notes="No deployment (paused)",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
