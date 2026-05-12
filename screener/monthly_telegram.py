"""
Monthly Telegram alert with new screener picks.

Designed to run as a cron / scheduled task on the 1st business day of
every month at 09:20 IST (5 minutes after market open).

Usage:
    python -m screener.monthly_telegram             # send today's picks
    python -m screener.monthly_telegram --dry-run   # render to stdout only

You can also wire this up to a Windows Task Scheduler / cron job to
fire automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root on path so 'notifications' resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notifications import telegram_notifier as tg


def format_picks(payload: dict) -> str:
    if "error" in payload:
        return f"<b>Stock Screener Error</b>\n{payload['error']}"
    picks = payload.get("picks", [])
    if not picks:
        return "<b>Stock Screener — no qualifying picks this month</b>"

    lines = []
    lines.append("<b>STOCK SCREENER — MONTHLY PICKS</b>")
    lines.append(f"<i>{datetime.now().strftime('%d %b %Y')}</i>")
    lines.append("")
    lines.append(f"Capital: Rs {payload.get('capital', 0):,.0f}")
    lines.append(f"Allocation: {payload.get('config', {}).get('allocation_split', '?')}")
    lines.append(f"Hold: {payload.get('config', {}).get('hold_days_default', 42)} days")
    bs = payload.get("backtest_summary", {})
    if bs:
        lines.append(f"Train CAGR: {bs.get('train_cagr', 0)*100:.1f}%  Sharpe {bs.get('train_sharpe', 0):.2f}")
    lines.append("")

    total_deploy = 0
    total_risk = 0
    for p in picks:
        verdict_emoji = {"BUY": "✅", "CAUTION": "⚠️", "SKIP": "❌"}.get(p.get("ai_verdict", ""), "")
        lines.append(
            f"<b>{p['symbol']}</b> {verdict_emoji} <i>{p['cap_tier']} / {p['sector']}</i>"
        )
        rr = (p["target"] - p["entry"]) / max(1e-6, p["entry"] - p["stop_loss"])
        lines.append(
            f"  Entry Rs {p['entry']}  SL Rs {p['stop_loss']} ({p['stop_distance_pct']*100:.1f}%)"
        )
        lines.append(
            f"  Target Rs {p['target']} ({p['target_pct']*100:.1f}%)  R:R {rr:.1f}"
        )
        lines.append(
            f"  Qty {p['qty']} | Deploy Rs {p['capital_deployed']:,.0f} | Risk Rs {p['risk_inr']:,.0f}"
        )
        if p.get("ai_reasoning"):
            lines.append(f"  <i>{p['ai_reasoning'][:120]}</i>")
        lines.append("")
        total_deploy += p["capital_deployed"]
        total_risk += p["risk_inr"]

    lines.append(f"Total deployed: Rs {total_deploy:,.0f}")
    lines.append(f"Total risk: Rs {total_risk:,.0f}")
    lines.append("")
    lines.append("<i>Hold until 1st of next month, or until SL/target hit.</i>")
    lines.append("<i>This is signal generation only — execute manually.</i>")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", type=float, default=100_000)
    parser.add_argument("--tier", default="BLEND")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print message instead of sending")
    args = parser.parse_args()

    from .live_picks_v2 import generate
    tier = args.tier.upper()
    n_large = 2
    n_mid = 1
    if tier == "LARGE":
        n_large, n_mid = 3, 0
    elif tier == "MID":
        n_large, n_mid = 0, 3

    print(f"Generating picks: capital=Rs {args.capital:,.0f}, tier={tier}, AI={not args.no_ai}")
    payload = generate(
        capital=args.capital,
        n_large=n_large, n_mid=n_mid,
        enable_ai=not args.no_ai,
        force_refresh=True,
    )

    msg = format_picks(payload)

    if args.dry_run:
        print("=" * 70)
        # strip HTML for stdout
        import re
        plain = re.sub(r"<[^>]+>", "", msg)
        print(plain)
        print("=" * 70)
        return

    # Real send
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("[error] TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required in env.")
        return 1
    tg.configure(bot_token, chat_id)
    tg.notify(msg)
    print("[OK] Sent Telegram alert.")


if __name__ == "__main__":
    sys.exit(main() or 0)
