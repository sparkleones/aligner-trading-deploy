"""Notifications package — Telegram alerts for the Aligner trading system."""
from notifications.telegram_notifier import (
    configure,
    notify,
    on_system_start,
    on_system_stop,
    on_trade_entry,
    on_trade_exit,
    on_kill_switch,
    on_daily_summary,
    on_heartbeat,
    on_error,
    maybe_heartbeat,
)

__all__ = [
    "configure",
    "notify",
    "on_system_start",
    "on_system_stop",
    "on_trade_entry",
    "on_trade_exit",
    "on_kill_switch",
    "on_daily_summary",
    "on_heartbeat",
    "on_error",
    "maybe_heartbeat",
]
