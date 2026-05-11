"""
Timezone-aware helpers for live trading.

Indian markets run on IST (Asia/Kolkata) regardless of where the trading
machine is hosted. Windows ignores the TZ=Asia/Kolkata env var, so we
explicitly anchor wall-clock-now to IST and strip tzinfo for downstream
naive-datetime arithmetic compatibility.

Usage:
    from config.timing import now_ist, IST
    now = now_ist()                 # naive datetime in IST
    if now.time() >= dt_time(9, 15):
        ...
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")
except Exception:
    # Fallback for Windows / Python builds without IANA tzdata
    IST = timezone(timedelta(hours=5, minutes=30), name="IST")


def now_ist() -> datetime:
    """Current wall-clock IST as a NAIVE datetime (tzinfo stripped).

    Existing engine code uses naive datetime arithmetic (combine, time(),
    timedelta subtraction). Returning an aware datetime would raise
    TypeError when subtracted from naive datetimes elsewhere. We anchor
    to IST and strip tzinfo so the value is always 'IST wall clock'
    regardless of host system timezone.
    """
    return datetime.now(IST).replace(tzinfo=None)


def now_ist_aware() -> datetime:
    """Current IST as a tz-aware datetime — for code that does its own
    tz handling explicitly. Most engine code should use `now_ist()`."""
    return datetime.now(IST)
