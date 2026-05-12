"""
DCA (Dollar-Cost Average) plan state manager.

Tracks a multi-week deployment plan: how much capital total, how many
tranches, how much already deployed, what happened each week. Persists
to JSON so the weekly cron is stateless.

Plan lifecycle:
  ACTIVE  - tranches still remaining, keep deploying weekly
  DONE    - all tranches deployed, switch to MAINTENANCE mode
  PAUSED  - stop trigger fired (VIX spike, breach below 23000 etc.)
  STOPPED - user explicitly halted
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

STATE_FILE = Path(__file__).resolve().parent.parent / "data" / "dca_plan.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class WeeklyRecord:
    week: int
    date: str
    tranche_inr: float
    deployed_running_total: float
    triggers_fired: list[str] = field(default_factory=list)
    market_score: int = 0
    market_verdict: str = ""
    picks: list[dict] = field(default_factory=list)
    notes: str = ""


@dataclass
class DCAPlan:
    plan_id: str
    started_at: str
    total_capital: float = 100_000.0
    base_tranches: int = 4
    deployed_so_far: float = 0.0
    weeks_completed: int = 0
    status: str = "ACTIVE"       # ACTIVE | DONE | PAUSED | STOPPED
    pause_reason: Optional[str] = None
    history: list[dict] = field(default_factory=list)

    def remaining_capital(self) -> float:
        return max(0.0, self.total_capital - self.deployed_so_far)

    def remaining_tranches(self) -> int:
        return max(0, self.base_tranches - self.weeks_completed)

    def base_tranche_size(self) -> float:
        """Equal-weight tranche size. Bigger if we've fallen behind."""
        rem_tr = self.remaining_tranches()
        if rem_tr == 0:
            return 0.0
        return self.remaining_capital() / rem_tr

    def to_dict(self) -> dict:
        return asdict(self)


def load_plan() -> Optional[DCAPlan]:
    """Load existing plan from disk. None if no plan started yet."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DCAPlan(**data)
    except Exception as e:
        print(f"[dca_state] Failed to load plan: {e}")
        return None


def save_plan(plan: DCAPlan) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(plan.to_dict(), f, indent=2, default=str)
    except Exception as e:
        print(f"[dca_state] Failed to save plan: {e}")


def new_plan(total_capital: float = 100_000.0, base_tranches: int = 4) -> DCAPlan:
    """Create a fresh DCA plan."""
    plan = DCAPlan(
        plan_id=datetime.now().strftime("dca_%Y%m%d_%H%M%S"),
        started_at=datetime.now().isoformat(),
        total_capital=total_capital,
        base_tranches=base_tranches,
    )
    save_plan(plan)
    return plan


def reset_plan() -> None:
    """Wipe the existing plan."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def record_week(
    plan: DCAPlan,
    tranche: float,
    triggers_fired: list[str],
    market_score: int,
    market_verdict: str,
    picks: list[dict],
    notes: str = "",
) -> None:
    """Record a weekly tranche and update plan state."""
    plan.weeks_completed += 1
    plan.deployed_so_far += tranche
    rec = WeeklyRecord(
        week=plan.weeks_completed,
        date=datetime.now().date().isoformat(),
        tranche_inr=tranche,
        deployed_running_total=plan.deployed_so_far,
        triggers_fired=triggers_fired,
        market_score=market_score,
        market_verdict=market_verdict,
        picks=picks,
        notes=notes,
    )
    plan.history.append(asdict(rec))
    # Mark DONE when fully deployed
    if plan.deployed_so_far >= plan.total_capital * 0.95:
        plan.status = "DONE"
    save_plan(plan)


def pause_plan(plan: DCAPlan, reason: str) -> None:
    plan.status = "PAUSED"
    plan.pause_reason = reason
    save_plan(plan)


def resume_plan(plan: DCAPlan) -> None:
    if plan.status == "PAUSED":
        plan.status = "ACTIVE"
        plan.pause_reason = None
        save_plan(plan)
