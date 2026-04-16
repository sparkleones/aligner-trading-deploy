"""Cross-session position tracker for autonomous trading.

Persists position state to disk so the system can:
  1. Recover from crashes (reload open positions)
  2. Track BTST positions across trading sessions
  3. Reconcile internal state vs broker positions
  4. Generate audit trails for compliance

State file: data/position_state.json
BTST file:  data/btst_positions.json
"""

import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "position_state.json"
BTST_FILE = DATA_DIR / "btst_positions.json"
DAILY_REPORT_DIR = DATA_DIR / "daily_reports"


class PositionTracker:
    """Tracks positions across sessions for crash recovery and BTST.

    Position lifecycle:
      OPEN → (intraday exit or) → BTST → (next day exit) → CLOSED

    BTST criteria (from backtesting analysis):
      - Position is profitable (unrealized PnL > 0)
      - Strong trend alignment (VIX < 20, momentum in trade direction)
      - Not expiry day (risk of gap against on expiry)
      - Favorable gap probability for direction (bearish = gap down likely)
      - Trade was entered before 2 PM (had time to build profit)
    """

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DAILY_REPORT_DIR.mkdir(parents=True, exist_ok=True)

        self._open_positions: list[dict] = []
        self._btst_positions: list[dict] = []
        self._closed_today: list[dict] = []
        self._daily_pnl: float = 0.0

    # ── Persistence ─────────────────────────────────────────────────

    def save_state(self) -> None:
        """Persist current state to disk."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "date": date.today().isoformat(),
            "open_positions": self._open_positions,
            "closed_today": self._closed_today,
            "daily_pnl": self._daily_pnl,
        }
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error("Failed to save position state: %s", e)

    def load_state(self) -> None:
        """Load state from disk. Only loads today's state.

        If the state file is from a previous day, it is discarded and
        overwritten with a clean empty state on disk. This prevents stale
        positions (e.g. from a prior-day broker auto-squareoff that was
        never recorded) from lingering in the file and confusing dashboards
        or audits that read the raw JSON.
        """
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    state = json.load(f)
                if state.get("date") == date.today().isoformat():
                    self._open_positions = state.get("open_positions", [])
                    self._closed_today = state.get("closed_today", [])
                    self._daily_pnl = state.get("daily_pnl", 0.0)
                    logger.info(
                        "Loaded position state | open=%d closed=%d pnl=%.2f",
                        len(self._open_positions), len(self._closed_today),
                        self._daily_pnl,
                    )
                else:
                    stale_open = len(state.get("open_positions", []))
                    stale_date = state.get("date", "unknown")
                    logger.info(
                        "Position state is from %s (stale, %d open) — "
                        "discarding and starting fresh",
                        stale_date, stale_open,
                    )
                    # Clear in-memory state and persist clean state to disk
                    self._open_positions = []
                    self._closed_today = []
                    self._daily_pnl = 0.0
                    self.save_state()
        except Exception as e:
            logger.error("Failed to load position state: %s", e)

    def save_btst(self) -> None:
        """Save BTST positions for next-day exit."""
        btst_state = {
            "created_date": date.today().isoformat(),
            "positions": self._btst_positions,
        }
        try:
            with open(BTST_FILE, "w") as f:
                json.dump(btst_state, f, indent=2, default=str)
            logger.info("Saved %d BTST positions for tomorrow", len(self._btst_positions))
        except Exception as e:
            logger.error("Failed to save BTST positions: %s", e)

    def load_btst(self) -> list[dict]:
        """Load BTST positions from previous day.

        Returns list of positions that need to be exited today.
        Only returns positions created yesterday (or most recent trading day).
        """
        try:
            if not BTST_FILE.exists():
                return []
            with open(BTST_FILE) as f:
                state = json.load(f)

            created = state.get("created_date", "")
            positions = state.get("positions", [])

            if not positions:
                return []

            # Accept BTST from any previous day (could be Friday → Monday)
            if created == date.today().isoformat():
                logger.info("BTST file is from today — already processed")
                return []

            logger.info(
                "Loaded %d BTST positions from %s for exit today",
                len(positions), created,
            )
            self._btst_positions = positions
            return positions

        except Exception as e:
            logger.error("Failed to load BTST positions: %s", e)
            return []

    def clear_btst(self) -> None:
        """Clear BTST file after positions are exited."""
        self._btst_positions = []
        try:
            if BTST_FILE.exists():
                BTST_FILE.unlink()
                logger.info("BTST file cleared")
        except Exception as e:
            logger.error("Failed to clear BTST file: %s", e)

    # ── Position Tracking ───────────────────────────────────────────

    def add_position(
        self,
        symbol: str,
        side: str,
        qty: int,
        entry_price: float,
        order_id: str,
        strategy: str,
        entry_type: str,
        product: str = "MIS",
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a new open position."""
        pos = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "order_id": order_id,
            "strategy": strategy,
            "entry_type": entry_type,
            "product": product,
            "entry_time": datetime.now().isoformat(),
            "current_price": entry_price,
            "unrealized_pnl": 0.0,
            "metadata": metadata or {},
        }
        self._open_positions.append(pos)
        self.save_state()
        logger.info(
            "Position opened | %s %s %s qty=%d @ %.2f | %s",
            side, symbol, product, qty, entry_price, strategy,
        )

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        realized_pnl: float,
        exit_reason: str,
    ) -> Optional[dict]:
        """Close an open position and move to closed list."""
        for pos in self._open_positions:
            if pos["symbol"] == symbol:
                pos["exit_price"] = exit_price
                pos["realized_pnl"] = realized_pnl
                pos["exit_reason"] = exit_reason
                pos["exit_time"] = datetime.now().isoformat()

                # Calculate hold_minutes from entry_time to exit_time
                try:
                    entry_dt = datetime.fromisoformat(pos["entry_time"])
                    pos["hold_minutes"] = round((datetime.now() - entry_dt).total_seconds() / 60, 1)
                except Exception:
                    pos["hold_minutes"] = 0

                self._closed_today.append(pos)
                self._open_positions.remove(pos)
                self._daily_pnl += realized_pnl
                self.save_state()

                logger.info(
                    "Position closed | %s | pnl=%.2f | reason=%s",
                    symbol, realized_pnl, exit_reason,
                )
                return pos
        return None

    def update_prices(self, price_map: dict[str, float]) -> None:
        """Update current prices and unrealized P&L for all positions."""
        for pos in self._open_positions:
            sym = pos["symbol"]
            if sym in price_map:
                pos["current_price"] = price_map[sym]
                if pos["side"] == "BUY":
                    pos["unrealized_pnl"] = (price_map[sym] - pos["entry_price"]) * pos["qty"]
                else:
                    pos["unrealized_pnl"] = (pos["entry_price"] - price_map[sym]) * pos["qty"]

    # ── BTST Decision ───────────────────────────────────────────────

    def evaluate_btst(
        self,
        vix: float,
        is_expiry_day: bool,
        market_trend: str = "",
    ) -> list[dict]:
        """Evaluate which open positions qualify for BTST (overnight hold).

        BTST criteria (from backtest analysis — 100% WR on BTST in V3):
          1. Position is profitable (unrealized PnL > 0)
          2. VIX < 20 (overnight gaps are manageable in normal VIX)
          3. NOT expiry day (expiry gaps are unpredictable)
          4. Trade was entered in first 4 hours (has momentum confirmation)
          5. BUY_PUT + bearish trend OR BUY_CALL + bullish trend (alignment)

        Returns list of positions to hold overnight.
        """
        if is_expiry_day:
            logger.info("BTST: Skipping — expiry day (unpredictable gaps)")
            return []

        if vix >= 20:
            logger.info("BTST: Skipping — VIX %.1f too high (gap risk)", vix)
            return []

        btst_candidates = []
        for pos in self._open_positions:
            # V17 DYNAMIC PRODUCT: positions that entered directly as NRML
            # (V17 btst-favorable layer) are ALREADY destined for overnight
            # carry. The 15:28 square-off path moves them into _btst_positions
            # via mark_btst(); this legacy MIS→NRML converter must not touch
            # them (it would close the MIS leg that doesn't exist and reopen
            # a duplicate NRML leg → double exposure).
            if pos.get("product") == "NRML":
                continue

            pnl = pos.get("unrealized_pnl", 0)

            # Must be profitable
            if pnl <= 0:
                continue

            # Trend alignment check
            action = pos.get("metadata", {}).get("original_action", pos.get("side", ""))
            if market_trend:
                if "PUT" in action and market_trend not in ("BEARISH", "STRONG_BEARISH"):
                    continue
                if "CALL" in action and market_trend not in ("BULLISH", "STRONG_BULLISH"):
                    continue

            # Profit threshold: at least 1% of entry value
            entry_val = pos["entry_price"] * pos["qty"]
            if entry_val > 0 and (pnl / entry_val) < 0.01:
                continue

            btst_candidates.append(pos)
            logger.info(
                "BTST CANDIDATE | %s | pnl=%.2f | entry_type=%s",
                pos["symbol"], pnl, pos.get("entry_type", "unknown"),
            )

        return btst_candidates

    def mark_btst(self, positions: list[dict]) -> None:
        """Mark positions for BTST — change product to NRML and persist."""
        for pos in positions:
            pos["product"] = "NRML"
            pos["btst"] = True
            pos["btst_marked_at"] = datetime.now().isoformat()

        self._btst_positions = positions
        # Remove from open_positions (they're now BTST)
        btst_symbols = {p["symbol"] for p in positions}
        self._open_positions = [
            p for p in self._open_positions if p["symbol"] not in btst_symbols
        ]
        self.save_btst()
        self.save_state()

    # ── Reconciliation ──────────────────────────────────────────────

    def reconcile_with_broker(self, broker_positions: list[dict]) -> dict:
        """Compare internal state with broker's actual positions.

        Returns a report of discrepancies.
        """
        internal_map = {p["symbol"]: p for p in self._open_positions}
        broker_map = {
            p["symbol"]: p for p in broker_positions
            if p.get("qty", p.get("quantity", 0)) != 0
        }

        missing_in_broker = []
        missing_in_internal = []
        qty_mismatch = []

        # Positions we think are open but broker doesn't have
        for sym, pos in internal_map.items():
            if sym not in broker_map:
                missing_in_broker.append(sym)
            elif abs(broker_map[sym].get("qty", broker_map[sym].get("quantity", 0))) != pos["qty"]:
                qty_mismatch.append({
                    "symbol": sym,
                    "internal_qty": pos["qty"],
                    "broker_qty": broker_map[sym].get("qty", broker_map[sym].get("quantity", 0)),
                })

        # Positions broker has that we don't know about
        for sym in broker_map:
            if sym not in internal_map:
                # Could be BTST from yesterday
                btst_syms = {p["symbol"] for p in self._btst_positions}
                if sym not in btst_syms:
                    missing_in_internal.append(sym)

        report = {
            "timestamp": datetime.now().isoformat(),
            "internal_count": len(internal_map),
            "broker_count": len(broker_map),
            "missing_in_broker": missing_in_broker,
            "missing_in_internal": missing_in_internal,
            "qty_mismatch": qty_mismatch,
            "is_synced": (
                not missing_in_broker and
                not missing_in_internal and
                not qty_mismatch
            ),
        }

        if not report["is_synced"]:
            logger.warning("POSITION MISMATCH: %s", json.dumps(report, indent=2))
        else:
            logger.info("Position reconciliation: SYNCED (%d positions)", len(internal_map))

        return report

    def adopt_broker_positions(self, broker_positions: list[dict]) -> list[str]:
        """Auto-adopt broker positions not tracked internally.

        This handles the case where a previous engine instance placed a trade
        but was killed before the new instance could learn about it.
        Critical on expiry days to ensure positions are managed.

        Returns list of adopted symbol names.
        """
        internal_syms = {p["symbol"] for p in self._open_positions}
        btst_syms = {p["symbol"] for p in self._btst_positions}
        adopted = []

        for bp in broker_positions:
            sym = bp.get("symbol", "")
            qty = bp.get("qty", bp.get("quantity", 0))
            if qty == 0 or not sym:
                continue
            if sym in internal_syms or sym in btst_syms:
                continue

            # Parse option type and strike from tradingsymbol
            # e.g. NIFTY2640722950CE → strike=22950, opt_type=CE
            opt_type = ""
            strike = 0.0
            if sym.endswith("CE") or sym.endswith("PE"):
                opt_type = sym[-2:]
                # Try to extract strike - last digits before CE/PE
                strike_str = ""
                for ch in reversed(sym[:-2]):
                    if ch.isdigit() or ch == ".":
                        strike_str = ch + strike_str
                    else:
                        break
                try:
                    strike = float(strike_str)
                except ValueError:
                    pass

            pos = {
                "symbol": sym,
                "side": "BUY" if qty > 0 else "SELL",
                "qty": abs(qty),
                "entry_price": bp.get("average_price", 0.0),
                "order_id": "adopted",
                "strategy": "adopted_from_broker",
                "entry_type": "adopted",
                "product": bp.get("product", "MIS"),
                "entry_time": datetime.now().isoformat(),
                "current_price": bp.get("ltp", bp.get("average_price", 0.0)),
                "unrealized_pnl": bp.get("pnl", 0.0),
                "opt_type": opt_type,
                "strike": strike,
                "metadata": {"adopted": True, "original_pnl": bp.get("pnl", 0.0)},
            }
            self._open_positions.append(pos)
            adopted.append(sym)
            logger.warning(
                "ADOPTED broker position: %s qty=%d avg=%.2f ltp=%.2f pnl=%.2f",
                sym, abs(qty), bp.get("average_price", 0), bp.get("ltp", 0), bp.get("pnl", 0),
            )

        if adopted:
            self.save_state()

        return adopted

    # ── Daily Report ────────────────────────────────────────────────

    def generate_daily_report(self) -> dict:
        """Generate end-of-day report."""
        report = {
            "date": date.today().isoformat(),
            "generated_at": datetime.now().isoformat(),
            "total_trades": len(self._closed_today),
            "open_positions": len(self._open_positions),
            "btst_positions": len(self._btst_positions),
            "daily_pnl": round(self._daily_pnl, 2),
            "trades": self._closed_today,
            "btst": self._btst_positions,
            "winners": sum(1 for t in self._closed_today if t.get("realized_pnl", 0) > 0),
            "losers": sum(1 for t in self._closed_today if t.get("realized_pnl", 0) < 0),
        }

        total = report["total_trades"]
        if total > 0:
            report["win_rate"] = round(report["winners"] / total * 100, 1)
            report["avg_pnl"] = round(self._daily_pnl / total, 2)
        else:
            report["win_rate"] = 0.0
            report["avg_pnl"] = 0.0

        # Save to daily reports
        report_path = DAILY_REPORT_DIR / f"report_{date.today().isoformat()}.json"
        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            logger.info("Daily report saved: %s", report_path)
        except Exception as e:
            logger.error("Failed to save daily report: %s", e)

        return report

    def reset_daily(self) -> None:
        """Reset for a new trading day."""
        self._open_positions.clear()
        self._closed_today.clear()
        self._daily_pnl = 0.0
        self.save_state()

    # ── Accessors ───────────────────────────────────────────────────

    @property
    def open_positions(self) -> list[dict]:
        return list(self._open_positions)

    @property
    def btst_positions(self) -> list[dict]:
        return list(self._btst_positions)

    @property
    def daily_pnl(self) -> float:
        return self._daily_pnl

    @property
    def trade_count(self) -> int:
        return len(self._closed_today)
