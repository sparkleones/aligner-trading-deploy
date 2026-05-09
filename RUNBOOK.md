# Operations Runbook — Aligner Trading System

**Last updated**: 2026-05-09
**Current live config**: V15_CONFIG with vix_floor=12, vix_ceil=25, directional_gate_threshold=0.5, lookback=3
**Live AWS endpoint**: http://51.21.206.33:8510/terminal
**Container name**: `aligner-trading`

---

## 1. Daily Operations

### Morning routine (every trading day, before 09:15 IST)

```bash
# 1. Open dashboard in browser
http://51.21.206.33:8510/terminal

# 2. Check Kite Broker panel (top of right rail)
#    - If badge shows DISCONNECTED: click AUTO-LOGIN button
#    - Wait 5-10 seconds, badge turns green with user name + balance

# 3. Verify engine status
#    - "TRADING" badge top-left should be LIVE
#    - VIX should show current value (not 0.0)
#    - NIFTY price should be live tick (not stale)

# 4. (Optional) Tail logs from EC2:
ssh <ec2>
docker logs -f aligner-trading 2>&1 | grep -iE "V14 SCORED|V14 ENTRY|directional_gate|MONITOR EXIT"
```

### What "all clear" looks like

- **Dashboard**: Kite Broker panel CONNECTED green, balance displayed
- **VIX**: Live value (typically 12-18 in current regime)
- **NIFTY**: Updates every ~5 seconds during market hours
- **Engine logs**: Periodic `V14 SCORED` or `V14 NO_SIGNAL` lines (proves engine is processing bars)
- **TODAY'S P&L**: 0 at market open, updates as trades close

### After market close (15:30 IST)

- Telegram should send EOD summary (auto)
- Verify TODAY'S P&L on dashboard matches your expectation
- Container keeps running overnight (no daily restart needed unless deploying changes)

---

## 2. Kite Token Refresh

### Automatic (recommended)

1. Open dashboard → Kite Broker panel
2. If DISCONNECTED → click **AUTO-LOGIN**
3. Done

### CLI fallback (if dashboard is unreachable)

```bash
ssh <ec2>
docker exec aligner-trading python -m broker.kite_auto_login
# Should print "✅ SUCCESS!" and the new access_token
```

### Manual (last resort, if auto-login fails)

1. Open https://kite.zerodha.com/connect/login?api_key=YOUR_API_KEY
2. Login with credentials + 2FA in browser
3. Kite redirects to your registered redirect_uri with `?request_token=XXX`
4. Copy that request_token
5. SSH to EC2:
   ```bash
   docker exec aligner-trading python -c "
   from kiteconnect import KiteConnect
   import os
   kite = KiteConnect(api_key=os.getenv('KITE_API_KEY'))
   data = kite.generate_session('PASTE_REQUEST_TOKEN_HERE', api_secret=os.getenv('KITE_API_SECRET'))
   print('access_token:', data['access_token'])
   "
   ```
6. Update `.env`: `nano ~/trading/.env` → set `KITE_ACCESS_TOKEN=<new_token>`
7. Restart: `docker compose -f ~/trading/deploy/docker-compose.prod.yml restart aligner-trading`

---

## 3. Monitoring — what to watch for

### Healthy operation indicators

| What | How to check | Expected |
|---|---|---|
| Container alive | `docker ps` | `aligner-trading` Up X minutes (healthy) |
| Engine processing bars | `docker logs aligner-trading \| tail -10` | Recent `V14 SCORED` lines |
| Kite API working | dashboard top right | Connection green, balance live |
| Trade dispatch | Telegram bot | "Trade entered" / "Trade closed" messages |
| Daily DD breaker | engine logs | No `circuit breaker` messages (means daily loss < 3%) |

### After directional gate activates (Thu May 14 onwards)

Look for these log patterns when entries get blocked:

```
V14 SKIP DIRECTIONAL_GATE: 5d return X.XX% > +0.50% threshold (blocking PUT in uptrend)
V14 SKIP DIRECTIONAL_GATE: 5d return Y.YY% < -0.50% threshold (blocking CALL in downtrend)
```

Frequency: typically 1-3 blocks per week (the gate is selective, only fires on directional days).

### After Monitor activates (any tail-risk trade)

```
V14 MONITOR EXIT: BUY_CALL 24500CE | reason=emergency_floor | entry=120.5 peak=145.2 cur=63.5 (peak/entry=1.21x cur/peak=0.44x)
```

Frequency: rare (only fires when premium peaks then collapses fast). 1-3 per month typical.

### Red flags — investigate immediately

| Symptom | Likely cause | Fix |
|---|---|---|
| Dashboard shows VIX=0.0 during market hours | yfinance/Kite VIX feed broken | Restart container; if persists, check yfinance status |
| Engine logs say "ERROR: invalid token" repeatedly | Kite token expired mid-session (rare) | AUTO-LOGIN from dashboard |
| Dashboard shows DISCONNECTED but trades still firing | Stale UI state, real engine OK | Refresh browser; if still wrong, restart container |
| Today's P&L crosses -3% | Daily DD circuit breaker should engage | Engine should auto-halt entries; verify in logs |
| Container "Up X minutes (unhealthy)" | Health check failing | `docker logs aligner-trading \| tail -50` for cause |
| No `V14 SCORED` lines in last 5 min during market hours | Engine stuck or crashed | Restart: `docker compose -f ~/trading/deploy/docker-compose.prod.yml restart` |

---

## 4. Emergency Procedures

### Kill switch (stop ALL trading immediately)

**From dashboard**:
- Click red "Kill Switch" button (top right corner)
- Confirms intent → engine halts new entries → existing positions unchanged

**From CLI**:
```bash
ssh <ec2>
echo "kill" > ~/trading/data/engine_control.json
# Or for emergency full container kill:
docker stop aligner-trading
```

### Force-close all open positions

The engine auto-square-offs at 15:15 IST daily. To force-close before that:

```bash
ssh <ec2>
docker exec aligner-trading python -c "
import os
from kiteconnect import KiteConnect
kite = KiteConnect(api_key=os.getenv('KITE_API_KEY'))
kite.set_access_token(os.getenv('KITE_ACCESS_TOKEN'))
positions = kite.positions().get('net', [])
for p in positions:
    if p['quantity'] != 0:
        # Place opposite order to close
        kite.place_order(
            variety='regular',
            tradingsymbol=p['tradingsymbol'],
            exchange=p['exchange'],
            transaction_type='SELL' if p['quantity'] > 0 else 'BUY',
            quantity=abs(p['quantity']),
            order_type='MARKET',
            product=p['product'],
        )
        print(f'Squared off: {p[\"tradingsymbol\"]} qty={p[\"quantity\"]}')
"
```

### Roll back to previous deployed version

If today's deploy breaks something:

```bash
ssh <ec2>
cd ~/trading
git log --oneline -10           # find prior good commit (e.g., 7b81710 = Option B without gate)
git reset --hard 7b81710        # WARNING: destructive, undoes local changes
cd deploy
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up -d --build
```

To "soft" disable the directional gate without rollback:

```bash
nano ~/trading/scoring/config.py
# Find: "directional_gate_threshold": 0.5
# Change to: "directional_gate_threshold": None  (or comment out the line)
# Save and:
docker compose -f ~/trading/deploy/docker-compose.prod.yml restart
```

---

## 5. Capital & Sizing

### Where capital is configured

```bash
grep CAPITAL ~/trading/.env
# Default: CAPITAL=30000 (Rs 30K)
```

### When to update CAPITAL

- After deposits/withdrawals to your Kite account
- If broker balance differs significantly from configured CAPITAL
- The engine sizes lots based on configured CAPITAL, not live broker balance

### Position-size guards (already enforced by V14 config)

- `max_lots_cap`: 27 (hard cap regardless of capital)
- `max_concurrent`: 3 simultaneous positions
- `max_trades_per_day`: 7
- Hard SL: per V15 trail params (0.7-1.5% trail-pct)
- Daily DD circuit breaker: 3% of capital (halts new entries)
- LiveTradeMonitor emergency floor: peak_mult=1.20x, floor_frac=0.45 (catches runaway losses)

---

## 6. Strategy Reference (currently deployed)

| Filter | Setting | Purpose |
|---|---|---|
| `avoid_days` | `[0, 2]` (Mon, Wed) | Validated worst days for V14 entries |
| `vix_floor` | 12 | Skip ultra-low-vol days |
| `vix_ceil` | 25 | Skip extreme-vol days |
| `directional_gate_threshold` | 0.5 | Block PUT in uptrend, CALL in downtrend |
| `directional_gate_lookback_days` | 3 | 3-day rolling spot trend |
| DTE target | Tuesday | Post-Sep 2025 SEBI cutover |
| Strike | ATM | Round to nearest 50 |
| Position size | min(3 lots, 50% capital / cost_per_lot) | Conservative sizing |

### Walk-forward validation (21mo, 6 windows)

| Metric | No gate (Option B) | With gate (current) |
|---|---:|---:|
| 21mo PnL | +Rs 57.30L | +Rs 86.72L (+51%) |
| PF | 1.94 | 3.53 |
| WR | 42.3% | 52.2% |
| Max DD | -Rs 6.08L | -Rs 5.06L |
| Walk-forward | n/a | **6/6 STRONG EDGE** |

**Forward expectation** (live execution friction): PF 2.0-2.5, +Rs 8-15L/year over no-gate baseline.

---

## 7. Cold Start Behaviors

### After container rebuild

- `_daily_close_history` is empty (gate is no-op)
- Gate needs **3 trading days** of close history before activating
- Behavior matches Option B baseline during cold start

### After Kite token expiry (midnight every day)

- Engine continues processing bars but can't place orders
- Kite Broker panel shows DISCONNECTED
- Click AUTO-LOGIN to refresh

### After market open (09:15 IST daily)

- Engine starts scoring bars at bar 0 (09:15) but most entries are blocked until bar 3 (composite entry window)
- First V8-scoring entries possible at bar ≥ 3 (09:30 IST)
- ORB high/low locked at bar 3 (after 15-min opening range)

---

## 8. File Locations

| File | Purpose |
|---|---|
| `~/trading/.env` | Secrets, capital, broker creds |
| `~/trading/scoring/config.py` | V14/V15/V17 strategy configs |
| `~/trading/orchestrator/strategy_agents/v14_live_agent.py` | Live execution agent |
| `~/trading/data/engine_control.json` | Kill switch flag |
| `~/trading/deploy/docker-compose.prod.yml` | Container orchestration |
| `~/trading/deploy/Dockerfile.prod` | Build recipe |

---

## 9. Reference Commits

| Commit | What |
|---|---|
| `4c18199` | Option A: vix_floor 13→12 |
| `7b81710` | Option B: vix_ceil 35→25 |
| `962160b` | Push #2 v1: DTE + Monitor + gate (lb=5, thr=1.0) |
| `3548a68` | Push #2 amendment: gate refined to lb=3, thr=0.5 |
| `d44bda0` | Auto-login + dashboard auth panel |

---

## 10. Quick Sanity Checks

```bash
# Container running?
docker ps | grep aligner-trading

# Latest commit deployed?
docker exec aligner-trading git log --oneline -1
# Should match: d44bda0 or whatever is HEAD on github

# Config loaded correctly?
docker exec aligner-trading python -c "
from scoring.config import V15_CONFIG
for k in ['vix_floor', 'vix_ceil', 'directional_gate_threshold', 'directional_gate_lookback_days', 'avoid_days']:
    print(f'{k}: {V15_CONFIG.get(k)}')
"

# Kite connected?
curl -s http://localhost:8510/api/broker/status | python -m json.tool

# Engine processing bars (last 10 lines)?
docker logs --tail 10 aligner-trading
```

---

## 11. Known Gotchas

1. **Saturday/Sunday**: dashboard shows stale Friday close data. Normal.
2. **Container rebuild resets `_daily_close_history`**: gate has 3-day cold start.
3. **`CAPITAL` env var ≠ live broker balance**: engine uses configured value, not live. Update `.env` if broker balance changes significantly.
4. **`KITE_ACCESS_TOKEN` expires daily at midnight IST**: must refresh every trading day (use AUTO-LOGIN button).
5. **Telegram notifier requires `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in .env**: if not set, you won't get trade alerts.
6. **Disk fills up over time** from Docker images: run `docker system prune -af` weekly to reclaim space.
7. **`_estimate_dte` targets Tuesday**: works for current weekly expiry. If SEBI changes again, update v14_live_agent.py line 303.
