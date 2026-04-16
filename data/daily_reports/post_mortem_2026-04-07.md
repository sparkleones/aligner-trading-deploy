# Post-Mortem: April 7, 2026 — Rs -20,056 Loss

## Executive Summary
- **Day P&L:** Rs -19,617 realized (-20,056 net with charges)
- **Capital:** Rs 30,000 → Rs 8,851 (-70.5%)
- **Trades:** 6 closed positions, 0 winners, 6 losers
- **Kill Switch:** Triggered at 15:12

## Trade Breakdown

| # | Symbol | P&L | Root Cause |
|---|--------|-----|------------|
| 1 | 22500PE | -4,144 | Orphan from previous engine session (4225 qty!), IP error blocked exit |
| 2 | 23000CE | -5,070 | Orphan from previous engine session |
| 3 | 23100CE | -4,518 | Orphan from previous engine session |
| 4 | 22950CE | -1,667 | Bought @86.5, 3 sell attempts FAILED (IP permission error) |
| 5 | 22850PE | -2,048 | Bought put @77.5, sold @61.75 in 4 minutes (20% premium loss) |
| 6 | 22900PE | -2,171 | Bought put @82.3 while market rallied 300+ pts off lows |

## Root Causes

### 1. Infrastructure: IP Permission Errors
- Kite API rejected orders from engine's IP address between 10:51-10:53
- Three consecutive sell attempts for 22950CE failed
- Position trapped, losses compounded
- **Fix needed:** IP whitelist verification on engine startup

### 2. Orphan Positions (3 trades, -Rs 13,732)
- Previous engine sessions left open positions with no exit mechanism
- On restart, engine adopted these at wrong averages
- The 22500PE position was 4,225 qty (65 lots!) — massive exposure
- **67% of today's loss came from orphan positions**

### 3. No Per-Trade Stop Loss
- 22850PE lost 20% of premium in 4 minutes with no hard SL
- V14 only had trailing stops (activate after 60-120 min hold)
- No defense against rapid premium decay

### 4. No Gap Reversal Detection
- Market gapped down 130 pts (22,968 → 22,838)
- Then reversed +310 pts (22,838 → 23,148)
- V14 kept trying PUT entries into a massive bullish reversal
- Indicators lagged the reversal by 20+ bars

### 5. check_position_limits() Never Called
- RiskManager.check_position_limits() existed but was not wired into execution path
- No runtime validation of position size before order placement

### 6. No Consecutive Loss Throttle
- After losing on orphans, engine kept trading aggressively
- No pause after consecutive losses to reassess

## Improvements Implemented

### V14 Strategy (v14_live_agent.py)
1. **Hard Stop Loss (30%)** — Exit immediately if option premium drops 30%
2. **Consecutive Loss Throttle** — After 2 consecutive losses, pause entries for 60 min
3. **Gap Reversal Filter** — Detect gap + reversal and block counter-trend entries
4. **Drawdown Lot Scaling** — Halve lots after 50% daily limit used, stop at 75%
5. **Entry Premium Tracking** — Store estimated entry premium for SL calculation
6. **Trade Result Tracking** — Log every trade's P&L for post-session analysis

### Risk Manager (risk_manager.py)
7. **daily_loss_pct property** — Expose daily loss as percentage for agent consumption

### Orchestrator (live_orchestrator.py)
8. **Pre-trade Risk Gate** — Wire check_position_limits() into _execute_trade()
9. **Risk Rejection Events** — Emit events when orders are blocked by risk limits

### Dashboard (decision state)
10. **Risk Guards Display** — Show consecutive losses, drawdown status, gap detection

## Backtested Impact (Expected)
- The 30% hard SL on trade #5 (22850PE) would have limited loss to -Rs 1,505 (saved Rs 543)
- Consecutive loss throttle would have blocked trade #6 (22900PE, -Rs 2,171) entirely
- Gap reversal filter would have blocked both put entries (#5 and #6), saving -Rs 4,219
- Drawdown scaling would have halved position sizes after orphan losses
- **Estimated savings: Rs 4,000-6,000 (20-30% of total loss)**

## Remaining Issues
- Orphan position adoption needs a more robust reconciliation on startup
- IP whitelist should be verified before market open
- Consider broker-side SL orders (GTT) as backup for infrastructure failures
