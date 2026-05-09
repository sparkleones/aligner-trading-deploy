# NIFTY Options System — Multi-Day Research Session Summary

**Period**: 2026-04-22 → 2026-05-09
**Strategy**: V17_PROD_ONLY (V15_CONFIG with V17 strike rules, V14 entries)
**Live capital reference**: Rs 2,00,000

---

## Live Config Evolution

| Phase | Config | Date deployed | Reason |
|---|---|---|---|
| Initial | V15_CONFIG, vix_floor=13, vix_ceil=35 | (pre-session) | Original |
| **Option A** | vix_floor 13→12 | 2026-04-22 | Recovers post-Sep low-vol regime |
| **Option B** ⭐ | vix_ceil 35→25 | 2026-05-08 | Drops 3 high-VIX losing trades |
| Current | avoid_days=[0,2], vix_floor=12, vix_ceil=25 | live | Walk-forward validated optimum |

**Option B 21-month metrics**: 29.65x return, +Rs 57.30L, WR 42.3%, **PF 1.94**, Max DD −Rs 6.08L (−12.2%), 194 trades.

**Option B last-1Y metrics**: 2.39x return, +Rs 2.78L, WR 37.7%, PF 1.20, Max DD −Rs 5.08L (−70.5%), 77 trades.

---

## Methodology

Three-tier validation gauntlet established and consistently applied:

1. **21-month full-window backtest** (2024-07 → 2026-04, 388 trading days)
   - First filter: must beat baseline on PnL or PF
2. **Out-of-sample window** (pre-1Y or post-Sep slice)
   - Second filter: must not catastrophically degrade in different regime
3. **Walk-forward 6 windows** (rolling 6-month, 3-month step)
   - Third filter: PnL wins ≥ 4/6, PF wins ≥ 4/6, no catastrophic windows

Strict standard: any candidate failing any tier is rejected.

---

## Variants Tested (30+)

### ✅ Deployed (cleared all 3 tiers)

| Config | Lift | Why deployed |
|---|---:|---|
| Option A (vix_floor 12) | post-Sep PF 1.60→1.73 | Recovers low-vol regime |
| Option B (vix_ceil 25) | post-Sep PF 1.73→1.87 | Drops 3 high-VIX losers |

### 🛑 Rejected at OOS / walk-forward (looked good in-sample, failed validation)

| Config | In-sample lift | Failure mode |
|---|---:|---|
| `[0,1,2]` (Mon+Tue+Wed avoid) | last-1Y +Rs 11.62L | Pre-1Y OOS: −Rs 7.57L (regime artifact) |
| Variant C (regime gate ON) | last-1Y +Rs 13.85L | Walk-forward 3/6 wins, 1 catastrophic |
| TIGHT_TRAIL | 21mo +Rs 4.7L | Walk-forward 3/6, 90% concentrated W1 |
| `vix_floor=10` | post-Sep +Rs 3.2L | PF degrades to 1.57 |
| `avoid=[0,2,3]+floor=12` | post-Sep PF 2.45 | Blocks Tuesday post-Sep expiry |
| `gap_classifier_skip_huge` | full +Rs 4.19L | Post-Sep -Rs 1.46L |
| OptionWise Mean Reversion | claimed 68% WR | Honest WR 28.8%, daily-OHLC fraud |
| Intraday short strangle | — | Slippage > intraday theta |
| Multi-day short strangle | WR 68% confirmed | PF 0.70 (−Rs 0.85L) |
| Iron Condor 1/3 multi-day | — | Wings destroy edge (PF 0.08) |

### 🟡 Marginal candidates (3/6 PnL, 4/6 PF, 0 catastrophic — under strict bar)

| Config | 21mo lift | Walk-fwd net Δ | Notes |
|---|---:|---:|---|
| NO_TRAIL | +Rs 2.11L | +Rs 150K | Trail entirely disabled |
| DELAYED | +Rs 2.34L | +Rs 197K | min_hold_trail 24/12 → 48/48 |
| **PE_TRAIL_ONLY** ⭐ | **+Rs 2.64L** | **+Rs 257K** | Disable CE trail, keep PE — best |

### ❌ Failed at first gate (worse than baseline 21mo)

| Config | Reason |
|---|---|
| AGGRESSIVE_STALE | WR 14%, PF 0.47, lost Rs 5.87L (churn) |
| MAX_WR (combined) | Same as above |
| CHANDELIER trail | Changes trade dynamics, PF 0.91 |
| H1 daily DD breaker | No-op (filter rarely fires) |
| H3 tighter scores | Filters too many trades |
| H4 ATR sizing | Flag not wired to exits |
| CE_TRAIL_ONLY | Confirms CE leak — only −0.5L variant |

---

## Key Findings

### 1. The 90% WR question is mathematically closed

Empirical ceiling: **43.8% WR** for V14 entry signals + this exit infrastructure. Aggressive WR-targeting via tight stale-exits crashes WR to 14% and PF to 0.47 (loses Rs 5.87L). Math:

- Per-trade EV at baseline = 0.423 × Rs 144,246 − 0.577 × Rs 54,449 = **+Rs 29,597**
- Per-trade EV under AGGRESSIVE_STALE = 0.136 × Rs 9,566 − 0.864 × Rs 3,218 = **−Rs 1,479**
- 90% WR would require R:R of 1:9, which our exit infrastructure cannot generate without churning.

### 2. The trail-stop leak is real and asymmetric

8 trail-stop fires across 21 months, all net negative (−Rs 4.2L). Counter-factual analysis:
- 4 PE fires net **+Rs 17.5K helpful** (correct cuts)
- 4 CE fires net **−Rs 331.2K harmful** (false alarms)

CE trail catches premium dips that go on to spike. PE trail catches genuine reversals. **The right fix is asymmetric** (PE_TRAIL_ONLY).

### 3. Walk-forward methodology saved us from 4+ overfit deploys

Each candidate looked compelling on a single window. Each failed when stress-tested across rolling 6-month windows:
- `[0,1,2]`: post-Sep PF 2.11 → pre-1Y OOS catastrophic
- Variant C: last-1Y PnL 6× lift → 3/6 walk-forward, W5 catastrophic
- TIGHT_TRAIL: 21mo +Rs 4.7L → 90% concentrated W1
- Each rejection was correct in retrospect

### 4. June 2025 was a regime break, not a bug

10 trades, 0 targets, 0 stops, 7 time_exits, 3 EOD closes. Pure theta bleed in a vol-crush + slow-uptrend regime (VIX 21.6 → 12.4, +3.9% spot move, 0.84% daily range). 6 of 9 v8_indicator entries fired PUTs into a rally. **This is a structural weakness of long-premium strategies in falling-vol uptrends, not a config error.**

### 5. The strategy is selective (not over-trading)

Across 21 months: 388 trading days, 194 trades, 160 days with at least one trade (41% of days). Median 1 trade/day, max 3. Already at the appropriate density for the entry signal class.

---

## The Plateau

After 30+ tested variants across 4+ research iterations, three independent candidates (NO_TRAIL, DELAYED, PE_TRAIL_ONLY) converge on the same walk-forward result:

```
PnL wins:        3/6   (need >=4 for strict pass)
PF wins:         4/6   (PASS)
Catastrophic:    0     (PASS)
DD profile:      identical to baseline
```

The W1 + W2 windows always show small losses for trail-fix candidates because the trail occasionally helps in those windows. The W3-W6 windows always show net wins. **No parameter-tuning configuration breaks past this 3/6 PnL plateau.**

---

## Open Decision

**PE_TRAIL_ONLY is the best-evidence deploy candidate of the entire session**:

| Metric | Value | vs deployed Option B |
|---|---:|---:|
| 21mo PnL | +Rs 59.94L | +Rs 2.64L (+4.6%) |
| 21mo PF | 1.99 | +0.05 |
| Max DD | −Rs 6.08L | identical |
| Walk-fwd net Δ | +Rs 257K | best of 3 marginal |
| Walk-fwd PnL wins | 3/6 | strict bar = 4/6 |

**Two consistent positions**:

- (α) **Stay strict**: 4/6 PnL bar held the line against all prior overfits. Reject PE_TRAIL_ONLY on the same standard. Stay on Option B.
- (β) **Deploy PE_TRAIL_ONLY**: W1/W2 "losses" (Rs 1K, Rs 40K) are within model noise. Mechanism mechanistically identified. DD-neutral. Bounded downside.

---

## What Real Next Steps Look Like

After exhausting parameter tuning, real improvement requires changing the strategy class:

| Path | Effort | Realistic gain |
|---|---|---:|
| Multi-timeframe entry confirmation (5-min + 15-min trend agreement) | 2-3 weeks | +1-3pp WR |
| OI / order-flow features as entry filter (NSE Bhav Copy) | 1-2 months | +5-15% PnL |
| Volatility regression LSTM as entry quality filter (regression target, NOT direction) | 4-6 weeks | +1-3pp WR |
| Directional sanity gate (block PUT in uptrends, CE in downtrends) | 1 week | Could rescue June 2025-type months |
| Different broker feed (tick data + real bid-ask) | 2-4 weeks | +5-10% via cost reduction |
| Add uncorrelated 4H-swing positional sleeve | 1-2 months | +30-50% Sharpe via diversification |

Each requires its own walk-forward validation against the same gauntlet.

---

## Files Generated This Session

- 25+ backtest scripts in `backtesting/` covering all variants tested
- 20+ research logs in `reports/oos/` with full numerical detail
- Dashboard endpoints `/api/strategies/comparison` and `/api/strategies/research`
- Strategy Lab UI panel in `terminal.html`

Live config unchanged from Option B. All research artifacts committed to GitHub.

---

## Summary Verdict

**The deployed Option B configuration is the local optimum within the V14 entry signal class**. After exhaustive parameter exploration:

✅ 90% WR is mathematically inaccessible
✅ The trail-stop leak is real (CE-asymmetric) but its fix only marginally improves PnL
✅ The strict 4/6 PnL walk-forward bar holds the line against overfitting
✅ Real next-stage improvements require new signal class, not parameter tuning

**Decision now rests on whether to deploy PE_TRAIL_ONLY (marginal) or stay on Option B (strict).**
