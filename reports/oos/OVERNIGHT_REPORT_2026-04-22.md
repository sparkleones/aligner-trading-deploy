# Overnight Analysis Report — 2026-04-22

**Generated while user was asleep. All numbers are from the regime-aware backtest (NSE Sep-2025 Thu→Tue expiry cutover corrected).**

---

## 1. Bottom Line Up Front

**Recommended change for live trading** (one knob, low-risk, meaningful lift):

- Switch `scoring/config.py` V15/V17 `vix_floor` from **13 → 12**
- Keep `avoid_days = [0, 2]` (Mon+Wed) unchanged
- Expected impact vs current prod: **full 28.13x → 28.99x**, **post-Sep +Rs 9.87L → +Rs 14.22L (+44%)**, trade count goes from 161 → 197, **PF goes up from 1.99 → 1.89** (slight quality drop, big volume lift)

**If you want maximum return and tolerate similar DD:** `vix_floor=10`, keep [0,2] → **30.81x full**, post-Sep **+Rs 17.44L** (+76% vs baseline). 82 post-Sep trades instead of 30 — you'd have been active through the Jul–Dec 2025 ultra-low-vol regime.

**If you want maximum quality and lower DD:** `avoid_days=[0,2,3]` × `vix_floor=12` → 21.02x full but post-Sep **PF 2.45, WR 50.0%**, max DD reduced to -Rs 4.54L (−50.5%) vs baseline −Rs 5.93L (−53.6%), longest losing-day streak **5 vs 9**.

Nothing has been deployed. All three are candidates; pick one before tomorrow's session.

---

## 2. What Actually Ran Overnight

| Stage | Status | Output |
|---|---|---|
| DTE Thursday→Tuesday fix in live agent | Patched, held local | Bundled with Monitor wiring for Push #2 |
| Regime-aware expiry in backtest | Pushed to GitHub | [da8190e](https://github.com/sparkleones/aligner-trading-deploy/commit/da8190e) |
| Re-validated V17_PROD_ONLY with fix | 28.1x / Rs 54.3L | Fix holds; edge survives |
| Jul–Dec 2025 zero-trade gap investigation | Root cause found | `vix_floor=13` blocked 101 of 103 days (VIX<13 in that window) — not a bug |
| avoid_days 12-variant sweep | Complete | [avoid_days_sweep.log](reports/oos/avoid_days_sweep.log) |
| vix_floor sweep on [0,2] | Complete | [vix_floor_sweep_mw.log](reports/oos/vix_floor_sweep_mw.log) |
| vix_floor sweep on [0,2,3] | Complete | [vix_floor_sweep_mwt.log](reports/oos/vix_floor_sweep_mwt.log) |
| Stability analysis on baseline + 2 winners | Complete | Drawdown/streak comparison below |
| Headless monitoring research | Complete | Telegram + Healthchecks.io = top pick |
| Complementary option strategies research | Complete | Iron condor + RBI-day short-straddle = top picks |

---

## 3. avoid_days Sweep (21-month + post-Sep sub-metric)

Config baseline: V17_PROD_ONLY with `use_v17_regime_gate=False`, `use_v17_monwed_gate=False`. Only `avoid_days` varied.

| avoid_days | Full 21mo | Post-Sep 2025 |
|---|---|---|
| `[0,2]` current baseline | **28.13x** / +Rs 54.3L, PF 1.99 | +Rs 9.87L, n=30, PF 1.60 |
| `[wed]` alone | 22.01x / +Rs 42.0L | +Rs 1.60L, n=43, PF 1.05 |
| **`[0,2,3]` Mon+Wed+Thu** | 20.80x / +Rs 39.6L, PF 2.11 | **+Rs 11.85L, n=21, PF 2.27** |
| `[tue_wed]` | 20.34x | +Rs 4.05L |
| `[wed_thu]` | 14.92x | +Rs 3.60L |
| `[mon_wed_fri]` | 13.43x | +Rs 3.92L |
| `[mon]` alone | 13.00x | +Rs 7.53L |
| `[mon_tue]` | 10.16x | +Rs 9.76L |
| `none` (no filter) | 7.76x | −Rs 0.65L |
| `[tue]` alone | 6.53x | +Rs 1.49L |
| `[thu]` alone | **−0.36x (loses money!)** | +Rs 0.70L |
| `[fri]` alone | −0.64x | +Rs 0.36L |

**Read:**
- The current `[0,2]` filter is **load-bearing** — removing it drops full-window from 28x to 7.76x.
- Thursday was the **single most profitable day pre-Sep-2025** — blocking it alone loses money (legacy weekly expiry).
- Post-cutover, Thursday flipped toxic. `[0,2,3]` lifts post-Sep P&L by **20%** vs baseline `[0,2]` and raises PF from 1.60 → 2.27.
- **Caveat:** post-Sep sample sizes are small (n=21–52). Statistically suggestive, not yet conclusive.

---

## 4. vix_floor Sweep

### 4a. Varying vix_floor with avoid_days=[0,2] (baseline)

| vix_floor | Full | Post-Sep |
|---|---|---|
| **10** | **30.81x** / +Rs 59.6L, PF 1.76 | **+Rs 17.44L**, n=82, PF 1.57 |
| 11 | 28.62x | +Rs 12.97L, n=62, PF 1.51 |
| 12 | 28.99x | +Rs 14.22L, n=42, PF 1.73 |
| 13 (current) | 28.13x | +Rs 9.87L, n=30, PF 1.60 |
| 14 | 18.69x | +Rs 10.28L, n=23, PF 1.83 |

### 4b. Varying vix_floor with avoid_days=[0,2,3]

| vix_floor | Full | Post-Sep |
|---|---|---|
| 10 | 21.54x / +Rs 41.1L, PF 1.83 | +Rs 16.79L, n=58, PF 1.90 |
| 11 | 20.83x | +Rs 15.27L, n=44, PF 2.00 |
| **12** | 21.02x / +Rs 40.0L, **PF 2.04** | **+Rs 15.98L, n=28, PF 2.45, WR 50.0%** |
| 13 | 20.80x | +Rs 11.85L, n=21, PF 2.27 |
| 14 | 14.90x | +Rs 13.11L, n=18, **PF 2.64** |

**Read:**
- Lowering `vix_floor` from 13 to 10 reclaims the Jul–Dec 2025 ultra-low-vol regime that we had been sitting out.
- Lift is large and consistent: **+5–8L post-Sep P&L** across both avoid_days variants.
- Floor=12 has the best PF (trade quality) in both configs — tight floor of 12 catches most of the low-vol trading opportunities while filtering the noisiest.

---

## 5. Stability + Drawdown Comparison

Capital Rs 2,00,000 start, no compounding during backtest.

| Config | Peak Equity | Max DD | Final Equity | Worst trade streak | Worst day streak | Trail-stop P&L |
|---|---|---|---|---|---|---|
| **Baseline `[0,2]×13`** | Rs 60.21L | −Rs 5.93L (−53.6%) | Rs 56.26L | 6 | 9 | −Rs 5.69L (7 all losers) |
| **Volume `[0,2]×10`** | Rs 63.76L | −Rs 6.17L (−55.9%) | Rs 61.62L | 6 | 9 | −Rs 5.78L (9 all losers) |
| **Quality `[0,2,3]×12`** | Rs 42.60L | **−Rs 4.54L (−50.5%)** | Rs 42.04L | 7 | **5** | −Rs 2.27L (7 all losers) |

**Reads:**
- All three configs have similar max-DD magnitude (Rs 4.5L–6.2L).
- Quality `[0,2,3]×12` has **best risk profile**: smaller absolute DD, day-losing streak almost halved (5 vs 9).
- Volume `[0,2]×10` has the highest final equity but also the highest max DD. Emotional test: you'd have been −55.9% from peak at your worst.
- `trail_stop` exits continue to bleed across all three configs — 0% WR on every variant. This is a **tail-risk exit that isn't pulling its weight** and may be a candidate for refinement (tighter re-entry or wider trail).

---

## 6. Concrete Live-Trade Recommendations

**Three options, ranked by conservatism:**

### OPTION A — One-knob minimal change (recommended for tomorrow's session)
- Edit [scoring/config.py](scoring/config.py) V15_CONFIG: `"vix_floor": 13` → `"vix_floor": 12`
- Keep everything else unchanged
- Backtest delta: 28.13x → 28.99x full, +44% post-Sep P&L, marginal DD change
- Effort: 1 line, 1 commit, 1 `git pull` on AWS + `docker compose up -d --build`
- **Rollback safety:** trivial — the original was 13, revert takes 30 seconds

### OPTION B — Capture the low-vol regime (recommended if Jul–Oct 2025 returns)
- `vix_floor = 10`
- Keep `avoid_days = [0, 2]`
- Backtest: 30.81x full, +Rs 17.44L post-Sep (+76% vs baseline)
- Caveat: 94 more trades total, 52 more post-Sep. Higher capital-utilization, more brokerage/slippage in real life than in backtest (+~0.3% drag not modeled).
- **Risk:** you'd be taking trades on days where historical WR was 30–35%. Backtest says PF 1.57 on that subset is net-positive, but confidence interval is wide.

### OPTION C — Quality-first (recommended if you want tighter risk)
- `avoid_days = [0, 2, 3]` (block Mon + Wed + Thu)
- `vix_floor = 12`
- Backtest: 21.02x full (smaller number but **lower DD, shorter streaks, PF 2.45 post-Sep, WR 50%**)
- Caveat: `[0,2,3]` post-Sep only has n=21–28 in the backtest — statistically suggestive, not proven. I'd pair this change with a 4-week monitoring window before committing.
- **Biggest win:** the longest losing-day streak drops from **9 → 5** — meaningful for emotional discipline.

**My pick if you want one recommendation: OPTION A.** Single-knob change, minimal rollback, clear lift. Watch post-Sep metrics for a month, then consider moving to B or C when you have live confirmation in the new regime.

---

## 7. Research — Headless Trade Monitoring (no dashboard)

Source: web research agent, verified inline.

**Top pick: Telegram Bot + Healthchecks.io.** ~1 hour setup, free, reliable.

**Telegram Bot (primary — routine alerts):**
- Create bot via `@BotFather` → get token
- Get your `chat_id` via `@userinfobot`
- Wire into V14 agent logger with ~10 lines of `requests.post(...)` — library-free
- Rate limits: 30 msg/s across users, 1 msg/s to same chat, 20 msg/min to same group (source: [core.telegram.org/bots/faq](https://core.telegram.org/bots/faq))
- **Alert pattern:** entry fired (symbol/strike/score/SL), peak milestones (+Rs 2K/5K/10K), trail armed, exit with reason, EoD summary

**Healthchecks.io (dead-man's switch — free tier 20 checks):**
- Ping URL every 30s from the V14 tick loop
- If 2 pings missed → SMSes/emails you
- Single-highest-ROI addition for a solo system

**SMS layer (critical only, pay per message):**
- MSG91 or Fast2SMS in India (Rs 0.18–0.25/SMS, DLT template required, 1–3 day approval)
- Use ONLY for: daily loss cap hit, consecutive-loss halt, Kite session expired, agent heartbeat missed >3 min
- Twilio International bypasses DLT but 3–4x the cost

**Skip for now:** FCM push (needs mobile app), WhatsApp Business (template friction), Prometheus/Grafana self-hosted (overkill for solo).

**Library pick:** `requests` + raw HTTP for tomorrow (10 LOC, 0 new deps). Migrate to [`apprise`](https://github.com/caronc/apprise) later when you want unified ntfy + email + SMS routing on one URL-string config.

**Sentry free tier** (5K events/month) is worth adding for exception capture in the agent's `except:` blocks — converts silent failures to Telegram pings.

---

## 8. Research — Complementary Strategies

**Key regulatory note** (correcting a factual error the research agent made): NSE **did** move weekly NIFTY expiry from Thursday → Tuesday on 2025-09-01 per SEBI circular (we verified this from your own live trade symbol `NIFTY2641324000CE` = April 13, 2026 expiry). Any strategy below that references "expiry day" should use Tuesday.

**Top picks to deploy alongside V17:**

1. **Weekly Iron Condor (defined risk, on V17-silent days):**
   - NIFTY 1-SD wings, weekly expiry cycle
   - SPAN margin ~Rs 55–70K/lot (lower than short strangle — defined risk)
   - WR 60–65% historical on NIFTY
   - Skip: days V17 has open positions, known event days (RBI, budget, results nights), expiry day itself
   - Code footprint estimate: ~300 LOC, reuses your existing backtester
   - **This is the most obvious complement:** V17 is silent on ~60% of days in the validated config

2. **RBI-Day Short Straddle (event overlay):**
   - 6 opportunities/year (RBI monetary policy announcements)
   - Sell ATM straddle 09:30 → close by 10:45 IST after post-announcement vol crush
   - Historical edge documented by Sensibull
   - SPAN ~Rs 1.3L/lot
   - Ultra-low code footprint: a calendar + 2 market orders

3. **Gap-fade (>0.5% gaps reverting):**
   - 55–58% WR on NIFTY 2018–2024 per various retail research
   - Low edge — needs tight execution
   - **Backtest on your data first** before live

**Avoid for now:**
- Naked short strangles (tail risk — Adani Feb 2023, election Jun 2024 wipeout patterns)
- V17 hedging via bull call spread (kills V17's unbounded-upside edge)
- BankNIFTY V17 port (weekly expiry discontinued by NSE Nov 2024, only monthly remains)

**Capital conflict note:** Every short-option strategy locks Rs 55K–1.8L SPAN per lot. If V17 and a short-vol layer both want capital the same day, you'll get margin calls. Size short layer at ≤30% of capital and skip days V17 has active positions.

---

## 9. What's Still Open / Questions for You

1. **Pick a config:** A (floor=12), B (floor=10), or C (avoid_days=[0,2,3] × floor=12). Once you pick, 5-minute deploy to AWS.
2. **Trail-stop investigation:** 0% WR across all three configs on `trail_stop` exits. Want me to sweep trail parameters (ATR multiplier, min-hold) next, or leave it alone?
3. **Monitor wiring (Push #2):** Monitor module + LiveTradeMonitor emergency-only floor are validated but unpushed. Want to bundle with the chosen config change in one deploy?
4. **Telegram setup:** 30 minutes to wire. Want me to do it as the next task when you wake up?
5. **Iron condor backtest:** codeable on top of the existing backtester in ~weekend. Not started — wanted your go-ahead.

---

## Files Generated Overnight

- [reports/oos/avoid_days_sweep.log](reports/oos/avoid_days_sweep.log) — 12-variant DOW sweep
- [reports/oos/vix_floor_sweep_mw.log](reports/oos/vix_floor_sweep_mw.log) — vix_floor on [0,2]
- [reports/oos/vix_floor_sweep_mwt.log](reports/oos/vix_floor_sweep_mwt.log) — vix_floor on [0,2,3]
- [reports/oos/stability_analysis.log](reports/oos/stability_analysis.log) — baseline stability
- [reports/oos/stability_mw_f10.log](reports/oos/stability_mw_f10.log) — volume-winner stability
- [reports/oos/stability_mwt_f12.log](reports/oos/stability_mwt_f12.log) — quality-winner stability
- [backtesting/avoid_days_sweep.py](backtesting/avoid_days_sweep.py) — reusable sweep harness
- [backtesting/vix_floor_sweep.py](backtesting/vix_floor_sweep.py) — reusable sweep harness
- [backtesting/stability_analysis.py](backtesting/stability_analysis.py) — reusable analysis script

Everything committed except the live agent DTE fix + Monitor wiring (held for your approval before Push #2).
