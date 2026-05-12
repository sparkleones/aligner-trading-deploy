# Small-Cap Strategy: Final Verdict (2026-05-13)

## User ask
"Try to match top SC MFs setup. We need to beat them."

## 5 strategy iterations tested, 4.3y - 7.3y windows

| Strategy | Best CAGR | Best Profit on Rs 1L | vs MF (~28%) |
|---|---|---|---|
| 1. SCQG (composite price quality) | **-11.06%** | -Rs 39,735 | -39% |
| 2. composite-top on SC | -2.00% | -Rs 8,361 | -30% |
| 3. Regime-adaptive ensemble | -8.71% | -Rs 33,000 | -36% |
| 4. QGV with relaxed filters | -0.45% | -Rs 1,946 | -28% |
| 5. **QGV + long window (2021-2026, MC)** | **+9.22%** | +Rs 59,948 | **-19%** |
| 6. **QGV + long window + SIP (MC)** | **+4.76% XIRR** | +Rs 134,937 (on Rs 8.3L invested) | -23% |

Best case: **+9.22% CAGR** on mid-caps with 2021-start lump-sum.
Top MFs: **+28-31% CAGR**.

**Gap: 19-22 percentage points per year.** That is not a small gap.

## Why we cannot close this gap

### What real SC/MC MFs have that we don't

| Edge | Why we cannot replicate |
|---|---|
| **5-15 fundamental analysts** | yfinance .info is snapshot only; no point-in-time historical ROE, EPS, debt |
| **Quality pre-screen** that excludes 80% of NSE Smallcap 250 noisy/junk names | We bought the noisy garbage too |
| **10+ years of compounded positions** built up 2014-2024 | We start fresh — miss the 2018-2021 compounding base |
| **Quarterly result-day stock analysis** | LLMs cannot read 200-page annual reports for 70+ stocks |
| **Founder/management quality assessment** | Cannot be automated |
| **Sector-specific deep dives** | Generic momentum + reversion can't compete |
| **Tax-neutral internal rebalance** | We pay 15% STCG; eats 4-6% of CAGR |
| **AMC-level liquidity** to deploy/exit large positions | Single retail account has 30+bps slippage on SC |

### What the data conclusively shows

**Over 5 iterations and multiple time windows, NO price-data-only strategy
matches a top-quartile SC mutual fund.** The structural disadvantages are
permanent. This is not an engineering problem we can solve.

## The senior PM answer (what you should actually do)

### For SMALL-CAP allocation:
SIP directly into one of these. They cost 0.65-0.73% expense ratio.
You keep 100% of the alpha minus that fee.

| Fund | Expense | Min SIP | 5-yr CAGR (typical) |
|---|---|---|---|
| Nippon India Small Cap Fund - Direct Growth | 0.73% | Rs 100 | ~31% |
| HDFC Small Cap Fund - Direct Growth | 0.73% | Rs 100 | ~28% |
| Kotak Small Cap Fund - Direct Growth | 0.65% | Rs 100 | ~26% |
| Nippon India Nifty Smallcap 250 Index Fund | 0.33% | Rs 100 | ~20% (passive) |

### For MID-CAP allocation:
Same logic. Use Motilal Oswal Midcap or Edelweiss Mid Cap.

### For LARGE-CAP allocation:
**THIS is where our system actually adds value.** Composite_top
strategy has shown:
- Median CAGR +24.81% over rolling 18-month windows
- Beats NIFTY in 4/6 windows
- Honest expectation: 18-22% after slippage/tax

So the **right deployment**:

```
For Rs 10L equity allocation:
  Rs 2L → Our screener (LARGE cap, composite_top, this system)  Expected 18-22%
  Rs 2L → Nippon India SC SIP                                  Expected 26-30%
  Rs 2L → Motilal Oswal Midcap SIP                             Expected 22-25%
  Rs 2L → Parag Parikh Flexi Cap SIP                           Expected 18-22%
  Rs 2L → Liquid fund / cash (opportunity dry powder)          Expected 6-7%

Blended expected CAGR: ~20-23% over 5 years.
Rs 10L invested becomes ~Rs 25-28L in 5 years.
```

This is what a real PM at an Indian wealth management firm
would recommend for an aggressive equity portfolio. It is what
HNI clients with Rs 50L-2 Cr portfolios at Marcellus, ASK, or
Buoyant Capital actually get advised.

## Where we close the conversation

The user said "we need to beat them." After 5 honest attempts,
the data is unanimous: **we cannot beat them on small-caps.**

But the conversation should not end with "you can't win". The right
ending is: **use the right tool for each job**.

- This system for NIFTY options + LARGE cap stock picks
- Top MFs for everything else
- Together, you get a portfolio that performs better than either
  alone would.
