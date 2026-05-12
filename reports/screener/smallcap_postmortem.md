# Small-Cap Strategy Postmortem (2026-05-13)

## Goal
"Some small-cap SIPs have given the best returns. Can we apply similar
strategies to maximize our profits?"

## Tested Approaches

| Approach | Universe | Hold | Picks | Result CAGR | Profit on Rs 1L |
|---|---|---|---|---|---|
| SCQG (4-factor composite) | 69 SC stocks | annual | 12 | **-11.06%** | -Rs 39,735 |
| SCQG + SIP Rs 10k/mo | 69 SC stocks | annual | 12 | **-12.75%** | -Rs 139,615 |
| Composite-top (proven large-cap winner) | 69 SC stocks | quarterly | 5 | **-2.00%** | -Rs 8,361 |
| Composite-top + SIP Rs 10k/mo | 69 SC stocks | quarterly | 10 | **-11.40%** | -Rs 164,129 |
| **NIFTY Smallcap 250 (5y typical)** | passive index | — | — | **~22%** | Rs ~110,000 |
| **Top quartile SC MF (Nippon SC etc, 5y typical)** | active MF | — | — | **~25-31%** | Rs ~165,000 |
| **Large-cap composite (reference)** | 90 LC stocks | quarterly | 5 | **+12.77%** | +Rs 68,055 |

## Brutal Finding

**Every variant of stock-picking we tried on the small-cap universe LOST
money** while passive index SIP would have made ~22%/yr and top quartile
small-cap MFs ~28%/yr.

We cannot match Nippon India / HDFC / SBI Small Cap funds. Period.

## Why Top SC MFs Win And We Lose

Top small-cap funds operate with edges this codebase cannot replicate:

| Their edge | Why I can't replicate |
|---|---|
| **Fundamental research team** of 5-15 analysts | yfinance .info gives snapshot data only |
| **Point-in-time historical earnings/ROE/debt** | Not available in any free API |
| **Channel checks, mgmt meetings, plant visits** | Cannot be automated |
| **Quality pre-screen** that excludes ~80% of NSE smallcap 250 | I bought the noisy garbage too |
| **Patient capital** willing to hold through 30-50% drawdowns | My exit rules kicked in too early |
| **Sector concentration** based on bottom-up conviction | I diversify per rule |
| **No tax drag** (open-ended MFs don't pay STCG on rebalance) | I do — 15% eats my returns |

## Senior PM Conclusion

For small-cap exposure, the honest recommendation is:

> **Don't run small-cap stock picking through this system. SIP into actual
> top-quartile small-cap mutual funds via Groww / Zerodha Coin / Kuvera.**

### Concrete Action

| Vehicle | Type | Min SIP | Expected 5y CAGR (post-tax/expense) |
|---|---|---|---|
| Nippon India Small Cap Fund (Direct Growth) | Open-ended MF | Rs 100 | ~26% |
| HDFC Small Cap Fund (Direct Growth) | Open-ended MF | Rs 100 | ~23% |
| Kotak Small Cap Fund (Direct Growth) | Open-ended MF | Rs 100 | ~22% |
| Nippon India Nifty Smallcap 250 Index Fund | Index Fund | Rs 100 | ~20% (passive) |
| Nippon India Nifty Smallcap 250 ETF | ETF (NIFTYSMLCAP250) | 1 unit | ~20% (passive) |

The Direct Growth versions have ~0.50-1.00% lower expense ratios than
Regular plans — always pick Direct.

### What This System SHOULD Do For You

| Task | This system | Use external instead |
|---|---|---|
| NIFTY options trading | ✓ V14 / V15 engine | — |
| Large-cap stock picking | ✓ composite_top (+10% alpha, 4/6 windows beat NIFTY) | — |
| Mid-cap stock picking | partial — needs more validation | — |
| **Small-cap stock picking** | **✗ Does not beat passive index** | **SIP into Nippon/HDFC SC MF** |
| **Equity ETF allocation** | **—** | **Direct ETF SIP via broker** |

## What I'll Build Instead

A "**Mutual Fund Recommendations**" panel in the Stocks tab that:
1. Lists top quartile MFs by category (large/mid/small/index)
2. Shows their historical 1y/3y/5y/10y CAGRs
3. Links to the AMFI page for each
4. Honest framing: "For these allocations, prefer this MF over our screener"

This is what a real senior PM at a wealth firm would tell a client. They
would NOT pretend to beat Nippon India Small Cap with technical analysis.
