# NIFTY Technical Analysis Research - Programmable Parameters
## Period: Oct 2025 - Apr 2026 | Compiled: April 5, 2026

---

## 1. CURRENT NIFTY TECHNICAL LEVELS (as of Apr 2, 2026)

### Price: 22,713.10

### Moving Averages (SMA)
| MA       | Value      | Signal  | Distance from Price |
|----------|-----------|---------|---------------------|
| 5 SMA    | 22,470.41 | Above   | +1.08%              |
| 10 SMA   | 22,503.63 | Above   | +0.93%              |
| 20 SMA   | 22,554.46 | Above   | +0.70%              |
| 50 SMA   | 22,775.89 | Below   | -0.28%              |
| 100 SMA  | 23,104.88 | Below   | -1.70%              |
| 200 SMA  | 24,051.09 | Below   | -5.56%              |

### Moving Averages (EMA)
| MA       | Value      | Signal  | Distance from Price |
|----------|-----------|---------|---------------------|
| 5 EMA    | 22,564.88 | Above   | +0.66%              |
| 10 EMA   | 22,520.55 | Above   | +0.86%              |
| 20 EMA   | 22,614.30 | Above   | +0.44%              |
| 50 EMA   | 22,795.99 | Below   | -0.36%              |
| 100 EMA  | 23,200.34 | Below   | -2.10%              |
| 200 EMA  | 23,775.05 | Below   | -4.47%              |

### KEY OBSERVATION: Price is below ALL long-term MAs (50+), confirming bearish regime.
### Price recently reclaimed short-term MAs (5/10/20), signaling potential relief rally.

### Key Support Levels
| Level       | Type                    | Value      |
|-------------|------------------------|-----------|
| Immediate   | Fibonacci/Pivot         | 22,500    |
| Near-term   | Technical               | 22,317    |
| Critical    | March 2026 low          | 22,071    |
| Major       | Swing support           | 21,750    |
| Extreme     | Fibonacci extension     | 21,000    |

### Key Resistance Levels
| Level       | Type                    | Value      |
|-------------|------------------------|-----------|
| Immediate   | Round number / OI wall  | 23,000    |
| Near-term   | Technical               | 23,110    |
| Fibonacci   | Retracement             | 23,355    |
| 50 SMA      | Moving average          | 22,776    |
| 100 SMA     | Moving average          | 23,105    |
| 200 SMA     | Major structural        | 24,051    |
| Bull/Bear   | Decisive zone           | 25,000-25,300 |

### Fibonacci Pivot Point: 22,602.97

---

## 2. INDICATORS - CURRENT VALUES & PROGRAMMABLE PARAMETERS

### RSI (14-period)
| Parameter           | Value    |
|---------------------|----------|
| Current RSI(14)     | 51.788   |
| Oversold threshold  | 30       |
| Overbought threshold| 70       |
| Bearish regime sell | RSI > 60 (in downtrend, rallies fail at lower RSI) |
| Recent low (Mar 13) | 24.15 (extreme oversold)  |
| Current recovery    | 38.90 -> 51.79            |

**BULLISH DIVERGENCE DETECTED (Mar 25, 2026):**
- Price made lower low at 22,471
- RSI made higher low vs 24.15 nadir on Mar 13
- Result: +350pt rally (~1.5%) to 23,350
- PROGRAMMABLE: Flag bullish divergence when price makes new low but RSI(14) > prior RSI low

### MACD (12, 26, 9)
| Parameter           | Value    |
|---------------------|----------|
| MACD Line           | -87.58   |
| Signal               | Bearish (MACD below zero) |
| Fast EMA             | 12       |
| Slow EMA             | 26       |
| Signal Line          | 9        |

**PROGRAMMABLE RULES:**
- Buy signal: MACD line crosses above signal line AND MACD rising from below -100
- Sell signal: MACD line crosses below signal line
- Trend filter: Only take longs when MACD > 0, shorts when MACD < 0

### Stochastic (9, 6)
| Parameter           | Value    |
|---------------------|----------|
| Current value       | 96.373   |
| StochRSI(14)        | 91.052   |
| Overbought          | > 80     |
| Oversold            | < 20     |

### ADX (14-period) - Trend Strength
| Parameter           | Value    |
|---------------------|----------|
| Current ADX         | 40.918   |
| Interpretation      | STRONG TREND |

**ADX REGIME CLASSIFICATION:**
| ADX Range  | Market State          | Strategy                    |
|------------|----------------------|----------------------------|
| < 20       | No trend / range     | Use mean reversion, sell strangles |
| 20-25      | Transition zone      | Watch for breakout          |
| 25-50      | Strong trend         | Use trend-following (Supertrend, MA crossovers) |
| > 50       | Very strong trend    | Trail stops, avoid counter-trend |

**CURRENT: ADX = 40.9 = STRONG BEARISH TREND. Use trend-following strategies.**

### ATR (14-period) - Volatility / Stop Loss
| Parameter           | Value       |
|---------------------|-------------|
| Current ATR(14)     | 193.90 pts  |
| ATR as % of price   | 0.85%       |

**ATR-BASED STOP LOSS PARAMETERS:**
| Multiplier | Stop Distance (pts) | Stop Distance (%) | Use Case              |
|------------|--------------------|--------------------|----------------------|
| 1.0x ATR   | 194 pts            | 0.85%              | Aggressive scalping   |
| 1.5x ATR   | 291 pts            | 1.28%              | Standard intraday     |
| 2.0x ATR   | 388 pts            | 1.71%              | Swing trading         |
| 2.5x ATR   | 485 pts            | 2.14%              | Position trading      |
| 3.0x ATR   | 582 pts            | 2.56%              | Chandelier exit       |

**CRITICAL: Your current 0.3% trail = ~68 pts. ATR is 194 pts. You are getting stopped out by normal noise.**
**RECOMMENDED: Minimum 1.0x ATR (194 pts / 0.85%) for intraday, 1.5x ATR (291 pts / 1.28%) for swing.**

### CCI (14-period)
| Parameter           | Value    |
|---------------------|----------|
| Current CCI         | 34.97    |
| Overbought          | > +100   |
| Oversold            | < -100   |

### Williams %R
| Parameter           | Value    |
|---------------------|----------|
| Current value       | -1.855   |
| Overbought          | > -20    |
| Oversold            | < -80    |
| CURRENT STATUS      | OVERBOUGHT (near-term pullback likely) |

### Bollinger Bands (20, 2)
| Parameter                | Value / Setting |
|--------------------------|----------------|
| Period                   | 20             |
| Standard deviations      | 2              |
| Intraday settings        | (10, 1.5) on 5-15min charts |
| Swing settings           | (50, 2.5)     |

**SQUEEZE DETECTION:**
| Metric                   | Threshold      |
|--------------------------|---------------|
| BBW < 2%                 | Extreme squeeze - breakout imminent |
| BBW 10-20%               | Normal conditions |
| BBW > 20%                | High volatility expansion |
| Squeeze lookback         | 120 periods (valid squeeze = lowest BBW in 120 bars) |
| Breakout confirmation    | Close beyond band + volume 1.5-2x average |

### Supertrend Parameters
| Setting       | Period | Multiplier | Use Case          |
|---------------|--------|------------|-------------------|
| Default       | 10     | 3          | Daily chart       |
| Intraday fast | 7      | 2          | 5-min scalping    |
| Intraday std  | 10     | 3          | 15-min trading    |
| NIFTY optimal | 4      | 10         | With 200 EMA filter |

**RULES:**
- Buy: Price crosses above Supertrend line AND MACD already positive
- Sell: Price crosses below Supertrend line AND MACD already negative
- Best on 5-min or 15-min intraday charts

### VWAP Parameters
| Parameter                | Value / Setting |
|--------------------------|----------------|
| Mean reversion threshold | 1-2% deviation from VWAP |
| Stabilization wait       | 30-60 min after open (skip 9:15-10:15) |
| Reversion win rate       | ~60% with 1:1 R:R |
| Bias filter              | Above VWAP = long bias, Below VWAP = short bias |

**PROGRAMMABLE VWAP RULES:**
- Long entry: Price > 1.5% below VWAP + exhaustion candle + falling volume
- Short entry: Price > 1.5% above VWAP + exhaustion candle + falling volume
- Target: VWAP level
- Stop: 0.5x ATR beyond entry

---

## 3. INTRADAY PATTERNS - STATISTICS

### Opening Range Breakout (ORB)
| Parameter                | Value           |
|--------------------------|----------------|
| ORB period               | First 15-30 min |
| Large gap-up into supply | Tends to trigger profit booking / liquidity grab |
| Mean reversion after 1%+ gap | 61% of the time within the following hour |

### Gap Fill Rates
| Metric                   | Value           |
|--------------------------|----------------|
| Overall gap fill rate    | 63%             |
| Bullish gap frequency    | 55% of all gaps |
| Bearish gap frequency    | 44% of all gaps |
| Common gaps              | Fill quickly (intraday) |
| Breakaway/exhaustion gaps| May not fill for weeks/months |

**PROGRAMMABLE:**
- After >1% opening gap: Enter mean-reversion trade with 61% probability
- Target: Previous close (gap fill level)
- Stop: 0.5x ATR beyond the gap extreme

### Hourly Patterns (10-year data, ~25,000 bars)
| Time Window    | Avg Move    | Green Prob  | Volatility  | Action            |
|----------------|-------------|-------------|-------------|-------------------|
| 9:15-10:15     | +/-0.19% SD | 54% green   | HIGHEST     | ORB setup zone    |
| 10:15-12:00    | Moderate    | ~50%        | Declining   | Trend continuation|
| 12:00-1:30     | Minimal     | ~50%        | LOWEST      | Lunch lull - avoid|
| 1:30-2:30      | Moderate    | ~50%        | Increasing  | Prepare for close |
| 2:30-3:30      | +0.21% avg  | 57% green   | Rising      | Closing rally bias|

**PROGRAMMABLE RULES:**
- Avoid new positions 12:00-1:30 (lunch lull, lowest volatility)
- Closing hour (2:30-3:30) has +0.21% avg return with 57% green probability
- After strong close (>0.75% up in last hour): Expect partial retracement next morning

### Last Hour Reversal
- Strong close (+0.75% in last hour): Partial retracement typical next morning
- Use this for overnight short positioning with tight stops

---

## 4. OPTIONS-SPECIFIC PATTERNS

### Put-Call Ratio (PCR) Extremes
| PCR Level   | Signal              | Action                     |
|-------------|---------------------|---------------------------|
| < 0.6       | Extreme call buying  | BEARISH - correction ahead |
| 0.6-0.8     | Bullish bias         | Mild resistance ahead      |
| 0.8-1.0     | Neutral              | No strong signal           |
| 1.0-1.2     | Bearish bias         | Mild support forming       |
| 1.2-1.6     | Strong put buying    | BULLISH - support strong   |
| > 1.6       | Extreme fear         | CONTRARIAN BUY signal      |

**REAL EXAMPLE (Mar 2026):**
- Feb 1, 2026: NIFTY 25,400, PCR = 0.72 (neutral)
- Mar 13, 2026: NIFTY 22,800, PCR spiked to 1.84 (extreme fear)
- Mar 17, 2026: NIFTY bounced +358 pts in 4 sessions
- PROGRAMMABLE: When PCR > 1.6, start scaling into longs

### OI Buildup Support/Resistance
| Pattern              | Condition                       | Signal    |
|----------------------|--------------------------------|-----------|
| Long Buildup         | Price up + OI up                | Bullish   |
| Short Buildup        | Price down + OI up              | Bearish   |
| Long Unwinding       | Price down + OI down            | Weak sell |
| Short Covering       | Price up + OI down              | Weak buy  |

**PROGRAMMABLE:**
- Resistance: Strike with highest CALL OI
- Support: Strike with highest PUT OI
- Expected range: [Highest Put OI strike] to [Highest Call OI strike]
- Update every 1-minute during market hours

### IV Crush Timing
| Event                | IV Behavior                     | Trading Rule             |
|----------------------|--------------------------------|--------------------------|
| Pre-Budget           | IV rises sharply               | Buy options 2+ weeks before |
| Post-Budget          | IV collapses (crush)           | Sell options into event   |
| Pre-RBI Policy       | IV rises 2-3 days before       | Sell straddles day before |
| Post-event           | IV drops 25-40%                | Don't buy options day before events |
| VIX 12 -> 21 spike   | Premiums nearly double         | This happened Nov 2025 - Jan 2026 |
| VIX 21 -> 11 drop    | Severe IV crush                | Happened by Mar 2026     |

**India VIX Current: 25.52 (moderate-high volatility)**

| VIX Level  | Interpretation        | Options Strategy         |
|------------|----------------------|--------------------------|
| < 12       | Low vol / complacent  | Buy options (cheap)      |
| 12-15      | Normal low            | Neutral                  |
| 15-18      | Normal                | Neutral                  |
| 18-25      | Elevated              | Sell premium cautiously  |
| 25-35      | High                  | Sell premium (expensive)  |
| > 35       | Panic                 | Contrarian buy signals   |

**CURRENT VIX = 25.52: Options are expensive. Favor selling strategies or debit spreads.**

### Volatility Skew
- OTM puts carry HIGHER IV than OTM calls (crash protection demand)
- Skew shape: "U" curve - OTM puts highest, ATM lowest, OTM calls moderate
- In bearish markets: Put skew steepens further

### Max Pain Effectiveness
| Metric                     | Value      |
|---------------------------|-----------|
| Max pain alignment rate    | 60-70% in liquid/calm markets |
| Monthly vs weekly          | Monthly shows STRONGER pinning |
| Best use                   | Final week before expiration |
| Reliability drops when     | High VIX, strong trends, macro events |

**PROGRAMMABLE:**
- Check max pain level daily
- If market is within 1% of max pain on Tuesday morning, expect pinning
- If VIX > 25 or ADX > 35, reduce max pain confidence to 40-50%

---

## 5. VOLUME ANALYSIS

### Volume Profile Framework
| Component          | Definition                        | Use                     |
|--------------------|----------------------------------|-------------------------|
| POC (Point of Control) | Price with highest volume    | Strongest S/R level     |
| Value Area High (VAH) | Upper 70% volume boundary    | Resistance              |
| Value Area Low (VAL)  | Lower 70% volume boundary    | Support                 |
| High Volume Node (HVN)| Clusters of heavy trading    | Sticky levels / consolidation |
| Low Volume Node (LVN) | Thin trading areas           | Price moves fast through these |

**PROGRAMMABLE:**
- Calculate daily POC from intraday volume data
- Value area = 70% of volume clustered around POC
- Expect price to spend 70% of time within value area
- LVN = potential breakout acceleration zones

### Delivery Percentage Signals
| Delivery %   | Interpretation                  |
|-------------|--------------------------------|
| > 50%       | Institutional buying/conviction |
| 30-50%      | Normal mixed activity           |
| < 30%       | Speculative/intraday dominated  |

### FII Index Futures OI
| Pattern                  | Signal                        |
|-------------------------|------------------------------|
| FII longs rising + OI up | Bullish institutional sentiment |
| FII shorts rising + OI up| Bearish institutional sentiment |
| FII long unwinding       | Institutions reducing risk    |

**Track daily via NSE participant-wise OI data.**

---

## 6. DAY-OF-WEEK PATTERNS (Post-Tuesday Expiry Shift)

### Weekly Return Patterns (10-year data)
| Day       | Avg Return | Win Rate | Key Pattern                    |
|-----------|-----------|----------|--------------------------------|
| Monday    | -0.09%    | 48%      | Slight negative bias, lowest volume |
| Tuesday   | Varies    | 56%      | EXPIRY DAY - highest volume, 40-60% theta decay in final hours |
| Wednesday | Neutral   | ~50%     | Post-expiry, new positions build |
| Thursday  | +/-0.41%  | 50%      | Largest average move           |
| Friday    | Moderate  | 47%      | End-of-week positioning, negative bias |

### Monthly Patterns
| Week          | Avg Return | Win Rate | Driver                      |
|---------------|-----------|----------|----------------------------|
| First week    | +0.32%    | 62%      | SIP inflows, new F&O positions |
| Last week     | -0.18%    | 55%      | Profit booking, rollover     |

### Pre-Holiday Pattern
| Metric         | Value    |
|----------------|----------|
| Day before holiday | +0.15% avg, 63% win rate |

### Tuesday Expiry Day Specifics
| Time              | Pattern                                    |
|-------------------|--------------------------------------------|
| 9:15-10:00 AM     | Directional confirmation window             |
| After 2:00 PM     | AVOID new positions                        |
| Final 30 min      | Settlement price calculation (3:00-3:30 PM)|
| Theta acceleration| 40-60% premium decay in final hours         |
| Volume spike      | 40-50% higher than other days              |

### Theta Decay Calendar
| Day from Expiry | Daily Decay Rate |
|-----------------|------------------|
| Thursday (T-5)  | 10-15%           |
| Friday (T-4)    | 15-20%           |
| Monday (T-1)    | 25-35%           |
| Tuesday (T-0)   | 40-60% final hrs |

**PROGRAMMABLE RULES:**
- Buy options on Thursday/Friday morning after directional confirmation
- Exit long options by Monday 3:00 PM (before Tuesday theta acceleration)
- Never sell naked ATM options on Tuesday morning (gamma risk)
- Stop-loss for option buys: 40-50% of premium paid

---

## 7. TRAILING STOP PARAMETERS

### Why 0.3% Trail is Too Tight
| Metric                   | Value           |
|--------------------------|----------------|
| Current ATR(14)          | 194 pts (0.85%)|
| Your current trail       | 0.3% (~68 pts) |
| Normal noise range       | ~1x ATR = 194 pts |
| Problem                  | 0.3% is only 0.35x ATR - stopped by noise |

### Recommended ATR-Based Trailing Stops
| Strategy         | ATR Multiple | Distance (pts) | Distance (%) | Use Case              |
|------------------|-------------|----------------|---------------|-----------------------|
| Tight scalp      | 1.0x        | 194            | 0.85%         | Sub-30min trades      |
| Standard intra   | 1.5x        | 291            | 1.28%         | Full-day intraday     |
| Swing trade      | 2.0x        | 388            | 1.71%         | 2-5 day holds         |
| Position trade   | 2.5x        | 485            | 2.14%         | Weekly holds          |
| Chandelier exit  | 3.0x        | 582            | 2.56%         | Major trend following |

### Chandelier Exit Settings
| Parameter    | Default | Aggressive | Conservative |
|-------------|---------|-----------|-------------|
| ATR Period   | 22      | 14        | 22          |
| Multiplier   | 3.0     | 2.0-2.5   | 3.5-4.0     |
| Long formula | Highest High(22) - 3.0 * ATR(22) | | |
| Short formula| Lowest Low(22) + 3.0 * ATR(22)  | | |

### Volatility-Adaptive Stop Rules
| ADX Reading | Market State | Recommended Stop |
|-------------|-------------|-----------------|
| < 20        | Range-bound  | Tighter: 1.0-1.5x ATR |
| 20-35       | Moderate     | Standard: 1.5-2.0x ATR |
| 35-50       | Strong trend | Wider: 2.0-2.5x ATR  |
| > 50        | Very strong  | Widest: 2.5-3.0x ATR |

**CURRENT ADX = 40.9: Use 2.0-2.5x ATR = 388-485 pts (1.71-2.14%)**

---

## 8. BREADTH INDICATORS

### Nifty 500 Stocks Above 200 DMA
| Threshold  | Signal                                      |
|-----------|---------------------------------------------|
| > 60%     | Healthy bull market                          |
| 40-60%    | Mixed / selective market                     |
| 20-40%    | Bearish conditions                           |
| < 20%     | HISTORICAL MARKET BOTTOM - contrarian buy    |

### Advance-Decline Ratio
| A/D Ratio  | Signal                                      |
|-----------|---------------------------------------------|
| > 2.0     | Strong bullish breadth                       |
| 1.0-2.0   | Moderate bullish                             |
| 0.5-1.0   | Bearish breadth                              |
| < 0.5     | Extreme bearish - potential bottom           |

**PROGRAMMABLE:**
- Track daily A/D ratio of Nifty 50 components
- If A/D < 0.5 AND PCR > 1.5: High-probability bounce setup
- If > 50% stocks above 50 DMA: Bullish bias
- If < 50% stocks above 50 DMA: Bearish bias

---

## 9. COMBINED SIGNAL SCORING SYSTEM (PROGRAMMABLE)

### Bullish Score (each condition = +1 point, max 10)
1. RSI(14) < 30 (oversold)
2. RSI bullish divergence present
3. PCR > 1.5
4. Price at/near highest PUT OI strike
5. India VIX > 25 (fear = opportunity)
6. ADX declining from > 40 (trend exhaustion)
7. Price within 1% of strong Fibonacci support
8. MACD histogram turning positive (momentum shift)
9. Nifty 500 stocks above 200 DMA < 20%
10. A/D ratio < 0.5

### Bearish Score (each condition = +1 point, max 10)
1. RSI(14) > 65 (overbought in downtrend)
2. PCR < 0.7
3. Price at/near highest CALL OI strike
4. India VIX < 12 (complacency)
5. ADX rising above 25 (trend strengthening)
6. Price below 200 SMA AND 200 EMA
7. MACD below zero and falling
8. Short buildup confirmed (falling price + rising OI)
9. Williams %R > -20 (overbought)
10. Price rejected at 50/100 SMA resistance

### Signal Interpretation
| Score    | Action                              |
|---------|-------------------------------------|
| 7-10    | Strong signal - full position        |
| 5-6     | Moderate signal - half position      |
| 3-4     | Weak signal - quarter position       |
| 0-2     | No signal - stay flat                |

---

## 10. REGIME-SPECIFIC PARAMETERS (Oct 2025 - Apr 2026 Bearish Market)

### Market Context
- NIFTY peaked at 26,104 in Oct 2025
- Declined to 22,471 low in Mar 2026 (-13.9% correction)
- Currently at 22,713 (Apr 2, 2026)
- Trading below ALL long-term moving averages
- ADX at 40.9 confirms STRONG downtrend
- India VIX at 25.52 (elevated fear)

### What Worked in This Regime
| Strategy                    | Effectiveness | Notes                      |
|-----------------------------|--------------|----------------------------|
| Trend following (short)     | HIGH         | ADX > 25 throughout        |
| PCR extreme contrarian buys | HIGH         | PCR 1.84 nailed the bottom |
| RSI divergence buys         | MODERATE     | Caught 350pt bounce        |
| Selling call spreads        | HIGH         | Bearish trend + high IV    |
| Mean reversion to VWAP      | MODERATE     | Works in consolidation phases |
| Max pain pinning            | LOW          | Unreliable in strong trends |
| ORB shorts                  | HIGH         | Gap-down opens continued   |

### Key Levels to Program
| Level          | Value     | Significance                |
|----------------|----------|-----------------------------|
| 200 SMA        | 24,051   | Must reclaim for bull case  |
| 200 EMA        | 23,775   | First major overhead barrier|
| 100 SMA        | 23,105   | Intermediate resistance     |
| 50 SMA         | 22,776   | Current nearby resistance   |
| Fibonacci pivot| 22,603   | Current pivot               |
| March 2026 low | 22,071   | Critical support            |
| Extreme support| 21,750   | If March low breaks         |

---

## SOURCES

- Investing.com NIFTY Technical Analysis: https://www.investing.com/indices/s-p-cnx-nifty-technical
- Trendlyne NIFTY 50 Technical: https://trendlyne.com/equity/technical-analysis/NIFTY/1887/nifty-50/
- Business Standard RSI Divergence Analysis: https://www.business-standard.com/markets/news/nifty-rsi-shows-positive-divergence-what-it-means-analysts-see-4-upside-126032500255_1.html
- PL Capital Tuesday Expiry Guide: https://www.plindia.com/blogs/nifty-weekly-options-strategy-tuesday-expiry-guide/
- NiftyInvest PCR Analysis: https://niftyinvest.com/put-call-ratio/NIFTY
- NiftyInvest Max Pain: https://niftyinvest.com/max-pain/NIFTY
- 5paisa NIFTY Outlook: https://www.5paisa.com/blog/nifty-outlook
- Multibagger Stock Ideas NIFTY Targets: https://www.multibaggerstockideas.com/2026/03/nifty-50-prediction-2026-targets-support-resistance.html
- AlphaEx Capital ATR Stop Loss Guide: https://www.alphaexcapital.com/stocks/technical-analysis-for-stock-trading/trading-strategies-using-technical-analysis/atr-based-stop-loss
- StockCharts Chandelier Exit: https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/chandelier-exit
- Elearnmarkets Supertrend Guide: https://blog.elearnmarkets.com/supertrend-indicator-strategy-trading/
- OneTradeJournal Pivot Points: https://onetradejournal.com/indicators/pivot-points
- Strike.money Max Pain: https://www.strike.money/options/max-pain-options
- Sahi.com Volatility & Options Pricing: https://www.sahi.com/blogs/7-volatility-and-options-pricing
- Quantsapp IV Skew: https://www.quantsapp.com/learn/articles/implied-volatility-and-skew-as-an-indicator-of-market-direction-187
- WealthBeats Market Breadth: https://wealthbeats.com/market-breadth-analysis/
- Medium NIFTY 50 Time Patterns (10yr analysis): https://medium.com/@stockdetails/unveiling-10-years-of-nifty-50-time-based-trading-patterns-what-the-data-really-tells-us-1c01d223875f
- Zerodha Varsity Open Interest: https://zerodha.com/varsity/chapter/open-interest-2/
- StockeZee FII DII Data: https://www.stockezee.com/fii-dii-data
- SCIRP Day of Week Effect Research: https://www.scirp.org/journal/paperinformation?paperid=86844
- TradingQnA Gap Fill Statistics: https://tradingqna.com/t/gap-filling-statistics/151053
