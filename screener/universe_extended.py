"""
Extended NSE universe spanning Large / Mid / Small cap tiers.

Large cap   = Nifty 100 (top 100 by market cap)
Mid cap     = Nifty Midcap 150
Small cap   = Nifty Smallcap 250 (filtered to most liquid 80 for quality)

All NSE tickers — use to_yahoo_symbol() from universe.py for data fetch.

References:
- NSE indices methodology (Apr 2026): top 100 free-float = large cap
- AMFI categorization: 1-100 = large, 101-250 = mid, 251-500 = small
"""

# ── LARGE CAP: Nifty 100 ──
LARGE_CAP = [
    # Banks & NBFC
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "PFC",
    "RECLTD", "SBILIFE", "HDFCLIFE", "ICICIPRULI", "BAJAJHLDNG",
    "SHRIRAMFIN", "IDFCFIRSTB", "PNB", "BANKBARODA", "CANBK",
    # IT
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "PERSISTENT",
    # FMCG & Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
    "GODREJCP", "MARICO", "COLPAL", "TATACONSUM", "VBL",
    # Retail / Discretionary
    "TITAN", "TRENT", "DMART", "ETERNAL",
    # Auto
    "MARUTI", "M&M", "BAJAJ-AUTO", "EICHERMOT", "MOTHERSON",
    "HEROMOTOCO", "TVSMOTOR",
    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "ZYDUSLIFE", "TORNTPHARM",
    # Metals & Energy
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "JINDALSTEL",
    "COALINDIA", "NMDC", "HINDZINC",
    "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL",
    "POWERGRID", "NTPC", "ADANIGREEN", "ADANIPOWER",
    "ADANIPORTS", "ADANIENT", "ATGL",
    # Cement / Capital Goods / Infra
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "SHREECEM",
    "LT", "SIEMENS", "ABB", "BHEL", "DLF", "GODREJPROP",
    # Telecom / Misc
    "BHARTIARTL", "JIOFIN",
    "PIDILITIND", "ASIANPAINT", "BERGEPAINT", "HAVELLS",
    "INDIGO", "IRCTC", "BEL",
]

# ── MID CAP: Nifty Midcap 150 (curated ~80 most liquid) ──
MID_CAP = [
    # Banks/NBFC
    "FEDERALBNK", "BANDHANBNK", "AUBANK", "RBLBANK",
    "MUTHOOTFIN", "MANAPPURAM", "LICHSGFIN", "SUNDARMFIN",
    "POONAWALLA",
    # IT/Services
    "MPHASIS", "COFORGE", "OFSS",
    "KPITTECH", "TATAELXSI", "LTTS",
    # FMCG / Consumer
    "PGHH", "EMAMILTD", "JUBLFOOD", "PAGEIND", "PRESTIGE",
    "IDEAFORGE", "DIXON",
    # Auto
    "ASHOKLEY", "BHARATFORG", "BALKRISIND",
    "BOSCHLTD", "TIINDIA", "EXIDEIND",
    # Pharma/Healthcare
    "LUPIN", "AUROPHARMA", "BIOCON", "ALKEM",
    "GLENMARK", "FORTIS", "MAXHEALTH", "LALPATHLAB",
    "METROPOLIS", "POLYMED",
    # Metals/Chemicals
    "SAIL", "NATIONALUM",
    "DEEPAKNTR", "AARTIIND", "NAVINFLUOR",
    "TATACHEM", "PIIND", "SRF", "VINATIORGA",
    # Capital Goods
    "CUMMINSIND", "HONAUT", "POLYCAB", "VOLTAS",
    "CGPOWER", "KAJARIACER", "ASTRAL",
    # Power/Infra
    "TORNTPOWER", "TATAPOWER", "JSWENERGY", "NHPC",
    "CONCOR", "GMRAIRPORT", "IRB", "RVNL",
    # Telecom/Misc
    "INDHOTEL", "NAUKRI", "INDIANB", "UNIONBANK",
    "HDFCAMC", "ABCAPITAL",
    "JUBLPHARMA", "GLAND",
]

# ── SMALL CAP CORE: Liquid Nifty Smallcap 250 names selected by:
#    - Min market cap > Rs 2,500 Cr (avoid microcap)
#    - 20D avg turnover > Rs 25 Cr/day
#    - F&O eligibility preferred (but not required)
#    - Known "compounder" candidates per AMFI / Value Research
SMALL_CAP = [
    # Banks/NBFC
    "CSBBANK", "DCBBANK", "EQUITASBNK", "UJJIVANSFB",
    "CREDITACC", "ICRA", "CRISIL", "CARERATING",
    # IT/Tech
    "BIRLASOFT", "RATEGAIN", "TANLA", "SUBEX",
    "INTELLECT", "MASTEK", "ECLERX", "HAPPSTMNDS",
    # Consumer/Retail
    "BATA", "RELAXO", "VGUARD", "ORIENTELEC",
    "VIPIND", "BAJAJELEC", "AMBER", "STYLAMIND",
    # Auto/Ancillary
    "ENDURANCE", "MINDACORP", "GABRIEL", "JBMA",
    "JAMNAAUTO", "SUBROS",
    # Pharma
    "STAR", "ERIS", "CAPLIPOINT", "NATCOPHARM",
    "GRANULES", "AJANTPHARM", "SOLARA", "JBCHEPHARM",
    # Chemicals
    "ALKYLAMINE", "GHCL", "ROSSARI", "GALAXYSURF",
    "SUDARSCHEM", "TANFACIND",
    # Capital Goods/Industrials
    "ELGIEQUIP", "GRINDWELL", "TIMKEN", "SKFINDIA",
    "FINOLEXIND", "FINPIPE", "PRINCEPIPE", "APOLLOPIPE",
    "CARBORUNIV", "FIEMIND",
    # Power/Mining
    "GUJGASLTD", "PETRONET", "MGL", "IGL",
    "GMDCLTD", "GREENPLY",
    # Media/Entertainment/Travel
    "DBCORP", "SAREGAMA", "NAZARA", "ZEEL", "SUNTV",
    "EASEMYTRIP", "TBOTEK", "MAHLOG",
    # Misc consumer/B2B
    "TASTYBITE", "DEVYANI", "WESTLIFE", "WONDERLA",
    "JINDWORLD", "SCHAEFFLER",
]

# ── Combined universe + cap tier lookup ──
ALL_STOCKS = LARGE_CAP + MID_CAP + SMALL_CAP

CAP_TIER = {}
for s in LARGE_CAP:
    CAP_TIER[s] = "LARGE"
for s in MID_CAP:
    CAP_TIER[s] = "MID"
for s in SMALL_CAP:
    CAP_TIER[s] = "SMALL"


def get_universe(tier: str = "ALL") -> list[str]:
    """Return universe filtered by cap tier."""
    tier = tier.upper()
    if tier == "LARGE":
        return list(LARGE_CAP)
    if tier == "MID":
        return list(MID_CAP)
    if tier == "SMALL":
        return list(SMALL_CAP)
    return list(ALL_STOCKS)


def get_cap_tier(symbol: str) -> str:
    """Return cap tier for a symbol."""
    return CAP_TIER.get(symbol, "UNKNOWN")


if __name__ == "__main__":
    print(f"LARGE: {len(LARGE_CAP)}")
    print(f"MID:   {len(MID_CAP)}")
    print(f"SMALL: {len(SMALL_CAP)}")
    print(f"TOTAL: {len(ALL_STOCKS)}")
