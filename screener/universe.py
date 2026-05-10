"""
NSE F&O equity universe — liquid names suitable for swing trading.

Source: NSE F&O list as of Apr 2026 (rebalanced quarterly).
Filtered to top ~100 by 20-day median turnover. Microcap and recently
listed names excluded. SME and T2T segments excluded.

Symbol format: NSE ticker WITHOUT exchange suffix. For Yahoo Finance
historical data we append `.NS`. For Kite execution we use the symbol
as-is.
"""

# Core liquid F&O universe — 80 names, manually curated for ₹30k-₹2L account size
# All have:
#   - 20D avg turnover > ₹100 Cr
#   - F&O eligible (no overnight gap blowups from circuit-only stocks)
#   - Market cap > ₹15,000 Cr (no penny-stock manipulation)
FNO_UNIVERSE = [
    # Banks & NBFC
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "INDUSINDBK", "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "PFC",
    "RECLTD", "M&MFIN", "SBILIFE", "HDFCLIFE", "ICICIPRULI",
    # IT
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT",
    "COFORGE", "MPHASIS",
    # FMCG & Consumer
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
    "GODREJCP", "MARICO", "COLPAL", "TITAN", "TRENT",
    # Auto
    "MARUTI", "M&M", "TATAMOTORS", "BAJAJ-AUTO", "EICHERMOT",
    "HEROMOTOCO", "TVSMOTOR", "ASHOKLEY", "BOSCHLTD",
    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN",
    "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM",
    # Metals & Mining
    "TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "JINDALSTEL",
    "NMDC", "COALINDIA", "HINDZINC",
    # Oil & Gas / Energy
    "RELIANCE", "ONGC", "BPCL", "IOC", "GAIL", "POWERGRID", "NTPC",
    "ADANIPORTS", "ADANIENT",
    # Cement
    "ULTRACEMCO", "GRASIM", "AMBUJACEM", "ACC", "SHREECEM",
    # Capital Goods / Infra
    "LT", "SIEMENS", "ABB", "BHARATFORG", "CUMMINSIND",
    # Telecom & Media
    "BHARTIARTL", "IDEA",
    # Chemicals & Misc
    "PIDILITIND", "ASIANPAINT", "BERGEPAINT", "UPL", "PIIND",
    "DEEPAKNTR", "TATACONSUM", "MCDOWELL-N",
]


def get_universe() -> list[str]:
    """Return the active F&O screening universe."""
    return list(FNO_UNIVERSE)


def to_yahoo_symbol(nse_symbol: str) -> str:
    """Convert NSE ticker → Yahoo Finance symbol."""
    # Yahoo uses `&` differently; normalize edge cases
    yahoo_safe = nse_symbol.replace("&", "_")
    # MCDOWELL-N on NSE is MCDOWELL-N.NS on Yahoo
    return f"{yahoo_safe}.NS"


def to_kite_symbol(nse_symbol: str) -> str:
    """Convert NSE ticker → Kite tradingsymbol (cash segment)."""
    # Cash segment uses raw symbol; exchange = NSE
    return nse_symbol


# Sector tags for diversification check (avoid 3 banks if all rank high)
SECTOR_MAP = {
    # Banks
    "HDFCBANK": "BANK", "ICICIBANK": "BANK", "SBIN": "BANK",
    "KOTAKBANK": "BANK", "AXISBANK": "BANK", "INDUSINDBK": "BANK",
    # NBFC / Insurance
    "BAJFINANCE": "NBFC", "BAJAJFINSV": "NBFC", "CHOLAFIN": "NBFC",
    "PFC": "NBFC", "RECLTD": "NBFC", "M&MFIN": "NBFC",
    "SBILIFE": "INSURANCE", "HDFCLIFE": "INSURANCE", "ICICIPRULI": "INSURANCE",
    # IT
    "TCS": "IT", "INFY": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    "MPHASIS": "IT",
    # FMCG
    "HINDUNILVR": "FMCG", "ITC": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "DABUR": "FMCG", "GODREJCP": "FMCG",
    "MARICO": "FMCG", "COLPAL": "FMCG", "TATACONSUM": "FMCG",
    "MCDOWELL-N": "FMCG",
    # Consumer Disc
    "TITAN": "CONSUMER", "TRENT": "CONSUMER",
    # Auto
    "MARUTI": "AUTO", "M&M": "AUTO", "TATAMOTORS": "AUTO",
    "BAJAJ-AUTO": "AUTO", "EICHERMOT": "AUTO", "HEROMOTOCO": "AUTO",
    "TVSMOTOR": "AUTO", "ASHOKLEY": "AUTO", "BOSCHLTD": "AUTO",
    # Pharma
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "LUPIN": "PHARMA", "AUROPHARMA": "PHARMA",
    "BIOCON": "PHARMA", "TORNTPHARM": "PHARMA", "ALKEM": "PHARMA",
    # Metals
    "TATASTEEL": "METAL", "HINDALCO": "METAL", "JSWSTEEL": "METAL",
    "VEDL": "METAL", "JINDALSTEL": "METAL", "NMDC": "METAL",
    "COALINDIA": "METAL", "HINDZINC": "METAL",
    # Energy
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "BPCL": "ENERGY",
    "IOC": "ENERGY", "GAIL": "ENERGY", "POWERGRID": "POWER",
    "NTPC": "POWER", "ADANIPORTS": "INFRA", "ADANIENT": "INFRA",
    # Cement
    "ULTRACEMCO": "CEMENT", "GRASIM": "CEMENT", "AMBUJACEM": "CEMENT",
    "ACC": "CEMENT", "SHREECEM": "CEMENT",
    # Capital Goods
    "LT": "CAPGOODS", "SIEMENS": "CAPGOODS", "ABB": "CAPGOODS",
    "BHARATFORG": "CAPGOODS", "CUMMINSIND": "CAPGOODS",
    # Telecom
    "BHARTIARTL": "TELECOM", "IDEA": "TELECOM",
    # Chemicals & Paints
    "PIDILITIND": "CHEMICAL", "ASIANPAINT": "PAINT", "BERGEPAINT": "PAINT",
    "UPL": "CHEMICAL", "PIIND": "CHEMICAL", "DEEPAKNTR": "CHEMICAL",
}


def get_sector(symbol: str) -> str:
    """Return sector tag for a symbol; 'OTHER' if unknown."""
    return SECTOR_MAP.get(symbol, "OTHER")
