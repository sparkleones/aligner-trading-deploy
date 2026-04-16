"""
Market constants for Indian NSE options trading.
Covers freeze limits, STT rates, trading hours, and SEBI compliance thresholds.
"""

# ─── NSE Quantity Freeze Limits (Revised March 2, 2026) ─────────────────────
FREEZE_LIMITS = {
    "NIFTY": 1800,
    "BANKNIFTY": 600,
    "FINNIFTY": 1200,
    "MIDCPNIFTY": 2800,
    "NIFTYNXT50": 600,
}

# ─── Securities Transaction Tax (Post-2026 Budget) ──────────────────────────
STT_RATES = {
    "options_sell": 0.0015,      # 0.15% on premium (sell side)
    "options_exercised": 0.0015, # 0.15% on intrinsic value
    "futures_sell": 0.0005,      # 0.05% on turnover (sell side)
}

# ─── Other Transaction Costs ────────────────────────────────────────────────
SEBI_TURNOVER_FEE = 0.000001    # ₹10 per crore
NSE_TRANSACTION_CHARGE = 0.0003553  # Options (₹35.53 per lakh turnover)
STAMP_DUTY_BUY = 0.00003       # 0.003% on buy side
GST_RATE = 0.18                 # 18% on brokerage + transaction charges

# ─── SEBI Compliance: Rate Limiting ─────────────────────────────────────────
MAX_ORDERS_PER_SECOND = 9       # Hard cap below 10 OPS SEBI threshold
SEBI_OPS_THRESHOLD = 10         # SEBI's formal threshold

# ─── Order Slicing ──────────────────────────────────────────────────────────
SLICE_DELAY_MIN_MS = 50         # Minimum delay between sliced orders
SLICE_DELAY_MAX_MS = 150        # Maximum delay between sliced orders

# ─── Trading Hours (IST) ────────────────────────────────────────────────────
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
PRE_OPEN_START_HOUR = 9
PRE_OPEN_START_MINUTE = 0

# Strategy-specific timing
STRADDLE_ENTRY_HOUR = 9
STRADDLE_ENTRY_MINUTE = 20      # 0920 straddle entry time

# ─── Index Configuration (NSE Lot Sizes effective Jan 2026) ─────────────────
# Ref: NSE circular Jan 2026 — lot sizes revised to align with index levels
INDEX_CONFIG = {
    "NIFTY": {
        "lot_size": 65,              # Revised SEBI Feb 2026 (was 75 → 65)
        "strike_interval": 50,
        "weekly_expiry_day": "Tuesday",  # SEBI Nov 2025: moved from Thursday to Tuesday
        "exchange": "NFO",
        "underlying_symbol": "NIFTY 50",
        "freeze_qty": 1800,
    },
    "BANKNIFTY": {
        "lot_size": 30,              # Revised Jan 2026 (was 25 → 30)
        "strike_interval": 100,
        "weekly_expiry_day": "Tuesday",  # SEBI Nov 2025: only 1 weekly expiry per exchange (NSE=Tue)
        "exchange": "NFO",
        "underlying_symbol": "NIFTY BANK",
        "freeze_qty": 600,
    },
    "FINNIFTY": {
        "lot_size": 65,              # Revised Jan 2026 (was 40 → 65)
        "strike_interval": 50,
        "weekly_expiry_day": "Tuesday",  # SEBI Nov 2025: consolidated to Tuesday
        "exchange": "NFO",
        "underlying_symbol": "NIFTY FIN SERVICE",
        "freeze_qty": 1200,
    },
}

# ─── Greeks Thresholds ──────────────────────────────────────────────────────
ATM_DELTA_THRESHOLD = 0.40      # Options with delta >= 0.40 considered ATM
OTM_DELTA_THRESHOLD = 0.15      # Options with delta <= 0.15 considered deep OTM

# ─── Risk Management Defaults ───────────────────────────────────────────────
DEFAULT_MAX_DAILY_LOSS_PCT = 0.03    # 3% of total capital
DEFAULT_MAX_POSITION_SIZE_PCT = 0.20 # 20% of capital per position (small ₹25K acct)
DEFAULT_MAX_OPEN_POSITIONS = 5       # Limited for small accounts
TRAILING_SL_ACTIVATION_PCT = 0.005   # 0.5% profit to activate trailing SL
TRAILING_SL_DISTANCE_PCT = 0.003     # 0.3% trailing distance

# ─── DDQN Agent Defaults ────────────────────────────────────────────────────
DDQN_STATE_DIM = 64
DDQN_ACTION_DIM = 5              # hold, buy_call, buy_put, sell_call, sell_put
DDQN_LEARNING_RATE = 0.0001
DDQN_GAMMA = 0.99
DDQN_EPSILON_START = 1.0
DDQN_EPSILON_END = 0.01
DDQN_EPSILON_DECAY = 0.9995
DDQN_BATCH_SIZE = 64
DDQN_MEMORY_SIZE = 100_000
DDQN_TARGET_UPDATE_FREQ = 10
DDQN_TRADE_PENALTY = 0.002      # Penalty per trade to discourage overtrading

# ─── India VIX Regime Thresholds ────────────────────────────────────────────
VIX_LOW = 12.0          # < 12: low vol — sell premium (straddles, condors)
VIX_NORMAL_LOW = 15.0   # 12-15: normal-low — sell spreads
VIX_NORMAL_HIGH = 20.0  # 15-20: normal-high — balanced
VIX_HIGH = 25.0         # 20-25: high — buy protection, tighten SL
VIX_EXTREME = 30.0      # > 30: extreme — reduce size, buy hedges only

# ─── Put-Call Ratio (PCR) Interpretation ────────────────────────────────────
PCR_OVERSOLD = 0.7      # < 0.7: extreme bullish — contrarian sell signal
PCR_BULLISH = 0.9       # 0.7-0.9: bullish positioning
PCR_NEUTRAL_LOW = 0.9   # 0.9-1.1: balanced
PCR_NEUTRAL_HIGH = 1.1
PCR_BEARISH = 1.3       # 1.1-1.3: bearish positioning
PCR_OVERBOUGHT = 1.5    # > 1.5: extreme bearish — contrarian buy signal

# ─── Max Pain Configuration ────────────────────────────────────────────────
MAX_PAIN_RELIABILITY_RANGE = 100  # Nifty settles within ±100 pts ~60% of time
MAX_PAIN_ENTRY_AFTER_HOUR = 13    # Use Max Pain signals after 1:30 PM
MAX_PAIN_ENTRY_AFTER_MINUTE = 30
MAX_PAIN_VIX_THRESHOLD = 18.0    # Only use Max Pain when VIX < 18

# ─── Open Interest (OI) Analysis ───────────────────────────────────────────
OI_SUPPORT_THRESHOLD = 0.15      # Put OI > 15% of total = strong support
OI_RESISTANCE_THRESHOLD = 0.15   # Call OI > 15% of total = strong resistance
OI_CHANGE_SIGNIFICANT = 0.10     # 10% OI change = significant shift
OI_BUILDUP_THRESHOLD = 1.5       # OI increase > 1.5x avg = fresh buildup

# ─── FII/DII Flow Thresholds (₹ crore) ────────────────────────────────────
FII_HEAVY_SELLING = -3000        # FII net < -3000 cr = heavy selling pressure
FII_HEAVY_BUYING = 3000          # FII net > +3000 cr = strong buying
DII_SUPPORT_THRESHOLD = 2000     # DII net > +2000 cr = domestic support

# ─── Option Chain & Greeks ─────────────────────────────────────────────────
IV_PERCENTILE_HIGH = 70          # IV > 70th percentile = sell premium
IV_PERCENTILE_LOW = 30           # IV < 30th percentile = buy premium
IV_SKEW_THRESHOLD = 0.05         # |put_iv - call_iv| / atm_iv > 5% = skew
GAMMA_RISK_THRESHOLD = 0.05      # High gamma near expiry = position risk

# ─── Intraday Timing Windows (IST) ────────────────────────────────────────
# First 15 min: high vol, avoid entries (gap and trap)
AVOID_ENTRY_UNTIL_HOUR = 9
AVOID_ENTRY_UNTIL_MINUTE = 30
# Last 30 min: theta accelerates, best for premium sellers
THETA_ACCELERATION_HOUR = 15
THETA_ACCELERATION_MINUTE = 0
# Expiry day sweet spot: 1:30-3:15 PM
EXPIRY_SWEET_SPOT_START_HOUR = 13
EXPIRY_SWEET_SPOT_START_MINUTE = 30

# ─── Network / Deployment ───────────────────────────────────────────────────
GRPC_PORT = 50051
WEBSOCKET_RECONNECT_BASE_DELAY_S = 1.0
WEBSOCKET_RECONNECT_MAX_DELAY_S = 60.0
WEBSOCKET_RECONNECT_MULTIPLIER = 2.0
