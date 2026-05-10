"""
Stock Screener module — separate from the V15 NIFTY options engine.

Picks 2-3 conviction equity longs from the NSE F&O universe using a
composite factor score (momentum + low-vol + quality + liquidity).

Runs as its own process and writes signals to dashboard endpoints.
Does NOT share capital, lock files, or order routing with the options
engine. Kill switch is independent.
"""

__version__ = "0.1.0"
