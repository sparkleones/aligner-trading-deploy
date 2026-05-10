"""Quick NIFTY benchmark for the failing window."""
from .config_sweep import benchmark_nifty

for start, end in [
    ("2022-01-01", "2023-06-30"),
    ("2022-07-01", "2023-12-31"),
    ("2023-01-01", "2024-06-30"),
    ("2024-01-01", "2025-06-30"),
    ("2024-07-01", "2026-04-30"),
]:
    n = benchmark_nifty(start, end)
    if "cagr_pct" in n:
        print(f"  {start[:7]}..{end[:7]}  NIFTY CAGR={n['cagr_pct']*100:>6.2f}%  "
              f"DD={n['max_drawdown_pct']*100:>6.2f}%  Sharpe={n['sharpe']:.2f}")
