"""Monte Carlo Retrospective Simulation — robustness testing for strategies.

From research: "Use non-parametric Brownian bridge methods to simulate 1,000+
alternate, statistically plausible historical price paths. Only deploy parameter
sets that show consistent profitability, low equity curve variance, and tightly
controlled Value-at-Risk (VaR) across the majority of simulated paths."

This module generates synthetic price paths that preserve the statistical properties
of the original data (mean, variance, autocorrelation, fat tails) while exploring
alternate trajectories the market COULD have taken.

Usage:
    from backtesting.monte_carlo import MonteCarloSimulator

    mc = MonteCarloSimulator(historical_closes)
    results = mc.run_strategy(strategy_fn, n_paths=1000)
    mc.print_robustness_report(results)
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PathResult:
    """Result from running a strategy on a single synthetic path."""
    path_id: int
    final_pnl: float
    max_drawdown_pct: float
    n_trades: int
    win_rate: float
    sharpe_ratio: float
    profit_factor: float
    equity_curve: list[float] = field(default_factory=list)


@dataclass
class MonteCarloResult:
    """Aggregated results from Monte Carlo simulation across all paths."""
    n_paths: int
    n_profitable: int
    median_pnl: float
    mean_pnl: float
    std_pnl: float
    percentile_5: float       # 5th percentile (worst case)
    percentile_25: float
    percentile_75: float
    percentile_95: float      # 95th percentile (best case)
    median_sharpe: float
    median_max_dd: float
    var_95: float             # Value at Risk (5% confidence)
    cvar_95: float            # Conditional VaR (expected loss beyond VaR)
    consistency_pct: float    # % of paths that are profitable
    mean_win_rate: float
    path_results: list[PathResult] = field(default_factory=list)


class MonteCarloSimulator:
    """Generate synthetic price paths and test strategy robustness.

    Methods
    -------
    generate_paths(n_paths, method)
        Generate synthetic price paths from historical data.
    run_strategy(strategy_fn, n_paths)
        Run a strategy across multiple synthetic paths and report results.
    """

    def __init__(
        self,
        historical_closes: np.ndarray,
        bars_per_day: int = 75,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        historical_closes : array
            Array of historical close prices (e.g., 5-min bars).
        bars_per_day : int
            Number of bars per trading day (75 for 5-min bars, 9:15-15:30).
        seed : int, optional
            Random seed for reproducibility.
        """
        self.prices = np.asarray(historical_closes, dtype=np.float64)
        self.bars_per_day = bars_per_day
        self.rng = np.random.default_rng(seed)

        # Pre-compute return statistics
        self.log_returns = np.diff(np.log(self.prices))
        self.mu = np.mean(self.log_returns)
        self.sigma = np.std(self.log_returns, ddof=1)
        self.n_bars = len(self.prices)

        # Fat tail parameters (for Student-t simulation)
        if len(self.log_returns) > 30:
            from scipy import stats
            try:
                self.df, _, _ = stats.t.fit(self.log_returns)
                self.df = max(3, min(30, self.df))  # Clamp between 3-30
            except Exception:
                self.df = 5.0  # Default degrees of freedom
        else:
            self.df = 5.0

        logger.info(
            "MonteCarloSimulator | %d bars | mu=%.6f sigma=%.6f df=%.1f",
            self.n_bars, self.mu, self.sigma, self.df,
        )

    def generate_paths(
        self,
        n_paths: int = 1000,
        method: str = "bootstrap",
    ) -> np.ndarray:
        """Generate synthetic price paths.

        Parameters
        ----------
        n_paths : int
            Number of synthetic paths to generate.
        method : str
            Generation method:
            - "bootstrap": resample historical returns with replacement
            - "gbm": Geometric Brownian Motion with fat tails (Student-t)
            - "brownian_bridge": constrained paths matching start/end prices

        Returns
        -------
        paths : ndarray of shape (n_paths, n_bars)
            Synthetic price paths starting from historical start price.
        """
        n = self.n_bars

        if method == "bootstrap":
            return self._bootstrap_paths(n_paths, n)
        elif method == "gbm":
            return self._gbm_paths(n_paths, n)
        elif method == "brownian_bridge":
            return self._brownian_bridge_paths(n_paths, n)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _bootstrap_paths(self, n_paths: int, n: int) -> np.ndarray:
        """Block bootstrap: resample day-length blocks of returns.

        Preserves intraday autocorrelation structure while randomizing
        day-to-day sequence. More realistic than individual bar resampling.
        """
        block_size = self.bars_per_day
        n_returns = n - 1
        n_blocks = len(self.log_returns) // block_size

        if n_blocks < 3:
            # Not enough data for block bootstrap — fall back to individual
            return self._gbm_paths(n_paths, n)

        # Extract complete day blocks
        blocks = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            if end <= len(self.log_returns):
                blocks.append(self.log_returns[start:end])

        paths = np.zeros((n_paths, n))
        paths[:, 0] = self.prices[0]

        for p in range(n_paths):
            # Randomly sample day blocks
            returns = []
            while len(returns) < n_returns:
                block_idx = self.rng.integers(0, len(blocks))
                returns.extend(blocks[block_idx].tolist())
            returns = np.array(returns[:n_returns])

            # Build price path from returns
            log_prices = np.log(self.prices[0]) + np.concatenate([[0], np.cumsum(returns)])
            paths[p] = np.exp(log_prices[:n])

        return paths

    def _gbm_paths(self, n_paths: int, n: int) -> np.ndarray:
        """Geometric Brownian Motion with Student-t innovations (fat tails).

        Uses the historical drift and volatility but draws innovations
        from a Student-t distribution to capture fat tails observed in
        Indian index returns (negative skewness, excess kurtosis).
        """
        from scipy.stats import t as t_dist

        paths = np.zeros((n_paths, n))
        paths[:, 0] = self.prices[0]

        for p in range(n_paths):
            # Student-t innovations (heavier tails than normal)
            innovations = t_dist.rvs(
                df=self.df, size=n - 1, random_state=self.rng
            )
            # Scale to match historical volatility
            innovations = innovations * self.sigma / np.sqrt(self.df / (self.df - 2))
            # Add drift
            returns = self.mu + innovations

            log_prices = np.log(self.prices[0]) + np.concatenate([[0], np.cumsum(returns)])
            paths[p] = np.exp(log_prices[:n])

        return paths

    def _brownian_bridge_paths(self, n_paths: int, n: int) -> np.ndarray:
        """Brownian bridge: paths constrained to match start and end prices.

        From research: "non-parametric Brownian bridge methods to simulate
        alternate, statistically plausible historical price paths."

        These paths start at the same price and end at the same price as
        the historical data, but take different routes in between.
        """
        start_price = self.prices[0]
        end_price = self.prices[-1]
        start_log = np.log(start_price)
        end_log = np.log(end_price)

        paths = np.zeros((n_paths, n))
        paths[:, 0] = start_price

        for p in range(n_paths):
            # Generate a standard Brownian motion
            increments = self.rng.normal(0, self.sigma, n - 1)
            bm = np.concatenate([[0], np.cumsum(increments)])

            # Apply Brownian bridge conditioning
            # B_bridge(t) = B(t) - (t/T) * B(T) + (t/T) * (end_log - start_log)
            t = np.arange(n) / (n - 1)  # Normalized time [0, 1]
            bridge = bm - t * bm[-1] + t * (end_log - start_log) + start_log

            paths[p] = np.exp(bridge)

        return paths

    def run_strategy(
        self,
        strategy_fn: Callable,
        n_paths: int = 1000,
        starting_capital: float = 30000.0,
        method: str = "bootstrap",
    ) -> MonteCarloResult:
        """Run a strategy across multiple synthetic paths.

        Parameters
        ----------
        strategy_fn : callable
            Function with signature: strategy_fn(prices: np.ndarray, capital: float)
            -> dict with keys: {final_pnl, max_drawdown_pct, n_trades, win_rate,
                                sharpe_ratio, profit_factor, equity_curve}
        n_paths : int
            Number of synthetic paths to test.
        starting_capital : float
            Starting capital for each path run.
        method : str
            Path generation method ("bootstrap", "gbm", "brownian_bridge").

        Returns
        -------
        MonteCarloResult with aggregated statistics.
        """
        logger.info(
            "Monte Carlo run | paths=%d method=%s capital=%.0f",
            n_paths, method, starting_capital,
        )

        paths = self.generate_paths(n_paths, method)
        path_results = []

        for i in range(n_paths):
            try:
                result = strategy_fn(paths[i], starting_capital)
                pr = PathResult(
                    path_id=i,
                    final_pnl=result.get("final_pnl", 0),
                    max_drawdown_pct=result.get("max_drawdown_pct", 0),
                    n_trades=result.get("n_trades", 0),
                    win_rate=result.get("win_rate", 0),
                    sharpe_ratio=result.get("sharpe_ratio", 0),
                    profit_factor=result.get("profit_factor", 0),
                    equity_curve=result.get("equity_curve", []),
                )
                path_results.append(pr)
            except Exception as e:
                logger.warning("Path %d failed: %s", i, e)

            if (i + 1) % 100 == 0:
                logger.info("  Monte Carlo progress: %d/%d paths", i + 1, n_paths)

        if not path_results:
            return MonteCarloResult(
                n_paths=n_paths, n_profitable=0,
                median_pnl=0, mean_pnl=0, std_pnl=0,
                percentile_5=0, percentile_25=0, percentile_75=0, percentile_95=0,
                median_sharpe=0, median_max_dd=0,
                var_95=0, cvar_95=0, consistency_pct=0, mean_win_rate=0,
            )

        # Aggregate statistics
        pnls = np.array([r.final_pnl for r in path_results])
        sharpes = np.array([r.sharpe_ratio for r in path_results])
        drawdowns = np.array([r.max_drawdown_pct for r in path_results])
        win_rates = np.array([r.win_rate for r in path_results])

        n_profitable = int(np.sum(pnls > 0))

        # VaR and CVaR
        sorted_pnls = np.sort(pnls)
        var_idx = max(0, int(0.05 * len(sorted_pnls)))
        var_95 = float(sorted_pnls[var_idx])
        cvar_95 = float(np.mean(sorted_pnls[:var_idx + 1])) if var_idx > 0 else var_95

        result = MonteCarloResult(
            n_paths=len(path_results),
            n_profitable=n_profitable,
            median_pnl=float(np.median(pnls)),
            mean_pnl=float(np.mean(pnls)),
            std_pnl=float(np.std(pnls)),
            percentile_5=float(np.percentile(pnls, 5)),
            percentile_25=float(np.percentile(pnls, 25)),
            percentile_75=float(np.percentile(pnls, 75)),
            percentile_95=float(np.percentile(pnls, 95)),
            median_sharpe=float(np.median(sharpes)),
            median_max_dd=float(np.median(drawdowns)),
            var_95=var_95,
            cvar_95=cvar_95,
            consistency_pct=n_profitable / len(path_results) * 100,
            mean_win_rate=float(np.mean(win_rates)),
            path_results=path_results,
        )

        logger.info(
            "Monte Carlo DONE | profitable=%d/%d (%.1f%%) | "
            "median_pnl=%.0f | VaR95=%.0f | CVaR95=%.0f | Sharpe=%.2f",
            n_profitable, len(path_results), result.consistency_pct,
            result.median_pnl, result.var_95, result.cvar_95, result.median_sharpe,
        )

        return result

    @staticmethod
    def print_robustness_report(result: MonteCarloResult) -> str:
        """Format a robustness report from Monte Carlo results."""
        lines = [
            "=" * 60,
            "MONTE CARLO ROBUSTNESS REPORT",
            "=" * 60,
            f"Paths simulated:     {result.n_paths}",
            f"Profitable paths:    {result.n_profitable}/{result.n_paths} ({result.consistency_pct:.1f}%)",
            "",
            "P&L Distribution:",
            f"  5th percentile:    {result.percentile_5:>12,.0f}",
            f"  25th percentile:   {result.percentile_25:>12,.0f}",
            f"  Median:            {result.median_pnl:>12,.0f}",
            f"  Mean:              {result.mean_pnl:>12,.0f}",
            f"  75th percentile:   {result.percentile_75:>12,.0f}",
            f"  95th percentile:   {result.percentile_95:>12,.0f}",
            f"  Std Dev:           {result.std_pnl:>12,.0f}",
            "",
            "Risk Metrics:",
            f"  VaR (95%):         {result.var_95:>12,.0f}",
            f"  CVaR (95%):        {result.cvar_95:>12,.0f}",
            f"  Median Max DD:     {result.median_max_dd:>10.1f}%",
            f"  Median Sharpe:     {result.median_sharpe:>10.2f}",
            f"  Mean Win Rate:     {result.mean_win_rate:>10.1f}%",
            "",
            "Deployment Verdict:",
        ]

        # Verdict
        if result.consistency_pct >= 70 and result.median_sharpe > 0.5:
            lines.append("  ROBUST - Strategy passes Monte Carlo validation")
        elif result.consistency_pct >= 50:
            lines.append("  MARGINAL - Strategy has edge but inconsistent across paths")
        else:
            lines.append("  FRAGILE - Strategy likely overfit to historical data")

        lines.append("=" * 60)
        report = "\n".join(lines)
        print(report)
        return report
