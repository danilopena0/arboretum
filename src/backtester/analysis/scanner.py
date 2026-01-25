"""Strategy parameter scanning with train/test validation.

Provides tools to:
1. Run parameter grid searches on strategies
2. Split data into train/test periods
3. Compare in-sample vs out-of-sample performance
4. Detect overfitting through performance degradation
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from itertools import product
from typing import Any

import polars as pl

from backtester.analysis.metrics import (
    PerformanceMetrics,
    calculate_metrics,
)
from backtester.core.engine import BacktestConfig, BacktestEngine, BacktestResult
from backtester.core.execution import SimulatedBroker
from backtester.data.handler import DataHandler
from backtester.strategies.base import Strategy


@dataclass
class ScanResult:
    """Result from a single parameter combination scan."""

    params: dict[str, Any]
    train_result: BacktestResult
    test_result: BacktestResult
    train_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics

    @property
    def sharpe_degradation(self) -> float:
        """How much Sharpe dropped from train to test (positive = degradation)."""
        if self.train_metrics.sharpe_ratio == 0:
            return 0.0
        return self.train_metrics.sharpe_ratio - self.test_metrics.sharpe_ratio

    @property
    def sharpe_ratio_train(self) -> float:
        return self.train_metrics.sharpe_ratio

    @property
    def sharpe_ratio_test(self) -> float:
        return self.test_metrics.sharpe_ratio

    @property
    def is_overfit(self) -> bool:
        """Simple overfit detection: test Sharpe < 50% of train Sharpe."""
        if self.train_metrics.sharpe_ratio <= 0:
            return False
        ratio = self.test_metrics.sharpe_ratio / self.train_metrics.sharpe_ratio
        return ratio < 0.5


@dataclass
class ScanSummary:
    """Summary of a parameter scan."""

    ticker: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    results: list[ScanResult]
    param_names: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pl.DataFrame:
        """Convert results to a DataFrame for analysis."""
        if not self.results:
            return pl.DataFrame()

        rows = []
        for r in self.results:
            row = {
                **r.params,
                "train_sharpe": r.train_metrics.sharpe_ratio,
                "train_return": r.train_metrics.total_return,
                "train_max_dd": r.train_metrics.max_drawdown,
                "train_num_trades": r.train_metrics.num_trades,
                "train_win_rate": r.train_metrics.win_rate,
                "test_sharpe": r.test_metrics.sharpe_ratio,
                "test_return": r.test_metrics.total_return,
                "test_max_dd": r.test_metrics.max_drawdown,
                "test_num_trades": r.test_metrics.num_trades,
                "test_win_rate": r.test_metrics.win_rate,
                "sharpe_degradation": r.sharpe_degradation,
                "is_overfit": r.is_overfit,
            }
            rows.append(row)

        return pl.DataFrame(rows)

    def best_by_test_sharpe(self, n: int = 5) -> pl.DataFrame:
        """Get top N parameter combinations by out-of-sample Sharpe."""
        df = self.to_dataframe()
        if df.is_empty():
            return df
        return df.sort("test_sharpe", descending=True).head(n)

    def best_robust(self, n: int = 5) -> pl.DataFrame:
        """Get top N by test Sharpe, excluding overfit combinations."""
        df = self.to_dataframe()
        if df.is_empty():
            return df
        return df.filter(pl.col("is_overfit").not_()).sort("test_sharpe", descending=True).head(n)


class StrategyScanner:
    """Scans parameter combinations for a strategy with train/test validation.

    Example:
        scanner = StrategyScanner(
            data_handler=YFinanceDataHandler(),
            strategy_factory=lambda p: MovingAverageCrossover(
                fast_period=p["fast"],
                slow_period=p["slow"],
            ),
            param_grid={"fast": [5, 10, 15, 20], "slow": [20, 30, 50, 100]},
        )

        summary = scanner.scan(
            ticker="SPY",
            train_start=date(2015, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2024, 12, 31),
        )

        print(summary.best_by_test_sharpe())
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy_factory: Callable[[dict[str, Any]], Strategy],
        param_grid: dict[str, list[Any]],
        config: BacktestConfig | None = None,
        broker: SimulatedBroker | None = None,
    ):
        """Initialize the scanner.

        Args:
            data_handler: Data source for backtests
            strategy_factory: Function that takes param dict and returns Strategy
            param_grid: Dict of param_name -> list of values to try
            config: Backtest configuration (shared across all runs)
            broker: Broker to use (shared across all runs)
        """
        self.data_handler = data_handler
        self.strategy_factory = strategy_factory
        self.param_grid = param_grid
        self.config = config or BacktestConfig()
        self.broker = broker or SimulatedBroker()

    def _generate_param_combinations(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from the grid."""
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))

        return combinations

    def _run_single(
        self,
        params: dict[str, Any],
        ticker: str,
        start: date,
        end: date,
    ) -> BacktestResult:
        """Run a single backtest with given parameters."""
        strategy = self.strategy_factory(params)

        engine = BacktestEngine(
            data_handler=self.data_handler,
            strategy=strategy,
            broker=self.broker,
            config=self.config,
        )

        return engine.run(
            tickers=[ticker],
            start=start,
            end=end,
        )

    def scan(
        self,
        ticker: str,
        train_start: date,
        train_end: date,
        test_start: date,
        test_end: date,
        verbose: bool = True,
    ) -> ScanSummary:
        """Run parameter scan with train/test split.

        Args:
            ticker: Ticker symbol to scan
            train_start: Training period start
            train_end: Training period end
            test_start: Test period start
            test_end: Test period end
            verbose: Print progress

        Returns:
            ScanSummary with all results
        """
        combinations = self._generate_param_combinations()
        results: list[ScanResult] = []

        if verbose:
            print(f"Scanning {len(combinations)} parameter combinations for {ticker}")
            print(f"Train: {train_start} to {train_end}")
            print(f"Test:  {test_start} to {test_end}")
            print("-" * 50)

        for i, params in enumerate(combinations):
            if verbose:
                print(f"[{i + 1}/{len(combinations)}] {params}", end=" ... ")

            # Run train period
            train_result = self._run_single(params, ticker, train_start, train_end)
            train_metrics = calculate_metrics(
                snapshots=train_result.snapshots,
                trades=train_result.trades,
                initial_capital=self.config.initial_capital,
            )

            # Run test period
            test_result = self._run_single(params, ticker, test_start, test_end)
            test_metrics = calculate_metrics(
                snapshots=test_result.snapshots,
                trades=test_result.trades,
                initial_capital=self.config.initial_capital,
            )

            scan_result = ScanResult(
                params=params,
                train_result=train_result,
                test_result=test_result,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
            )

            results.append(scan_result)

            if verbose:
                print(
                    f"Train Sharpe: {train_metrics.sharpe_ratio:.2f}, "
                    f"Test Sharpe: {test_metrics.sharpe_ratio:.2f}"
                    + (" [OVERFIT]" if scan_result.is_overfit else "")
                )

        if verbose:
            print("-" * 50)
            print(f"Scan complete. {len(results)} combinations tested.")

        return ScanSummary(
            ticker=ticker,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            results=results,
            param_names=list(self.param_grid.keys()),
        )


def scan_ma_crossover(
    ticker: str,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    fast_periods: list[int] | None = None,
    slow_periods: list[int] | None = None,
    data_handler: DataHandler | None = None,
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> ScanSummary:
    """Convenience function to scan MA crossover parameters.

    Args:
        ticker: Ticker to scan (e.g., "SPY")
        train_start: Training period start
        train_end: Training period end
        test_start: Test period start
        test_end: Test period end
        fast_periods: List of fast MA periods to try (default: [5, 10, 15, 20])
        slow_periods: List of slow MA periods to try (default: [20, 30, 50, 100])
        data_handler: Data handler (default: creates YFinanceDataHandler)
        config: Backtest config (default: 100k capital, 20% position size)
        verbose: Print progress

    Returns:
        ScanSummary with results

    Example:
        summary = scan_ma_crossover(
            ticker="SPY",
            train_start=date(2015, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2024, 12, 31),
        )

        # Best parameters by out-of-sample performance
        print(summary.best_by_test_sharpe())

        # Best robust parameters (excluding overfit)
        print(summary.best_robust())
    """
    from backtester.data.yfinance_handler import YFinanceDataHandler
    from backtester.strategies.examples.moving_average_crossover import (
        MovingAverageCrossover,
    )

    # Defaults
    if fast_periods is None:
        fast_periods = [5, 10, 15, 20]
    if slow_periods is None:
        slow_periods = [20, 30, 50, 100]
    if data_handler is None:
        data_handler = YFinanceDataHandler()
    if config is None:
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.2,
        )

    # Filter invalid combinations (fast must be < slow)
    valid_fast = []
    valid_slow = []
    for f in fast_periods:
        for s in slow_periods:
            if f < s:
                if f not in valid_fast:
                    valid_fast.append(f)
                if s not in valid_slow:
                    valid_slow.append(s)

    def strategy_factory(params: dict) -> Strategy:
        return MovingAverageCrossover(
            fast_period=params["fast"],
            slow_period=params["slow"],
        )

    # Build param grid filtering invalid combinations
    param_grid = {
        "fast": valid_fast,
        "slow": valid_slow,
    }

    scanner = StrategyScanner(
        data_handler=data_handler,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        config=config,
    )

    # Run scan
    summary = scanner.scan(
        ticker=ticker,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        verbose=verbose,
    )

    return summary


def scan_ema_crossover(
    ticker: str,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date,
    fast_periods: list[int] | None = None,
    slow_periods: list[int] | None = None,
    data_handler: DataHandler | None = None,
    config: BacktestConfig | None = None,
    verbose: bool = True,
) -> ScanSummary:
    """Convenience function to scan EMA crossover parameters.

    Args:
        ticker: Ticker to scan (e.g., "SPY")
        train_start: Training period start
        train_end: Training period end
        test_start: Test period start
        test_end: Test period end
        fast_periods: List of fast EMA periods to try (default: [5, 10, 15, 20])
        slow_periods: List of slow EMA periods to try (default: [20, 30, 50, 100])
        data_handler: Data handler (default: creates YFinanceDataHandler)
        config: Backtest config (default: 100k capital, 20% position size)
        verbose: Print progress

    Returns:
        ScanSummary with results

    Example:
        summary = scan_ema_crossover(
            ticker="SPY",
            train_start=date(2015, 1, 1),
            train_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2024, 12, 31),
        )

        # Best parameters by out-of-sample performance
        print(summary.best_by_test_sharpe())

        # Best robust parameters (excluding overfit)
        print(summary.best_robust())
    """
    from backtester.data.yfinance_handler import YFinanceDataHandler
    from backtester.strategies.examples.ema_crossover import EMACrossover

    # Defaults
    if fast_periods is None:
        fast_periods = [5, 10, 15, 20]
    if slow_periods is None:
        slow_periods = [20, 30, 50, 100]
    if data_handler is None:
        data_handler = YFinanceDataHandler()
    if config is None:
        config = BacktestConfig(
            initial_capital=100_000.0,
            position_size=0.2,
        )

    # Filter invalid combinations (fast must be < slow)
    valid_fast = []
    valid_slow = []
    for f in fast_periods:
        for s in slow_periods:
            if f < s:
                if f not in valid_fast:
                    valid_fast.append(f)
                if s not in valid_slow:
                    valid_slow.append(s)

    def strategy_factory(params: dict) -> Strategy:
        return EMACrossover(
            fast_period=params["fast"],
            slow_period=params["slow"],
        )

    # Build param grid filtering invalid combinations
    param_grid = {
        "fast": valid_fast,
        "slow": valid_slow,
    }

    scanner = StrategyScanner(
        data_handler=data_handler,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        config=config,
    )

    # Run scan
    summary = scanner.scan(
        ticker=ticker,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        verbose=verbose,
    )

    return summary
