"""Moving Average Crossover Strategy.

A classic trend-following strategy that generates:
- BUY signal when fast MA crosses above slow MA
- SELL signal when fast MA crosses below slow MA
"""

from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy
from backtester.strategies.indicators.moving_averages import IncrementalSMA


class MovingAverageCrossover(Strategy):
    """Moving Average Crossover Strategy.

    Goes long when the fast moving average crosses above the slow moving
    average, and exits when it crosses below.

    Attributes:
        fast_period: Period for fast moving average
        slow_period: Period for slow moving average
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        strategy_id: str = "ma_crossover",
    ):
        """Initialize the strategy.

        Args:
            fast_period: Fast MA period (default: 10)
            slow_period: Slow MA period (default: 30)
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Per-ticker indicators and state
        self._fast_mas: dict[str, IncrementalSMA] = {}
        self._slow_mas: dict[str, IncrementalSMA] = {}
        self._prev_fast: dict[str, float | None] = {}
        self._prev_slow: dict[str, float | None] = {}
        self._position: dict[str, bool] = {}  # True if long

    def set_tickers(self, tickers: list[str]) -> None:
        """Initialize indicators for each ticker.

        Args:
            tickers: List of tickers to trade
        """
        super().set_tickers(tickers)
        for ticker in tickers:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None
            self._position[ticker] = False

    def reset(self) -> None:
        """Reset strategy state."""
        for ticker in self._tickers:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """Process market data and generate signals.

        Args:
            event: Market data event

        Returns:
            Signal if crossover detected, None otherwise
        """
        ticker = event.ticker

        # Initialize indicators if needed (e.g., ticker not in set_tickers)
        if ticker not in self._fast_mas:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None
            self._position[ticker] = False

        # Update indicators
        fast_ma = self._fast_mas[ticker].update(event.close)
        slow_ma = self._slow_mas[ticker].update(event.close)

        # Need both MAs to be ready
        if fast_ma is None or slow_ma is None:
            return None

        # Get previous values
        prev_fast = self._prev_fast[ticker]
        prev_slow = self._prev_slow[ticker]

        # Store current values for next iteration
        self._prev_fast[ticker] = fast_ma
        self._prev_slow[ticker] = slow_ma

        # Need previous values to detect crossover
        if prev_fast is None or prev_slow is None:
            return None

        # Detect crossover
        signal = None

        # Bullish crossover: fast crosses above slow (only if not already long)
        if prev_fast <= prev_slow and fast_ma > slow_ma and not self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=1.0,
            )
            self._position[ticker] = True

        # Bearish crossover: fast crosses below slow (only if currently long)
        elif prev_fast >= prev_slow and fast_ma < slow_ma and self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class DualMovingAverageCrossover(Strategy):
    """Dual MA Crossover with both long and short positions.

    Similar to MovingAverageCrossover but also takes short positions
    when the fast MA crosses below the slow MA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        strategy_id: str = "dual_ma_crossover",
    ):
        """Initialize the strategy.

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.fast_period = fast_period
        self.slow_period = slow_period

        self._fast_mas: dict[str, IncrementalSMA] = {}
        self._slow_mas: dict[str, IncrementalSMA] = {}
        self._prev_fast: dict[str, float | None] = {}
        self._prev_slow: dict[str, float | None] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

    def reset(self) -> None:
        for ticker in self._tickers:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._fast_mas:
            self._fast_mas[ticker] = IncrementalSMA(self.fast_period)
            self._slow_mas[ticker] = IncrementalSMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

        fast_ma = self._fast_mas[ticker].update(event.close)
        slow_ma = self._slow_mas[ticker].update(event.close)

        if fast_ma is None or slow_ma is None:
            return None

        prev_fast = self._prev_fast[ticker]
        prev_slow = self._prev_slow[ticker]

        self._prev_fast[ticker] = fast_ma
        self._prev_slow[ticker] = slow_ma

        if prev_fast is None or prev_slow is None:
            return None

        # Bullish crossover -> go long
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=1.0,
            )

        # Bearish crossover -> go short
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.SHORT,
                strength=1.0,
            )

        return None
