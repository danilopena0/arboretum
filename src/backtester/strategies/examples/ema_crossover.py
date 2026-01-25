"""EMA Crossover Strategy.

A trend-following strategy using Exponential Moving Averages that generates:
- BUY signal when fast EMA crosses above slow EMA
- SELL signal when fast EMA crosses below slow EMA

EMA gives more weight to recent prices compared to SMA, resulting in:
- Faster signals with less lag
- Better trend responsiveness
- Potentially more whipsaws in sideways markets
"""

from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy
from backtester.strategies.indicators.moving_averages import IncrementalEMA


class EMACrossover(Strategy):
    """EMA Crossover Strategy.

    Goes long when the fast EMA crosses above the slow EMA,
    and exits when it crosses below.

    Attributes:
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        strategy_id: str = "ema_crossover",
    ):
        """Initialize the strategy.

        Args:
            fast_period: Fast EMA period (default: 10)
            slow_period: Slow EMA period (default: 30)
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.fast_period = fast_period
        self.slow_period = slow_period

        # Per-ticker indicators and state
        self._fast_emas: dict[str, IncrementalEMA] = {}
        self._slow_emas: dict[str, IncrementalEMA] = {}
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
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None
            self._position[ticker] = False

    def reset(self) -> None:
        """Reset strategy state."""
        for ticker in self._tickers:
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
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
        if ticker not in self._fast_emas:
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None
            self._position[ticker] = False

        # Update indicators
        fast_ema = self._fast_emas[ticker].update(event.close)
        slow_ema = self._slow_emas[ticker].update(event.close)

        # EMA always returns a value, but wait for both to stabilize
        if not self._fast_emas[ticker].is_ready or not self._slow_emas[ticker].is_ready:
            # Still track values for crossover detection once ready
            self._prev_fast[ticker] = fast_ema
            self._prev_slow[ticker] = slow_ema
            return None

        # At this point EMA values are guaranteed to exist
        if fast_ema is None or slow_ema is None:
            return None

        # Get previous values
        prev_fast = self._prev_fast[ticker]
        prev_slow = self._prev_slow[ticker]

        # Store current values for next iteration
        self._prev_fast[ticker] = fast_ema
        self._prev_slow[ticker] = slow_ema

        # Need previous values to detect crossover
        if prev_fast is None or prev_slow is None:
            return None

        # Detect crossover
        signal = None

        # Bullish crossover: fast crosses above slow (only if not already long)
        if prev_fast <= prev_slow and fast_ema > slow_ema and not self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=1.0,
            )
            self._position[ticker] = True

        # Bearish crossover: fast crosses below slow (only if currently long)
        elif prev_fast >= prev_slow and fast_ema < slow_ema and self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class DualEMACrossover(Strategy):
    """Dual EMA Crossover with both long and short positions.

    Similar to EMACrossover but also takes short positions
    when the fast EMA crosses below the slow EMA.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        strategy_id: str = "dual_ema_crossover",
    ):
        """Initialize the strategy.

        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.fast_period = fast_period
        self.slow_period = slow_period

        self._fast_emas: dict[str, IncrementalEMA] = {}
        self._slow_emas: dict[str, IncrementalEMA] = {}
        self._prev_fast: dict[str, float | None] = {}
        self._prev_slow: dict[str, float | None] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

    def reset(self) -> None:
        for ticker in self._tickers:
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._fast_emas:
            self._fast_emas[ticker] = IncrementalEMA(self.fast_period)
            self._slow_emas[ticker] = IncrementalEMA(self.slow_period)
            self._prev_fast[ticker] = None
            self._prev_slow[ticker] = None

        fast_ema = self._fast_emas[ticker].update(event.close)
        slow_ema = self._slow_emas[ticker].update(event.close)

        # Wait for EMAs to stabilize
        if not self._fast_emas[ticker].is_ready or not self._slow_emas[ticker].is_ready:
            self._prev_fast[ticker] = fast_ema
            self._prev_slow[ticker] = slow_ema
            return None

        # At this point EMA values are guaranteed to exist
        if fast_ema is None or slow_ema is None:
            return None

        prev_fast = self._prev_fast[ticker]
        prev_slow = self._prev_slow[ticker]

        self._prev_fast[ticker] = fast_ema
        self._prev_slow[ticker] = slow_ema

        if prev_fast is None or prev_slow is None:
            return None

        # Bullish crossover -> go long
        if prev_fast <= prev_slow and fast_ema > slow_ema:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=1.0,
            )

        # Bearish crossover -> go short
        if prev_fast >= prev_slow and fast_ema < slow_ema:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.SHORT,
                strength=1.0,
            )

        return None
