"""Bollinger Band Strategies.

Two strategy variants based on Bollinger Bands:
1. Mean Reversion: Buy at lower band, sell at upper band (range-bound markets)
2. Breakout: Buy on close above upper band (trending markets)
"""

from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy
from backtester.strategies.indicators.volatility import IncrementalBollingerBands


class BollingerBandMeanReversion(Strategy):
    """Bollinger Band Mean Reversion Strategy.

    Assumes price will revert to the mean:
    - Goes long when price touches/crosses below the lower band
    - Exits when price returns to the middle band (SMA)
    - Optionally goes short when price touches/crosses above upper band

    Best suited for range-bound, non-trending markets.

    Attributes:
        window: Bollinger Band SMA window (default 20)
        num_std: Number of standard deviations (default 2.0)
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        strategy_id: str = "bb_mean_reversion",
    ):
        """Initialize the strategy.

        Args:
            window: Bollinger Band SMA window
            num_std: Number of standard deviations for bands
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.window = window
        self.num_std = num_std

        # Per-ticker indicators and state
        self._bands: dict[str, IncrementalBollingerBands] = {}
        self._position: dict[str, bool] = {}  # True if long

    def set_tickers(self, tickers: list[str]) -> None:
        """Initialize indicators for each ticker.

        Args:
            tickers: List of tickers to trade
        """
        super().set_tickers(tickers)
        for ticker in tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = False

    def reset(self) -> None:
        """Reset strategy state."""
        for ticker in self._tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """Process market data and generate signals.

        Args:
            event: Market data event

        Returns:
            Signal if band condition met, None otherwise
        """
        ticker = event.ticker

        # Initialize if needed
        if ticker not in self._bands:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = False

        # Update Bollinger Bands
        middle, upper, lower = self._bands[ticker].update(event.close)

        # Need bands to be ready
        if middle is None or upper is None or lower is None:
            return None

        signal = None
        price = event.close

        # Calculate band width for signal strength
        band_width = upper - lower
        if band_width == 0:
            return None

        # Buy signal: price at or below lower band (only if not already long)
        if price <= lower and not self._position[ticker]:
            # Signal strength based on how far below the band
            distance_below = (lower - price) / band_width
            strength = min(1.0, 0.5 + distance_below)

            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
            )
            self._position[ticker] = True

        # Exit signal: price returns to middle band (only if currently long)
        elif price >= middle and self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class BollingerBandBreakout(Strategy):
    """Bollinger Band Breakout Strategy.

    Follows trends by entering on breakouts:
    - Goes long when price closes above upper band
    - Exits when price falls below middle band

    Best suited for trending markets. Can be combined with
    band squeeze detection for higher probability entries.

    Attributes:
        window: Bollinger Band SMA window (default 20)
        num_std: Number of standard deviations (default 2.0)
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        strategy_id: str = "bb_breakout",
    ):
        """Initialize the strategy.

        Args:
            window: Bollinger Band SMA window
            num_std: Number of standard deviations for bands
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.window = window
        self.num_std = num_std

        self._bands: dict[str, IncrementalBollingerBands] = {}
        self._prev_close: dict[str, float | None] = {}
        self._prev_upper: dict[str, float | None] = {}
        self._position: dict[str, bool] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._prev_close[ticker] = None
            self._prev_upper[ticker] = None
            self._position[ticker] = False

    def reset(self) -> None:
        for ticker in self._tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._prev_close[ticker] = None
            self._prev_upper[ticker] = None
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._bands:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._prev_close[ticker] = None
            self._prev_upper[ticker] = None
            self._position[ticker] = False

        middle, upper, lower = self._bands[ticker].update(event.close)

        if middle is None or upper is None or lower is None:
            return None

        prev_close = self._prev_close[ticker]
        prev_upper = self._prev_upper[ticker]

        self._prev_close[ticker] = event.close
        self._prev_upper[ticker] = upper

        # Need previous values to detect breakout
        if prev_close is None or prev_upper is None:
            return None

        signal = None
        price = event.close

        # Breakout signal: price crosses above upper band
        if prev_close <= prev_upper and price > upper and not self._position[ticker]:
            # Signal strength based on breakout magnitude
            breakout_pct = (price - upper) / upper if upper > 0 else 0
            strength = min(1.0, 0.5 + breakout_pct * 10)

            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
            )
            self._position[ticker] = True

        # Exit signal: price falls below middle band
        elif price < middle and self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class DualBollingerBand(Strategy):
    """Dual Bollinger Band Strategy with long and short positions.

    Mean reversion strategy that trades both sides:
    - Goes long when price touches lower band
    - Goes short when price touches upper band
    - Exits at middle band
    """

    def __init__(
        self,
        window: int = 20,
        num_std: float = 2.0,
        strategy_id: str = "dual_bb",
    ):
        """Initialize the strategy.

        Args:
            window: Bollinger Band SMA window
            num_std: Number of standard deviations for bands
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.window = window
        self.num_std = num_std

        self._bands: dict[str, IncrementalBollingerBands] = {}
        self._position: dict[str, str] = {}  # "long", "short", or "flat"

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = "flat"

    def reset(self) -> None:
        for ticker in self._tickers:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = "flat"

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._bands:
            self._bands[ticker] = IncrementalBollingerBands(self.window, self.num_std)
            self._position[ticker] = "flat"

        middle, upper, lower = self._bands[ticker].update(event.close)

        if middle is None or upper is None or lower is None:
            return None

        price = event.close
        position = self._position[ticker]
        signal = None

        band_width = upper - lower
        if band_width == 0:
            return None

        # Entry signals when flat
        if position == "flat":
            if price <= lower:
                distance_below = (lower - price) / band_width
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, 0.5 + distance_below),
                )
                self._position[ticker] = "long"

            elif price >= upper:
                distance_above = (price - upper) / band_width
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, 0.5 + distance_above),
                )
                self._position[ticker] = "short"

        # Exit signals at middle band
        elif (position == "long" and price >= middle) or (position == "short" and price <= middle):
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = "flat"

        return signal
