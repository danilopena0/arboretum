"""RSI Mean Reversion Strategy.

A mean reversion strategy that generates signals based on RSI (Relative Strength Index)
crossing oversold/overbought thresholds:
- BUY when RSI crosses below oversold threshold (e.g., 30)
- SELL when RSI crosses above overbought threshold (e.g., 70)
"""

from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy
from backtester.strategies.indicators.momentum import IncrementalRSI


class RSIMeanReversion(Strategy):
    """RSI Mean Reversion Strategy.

    Goes long when RSI drops below the oversold threshold and exits
    when RSI rises above the overbought threshold. This strategy
    exploits short-term price overreactions.

    Attributes:
        rsi_period: Period for RSI calculation (default 14)
        oversold: RSI level considered oversold (default 30)
        overbought: RSI level considered overbought (default 70)
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_id: str = "rsi_mean_reversion",
    ):
        """Initialize the strategy.

        Args:
            rsi_period: RSI calculation period (default 14)
            oversold: RSI level to trigger buy (default 30)
            overbought: RSI level to trigger sell (default 70)
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

        # Per-ticker indicators and state
        self._rsi: dict[str, IncrementalRSI] = {}
        self._prev_rsi: dict[str, float | None] = {}
        self._position: dict[str, bool] = {}  # True if long

    def set_tickers(self, tickers: list[str]) -> None:
        """Initialize indicators for each ticker.

        Args:
            tickers: List of tickers to trade
        """
        super().set_tickers(tickers)
        for ticker in tickers:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = False

    def reset(self) -> None:
        """Reset strategy state."""
        for ticker in self._tickers:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """Process market data and generate signals.

        Args:
            event: Market data event

        Returns:
            Signal if RSI threshold crossed, None otherwise
        """
        ticker = event.ticker

        # Initialize if needed (e.g., ticker not in set_tickers)
        if ticker not in self._rsi:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = False

        # Update RSI
        current_rsi = self._rsi[ticker].update(event.close)

        # Need RSI to be ready
        if current_rsi is None:
            return None

        # Get previous RSI
        prev_rsi = self._prev_rsi[ticker]

        # Store current RSI for next iteration
        self._prev_rsi[ticker] = current_rsi

        signal = None

        # Buy signal: RSI is in oversold territory (only if not already long)
        # Trigger when RSI crosses below OR is already below on first valid reading
        is_oversold = current_rsi < self.oversold
        was_not_oversold = prev_rsi is None or prev_rsi >= self.oversold

        if is_oversold and was_not_oversold and not self._position[ticker]:
            # Signal strength based on how oversold (lower RSI = stronger signal)
            strength = min(1.0, (self.oversold - current_rsi) / self.oversold + 0.5)
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
            )
            self._position[ticker] = True

        # Exit signal: RSI enters overbought territory (only if currently long)
        is_overbought = current_rsi > self.overbought
        was_not_overbought = prev_rsi is None or prev_rsi <= self.overbought

        if is_overbought and was_not_overbought and self._position[ticker]:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class DualRSIMeanReversion(Strategy):
    """Dual RSI Mean Reversion with both long and short positions.

    Goes long when RSI is oversold and short when RSI is overbought.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_threshold: float = 50.0,
        strategy_id: str = "dual_rsi_mean_reversion",
    ):
        """Initialize the strategy.

        Args:
            rsi_period: RSI calculation period
            oversold: RSI level to trigger long
            overbought: RSI level to trigger short
            exit_threshold: RSI level to exit positions (default 50 - neutral)
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_threshold = exit_threshold

        self._rsi: dict[str, IncrementalRSI] = {}
        self._prev_rsi: dict[str, float | None] = {}
        self._position: dict[str, str] = {}  # "long", "short", or "flat"

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = "flat"

    def reset(self) -> None:
        for ticker in self._tickers:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = "flat"

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._rsi:
            self._rsi[ticker] = IncrementalRSI(self.rsi_period)
            self._prev_rsi[ticker] = None
            self._position[ticker] = "flat"

        current_rsi = self._rsi[ticker].update(event.close)

        if current_rsi is None:
            return None

        prev_rsi = self._prev_rsi[ticker]
        self._prev_rsi[ticker] = current_rsi

        position = self._position[ticker]
        signal = None

        # Detect zone entries (crossing or first valid reading already in zone)
        is_oversold = current_rsi < self.oversold
        was_not_oversold = prev_rsi is None or prev_rsi >= self.oversold
        is_overbought = current_rsi > self.overbought
        was_not_overbought = prev_rsi is None or prev_rsi <= self.overbought

        # Entry signals
        if position == "flat":
            # Go long when entering oversold territory
            if is_oversold and was_not_oversold:
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, (self.oversold - current_rsi) / self.oversold + 0.5),
                )
                self._position[ticker] = "long"

            # Go short when entering overbought territory
            elif is_overbought and was_not_overbought:
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.SHORT,
                    strength=min(
                        1.0, (current_rsi - self.overbought) / (100 - self.overbought) + 0.5
                    ),
                )
                self._position[ticker] = "short"

        # Exit signals - exit when RSI reverts to neutral zone
        elif position == "long":
            if current_rsi >= self.exit_threshold:
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.EXIT,
                    strength=1.0,
                )
                self._position[ticker] = "flat"

        elif position == "short" and current_rsi <= self.exit_threshold:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = "flat"

        return signal
