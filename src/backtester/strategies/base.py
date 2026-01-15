"""Base strategy interface.

Defines the abstract interface that all trading strategies must implement.
"""

from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime

from backtester.core.events import MarketEvent, SignalEvent, SignalType


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Strategies receive market events and generate trading signals.
    The strategy is responsible for maintaining any internal state
    (indicators, buffers, etc.) needed for decision making.

    Subclasses must implement:
    - on_market: Process market data and optionally return a signal

    Subclasses may override:
    - reset: Clear internal state
    - set_tickers: Initialize for specific tickers
    """

    def __init__(self, strategy_id: str = "default"):
        """Initialize the strategy.

        Args:
            strategy_id: Unique identifier for this strategy instance
        """
        self.strategy_id = strategy_id
        self._tickers: list[str] = []

    @abstractmethod
    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """Process a market event and optionally generate a signal.

        This is called for each bar of market data. The strategy
        should update internal state and decide whether to generate
        a trading signal.

        Args:
            event: Market data event

        Returns:
            SignalEvent if a trade signal is generated, None otherwise
        """
        ...

    def set_tickers(self, tickers: list[str]) -> None:
        """Initialize strategy for trading specific tickers.

        Called before the backtest starts. Override to perform
        ticker-specific initialization.

        Args:
            tickers: List of tickers to trade
        """
        self._tickers = tickers

    def reset(self) -> None:
        """Reset strategy state.

        Called at the start of each backtest. Override to clear
        any internal state. Default implementation clears ticker list.
        """
        self._tickers = []

    def create_signal(
        self,
        timestamp: datetime,
        ticker: str,
        signal_type: SignalType,
        strength: float = 1.0,
    ) -> SignalEvent:
        """Helper to create a signal event.

        Args:
            timestamp: Signal timestamp
            ticker: Stock ticker
            signal_type: LONG, SHORT, or EXIT
            strength: Signal strength (0.0 to 1.0)

        Returns:
            SignalEvent
        """
        return SignalEvent(
            timestamp=timestamp,
            ticker=ticker,
            signal_type=signal_type,
            strength=strength,
            strategy_id=self.strategy_id,
        )


class BarBuffer:
    """Fixed-size buffer for storing recent bars.

    Useful for strategies that need to look back at recent price history.
    Maintains separate buffers for each ticker.
    """

    def __init__(self, maxlen: int = 100):
        """Initialize the buffer.

        Args:
            maxlen: Maximum number of bars to store per ticker
        """
        self.maxlen = maxlen
        self._buffers: dict[str, deque[MarketEvent]] = {}

    def add(self, event: MarketEvent) -> None:
        """Add a bar to the buffer.

        Args:
            event: Market event to add
        """
        ticker = event.ticker
        if ticker not in self._buffers:
            self._buffers[ticker] = deque(maxlen=self.maxlen)
        self._buffers[ticker].append(event)

    def get(self, ticker: str) -> list[MarketEvent]:
        """Get all bars for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            List of bars (oldest first)
        """
        if ticker not in self._buffers:
            return []
        return list(self._buffers[ticker])

    def get_closes(self, ticker: str) -> list[float]:
        """Get closing prices for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            List of closing prices (oldest first)
        """
        return [bar.close for bar in self.get(ticker)]

    def get_latest(self, ticker: str, n: int = 1) -> list[MarketEvent]:
        """Get the n most recent bars for a ticker.

        Args:
            ticker: Stock ticker
            n: Number of bars to return

        Returns:
            List of most recent bars (oldest first)
        """
        bars = self.get(ticker)
        return bars[-n:] if len(bars) >= n else bars

    def count(self, ticker: str) -> int:
        """Get number of bars stored for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Number of bars
        """
        if ticker not in self._buffers:
            return 0
        return len(self._buffers[ticker])

    def is_ready(self, ticker: str, required: int) -> bool:
        """Check if buffer has enough bars.

        Args:
            ticker: Stock ticker
            required: Minimum bars needed

        Returns:
            True if buffer has at least required bars
        """
        return self.count(ticker) >= required

    def clear(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()

    def clear_ticker(self, ticker: str) -> None:
        """Clear buffer for a specific ticker.

        Args:
            ticker: Stock ticker
        """
        if ticker in self._buffers:
            self._buffers[ticker].clear()
