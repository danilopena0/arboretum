"""Abstract data handler interface.

Defines the protocol for data handlers that provide market data
to the backtesting engine.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import date, datetime

import polars as pl

from backtester.core.events import MarketEvent


class DataHandler(ABC):
    """Abstract base class for data handlers.

    Data handlers are responsible for:
    - Fetching historical market data
    - Caching data for efficient access
    - Providing data as an iterator of MarketEvents

    Subclasses must implement the abstract methods to provide
    data from specific sources (yfinance, CSV, database, etc.)
    """

    @abstractmethod
    def get_bars(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> pl.LazyFrame:
        """Fetch OHLCV bars for the specified tickers and date range.

        Args:
            tickers: List of stock ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            interval: Bar interval (e.g., "1d", "1h", "5m")

        Returns:
            LazyFrame with columns: timestamp, ticker, open, high, low,
            close, volume, adj_close
        """
        ...

    @abstractmethod
    def get_latest_bar(self, ticker: str) -> MarketEvent | None:
        """Get the most recent bar for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MarketEvent for the latest bar, or None if no data
        """
        ...

    @abstractmethod
    def iter_bars(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> Iterator[MarketEvent]:
        """Iterate over bars in chronological order.

        Yields MarketEvents in timestamp order, across all tickers.
        This is the primary interface for the backtesting engine.

        Args:
            tickers: List of stock ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            interval: Bar interval

        Yields:
            MarketEvent instances in chronological order
        """
        ...

    @abstractmethod
    def update_bars(self) -> bool:
        """Update the data handler with the next bar.

        For live/streaming data, this would fetch new data.
        For historical data, this advances the internal iterator.

        Returns:
            True if new data is available, False if exhausted
        """
        ...

    def get_bar_count(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> int:
        """Get the total number of bars for the given parameters.

        Useful for progress tracking and memory estimation.

        Args:
            tickers: List of stock ticker symbols
            start: Start date
            end: End date
            interval: Bar interval

        Returns:
            Total number of bars
        """
        lf = self.get_bars(tickers, start, end, interval)
        return lf.select(pl.len()).collect().item()
