"""YFinance data handler with DuckDB caching.

Fetches historical market data from Yahoo Finance with intelligent
caching to minimize API calls and improve performance.
"""

import logging
import time
from collections.abc import Iterator
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import yfinance as yf

from backtester.core.events import MarketEvent
from backtester.data.cache import DuckDBCache
from backtester.data.handler import DataHandler
from backtester.data.schemas import date_to_datetime, normalize_ohlcv_columns

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to avoid hitting API limits."""

    def __init__(self, calls_per_second: float = 2.0):
        """Initialize rate limiter.

        Args:
            calls_per_second: Maximum API calls per second
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call: float = 0.0

    def wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        elapsed = time.time() - self.last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_call = time.time()


class YFinanceDataHandler(DataHandler):
    """Data handler that fetches from Yahoo Finance with caching.

    Features:
    - DuckDB-based persistent cache
    - Smart cache hit detection (only fetch missing ranges)
    - Rate limiting to avoid API blocks
    - Lazy evaluation with Polars LazyFrames

    Attributes:
        cache: DuckDB cache instance
        rate_limiter: Rate limiter for API calls
    """

    def __init__(
        self,
        cache_path: str | Path = "data/market_cache.duckdb",
        calls_per_second: float = 2.0,
    ):
        """Initialize the YFinance data handler.

        Args:
            cache_path: Path to the DuckDB cache file
            calls_per_second: Maximum yfinance API calls per second
        """
        self.cache = DuckDBCache(cache_path)
        self.rate_limiter = RateLimiter(calls_per_second)
        self._current_data: pl.DataFrame | None = None
        self._current_index: int = 0

    def _fetch_from_yfinance(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pl.DataFrame:
        """Fetch data from yfinance API.

        Args:
            ticker: Stock ticker symbol
            start: Start datetime
            end: End datetime
            interval: Data interval

        Returns:
            DataFrame with OHLCV data
        """
        self.rate_limiter.wait()

        logger.info(f"Fetching {ticker} from {start} to {end}")

        try:
            # yfinance end date is exclusive, add 1 day
            yf_end = end + timedelta(days=1)

            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(
                start=start,
                end=yf_end,
                interval=interval,
                auto_adjust=False,  # Keep original and adjusted prices
            )

            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pl.DataFrame()

            # Convert pandas to polars
            pl_df = pl.from_pandas(df.reset_index())

            # Add ticker column
            pl_df = pl_df.with_columns(pl.lit(ticker).alias("ticker"))

            # Normalize column names
            pl_df = normalize_ohlcv_columns(pl_df)

            # Handle the Date index from yfinance
            if "Date" in pl_df.columns:
                pl_df = pl_df.rename({"Date": "timestamp"})
            elif "Datetime" in pl_df.columns:
                pl_df = pl_df.rename({"Datetime": "timestamp"})

            # Ensure timestamp is datetime
            if pl_df.schema.get("timestamp") == pl.Date:
                pl_df = pl_df.with_columns(
                    pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp")
                )

            return pl_df

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            raise

    def get_bars(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> pl.LazyFrame:
        """Fetch OHLCV bars with caching.

        First checks the cache, then fetches any missing data from yfinance.

        Args:
            tickers: List of stock ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            interval: Bar interval (currently only "1d" supported for caching)

        Returns:
            LazyFrame with OHLCV data for all tickers
        """
        start_dt = date_to_datetime(start)
        end_dt = date_to_datetime(end)

        all_data: list[pl.DataFrame] = []

        for ticker in tickers:
            # Check what's missing from cache
            missing_ranges = self.cache.get_missing_ranges(ticker, start_dt, end_dt)

            # Fetch missing data
            for range_start, range_end in missing_ranges:
                fetched = self._fetch_from_yfinance(
                    ticker, range_start, range_end, interval
                )
                if not fetched.is_empty():
                    self.cache.store_data(fetched)

            # Get data from cache
            cached = self.cache.get_cached_data(ticker, start_dt, end_dt)
            if cached is not None:
                all_data.append(cached.collect())

        if not all_data:
            # Return empty LazyFrame with correct schema
            from backtester.data.schemas import create_empty_ohlcv_frame

            return create_empty_ohlcv_frame().lazy()

        # Combine all data and sort by timestamp
        combined = pl.concat(all_data)
        return combined.sort(["timestamp", "ticker"]).lazy()

    def get_latest_bar(self, ticker: str) -> MarketEvent | None:
        """Get the most recent bar for a ticker from cache.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MarketEvent for the latest bar, or None if no data
        """
        cached_range = self.cache.get_cached_range(ticker)
        if cached_range is None:
            return None

        _, max_date = cached_range
        data = self.cache.get_cached_data(ticker, max_date, max_date)
        if data is None:
            return None

        df = data.collect()
        if df.is_empty():
            return None

        row = df.row(-1, named=True)
        return MarketEvent(
            timestamp=row["timestamp"],
            ticker=row["ticker"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            adj_close=row.get("adj_close"),
        )

    def iter_bars(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> Iterator[MarketEvent]:
        """Iterate over bars in chronological order.

        Yields MarketEvents sorted by timestamp, then by ticker.

        Args:
            tickers: List of stock ticker symbols
            start: Start date (inclusive)
            end: End date (inclusive)
            interval: Bar interval

        Yields:
            MarketEvent instances
        """
        # Fetch and cache all data first
        lf = self.get_bars(tickers, start, end, interval)
        df = lf.collect()

        # Iterate in chronological order
        for row in df.iter_rows(named=True):
            yield MarketEvent(
                timestamp=row["timestamp"],
                ticker=row["ticker"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                adj_close=row.get("adj_close"),
            )

    def update_bars(self) -> bool:
        """Advance to the next bar in the current dataset.

        Must call get_bars() or set_data() first to load data.

        Returns:
            True if advanced to next bar, False if exhausted
        """
        if self._current_data is None:
            return False

        if self._current_index >= len(self._current_data) - 1:
            return False

        self._current_index += 1
        return True

    def set_data(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> None:
        """Load data for iteration.

        Call this before using update_bars() and get_current_bar().

        Args:
            tickers: List of stock ticker symbols
            start: Start date
            end: End date
            interval: Bar interval
        """
        lf = self.get_bars(tickers, start, end, interval)
        self._current_data = lf.collect()
        self._current_index = 0

    def get_current_bar(self, ticker: str) -> MarketEvent | None:
        """Get the current bar for a specific ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            MarketEvent for the current bar, or None if not found
        """
        if self._current_data is None:
            return None

        # Get the current timestamp
        if self._current_index >= len(self._current_data):
            return None

        current_ts = self._current_data.row(self._current_index, named=True)["timestamp"]

        # Find the row for this ticker at this timestamp
        row_df = self._current_data.filter(
            (pl.col("ticker") == ticker) & (pl.col("timestamp") == current_ts)
        )

        if row_df.is_empty():
            return None

        row = row_df.row(0, named=True)
        return MarketEvent(
            timestamp=row["timestamp"],
            ticker=row["ticker"],
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            adj_close=row.get("adj_close"),
        )

    def reset(self) -> None:
        """Reset the iteration index to the beginning."""
        self._current_index = 0

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
