"""DuckDB-based caching layer for market data.

Provides persistent caching of OHLCV data to avoid repeated API calls.
Uses connection pooling for efficient database access.
"""

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import polars as pl

from backtester.data.schemas import (
    CACHE_TABLE_SCHEMA,
    OHLCV_COLUMNS,
    date_to_datetime,
    normalize_ohlcv_columns,
)


class DuckDBCache:
    """Persistent cache for market data using DuckDB.

    Provides efficient storage and retrieval of OHLCV data with
    support for range queries and batch inserts.

    Attributes:
        db_path: Path to the DuckDB database file
    """

    TABLE_NAME = "ohlcv_data"

    def __init__(self, db_path: str | Path = "data/market_cache.duckdb"):
        """Initialize the cache.

        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                    {CACHE_TABLE_SCHEMA}
                )
            """)
            # Create index for efficient range queries
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_ticker_timestamp
                ON {self.TABLE_NAME} (ticker, timestamp)
            """)

    @contextmanager
    def _get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a database connection from the pool."""
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def get_cached_data(
        self,
        ticker: str,
        start: date | datetime,
        end: date | datetime,
    ) -> pl.LazyFrame | None:
        """Retrieve cached data for a ticker and date range.

        Args:
            ticker: Stock ticker symbol
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            LazyFrame with cached data, or None if no data found
        """
        start_dt = date_to_datetime(start)
        end_dt = date_to_datetime(end)

        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT {", ".join(OHLCV_COLUMNS)}
                FROM {self.TABLE_NAME}
                WHERE ticker = ?
                  AND timestamp >= ?
                  AND timestamp <= ?
                ORDER BY timestamp
                """,
                [ticker, start_dt, end_dt],
            ).pl()

        if result.is_empty():
            return None

        return result.lazy()

    def get_cached_range(self, ticker: str) -> tuple[datetime, datetime] | None:
        """Get the date range of cached data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple of (min_date, max_date) or None if no data cached
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"""
                SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts
                FROM {self.TABLE_NAME}
                WHERE ticker = ?
                """,
                [ticker],
            ).fetchone()

        if result is None or result[0] is None:
            return None

        return (result[0], result[1])

    def get_missing_ranges(
        self,
        ticker: str,
        start: date | datetime,
        end: date | datetime,
    ) -> list[tuple[datetime, datetime]]:
        """Determine which date ranges are missing from the cache.

        Args:
            ticker: Stock ticker symbol
            start: Desired start date
            end: Desired end date

        Returns:
            List of (start, end) tuples representing missing ranges.
            Ranges are adjusted to avoid re-fetching cached boundary dates.
        """
        start_dt = date_to_datetime(start)
        end_dt = date_to_datetime(end)

        cached_range = self.get_cached_range(ticker)
        if cached_range is None:
            return [(start_dt, end_dt)]

        cache_start, cache_end = cached_range
        missing = []

        # Check if we need data before the cache
        # Request up to the day BEFORE cache starts to avoid refetching
        if start_dt < cache_start:
            missing.append((start_dt, cache_start - timedelta(days=1)))

        # Check if we need data after the cache
        # Request from the day AFTER cache ends to avoid refetching
        if end_dt > cache_end:
            missing.append((cache_end + timedelta(days=1), end_dt))

        return missing

    def store_data(self, df: pl.DataFrame) -> int:
        """Store OHLCV data in the cache.

        Uses UPSERT semantics - existing data with the same
        (ticker, timestamp) will be updated.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Number of rows stored
        """
        if df.is_empty():
            return 0

        # Normalize the data
        df = normalize_ohlcv_columns(df)

        # Add fetched_at timestamp (when this data was retrieved from the API)
        fetch_time = datetime.now(UTC).replace(tzinfo=None)
        if "fetched_at" not in df.columns:
            df = df.with_columns(pl.lit(fetch_time).alias("fetched_at"))

        # Ensure we have all required columns
        for col in OHLCV_COLUMNS:
            if col not in df.columns:
                if col == "adj_close":
                    df = df.with_columns(pl.lit(None).alias("adj_close"))
                elif col == "fetched_at":
                    df = df.with_columns(pl.lit(fetch_time).alias("fetched_at"))
                else:
                    raise ValueError(f"Missing required column: {col}")

        # Reorder columns
        df = df.select(OHLCV_COLUMNS)

        with self._get_connection() as conn:
            # Use INSERT OR REPLACE for upsert behavior
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.TABLE_NAME}
                SELECT * FROM df
            """)

        return len(df)

    def clear_ticker(self, ticker: str) -> int:
        """Remove all cached data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Number of rows deleted
        """
        with self._get_connection() as conn:
            # Count rows before deletion (DuckDB doesn't have changes() like SQLite)
            count_result = conn.execute(
                f"SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE ticker = ?",
                [ticker],
            ).fetchone()
            count = count_result[0] if count_result else 0

            conn.execute(
                f"DELETE FROM {self.TABLE_NAME} WHERE ticker = ?",
                [ticker],
            )
            return count

    def clear_all(self) -> None:
        """Remove all cached data."""
        with self._get_connection() as conn:
            conn.execute(f"DELETE FROM {self.TABLE_NAME}")

    def get_cached_tickers(self) -> list[str]:
        """Get list of all tickers with cached data.

        Returns:
            List of ticker symbols
        """
        with self._get_connection() as conn:
            result = conn.execute(
                f"SELECT DISTINCT ticker FROM {self.TABLE_NAME} ORDER BY ticker"
            ).fetchall()
        return [row[0] for row in result]

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._get_connection() as conn:
            result = conn.execute(f"""
                SELECT
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    MIN(timestamp) as earliest_date,
                    MAX(timestamp) as latest_date
                FROM {self.TABLE_NAME}
            """).fetchone()

        if result is None:
            return {
                "total_rows": 0,
                "unique_tickers": 0,
                "earliest_date": None,
                "latest_date": None,
            }

        return {
            "total_rows": result[0],
            "unique_tickers": result[1],
            "earliest_date": result[2],
            "latest_date": result[3],
        }
