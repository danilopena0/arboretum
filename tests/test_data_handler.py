"""Tests for data handlers and caching."""

from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from backtester.data.cache import DuckDBCache
from backtester.data.schemas import (
    OHLCV_COLUMNS,
    create_empty_ohlcv_frame,
    normalize_ohlcv_columns,
    validate_ohlcv_frame,
)


class TestDuckDBCache:
    """Tests for DuckDBCache."""

    def test_create_cache(self, temp_cache_path: Path) -> None:
        """Test cache creation."""
        cache = DuckDBCache(temp_cache_path)
        assert temp_cache_path.exists()
        stats = cache.get_stats()
        assert stats["total_rows"] == 0

    def test_store_and_retrieve_data(
        self, temp_cache: DuckDBCache, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test storing and retrieving data."""
        # Store data
        rows_stored = temp_cache.store_data(sample_ohlcv_data)
        assert rows_stored == 5

        # Retrieve data
        result = temp_cache.get_cached_data(
            "AAPL",
            datetime(2024, 1, 15),
            datetime(2024, 1, 19),
        )
        assert result is not None
        df = result.collect()
        assert len(df) == 5
        assert df["ticker"].unique().to_list() == ["AAPL"]

    def test_get_cached_range(self, populated_cache: DuckDBCache) -> None:
        """Test getting cached date range."""
        range_result = populated_cache.get_cached_range("AAPL")
        assert range_result is not None
        min_date, max_date = range_result
        assert min_date.date() == date(2024, 1, 15)
        assert max_date.date() == date(2024, 1, 19)

    def test_get_cached_range_missing_ticker(self, populated_cache: DuckDBCache) -> None:
        """Test cached range for non-existent ticker."""
        result = populated_cache.get_cached_range("MISSING")
        assert result is None

    def test_get_missing_ranges_full_miss(self, temp_cache: DuckDBCache) -> None:
        """Test missing ranges when no data cached."""
        missing = temp_cache.get_missing_ranges(
            "AAPL",
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )
        assert len(missing) == 1
        assert missing[0][0] == datetime(2024, 1, 1)
        assert missing[0][1] == datetime(2024, 1, 31)

    def test_get_missing_ranges_partial(self, populated_cache: DuckDBCache) -> None:
        """Test missing ranges with partial cache coverage."""
        # Cache has Jan 15-19, request Jan 10-25
        missing = populated_cache.get_missing_ranges(
            "AAPL",
            datetime(2024, 1, 10),
            datetime(2024, 1, 25),
        )
        assert len(missing) == 2
        # Before cached range
        assert missing[0][0] == datetime(2024, 1, 10)
        # After cached range
        assert missing[1][1] == datetime(2024, 1, 25)

    def test_upsert_behavior(
        self, temp_cache: DuckDBCache, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test that storing data with same key updates existing rows."""
        # Store initial data
        temp_cache.store_data(sample_ohlcv_data)

        # Store updated data with different close price
        updated = sample_ohlcv_data.with_columns(
            pl.col("close") * 1.1  # 10% increase
        )
        temp_cache.store_data(updated)

        # Verify update happened
        result = temp_cache.get_cached_data(
            "AAPL",
            datetime(2024, 1, 15),
            datetime(2024, 1, 15),
        )
        assert result is not None
        df = result.collect()
        # Original close was 186.5, updated should be 186.5 * 1.1
        assert df["close"][0] == pytest.approx(186.5 * 1.1)

        # Should still have 5 rows, not 10
        stats = temp_cache.get_stats()
        assert stats["total_rows"] == 5

    def test_clear_ticker(self, populated_cache: DuckDBCache) -> None:
        """Test clearing data for a specific ticker."""
        populated_cache.clear_ticker("AAPL")
        result = populated_cache.get_cached_data(
            "AAPL",
            datetime(2024, 1, 1),
            datetime(2024, 12, 31),
        )
        assert result is None or result.collect().is_empty()

    def test_get_cached_tickers(
        self, temp_cache: DuckDBCache, sample_ohlcv_data: pl.DataFrame
    ) -> None:
        """Test getting list of cached tickers."""
        # Add AAPL
        temp_cache.store_data(sample_ohlcv_data)

        # Add MSFT
        msft_data = sample_ohlcv_data.with_columns(pl.lit("MSFT").alias("ticker"))
        temp_cache.store_data(msft_data)

        tickers = temp_cache.get_cached_tickers()
        assert sorted(tickers) == ["AAPL", "MSFT"]


class TestSchemas:
    """Tests for schema utilities."""

    def test_create_empty_frame(self) -> None:
        """Test creating empty DataFrame with correct schema."""
        df = create_empty_ohlcv_frame()
        assert df.is_empty()
        assert list(df.columns) == OHLCV_COLUMNS

    def test_validate_valid_frame(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test validation of valid DataFrame."""
        assert validate_ohlcv_frame(sample_ohlcv_data) is True

    def test_validate_missing_column(self, sample_ohlcv_data: pl.DataFrame) -> None:
        """Test validation fails with missing column."""
        df = sample_ohlcv_data.drop("volume")
        assert validate_ohlcv_frame(df) is False

    def test_normalize_columns(self) -> None:
        """Test column name normalization."""
        df = pl.DataFrame(
            {
                "Date": [datetime(2024, 1, 15)],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000000.0],  # Float instead of int
                "Adj Close": [100.5],
                "Symbol": ["AAPL"],
            }
        )
        normalized = normalize_ohlcv_columns(df)

        assert "timestamp" in normalized.columns
        assert "open" in normalized.columns
        assert "ticker" in normalized.columns
        assert normalized.schema["volume"] == pl.Int64
