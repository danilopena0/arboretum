"""Pytest configuration and fixtures."""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from backtester.core.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    SignalEvent,
    SignalType,
)
from backtester.data.cache import DuckDBCache


@pytest.fixture
def sample_market_event() -> MarketEvent:
    """Create a sample MarketEvent for testing."""
    return MarketEvent(
        timestamp=datetime(2024, 1, 15, 9, 30, 0),
        ticker="AAPL",
        open=185.50,
        high=187.25,
        low=184.75,
        close=186.50,
        volume=45_000_000,
        adj_close=186.50,
    )


@pytest.fixture
def sample_signal_event() -> SignalEvent:
    """Create a sample SignalEvent for testing."""
    return SignalEvent(
        timestamp=datetime(2024, 1, 15, 9, 30, 0),
        ticker="AAPL",
        signal_type=SignalType.LONG,
        strength=0.85,
        strategy_id="ma_crossover",
    )


@pytest.fixture
def sample_order_event() -> OrderEvent:
    """Create a sample OrderEvent for testing."""
    return OrderEvent(
        timestamp=datetime(2024, 1, 15, 9, 30, 0),
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        order_id="order_001",
        limit_price=None,
    )


@pytest.fixture
def sample_fill_event() -> FillEvent:
    """Create a sample FillEvent for testing."""
    return FillEvent(
        timestamp=datetime(2024, 1, 15, 9, 30, 0),
        ticker="AAPL",
        side=OrderSide.BUY,
        quantity=100,
        fill_price=186.55,
        commission=0.0,
        order_id="order_001",
        slippage=0.05,
    )


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    return pl.DataFrame(
        {
            "timestamp": [
                datetime(2024, 1, 15),
                datetime(2024, 1, 16),
                datetime(2024, 1, 17),
                datetime(2024, 1, 18),
                datetime(2024, 1, 19),
            ],
            "ticker": ["AAPL"] * 5,
            "open": [185.0, 186.5, 187.0, 186.0, 188.0],
            "high": [187.0, 188.0, 189.5, 187.5, 190.0],
            "low": [184.5, 185.5, 186.5, 185.0, 187.0],
            "close": [186.5, 187.5, 186.5, 187.0, 189.5],
            "volume": [45_000_000, 42_000_000, 48_000_000, 40_000_000, 50_000_000],
            "adj_close": [186.5, 187.5, 186.5, 187.0, 189.5],
        }
    )


@pytest.fixture
def temp_cache_path(tmp_path: Path) -> Path:
    """Create a temporary path for DuckDB cache."""
    return tmp_path / "test_cache.duckdb"


@pytest.fixture
def temp_cache(temp_cache_path: Path) -> DuckDBCache:
    """Create a temporary DuckDB cache for testing."""
    return DuckDBCache(temp_cache_path)


@pytest.fixture
def populated_cache(temp_cache: DuckDBCache, sample_ohlcv_data: pl.DataFrame) -> DuckDBCache:
    """Create a cache populated with sample data."""
    temp_cache.store_data(sample_ohlcv_data)
    return temp_cache
