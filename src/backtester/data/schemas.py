"""Polars schema definitions for market data.

Defines consistent schemas for OHLCV data used throughout the backtester.
"""

from datetime import date, datetime

import polars as pl

# Schema for raw OHLCV data from yfinance
OHLCV_SCHEMA = {
    "timestamp": pl.Datetime("us"),
    "ticker": pl.Utf8,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "adj_close": pl.Float64,
    "fetched_at": pl.Datetime("us"),
}

# Schema for the DuckDB cache table
CACHE_TABLE_SCHEMA = """
    timestamp TIMESTAMP NOT NULL,
    ticker VARCHAR NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume BIGINT NOT NULL,
    adj_close DOUBLE,
    fetched_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ticker, timestamp)
"""

# Column order for consistent data handling
OHLCV_COLUMNS = ["timestamp", "ticker", "open", "high", "low", "close", "volume", "adj_close", "fetched_at"]


def validate_ohlcv_frame(df: pl.DataFrame | pl.LazyFrame) -> bool:
    """Validate that a DataFrame has the expected OHLCV schema.

    Args:
        df: Polars DataFrame or LazyFrame to validate

    Returns:
        True if schema matches, False otherwise
    """
    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema

    for col, expected_type in OHLCV_SCHEMA.items():
        if col not in schema:
            return False
        # Allow some type flexibility (e.g., Date vs Datetime)
        actual_type = schema[col]
        if col == "timestamp":
            if actual_type not in (pl.Datetime, pl.Date):
                return False
        elif col == "volume":
            if actual_type not in (pl.Int64, pl.Int32, pl.UInt64):
                return False
        elif col == "adj_close":
            # adj_close can be nullable
            continue
        elif actual_type != expected_type:
            return False
    return True


def create_empty_ohlcv_frame() -> pl.DataFrame:
    """Create an empty DataFrame with the OHLCV schema."""
    return pl.DataFrame(schema=OHLCV_SCHEMA)


def normalize_ohlcv_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize column names and types to match the expected schema.

    Handles common variations in column names from different data sources.

    Args:
        df: DataFrame with potentially non-standard column names

    Returns:
        DataFrame with normalized column names and types
    """
    # Common column name mappings
    column_mappings = {
        "date": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "adjusted_close": "adj_close",
        "Symbol": "ticker",
        "symbol": "ticker",
    }

    # Rename columns
    rename_dict = {}
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            rename_dict[old_name] = new_name

    if rename_dict:
        df = df.rename(rename_dict)

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        if df.schema["timestamp"] == pl.Date:
            df = df.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us")).alias("timestamp")
            )
        elif df.schema["timestamp"] == pl.Utf8:
            df = df.with_columns(
                pl.col("timestamp").str.to_datetime().alias("timestamp")
            )

    # Ensure volume is int64
    if "volume" in df.columns and df.schema["volume"] in (pl.Float64, pl.Float32):
        df = df.with_columns(pl.col("volume").cast(pl.Int64).alias("volume"))

    return df


def date_to_datetime(d: date | datetime) -> datetime:
    """Convert a date to datetime at midnight."""
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)
