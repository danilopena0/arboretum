"""Data handling: caching, fetching, and schema definitions."""

from backtester.data.cache import DuckDBCache
from backtester.data.handler import DataHandler
from backtester.data.yfinance_handler import YFinanceDataHandler

__all__ = [
    "DataHandler",
    "DuckDBCache",
    "YFinanceDataHandler",
]
