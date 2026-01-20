"""Trading calendar utilities.

Provides trading day detection and calculation by inferring
from cached market data.
"""

from datetime import date, datetime

import polars as pl

from backtester.data.cache import DuckDBCache


class TradingCalendar:
    """Trading calendar inferred from market data.

    Uses actual trading dates from cached data to determine
    trading days, holidays, and perform date calculations.

    Attributes:
        trading_days: Sorted set of known trading days
    """

    def __init__(self, trading_days: set[date] | None = None):
        """Initialize calendar.

        Args:
            trading_days: Set of known trading days
        """
        self._trading_days: set[date] = trading_days or set()
        self._sorted_days: list[date] | None = None

    @classmethod
    def from_cache(cls, cache: DuckDBCache, ticker: str | None = None) -> "TradingCalendar":
        """Create calendar from cached data.

        Args:
            cache: DuckDB cache instance
            ticker: Optional ticker to filter by (default: use all tickers)

        Returns:
            TradingCalendar populated with trading days from cache
        """
        with cache._get_connection() as conn:
            if ticker:
                result = conn.execute(
                    f"SELECT DISTINCT CAST(timestamp AS DATE) as trade_date FROM {cache.TABLE_NAME} WHERE ticker = ? ORDER BY trade_date",
                    [ticker],
                ).fetchall()
            else:
                result = conn.execute(
                    f"SELECT DISTINCT CAST(timestamp AS DATE) as trade_date FROM {cache.TABLE_NAME} ORDER BY trade_date"
                ).fetchall()

        trading_days = {row[0] for row in result}
        return cls(trading_days)

    @classmethod
    def from_dataframe(
        cls, df: pl.DataFrame, timestamp_col: str = "timestamp"
    ) -> "TradingCalendar":
        """Create calendar from a DataFrame.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            TradingCalendar populated with unique dates
        """
        if df.is_empty() or timestamp_col not in df.columns:
            return cls()

        dates = df.select(pl.col(timestamp_col).dt.date().unique()).to_series().to_list()
        trading_days = {d for d in dates if d is not None}
        return cls(trading_days)

    @classmethod
    def from_dates(cls, dates: list[date] | list[datetime]) -> "TradingCalendar":
        """Create calendar from a list of dates.

        Args:
            dates: List of dates or datetimes

        Returns:
            TradingCalendar populated with dates
        """
        trading_days = set()
        for d in dates:
            if isinstance(d, datetime):
                trading_days.add(d.date())
            else:
                trading_days.add(d)
        return cls(trading_days)

    def _get_sorted_days(self) -> list[date]:
        """Get sorted list of trading days (cached)."""
        if self._sorted_days is None:
            self._sorted_days = sorted(self._trading_days)
        return self._sorted_days

    def _invalidate_cache(self) -> None:
        """Invalidate sorted days cache."""
        self._sorted_days = None

    def add_trading_day(self, d: date | datetime) -> None:
        """Add a trading day to the calendar.

        Args:
            d: Date to add
        """
        if isinstance(d, datetime):
            d = d.date()
        self._trading_days.add(d)
        self._invalidate_cache()

    def add_trading_days(self, dates: list[date] | list[datetime]) -> None:
        """Add multiple trading days.

        Args:
            dates: Dates to add
        """
        for d in dates:
            if isinstance(d, datetime):
                self._trading_days.add(d.date())
            else:
                self._trading_days.add(d)
        self._invalidate_cache()

    def is_trading_day(self, d: date | datetime) -> bool:
        """Check if a date is a trading day.

        Args:
            d: Date to check

        Returns:
            True if date is a known trading day
        """
        if isinstance(d, datetime):
            d = d.date()
        return d in self._trading_days

    def is_weekend(self, d: date | datetime) -> bool:
        """Check if a date is a weekend.

        Args:
            d: Date to check

        Returns:
            True if Saturday or Sunday
        """
        if isinstance(d, datetime):
            d = d.date()
        return d.weekday() >= 5

    def is_likely_holiday(self, d: date | datetime) -> bool:
        """Check if a date is likely a market holiday.

        A weekday that's not in our trading days set is likely a holiday.
        Note: Only reliable within the date range of cached data.

        Args:
            d: Date to check

        Returns:
            True if weekday but not a trading day
        """
        if isinstance(d, datetime):
            d = d.date()
        return not self.is_weekend(d) and not self.is_trading_day(d)

    def trading_days_between(
        self,
        start: date | datetime,
        end: date | datetime,
        inclusive: bool = True,
    ) -> int:
        """Count trading days between two dates.

        Args:
            start: Start date
            end: End date
            inclusive: Include start and end dates if they're trading days

        Returns:
            Number of trading days in range
        """
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()

        if start > end:
            start, end = end, start

        count = 0
        for d in self._trading_days:
            if inclusive:
                if start <= d <= end:
                    count += 1
            else:
                if start < d < end:
                    count += 1
        return count

    def next_trading_day(self, d: date | datetime, include_current: bool = False) -> date | None:
        """Get the next trading day on or after a date.

        Args:
            d: Starting date
            include_current: If True, return d if it's a trading day

        Returns:
            Next trading day, or None if not found in calendar
        """
        if isinstance(d, datetime):
            d = d.date()

        if include_current and self.is_trading_day(d):
            return d

        sorted_days = self._get_sorted_days()
        for trading_day in sorted_days:
            if trading_day > d:
                return trading_day
        return None

    def prev_trading_day(self, d: date | datetime, include_current: bool = False) -> date | None:
        """Get the previous trading day on or before a date.

        Args:
            d: Starting date
            include_current: If True, return d if it's a trading day

        Returns:
            Previous trading day, or None if not found in calendar
        """
        if isinstance(d, datetime):
            d = d.date()

        if include_current and self.is_trading_day(d):
            return d

        sorted_days = self._get_sorted_days()
        for trading_day in reversed(sorted_days):
            if trading_day < d:
                return trading_day
        return None

    def trading_days_in_range(
        self,
        start: date | datetime,
        end: date | datetime,
    ) -> list[date]:
        """Get list of trading days in a range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            Sorted list of trading days in range
        """
        if isinstance(start, datetime):
            start = start.date()
        if isinstance(end, datetime):
            end = end.date()

        if start > end:
            start, end = end, start

        return [d for d in self._get_sorted_days() if start <= d <= end]

    def add_trading_days_to_date(self, d: date | datetime, n: int) -> date | None:
        """Add n trading days to a date.

        Args:
            d: Starting date
            n: Number of trading days to add (can be negative)

        Returns:
            Resulting date, or None if out of calendar range
        """
        if isinstance(d, datetime):
            d = d.date()

        sorted_days = self._get_sorted_days()
        if not sorted_days:
            return None

        # Find starting position
        try:
            idx = sorted_days.index(d)
        except ValueError:
            # d is not a trading day, find nearest
            if n >= 0:
                next_day = self.next_trading_day(d, include_current=True)
                if next_day is None:
                    return None
                idx = sorted_days.index(next_day)
            else:
                prev_day = self.prev_trading_day(d, include_current=True)
                if prev_day is None:
                    return None
                idx = sorted_days.index(prev_day)

        new_idx = idx + n
        if 0 <= new_idx < len(sorted_days):
            return sorted_days[new_idx]
        return None

    @property
    def min_date(self) -> date | None:
        """Earliest date in calendar."""
        if not self._trading_days:
            return None
        return min(self._trading_days)

    @property
    def max_date(self) -> date | None:
        """Latest date in calendar."""
        if not self._trading_days:
            return None
        return max(self._trading_days)

    @property
    def total_trading_days(self) -> int:
        """Total number of trading days in calendar."""
        return len(self._trading_days)

    def get_trading_days_per_year(self) -> float:
        """Estimate trading days per year from data.

        Returns:
            Average trading days per year, or 252.0 if insufficient data
        """
        if len(self._trading_days) < 2:
            return 252.0

        min_d = self.min_date
        max_d = self.max_date
        if min_d is None or max_d is None:
            return 252.0

        calendar_days = (max_d - min_d).days
        if calendar_days == 0:
            return 252.0

        years = calendar_days / 365.0
        return len(self._trading_days) / years

    def __len__(self) -> int:
        """Number of trading days."""
        return len(self._trading_days)

    def __contains__(self, d: date | datetime) -> bool:
        """Check if date is a trading day."""
        return self.is_trading_day(d)

    def __repr__(self) -> str:
        """String representation."""
        if not self._trading_days:
            return "TradingCalendar(empty)"
        return f"TradingCalendar({self.min_date} to {self.max_date}, {len(self)} days)"
