"""Tests for trading calendar."""

from datetime import date, datetime, timedelta

import polars as pl
import pytest

from backtester.data.calendar import TradingCalendar


class TestTradingCalendarBasic:
    """Basic trading calendar tests."""

    @pytest.fixture
    def sample_calendar(self) -> TradingCalendar:
        """Create a calendar with known trading days."""
        # A week of trading days (Mon-Fri, no holidays)
        trading_days = {
            date(2024, 1, 2),  # Tuesday
            date(2024, 1, 3),  # Wednesday
            date(2024, 1, 4),  # Thursday
            date(2024, 1, 5),  # Friday
            date(2024, 1, 8),  # Monday
            date(2024, 1, 9),  # Tuesday
            date(2024, 1, 10),  # Wednesday
        }
        return TradingCalendar(trading_days)

    def test_is_trading_day(self, sample_calendar: TradingCalendar) -> None:
        """Test trading day detection."""
        assert sample_calendar.is_trading_day(date(2024, 1, 2))
        assert not sample_calendar.is_trading_day(date(2024, 1, 1))  # Holiday
        assert not sample_calendar.is_trading_day(date(2024, 1, 6))  # Saturday
        assert not sample_calendar.is_trading_day(date(2024, 1, 7))  # Sunday

    def test_is_trading_day_datetime(self, sample_calendar: TradingCalendar) -> None:
        """Test trading day detection with datetime input."""
        assert sample_calendar.is_trading_day(datetime(2024, 1, 2, 10, 30))
        assert not sample_calendar.is_trading_day(datetime(2024, 1, 6, 10, 30))

    def test_is_weekend(self, sample_calendar: TradingCalendar) -> None:
        """Test weekend detection."""
        assert sample_calendar.is_weekend(date(2024, 1, 6))  # Saturday
        assert sample_calendar.is_weekend(date(2024, 1, 7))  # Sunday
        assert not sample_calendar.is_weekend(date(2024, 1, 8))  # Monday

    def test_is_likely_holiday(self, sample_calendar: TradingCalendar) -> None:
        """Test holiday detection."""
        # Jan 1, 2024 was Monday (New Year's Day)
        assert sample_calendar.is_likely_holiday(date(2024, 1, 1))
        # Regular trading day
        assert not sample_calendar.is_likely_holiday(date(2024, 1, 2))
        # Weekend (not a holiday, just weekend)
        assert not sample_calendar.is_likely_holiday(date(2024, 1, 6))

    def test_contains(self, sample_calendar: TradingCalendar) -> None:
        """Test __contains__ method."""
        assert date(2024, 1, 2) in sample_calendar
        assert date(2024, 1, 1) not in sample_calendar

    def test_len(self, sample_calendar: TradingCalendar) -> None:
        """Test __len__ method."""
        assert len(sample_calendar) == 7


class TestTradingCalendarNavigation:
    """Tests for date navigation methods."""

    @pytest.fixture
    def sample_calendar(self) -> TradingCalendar:
        """Create a calendar with gaps (holidays)."""
        trading_days = {
            date(2024, 1, 2),  # Tuesday
            date(2024, 1, 3),  # Wednesday
            date(2024, 1, 4),  # Thursday
            date(2024, 1, 5),  # Friday
            # Jan 6-7 weekend
            date(2024, 1, 8),  # Monday
            # Jan 9 missing (pretend holiday)
            date(2024, 1, 10),  # Wednesday
        }
        return TradingCalendar(trading_days)

    def test_next_trading_day(self, sample_calendar: TradingCalendar) -> None:
        """Test finding next trading day."""
        # From a trading day (not including current)
        assert sample_calendar.next_trading_day(date(2024, 1, 2)) == date(2024, 1, 3)

        # From a weekend
        assert sample_calendar.next_trading_day(date(2024, 1, 6)) == date(2024, 1, 8)

        # Include current
        assert sample_calendar.next_trading_day(
            date(2024, 1, 2), include_current=True
        ) == date(2024, 1, 2)

        # From a holiday (Jan 9)
        assert sample_calendar.next_trading_day(date(2024, 1, 9)) == date(2024, 1, 10)

    def test_prev_trading_day(self, sample_calendar: TradingCalendar) -> None:
        """Test finding previous trading day."""
        # From a trading day (not including current)
        assert sample_calendar.prev_trading_day(date(2024, 1, 3)) == date(2024, 1, 2)

        # From a weekend
        assert sample_calendar.prev_trading_day(date(2024, 1, 6)) == date(2024, 1, 5)

        # Include current
        assert sample_calendar.prev_trading_day(
            date(2024, 1, 3), include_current=True
        ) == date(2024, 1, 3)

        # From a holiday (Jan 9)
        assert sample_calendar.prev_trading_day(date(2024, 1, 9)) == date(2024, 1, 8)

    def test_next_trading_day_none(self) -> None:
        """Test next trading day returns None when out of range."""
        cal = TradingCalendar({date(2024, 1, 2)})
        assert cal.next_trading_day(date(2024, 1, 3)) is None

    def test_prev_trading_day_none(self) -> None:
        """Test prev trading day returns None when out of range."""
        cal = TradingCalendar({date(2024, 1, 5)})
        assert cal.prev_trading_day(date(2024, 1, 4)) is None


class TestTradingDaysCounting:
    """Tests for counting trading days."""

    @pytest.fixture
    def sample_calendar(self) -> TradingCalendar:
        """Create a two-week calendar."""
        trading_days = {
            date(2024, 1, 2),
            date(2024, 1, 3),
            date(2024, 1, 4),
            date(2024, 1, 5),
            date(2024, 1, 8),
            date(2024, 1, 9),
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12),
        }
        return TradingCalendar(trading_days)

    def test_trading_days_between_inclusive(self, sample_calendar: TradingCalendar) -> None:
        """Test counting trading days (inclusive)."""
        count = sample_calendar.trading_days_between(
            date(2024, 1, 2), date(2024, 1, 5), inclusive=True
        )
        assert count == 4

    def test_trading_days_between_exclusive(self, sample_calendar: TradingCalendar) -> None:
        """Test counting trading days (exclusive)."""
        count = sample_calendar.trading_days_between(
            date(2024, 1, 2), date(2024, 1, 5), inclusive=False
        )
        assert count == 2  # Jan 3, 4

    def test_trading_days_between_across_weekend(self, sample_calendar: TradingCalendar) -> None:
        """Test counting across weekend."""
        count = sample_calendar.trading_days_between(
            date(2024, 1, 4), date(2024, 1, 9), inclusive=True
        )
        assert count == 4  # Jan 4, 5, 8, 9

    def test_trading_days_in_range(self, sample_calendar: TradingCalendar) -> None:
        """Test getting list of trading days in range."""
        days = sample_calendar.trading_days_in_range(date(2024, 1, 4), date(2024, 1, 9))
        assert len(days) == 4
        assert days == [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 9)]

    def test_add_trading_days_to_date(self, sample_calendar: TradingCalendar) -> None:
        """Test adding trading days to a date."""
        # Add 2 trading days to Jan 2
        # Sorted: [Jan 2, Jan 3, Jan 4, Jan 5, Jan 8, Jan 9, Jan 10, Jan 11, Jan 12]
        # Jan 2 is index 0, +2 = index 2 = Jan 4
        result = sample_calendar.add_trading_days_to_date(date(2024, 1, 2), 2)
        assert result == date(2024, 1, 4)

        # Add across weekend
        # Jan 4 is index 2, +2 = index 4 = Jan 8
        result = sample_calendar.add_trading_days_to_date(date(2024, 1, 4), 2)
        assert result == date(2024, 1, 8)

        # Negative (go back)
        # Jan 9 is index 5, -2 = index 3 = Jan 5
        result = sample_calendar.add_trading_days_to_date(date(2024, 1, 9), -2)
        assert result == date(2024, 1, 5)

    def test_add_trading_days_out_of_range(self, sample_calendar: TradingCalendar) -> None:
        """Test adding trading days returns None when out of range."""
        result = sample_calendar.add_trading_days_to_date(date(2024, 1, 12), 10)
        assert result is None


class TestTradingCalendarCreation:
    """Tests for calendar creation methods."""

    def test_from_dates(self) -> None:
        """Test creating calendar from list of dates."""
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        cal = TradingCalendar.from_dates(dates)

        assert len(cal) == 3
        assert date(2024, 1, 2) in cal

    def test_from_dates_with_datetimes(self) -> None:
        """Test creating calendar from list of datetimes."""
        dates = [
            datetime(2024, 1, 2, 9, 30),
            datetime(2024, 1, 3, 9, 30),
            datetime(2024, 1, 2, 16, 0),  # Duplicate date
        ]
        cal = TradingCalendar.from_dates(dates)

        assert len(cal) == 2  # Duplicates removed

    def test_from_dataframe(self) -> None:
        """Test creating calendar from DataFrame."""
        df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 2, 9, 30),
                    datetime(2024, 1, 3, 9, 30),
                    datetime(2024, 1, 4, 9, 30),
                ],
                "close": [100.0, 101.0, 102.0],
            }
        )

        cal = TradingCalendar.from_dataframe(df)

        assert len(cal) == 3
        assert date(2024, 1, 2) in cal

    def test_from_dataframe_empty(self) -> None:
        """Test creating calendar from empty DataFrame."""
        df = pl.DataFrame(schema={"timestamp": pl.Datetime, "close": pl.Float64})
        cal = TradingCalendar.from_dataframe(df)

        assert len(cal) == 0


class TestTradingCalendarProperties:
    """Tests for calendar properties."""

    def test_min_max_date(self) -> None:
        """Test min and max date properties."""
        trading_days = {
            date(2024, 1, 2),
            date(2024, 1, 5),
            date(2024, 1, 10),
        }
        cal = TradingCalendar(trading_days)

        assert cal.min_date == date(2024, 1, 2)
        assert cal.max_date == date(2024, 1, 10)

    def test_min_max_date_empty(self) -> None:
        """Test min/max date for empty calendar."""
        cal = TradingCalendar()

        assert cal.min_date is None
        assert cal.max_date is None

    def test_total_trading_days(self) -> None:
        """Test total trading days count."""
        cal = TradingCalendar({date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)})
        assert cal.total_trading_days == 3

    def test_trading_days_per_year(self) -> None:
        """Test trading days per year estimation."""
        # Create a year's worth of weekdays (rough approximation)
        days = set()
        current = date(2023, 1, 2)
        while current < date(2024, 1, 2):
            if current.weekday() < 5:  # Weekday
                days.add(current)
            current += timedelta(days=1)

        cal = TradingCalendar(days)
        tpy = cal.get_trading_days_per_year()

        # Should be around 260 (52 weeks * 5 days, minus some holidays)
        assert 250 < tpy < 265

    def test_trading_days_per_year_insufficient_data(self) -> None:
        """Test default when insufficient data."""
        cal = TradingCalendar({date(2024, 1, 2)})
        assert cal.get_trading_days_per_year() == 252.0

    def test_repr(self) -> None:
        """Test string representation."""
        cal = TradingCalendar({date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)})
        repr_str = repr(cal)

        assert "TradingCalendar" in repr_str
        assert "2024-01-02" in repr_str
        assert "2024-01-04" in repr_str
        assert "3 days" in repr_str

    def test_repr_empty(self) -> None:
        """Test repr for empty calendar."""
        cal = TradingCalendar()
        assert repr(cal) == "TradingCalendar(empty)"


class TestTradingCalendarMutation:
    """Tests for adding trading days."""

    def test_add_trading_day(self) -> None:
        """Test adding a single trading day."""
        cal = TradingCalendar()
        cal.add_trading_day(date(2024, 1, 2))

        assert len(cal) == 1
        assert date(2024, 1, 2) in cal

    def test_add_trading_days(self) -> None:
        """Test adding multiple trading days."""
        cal = TradingCalendar()
        cal.add_trading_days([date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)])

        assert len(cal) == 3

    def test_add_trading_day_datetime(self) -> None:
        """Test adding datetime is converted to date."""
        cal = TradingCalendar()
        cal.add_trading_day(datetime(2024, 1, 2, 9, 30))

        assert date(2024, 1, 2) in cal
