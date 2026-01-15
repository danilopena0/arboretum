"""Tests for performance metrics."""

import math
from datetime import datetime, timedelta

import polars as pl
import pytest

from backtester.analysis.metrics import (
    PerformanceMetrics,
    RoundTripTrade,
    alpha,
    annualized_return,
    average_drawdown,
    average_trade_duration,
    average_win_loss_ratio,
    beta,
    cagr,
    calculate_metrics,
    calculate_returns,
    calmar_ratio,
    downside_deviation,
    drawdown_series,
    expectancy,
    extract_round_trips,
    max_drawdown,
    max_drawdown_duration,
    profit_factor,
    sharpe_ratio,
    snapshots_to_dataframe,
    sortino_ratio,
    total_return,
    trades_to_dataframe,
    volatility,
    win_rate,
)
from backtester.core.portfolio import PortfolioSnapshot, Trade


class TestDataPreparation:
    """Tests for data conversion functions."""

    def test_snapshots_to_dataframe(self) -> None:
        """Test conversion of snapshots to DataFrame."""
        snapshots = [
            PortfolioSnapshot(
                timestamp=datetime(2024, 1, 1),
                cash=90000.0,
                positions_value=10000.0,
                total_equity=100000.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
            ),
            PortfolioSnapshot(
                timestamp=datetime(2024, 1, 2),
                cash=90000.0,
                positions_value=11000.0,
                total_equity=101000.0,
                realized_pnl=0.0,
                unrealized_pnl=1000.0,
            ),
        ]

        df = snapshots_to_dataframe(snapshots)

        assert len(df) == 2
        assert df["total_equity"][0] == 100000.0
        assert df["total_equity"][1] == 101000.0

    def test_snapshots_to_dataframe_empty(self) -> None:
        """Test empty snapshot list."""
        df = snapshots_to_dataframe([])
        assert df.is_empty()
        assert "total_equity" in df.columns

    def test_trades_to_dataframe(self) -> None:
        """Test conversion of trades to DataFrame."""
        trades = [
            Trade(
                timestamp=datetime(2024, 1, 1),
                ticker="AAPL",
                quantity=100,
                price=150.0,
                commission=1.0,
                slippage=0.01,
                order_id="order1",
            ),
        ]

        df = trades_to_dataframe(trades)

        assert len(df) == 1
        assert df["ticker"][0] == "AAPL"
        assert df["side"][0] == "BUY"

    def test_calculate_returns(self) -> None:
        """Test return calculation from equity series."""
        equity = pl.Series([100, 110, 105, 115])
        returns = calculate_returns(equity)

        assert len(returns) == 4
        assert returns[0] == 0.0  # First return is 0
        assert abs(returns[1] - 0.10) < 0.001  # 10% gain
        assert abs(returns[2] - (-0.0455)) < 0.001  # ~4.5% loss


class TestReturnMetrics:
    """Tests for return calculation functions."""

    def test_total_return(self) -> None:
        """Test total return calculation."""
        assert total_return(100000, 110000) == pytest.approx(0.10)
        assert total_return(100000, 90000) == pytest.approx(-0.10)
        assert total_return(100000, 100000) == 0.0
        assert total_return(0, 100) == 0.0

    def test_cagr(self) -> None:
        """Test CAGR calculation."""
        # 10% return over 1 year = 10% CAGR
        result = cagr(100000, 110000, 365)
        assert result == pytest.approx(0.10, rel=0.01)

        # 21% return over 2 years = ~10% CAGR
        result = cagr(100000, 121000, 730)
        assert result == pytest.approx(0.10, rel=0.01)

        # Edge cases
        assert cagr(0, 100, 365) == 0.0
        assert cagr(100, 100, 0) == 0.0

    def test_annualized_return(self) -> None:
        """Test annualized return from period returns."""
        # Daily returns of 0.1% for 252 days
        daily_return = 0.001
        returns = pl.Series([daily_return] * 252)

        ann_ret = annualized_return(returns, periods_per_year=252)
        # Should be approximately (1.001)^252 - 1 ≈ 28.6%
        assert ann_ret == pytest.approx(0.286, rel=0.05)

        # Empty returns
        assert annualized_return(pl.Series([]), 252) == 0.0


class TestRiskMetrics:
    """Tests for risk metrics."""

    def test_volatility(self) -> None:
        """Test annualized volatility calculation."""
        # Create returns with known std
        returns = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])

        vol = volatility(returns, periods_per_year=252)

        # Should be annualized
        assert vol > 0
        assert vol == pytest.approx(returns.std() * math.sqrt(252), rel=0.01)

    def test_volatility_empty(self) -> None:
        """Test volatility with empty series."""
        assert volatility(pl.Series([]), 252) == 0.0
        assert volatility(pl.Series([1.0]), 252) == 0.0

    def test_downside_deviation(self) -> None:
        """Test downside deviation calculation."""
        # Mix of positive and negative returns
        returns = pl.Series([0.02, -0.01, 0.03, -0.02, -0.015, 0.01])

        dd = downside_deviation(returns, threshold=0.0, periods_per_year=252)

        # Should only consider negative returns
        assert dd > 0

        # With all positive returns, should be 0
        positive_returns = pl.Series([0.01, 0.02, 0.03])
        assert downside_deviation(positive_returns) == 0.0


class TestRiskAdjustedMetrics:
    """Tests for risk-adjusted return metrics."""

    def test_sharpe_ratio(self) -> None:
        """Test Sharpe ratio calculation."""
        # Returns with positive mean
        returns = pl.Series([0.01, 0.02, 0.01, 0.015, 0.01])

        sharpe = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Should be positive for positive returns
        assert sharpe > 0

    def test_sharpe_ratio_with_risk_free(self) -> None:
        """Test Sharpe with non-zero risk-free rate."""
        returns = pl.Series([0.01, 0.02, 0.01, 0.015, 0.01])

        sharpe_rf0 = sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_rf5 = sharpe_ratio(returns, risk_free_rate=0.05)

        # Higher risk-free rate = lower Sharpe
        assert sharpe_rf0 > sharpe_rf5

    def test_sortino_ratio(self) -> None:
        """Test Sortino ratio calculation."""
        returns = pl.Series([0.02, -0.01, 0.03, -0.02, 0.01])

        sortino = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

        # Should be defined
        assert not math.isnan(sortino)

    def test_calmar_ratio(self) -> None:
        """Test Calmar ratio calculation."""
        equity = pl.Series([100000.0, 110000.0, 105000.0, 115000.0, 120000.0])
        returns = calculate_returns(equity)

        calmar = calmar_ratio(returns, equity, periods_per_year=252)

        # Should be positive for net positive returns
        assert calmar > 0


class TestDrawdownMetrics:
    """Tests for drawdown calculations."""

    def test_drawdown_series(self) -> None:
        """Test drawdown series calculation."""
        equity = pl.Series([100.0, 110.0, 105.0, 115.0, 110.0])
        dd = drawdown_series(equity)

        assert len(dd) == 5
        assert dd[0] == 0.0  # At initial value (peak)
        assert dd[1] == 0.0  # At new peak
        assert dd[2] == pytest.approx(0.0455, rel=0.01)  # Below peak
        assert dd[3] == 0.0  # New peak
        assert dd[4] == pytest.approx(0.0435, rel=0.01)  # Below peak

    def test_max_drawdown(self) -> None:
        """Test maximum drawdown calculation."""
        # Equity goes: 100 -> 110 -> 90 -> 100
        equity = pl.Series([100.0, 110.0, 90.0, 100.0])

        mdd = max_drawdown(equity)

        # Max drawdown is (110 - 90) / 110 = 18.18%
        assert mdd == pytest.approx(0.1818, rel=0.01)

    def test_max_drawdown_no_drawdown(self) -> None:
        """Test max drawdown with monotonic increase."""
        equity = pl.Series([100.0, 110.0, 120.0, 130.0])
        assert max_drawdown(equity) == 0.0

    def test_max_drawdown_duration(self) -> None:
        """Test maximum drawdown duration."""
        equity = pl.Series([100.0, 110.0, 105.0, 100.0, 110.0, 115.0])
        timestamps = pl.Series(
            [
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                datetime(2024, 1, 3),
                datetime(2024, 1, 4),
                datetime(2024, 1, 5),
                datetime(2024, 1, 6),
            ]
        )

        duration = max_drawdown_duration(equity, timestamps)

        # Drawdown starts day 3 (below peak of 110), recovers day 5
        # Duration = Jan 5 - Jan 3 = 2 days
        assert duration.days == 2

    def test_average_drawdown(self) -> None:
        """Test average drawdown calculation."""
        equity = pl.Series([100.0, 90.0, 95.0, 100.0])

        avg_dd = average_drawdown(equity)

        assert avg_dd > 0
        assert avg_dd < max_drawdown(equity)


class TestTradeMetrics:
    """Tests for trade analysis metrics."""

    @pytest.fixture
    def sample_round_trips(self) -> list[RoundTripTrade]:
        """Create sample round trip trades."""
        return [
            RoundTripTrade(
                ticker="AAPL",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,  # Winner
                commission=2.0,
                side="LONG",
            ),
            RoundTripTrade(
                ticker="AAPL",
                entry_time=datetime(2024, 1, 10),
                exit_time=datetime(2024, 1, 12),
                entry_price=110.0,
                exit_price=105.0,
                quantity=100,
                pnl=-500.0,  # Loser
                commission=2.0,
                side="LONG",
            ),
            RoundTripTrade(
                ticker="MSFT",
                entry_time=datetime(2024, 1, 15),
                exit_time=datetime(2024, 1, 20),
                entry_price=200.0,
                exit_price=210.0,
                quantity=50,
                pnl=500.0,  # Winner
                commission=2.0,
                side="LONG",
            ),
        ]

    def test_win_rate(self, sample_round_trips: list[RoundTripTrade]) -> None:
        """Test win rate calculation."""
        wr = win_rate(sample_round_trips)
        # 2 winners, 1 loser = 66.7%
        assert wr == pytest.approx(0.667, rel=0.01)

    def test_win_rate_empty(self) -> None:
        """Test win rate with no trades."""
        assert win_rate([]) == 0.0

    def test_profit_factor(self, sample_round_trips: list[RoundTripTrade]) -> None:
        """Test profit factor calculation."""
        pf = profit_factor(sample_round_trips)
        # Gross profit: 1000 + 500 = 1500
        # Gross loss: 500
        # PF = 1500 / 500 = 3.0
        assert pf == pytest.approx(3.0)

    def test_profit_factor_no_losers(self) -> None:
        """Test profit factor with no losing trades."""
        winners = [
            RoundTripTrade(
                ticker="AAPL",
                entry_time=datetime(2024, 1, 1),
                exit_time=datetime(2024, 1, 5),
                entry_price=100.0,
                exit_price=110.0,
                quantity=100,
                pnl=1000.0,
                commission=2.0,
                side="LONG",
            ),
        ]
        assert profit_factor(winners) == float("inf")

    def test_average_win_loss_ratio(self, sample_round_trips: list[RoundTripTrade]) -> None:
        """Test average win/loss ratio."""
        ratio = average_win_loss_ratio(sample_round_trips)
        # Avg win: (1000 + 500) / 2 = 750
        # Avg loss: 500
        # Ratio: 1.5
        assert ratio == pytest.approx(1.5)

    def test_average_trade_duration(self, sample_round_trips: list[RoundTripTrade]) -> None:
        """Test average trade duration."""
        avg_dur = average_trade_duration(sample_round_trips)
        # Durations: 4, 2, 5 days -> avg = 3.67 days
        assert avg_dur.days == 3

    def test_expectancy(self, sample_round_trips: list[RoundTripTrade]) -> None:
        """Test trade expectancy."""
        exp = expectancy(sample_round_trips)
        # Total PnL: 1000 - 500 + 500 = 1000
        # Num trades: 3
        # Expectancy: 333.33
        assert exp == pytest.approx(333.33, rel=0.01)


class TestExtractRoundTrips:
    """Tests for round trip extraction."""

    def test_simple_round_trip(self) -> None:
        """Test extraction of simple buy-sell round trip."""
        trades = [
            Trade(
                timestamp=datetime(2024, 1, 1),
                ticker="AAPL",
                quantity=100,
                price=100.0,
                commission=1.0,
                slippage=0.0,
                order_id="1",
            ),
            Trade(
                timestamp=datetime(2024, 1, 5),
                ticker="AAPL",
                quantity=-100,
                price=110.0,
                commission=1.0,
                slippage=0.0,
                order_id="2",
            ),
        ]

        round_trips = extract_round_trips(trades)

        assert len(round_trips) == 1
        assert round_trips[0].ticker == "AAPL"
        assert round_trips[0].pnl == pytest.approx(998.0)  # 1000 - 2 commission
        assert round_trips[0].side == "LONG"

    def test_multiple_tickers(self) -> None:
        """Test round trips across multiple tickers."""
        trades = [
            Trade(
                datetime(2024, 1, 1),
                "AAPL",
                100,
                100.0,
                1.0,
                0.0,
                "1",
            ),
            Trade(
                datetime(2024, 1, 2),
                "MSFT",
                50,
                200.0,
                1.0,
                0.0,
                "2",
            ),
            Trade(
                datetime(2024, 1, 3),
                "AAPL",
                -100,
                110.0,
                1.0,
                0.0,
                "3",
            ),
            Trade(
                datetime(2024, 1, 4),
                "MSFT",
                -50,
                210.0,
                1.0,
                0.0,
                "4",
            ),
        ]

        round_trips = extract_round_trips(trades)

        assert len(round_trips) == 2
        tickers = {rt.ticker for rt in round_trips}
        assert tickers == {"AAPL", "MSFT"}


class TestBenchmarkMetrics:
    """Tests for benchmark comparison metrics."""

    def test_beta(self) -> None:
        """Test beta calculation."""
        # Portfolio moves with benchmark
        benchmark = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        # Portfolio has beta of ~2 (moves twice as much)
        portfolio = pl.Series([0.02, -0.02, 0.04, -0.04, 0.02])

        b = beta(portfolio, benchmark)
        assert b == pytest.approx(2.0, rel=0.1)

    def test_beta_uncorrelated(self) -> None:
        """Test beta with uncorrelated returns."""
        benchmark = pl.Series([0.01, -0.01, 0.02, -0.02, 0.01])
        portfolio = pl.Series([0.01, 0.01, 0.01, 0.01, 0.01])  # Constant

        b = beta(portfolio, benchmark)
        # Should be close to 0
        assert abs(b) < 0.1

    def test_alpha(self) -> None:
        """Test alpha calculation."""
        benchmark = pl.Series([0.001] * 252)  # 0.1% daily
        portfolio = pl.Series([0.002] * 252)  # 0.2% daily (outperformance)

        a = alpha(portfolio, benchmark, risk_free_rate=0.0)
        # Should be positive (portfolio beats benchmark)
        assert a > 0


class TestCalculateMetrics:
    """Tests for the combined metrics calculator."""

    def test_calculate_metrics_full(self) -> None:
        """Test full metrics calculation."""
        snapshots = [
            PortfolioSnapshot(
                datetime(2024, 1, i),
                90000.0,
                10000.0 + i * 100,
                100000.0 + i * 100,
                0.0,
                i * 100,
            )
            for i in range(1, 11)
        ]

        trades = [
            Trade(datetime(2024, 1, 1), "AAPL", 100, 100.0, 1.0, 0.0, "1"),
            Trade(datetime(2024, 1, 5), "AAPL", -100, 110.0, 1.0, 0.0, "2"),
        ]

        metrics = calculate_metrics(
            snapshots=snapshots,
            trades=trades,
            initial_capital=100000.0,
        )

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0
        assert metrics.num_trades == 1
        assert metrics.win_rate == 1.0

    def test_calculate_metrics_empty(self) -> None:
        """Test metrics with empty data."""
        metrics = calculate_metrics(
            snapshots=[],
            trades=[],
            initial_capital=100000.0,
        )

        assert metrics.total_return == 0.0
        assert metrics.num_trades == 0
