"""Tests for tearsheet generation."""

import json
from dataclasses import dataclass
from datetime import datetime

import polars as pl
import pytest

from backtester.analysis.tearsheet import (
    MonthlyReturns,
    Tearsheet,
    YearlyReturns,
    _calculate_monthly_returns,
    _calculate_yearly_returns,
    generate_tearsheet,
)
from backtester.core.portfolio import PortfolioSnapshot, Trade


@dataclass
class MockBacktestConfig:
    """Mock BacktestConfig for testing."""

    initial_capital: float = 100_000.0


@dataclass
class MockBacktestResult:
    """Mock BacktestResult for testing."""

    config: MockBacktestConfig
    snapshots: list[PortfolioSnapshot]
    trades: list[Trade]
    start_date: datetime | None
    end_date: datetime | None
    bars_processed: int
    final_equity: float


class TestMonthlyReturns:
    """Tests for monthly return calculation."""

    def test_calculate_monthly_returns(self) -> None:
        """Test monthly returns are calculated correctly."""
        snapshots = [
            PortfolioSnapshot(datetime(2024, 1, 1), 90000, 10000, 100000, 0, 0),
            PortfolioSnapshot(datetime(2024, 1, 15), 90000, 15000, 105000, 0, 5000),
            PortfolioSnapshot(datetime(2024, 1, 31), 90000, 20000, 110000, 0, 10000),
            PortfolioSnapshot(datetime(2024, 2, 15), 90000, 18000, 108000, 0, 8000),
            PortfolioSnapshot(datetime(2024, 2, 28), 90000, 22000, 112000, 0, 12000),
        ]

        from backtester.analysis.metrics import snapshots_to_dataframe

        df = snapshots_to_dataframe(snapshots)
        monthly = _calculate_monthly_returns(df)

        assert len(monthly) == 2
        assert monthly[0].year == 2024
        assert monthly[0].month == 1
        # Jan: 100000 -> 110000 = 10%
        assert monthly[0].return_pct == pytest.approx(0.10)
        assert monthly[1].year == 2024
        assert monthly[1].month == 2
        # Feb: 108000 -> 112000 (first to last in Feb)
        assert monthly[1].return_pct == pytest.approx(0.037, rel=0.01)

    def test_calculate_monthly_returns_empty(self) -> None:
        """Test empty snapshot handling."""
        from backtester.analysis.metrics import snapshots_to_dataframe

        df = snapshots_to_dataframe([])
        monthly = _calculate_monthly_returns(df)
        assert monthly == []


class TestYearlyReturns:
    """Tests for yearly return calculation."""

    def test_calculate_yearly_returns(self) -> None:
        """Test yearly returns are calculated correctly."""
        snapshots = [
            PortfolioSnapshot(datetime(2023, 1, 1), 90000, 10000, 100000, 0, 0),
            PortfolioSnapshot(datetime(2023, 12, 31), 90000, 30000, 120000, 0, 20000),
            PortfolioSnapshot(datetime(2024, 1, 1), 90000, 30000, 120000, 0, 20000),
            PortfolioSnapshot(datetime(2024, 12, 31), 90000, 50000, 140000, 0, 40000),
        ]

        from backtester.analysis.metrics import snapshots_to_dataframe

        df = snapshots_to_dataframe(snapshots)
        yearly = _calculate_yearly_returns(df)

        assert len(yearly) == 2
        assert yearly[0].year == 2023
        # 2023: 100000 -> 120000 = 20%
        assert yearly[0].return_pct == pytest.approx(0.20)
        assert yearly[1].year == 2024
        # 2024: 120000 -> 140000 = 16.67%
        assert yearly[1].return_pct == pytest.approx(0.167, rel=0.01)


class TestTearsheetSummary:
    """Tests for tearsheet text summary."""

    @pytest.fixture
    def sample_tearsheet(self) -> Tearsheet:
        """Create a sample tearsheet for testing."""
        from backtester.analysis.metrics import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            annualized_return=0.14,
            volatility=0.20,
            downside_deviation=0.12,
            sharpe_ratio=0.70,
            sortino_ratio=1.17,
            calmar_ratio=1.50,
            max_drawdown=0.10,
            max_drawdown_duration_days=30,
            average_drawdown=0.05,
            num_trades=10,
            win_rate=0.60,
            profit_factor=1.50,
            avg_win_loss_ratio=1.20,
            avg_trade_duration_days=5,
            expectancy=500.0,
            beta=1.1,
            alpha=0.02,
        )

        return Tearsheet(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_equity=115000.0,
            bars_processed=252,
            metrics=metrics,
            benchmark_ticker="SPY",
            benchmark_return=0.10,
            monthly_returns=[
                MonthlyReturns(2024, 1, 0.02),
                MonthlyReturns(2024, 2, 0.03),
            ],
            yearly_returns=[YearlyReturns(2024, 0.15)],
        )

    def test_summary_output(self, sample_tearsheet: Tearsheet) -> None:
        """Test summary produces expected text."""
        summary = sample_tearsheet.summary()

        assert "BACKTEST PERFORMANCE REPORT" in summary
        assert "Total Return:    +15.00%" in summary
        assert "Sharpe Ratio:    0.70" in summary
        assert "Max Drawdown:    10.00%" in summary
        assert "Win Rate:           60.0%" in summary

    def test_summary_benchmark(self, sample_tearsheet: Tearsheet) -> None:
        """Test benchmark info appears in summary."""
        summary = sample_tearsheet.summary()

        assert "Benchmark (SPY):" in summary
        assert "+10.00%" in summary
        assert "Excess Return:" in summary

    def test_summary_without_trades(self, sample_tearsheet: Tearsheet) -> None:
        """Test summary can exclude trade stats."""
        summary = sample_tearsheet.summary(include_trades=False)

        assert "TRADE STATISTICS" not in summary
        assert "Win Rate" not in summary


class TestTearsheetToDict:
    """Tests for tearsheet dict/JSON export."""

    @pytest.fixture
    def sample_tearsheet(self) -> Tearsheet:
        """Create a sample tearsheet for testing."""
        from backtester.analysis.metrics import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            annualized_return=0.14,
            volatility=0.20,
            downside_deviation=0.12,
            sharpe_ratio=0.70,
            sortino_ratio=1.17,
            calmar_ratio=1.50,
            max_drawdown=0.10,
            max_drawdown_duration_days=30,
            average_drawdown=0.05,
            num_trades=10,
            win_rate=0.60,
            profit_factor=float("inf"),  # Test infinity handling
            avg_win_loss_ratio=1.20,
            avg_trade_duration_days=5,
            expectancy=500.0,
        )

        return Tearsheet(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_equity=115000.0,
            bars_processed=252,
            metrics=metrics,
            benchmark_ticker=None,
            benchmark_return=None,
            monthly_returns=[MonthlyReturns(2024, 1, 0.02)],
            yearly_returns=[YearlyReturns(2024, 0.15)],
        )

    def test_to_dict_structure(self, sample_tearsheet: Tearsheet) -> None:
        """Test dict has expected structure."""
        d = sample_tearsheet.to_dict()

        assert "metadata" in d
        assert "returns" in d
        assert "risk" in d
        assert "risk_adjusted" in d
        assert "trades" in d
        assert "monthly_returns" in d
        assert "yearly_returns" in d

    def test_to_dict_values(self, sample_tearsheet: Tearsheet) -> None:
        """Test dict has correct values."""
        d = sample_tearsheet.to_dict()

        assert d["metadata"]["initial_capital"] == 100000.0
        assert d["returns"]["total_return"] == 0.15
        assert d["risk"]["max_drawdown"] == 0.10
        assert d["trades"]["num_trades"] == 10

    def test_to_dict_infinity_handling(self, sample_tearsheet: Tearsheet) -> None:
        """Test infinity values are converted to strings."""
        d = sample_tearsheet.to_dict()

        # profit_factor was set to infinity
        assert d["trades"]["profit_factor"] == "Infinity"

    def test_to_json_valid(self, sample_tearsheet: Tearsheet) -> None:
        """Test JSON output is valid."""
        json_str = sample_tearsheet.to_json()

        # Should not raise
        parsed = json.loads(json_str)
        assert parsed["metadata"]["final_equity"] == 115000.0


class TestTearsheetToHtml:
    """Tests for tearsheet HTML export."""

    @pytest.fixture
    def sample_tearsheet(self) -> Tearsheet:
        """Create a sample tearsheet for testing."""
        from backtester.analysis.metrics import PerformanceMetrics

        metrics = PerformanceMetrics(
            total_return=0.15,
            cagr=0.12,
            annualized_return=0.14,
            volatility=0.20,
            downside_deviation=0.12,
            sharpe_ratio=0.70,
            sortino_ratio=1.17,
            calmar_ratio=1.50,
            max_drawdown=0.10,
            max_drawdown_duration_days=30,
            average_drawdown=0.05,
            num_trades=10,
            win_rate=0.60,
            profit_factor=1.50,
            avg_win_loss_ratio=1.20,
            avg_trade_duration_days=5,
            expectancy=500.0,
        )

        return Tearsheet(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000.0,
            final_equity=115000.0,
            bars_processed=252,
            metrics=metrics,
            benchmark_ticker=None,
            benchmark_return=None,
            monthly_returns=[
                MonthlyReturns(2024, 1, 0.02),
                MonthlyReturns(2024, 2, -0.01),
            ],
            yearly_returns=[YearlyReturns(2024, 0.15)],
        )

    def test_to_html_structure(self, sample_tearsheet: Tearsheet) -> None:
        """Test HTML has expected structure."""
        html = sample_tearsheet.to_html()

        assert "<!DOCTYPE html>" in html
        assert "<title>Backtest Performance Report</title>" in html
        assert "+15.00%" in html
        assert "Monthly Returns" in html

    def test_to_html_file_write(self, sample_tearsheet: Tearsheet, tmp_path) -> None:
        """Test HTML can be written to file."""
        filepath = tmp_path / "report.html"
        sample_tearsheet.to_html(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "Backtest Performance Report" in content


class TestGenerateTearsheet:
    """Tests for generate_tearsheet function."""

    def test_generate_tearsheet_basic(self) -> None:
        """Test generating tearsheet from mock result."""
        snapshots = [
            PortfolioSnapshot(
                datetime(2024, 1, i), 90000.0, 10000.0 + i * 100, 100000.0 + i * 100, 0, i * 100
            )
            for i in range(1, 11)
        ]

        trades = [
            Trade(datetime(2024, 1, 1), "AAPL", 100, 100.0, 1.0, 0.0, "1"),
            Trade(datetime(2024, 1, 5), "AAPL", -100, 110.0, 1.0, 0.0, "2"),
        ]

        result = MockBacktestResult(
            config=MockBacktestConfig(initial_capital=100000.0),
            snapshots=snapshots,
            trades=trades,
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            bars_processed=10,
            final_equity=100900.0,
        )

        tearsheet = generate_tearsheet(result)

        assert tearsheet.initial_capital == 100000.0
        assert tearsheet.final_equity == 100900.0
        assert tearsheet.metrics.num_trades == 1
        assert tearsheet.start_date == datetime(2024, 1, 1)

    def test_generate_tearsheet_with_benchmark(self) -> None:
        """Test generating tearsheet with benchmark returns."""
        snapshots = [
            PortfolioSnapshot(datetime(2024, 1, i), 90000.0, 10000.0, 100000.0, 0, 0)
            for i in range(1, 11)
        ]

        result = MockBacktestResult(
            config=MockBacktestConfig(initial_capital=100000.0),
            snapshots=snapshots,
            trades=[],
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
            bars_processed=10,
            final_equity=100000.0,
        )

        benchmark_returns = pl.Series([0.001] * 10)

        tearsheet = generate_tearsheet(result, benchmark="SPY", benchmark_returns=benchmark_returns)

        assert tearsheet.benchmark_ticker == "SPY"
        assert tearsheet.benchmark_return is not None
        assert tearsheet.benchmark_return > 0

    def test_generate_tearsheet_empty(self) -> None:
        """Test generating tearsheet with empty data."""
        result = MockBacktestResult(
            config=MockBacktestConfig(initial_capital=100000.0),
            snapshots=[],
            trades=[],
            start_date=None,
            end_date=None,
            bars_processed=0,
            final_equity=100000.0,
        )

        tearsheet = generate_tearsheet(result)

        assert tearsheet.metrics.total_return == 0.0
        assert tearsheet.metrics.num_trades == 0
        assert tearsheet.monthly_returns == []
        assert tearsheet.yearly_returns == []
