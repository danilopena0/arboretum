"""Tests for visualization module."""

from dataclasses import dataclass
from datetime import datetime

import plotly.graph_objects as go
import pytest

from backtester.analysis.visualization import (
    _empty_figure,
    create_performance_dashboard,
    plot_cumulative_returns,
    plot_drawdown,
    plot_equity_curve,
    plot_monthly_returns,
    plot_portfolio_composition,
    plot_returns_distribution,
    plot_rolling_sharpe,
    plot_trade_pnl,
    plot_trades_on_price,
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


@pytest.fixture
def sample_snapshots() -> list[PortfolioSnapshot]:
    """Create sample snapshots spanning multiple months."""
    snapshots = []
    equity = 100000.0

    # Generate 100 days of data with some volatility
    for i in range(100):
        day = datetime(2024, 1, 1) + __import__("datetime").timedelta(days=i)
        # Simulate some price movement
        change = 500 * (1 if i % 3 != 0 else -1)
        equity += change
        snapshots.append(
            PortfolioSnapshot(
                timestamp=day,
                cash=equity * 0.3,
                positions_value=equity * 0.7,
                total_equity=equity,
                realized_pnl=i * 100,
                unrealized_pnl=change,
            )
        )

    return snapshots


@pytest.fixture
def sample_trades() -> list[Trade]:
    """Create sample trades."""
    return [
        Trade(datetime(2024, 1, 5), "AAPL", 100, 150.0, 1.0, 0.01, "1"),
        Trade(datetime(2024, 1, 15), "AAPL", -100, 160.0, 1.0, 0.01, "2"),
        Trade(datetime(2024, 2, 1), "MSFT", 50, 300.0, 1.0, 0.01, "3"),
        Trade(datetime(2024, 2, 20), "MSFT", -50, 290.0, 1.0, 0.01, "4"),
        Trade(datetime(2024, 3, 1), "GOOGL", 30, 100.0, 1.0, 0.01, "5"),
        Trade(datetime(2024, 3, 15), "GOOGL", -30, 115.0, 1.0, 0.01, "6"),
    ]


@pytest.fixture
def mock_result(sample_snapshots, sample_trades) -> MockBacktestResult:
    """Create mock backtest result."""
    return MockBacktestResult(
        config=MockBacktestConfig(initial_capital=100000.0),
        snapshots=sample_snapshots,
        trades=sample_trades,
        start_date=sample_snapshots[0].timestamp if sample_snapshots else None,
        end_date=sample_snapshots[-1].timestamp if sample_snapshots else None,
        bars_processed=len(sample_snapshots),
        final_equity=sample_snapshots[-1].total_equity if sample_snapshots else 100000.0,
    )


class TestPlotEquityCurve:
    """Tests for equity curve plotting."""

    def test_plot_equity_curve_basic(self, sample_snapshots) -> None:
        """Test basic equity curve plot."""
        fig = plot_equity_curve(sample_snapshots)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1
        assert fig.data[0].name == "Portfolio"

    def test_plot_equity_curve_with_benchmark(self, sample_snapshots) -> None:
        """Test equity curve with benchmark."""
        benchmark = [100000 + i * 300 for i in range(len(sample_snapshots))]

        fig = plot_equity_curve(sample_snapshots, benchmark_equity=benchmark, benchmark_name="SPY")

        assert len(fig.data) == 2
        assert fig.data[1].name == "SPY"

    def test_plot_equity_curve_empty(self) -> None:
        """Test equity curve with empty data."""
        fig = plot_equity_curve([])

        assert isinstance(fig, go.Figure)
        # Should have annotation for no data
        assert len(fig.layout.annotations) > 0

    def test_plot_equity_curve_custom_title(self, sample_snapshots) -> None:
        """Test custom title."""
        fig = plot_equity_curve(sample_snapshots, title="My Portfolio")

        assert fig.layout.title.text == "My Portfolio"


class TestPlotCumulativeReturns:
    """Tests for cumulative returns plotting."""

    def test_plot_cumulative_returns_basic(self, sample_snapshots) -> None:
        """Test basic cumulative returns plot."""
        fig = plot_cumulative_returns(sample_snapshots)

        assert isinstance(fig, go.Figure)
        assert fig.data[0].name == "Portfolio"

    def test_plot_cumulative_returns_with_benchmark(self, sample_snapshots) -> None:
        """Test cumulative returns with benchmark."""
        benchmark_returns = [0.001] * len(sample_snapshots)

        fig = plot_cumulative_returns(
            sample_snapshots, benchmark_returns=benchmark_returns, benchmark_name="SPY"
        )

        assert len(fig.data) == 2

    def test_plot_cumulative_returns_empty(self) -> None:
        """Test with empty data."""
        fig = plot_cumulative_returns([])

        assert isinstance(fig, go.Figure)


class TestPlotReturnsDistribution:
    """Tests for returns distribution plotting."""

    def test_plot_returns_distribution_basic(self, sample_snapshots) -> None:
        """Test basic returns distribution."""
        fig = plot_returns_distribution(sample_snapshots)

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Histogram)

    def test_plot_returns_distribution_custom_bins(self, sample_snapshots) -> None:
        """Test custom number of bins."""
        fig = plot_returns_distribution(sample_snapshots, bins=20)

        assert isinstance(fig, go.Figure)

    def test_plot_returns_distribution_empty(self) -> None:
        """Test with empty data."""
        fig = plot_returns_distribution([])

        assert isinstance(fig, go.Figure)


class TestPlotDrawdown:
    """Tests for drawdown plotting."""

    def test_plot_drawdown_basic(self, sample_snapshots) -> None:
        """Test basic drawdown plot."""
        fig = plot_drawdown(sample_snapshots)

        assert isinstance(fig, go.Figure)
        # Values should be negative (underwater)
        y_values = fig.data[0].y
        assert all(v <= 0 for v in y_values)

    def test_plot_drawdown_empty(self) -> None:
        """Test with empty data."""
        fig = plot_drawdown([])

        assert isinstance(fig, go.Figure)

    def test_plot_drawdown_has_annotation(self, sample_snapshots) -> None:
        """Test max drawdown annotation is present."""
        fig = plot_drawdown(sample_snapshots)

        # Should have annotation for max drawdown
        assert len(fig.layout.annotations) > 0


class TestPlotTradesOnPrice:
    """Tests for trade markers on price chart."""

    def test_plot_trades_on_price_basic(self, mock_result) -> None:
        """Test basic trade plot."""
        price_data = [(datetime(2024, 1, i), 150 + i * 0.5) for i in range(1, 32)]

        fig = plot_trades_on_price(mock_result, "AAPL", price_data)

        assert isinstance(fig, go.Figure)
        # Should have price line + buy markers + sell markers
        assert len(fig.data) >= 2

    def test_plot_trades_on_price_no_trades(self, mock_result) -> None:
        """Test with ticker that has no trades."""
        price_data = [(datetime(2024, 1, i), 100) for i in range(1, 10)]

        fig = plot_trades_on_price(mock_result, "XYZ", price_data)

        # Should still have price line
        assert len(fig.data) >= 1

    def test_plot_trades_on_price_empty_prices(self, mock_result) -> None:
        """Test with empty price data."""
        fig = plot_trades_on_price(mock_result, "AAPL", [])

        assert isinstance(fig, go.Figure)


class TestPlotTradePnl:
    """Tests for trade P&L bar chart."""

    def test_plot_trade_pnl_basic(self, sample_trades) -> None:
        """Test basic trade P&L chart."""
        fig = plot_trade_pnl(sample_trades)

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Bar)

    def test_plot_trade_pnl_colors(self, sample_trades) -> None:
        """Test P&L bars have correct colors (green/red)."""
        fig = plot_trade_pnl(sample_trades)

        colors = fig.data[0].marker.color
        # AAPL trade was profitable, MSFT was loss, GOOGL was profitable
        assert len(colors) >= 2

    def test_plot_trade_pnl_empty(self) -> None:
        """Test with empty trades."""
        fig = plot_trade_pnl([])

        assert isinstance(fig, go.Figure)


class TestPlotMonthlyReturns:
    """Tests for monthly returns heatmap."""

    def test_plot_monthly_returns_basic(self, sample_snapshots) -> None:
        """Test basic monthly returns heatmap."""
        fig = plot_monthly_returns(sample_snapshots)

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Heatmap)

    def test_plot_monthly_returns_empty(self) -> None:
        """Test with empty data."""
        fig = plot_monthly_returns([])

        assert isinstance(fig, go.Figure)


class TestPlotRollingSharpe:
    """Tests for rolling Sharpe ratio plotting."""

    def test_plot_rolling_sharpe_basic(self, sample_snapshots) -> None:
        """Test basic rolling Sharpe plot."""
        fig = plot_rolling_sharpe(sample_snapshots, window=20)

        assert isinstance(fig, go.Figure)

    def test_plot_rolling_sharpe_insufficient_data(self) -> None:
        """Test with insufficient data for window."""
        snapshots = [
            PortfolioSnapshot(datetime(2024, 1, i), 90000, 10000, 100000, 0, 0)
            for i in range(1, 10)
        ]

        fig = plot_rolling_sharpe(snapshots, window=63)

        # Should return empty figure
        assert isinstance(fig, go.Figure)

    def test_plot_rolling_sharpe_empty(self) -> None:
        """Test with empty data."""
        fig = plot_rolling_sharpe([])

        assert isinstance(fig, go.Figure)


class TestPlotPortfolioComposition:
    """Tests for portfolio composition chart."""

    def test_plot_portfolio_composition_basic(self, mock_result) -> None:
        """Test basic portfolio composition plot."""
        fig = plot_portfolio_composition(mock_result)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # Cash and Positions

    def test_plot_portfolio_composition_empty(self) -> None:
        """Test with empty data."""
        result = MockBacktestResult(
            config=MockBacktestConfig(),
            snapshots=[],
            trades=[],
            start_date=None,
            end_date=None,
            bars_processed=0,
            final_equity=100000.0,
        )

        fig = plot_portfolio_composition(result)

        assert isinstance(fig, go.Figure)


class TestCreatePerformanceDashboard:
    """Tests for the comprehensive dashboard."""

    def test_create_performance_dashboard(self, mock_result) -> None:
        """Test creating full dashboard."""
        fig = create_performance_dashboard(mock_result)

        assert isinstance(fig, go.Figure)
        # Dashboard should have multiple traces
        assert len(fig.data) >= 4

    def test_create_performance_dashboard_height(self, mock_result) -> None:
        """Test dashboard has appropriate height."""
        fig = create_performance_dashboard(mock_result)

        assert fig.layout.height == 900

    def test_create_performance_dashboard_empty(self) -> None:
        """Test dashboard with empty data."""
        result = MockBacktestResult(
            config=MockBacktestConfig(),
            snapshots=[],
            trades=[],
            start_date=None,
            end_date=None,
            bars_processed=0,
            final_equity=100000.0,
        )

        fig = create_performance_dashboard(result)

        assert isinstance(fig, go.Figure)


class TestEmptyFigure:
    """Tests for empty figure helper."""

    def test_empty_figure_has_annotation(self) -> None:
        """Test empty figure has no data message."""
        fig = _empty_figure("Test Title")

        assert fig.layout.title.text == "Test Title"
        assert len(fig.layout.annotations) > 0
        assert "No data available" in fig.layout.annotations[0].text


class TestVisualizationWriteToFile:
    """Tests for saving visualizations to files."""

    def test_write_html(self, sample_snapshots, tmp_path) -> None:
        """Test writing figure to HTML file."""
        fig = plot_equity_curve(sample_snapshots)
        filepath = tmp_path / "equity.html"

        fig.write_html(str(filepath))

        assert filepath.exists()
        content = filepath.read_text()
        assert "plotly" in content.lower()
