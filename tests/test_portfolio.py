"""Tests for portfolio management."""

from datetime import datetime

import pytest

from backtester.core.portfolio import Portfolio, Position


class TestPosition:
    """Tests for Position class."""

    def test_create_position(self) -> None:
        """Test basic position creation."""
        pos = Position(ticker="AAPL", quantity=100, avg_cost=150.0)
        assert pos.ticker == "AAPL"
        assert pos.quantity == 100
        assert pos.avg_cost == 150.0
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat

    def test_short_position(self) -> None:
        """Test short position detection."""
        pos = Position(ticker="AAPL", quantity=-100, avg_cost=150.0)
        assert pos.is_short
        assert not pos.is_long

    def test_flat_position(self) -> None:
        """Test flat position detection."""
        pos = Position(ticker="AAPL", quantity=0)
        assert pos.is_flat

    def test_unrealized_pnl_long(self) -> None:
        """Test unrealized P&L for long position."""
        pos = Position(ticker="AAPL", quantity=100, avg_cost=150.0)
        # Price went up
        assert pos.unrealized_pnl(160.0) == pytest.approx(1000.0)
        # Price went down
        assert pos.unrealized_pnl(140.0) == pytest.approx(-1000.0)

    def test_unrealized_pnl_short(self) -> None:
        """Test unrealized P&L for short position."""
        pos = Position(ticker="AAPL", quantity=-100, avg_cost=150.0)
        # Price went down (good for short)
        assert pos.unrealized_pnl(140.0) == pytest.approx(1000.0)
        # Price went up (bad for short)
        assert pos.unrealized_pnl(160.0) == pytest.approx(-1000.0)


class TestPortfolio:
    """Tests for Portfolio class."""

    def test_create_portfolio(self) -> None:
        """Test portfolio initialization."""
        portfolio = Portfolio(initial_capital=100_000.0)
        assert portfolio.cash == 100_000.0
        assert portfolio.total_equity == 100_000.0
        assert len(portfolio.positions) == 0

    def test_process_buy_fill(self) -> None:
        """Test processing a buy fill."""
        portfolio = Portfolio(initial_capital=100_000.0)

        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=150.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        assert portfolio.cash == pytest.approx(85_000.0)  # 100k - 15k
        position = portfolio.get_position("AAPL")
        assert position.quantity == 100
        assert position.avg_cost == 150.0

    def test_process_sell_fill(self) -> None:
        """Test processing a sell fill after buy."""
        portfolio = Portfolio(initial_capital=100_000.0)

        # Buy 100 shares at $150
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=150.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        # Sell 100 shares at $160
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 16),
            ticker="AAPL",
            quantity=-100,
            fill_price=160.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_002",
        )

        # Cash: 85k + 16k = 101k
        assert portfolio.cash == pytest.approx(101_000.0)
        position = portfolio.get_position("AAPL")
        assert position.quantity == 0
        assert position.realized_pnl == pytest.approx(1000.0)

    def test_average_cost_on_add(self) -> None:
        """Test average cost calculation when adding to position."""
        portfolio = Portfolio(initial_capital=100_000.0)

        # Buy 100 shares at $100
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=100.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        # Buy 100 more shares at $120
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 16),
            ticker="AAPL",
            quantity=100,
            fill_price=120.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_002",
        )

        position = portfolio.get_position("AAPL")
        assert position.quantity == 200
        # Average cost: (100*100 + 100*120) / 200 = 110
        assert position.avg_cost == pytest.approx(110.0)

    def test_partial_close_realized_pnl(self) -> None:
        """Test realized P&L on partial position close."""
        portfolio = Portfolio(initial_capital=100_000.0)

        # Buy 100 shares at $100
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=100.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        # Sell 50 shares at $120
        portfolio.process_fill(
            timestamp=datetime(2024, 1, 16),
            ticker="AAPL",
            quantity=-50,
            fill_price=120.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_002",
        )

        position = portfolio.get_position("AAPL")
        assert position.quantity == 50
        assert position.avg_cost == pytest.approx(100.0)  # Unchanged
        # Realized: 50 * (120 - 100) = 1000
        assert position.realized_pnl == pytest.approx(1000.0)

    def test_commission_deducted(self) -> None:
        """Test that commission is deducted from cash."""
        portfolio = Portfolio(initial_capital=100_000.0)

        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=100.0,
            commission=10.0,
            slippage=0.0,
            order_id="order_001",
        )

        # Cash: 100k - 10k - 10 commission = 89,990
        assert portfolio.cash == pytest.approx(89_990.0)

    def test_total_equity(self) -> None:
        """Test total equity calculation."""
        portfolio = Portfolio(initial_capital=100_000.0)

        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=100.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        # Update price to 110
        portfolio.update_price("AAPL", 110.0)

        # Cash: 90k, Positions: 100 * 110 = 11k
        assert portfolio.cash == pytest.approx(90_000.0)
        assert portfolio.positions_value == pytest.approx(11_000.0)
        assert portfolio.total_equity == pytest.approx(101_000.0)

    def test_can_afford(self) -> None:
        """Test affordability check."""
        portfolio = Portfolio(initial_capital=10_000.0)

        assert portfolio.can_afford("AAPL", 100, 50.0) is True  # 5k < 10k
        assert portfolio.can_afford("AAPL", 100, 150.0) is False  # 15k > 10k
        assert portfolio.can_afford("AAPL", -100, 150.0) is True  # Sells always OK

    def test_max_shares_affordable(self) -> None:
        """Test max affordable shares calculation."""
        portfolio = Portfolio(initial_capital=10_000.0)

        assert portfolio.max_shares_affordable(100.0) == 100  # 10k / 100 = 100
        assert portfolio.max_shares_affordable(100.0, commission=100.0) == 99  # (10k-100)/100

    def test_take_snapshot(self) -> None:
        """Test portfolio snapshot."""
        portfolio = Portfolio(initial_capital=100_000.0)

        snapshot = portfolio.take_snapshot(datetime(2024, 1, 15))

        assert snapshot.cash == 100_000.0
        assert snapshot.total_equity == 100_000.0
        assert len(portfolio.snapshots) == 1

    def test_reset(self) -> None:
        """Test portfolio reset."""
        portfolio = Portfolio(initial_capital=100_000.0)

        portfolio.process_fill(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            quantity=100,
            fill_price=100.0,
            commission=0.0,
            slippage=0.0,
            order_id="order_001",
        )

        portfolio.reset()

        assert portfolio.cash == 100_000.0
        assert len(portfolio.positions) == 0
        assert len(portfolio.trades) == 0
