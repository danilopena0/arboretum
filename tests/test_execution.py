"""Tests for execution models."""

from datetime import datetime

import pytest

from backtester.core.events import MarketEvent, OrderEvent, OrderSide
from backtester.core.execution import (
    FixedSlippage,
    PercentageSlippage,
    PerShareCommission,
    SimulatedBroker,
    TieredCommission,
    VolumeBasedSlippage,
    ZeroCommission,
    ZeroSlippage,
    create_order,
)


class TestSlippageModels:
    """Tests for slippage models."""

    @pytest.fixture
    def sample_order(self) -> OrderEvent:
        return create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

    @pytest.fixture
    def sample_market(self) -> MarketEvent:
        return MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1_000_000,
        )

    def test_zero_slippage(self, sample_order: OrderEvent, sample_market: MarketEvent) -> None:
        model = ZeroSlippage()
        assert model.calculate_slippage(sample_order, sample_market) == 0.0

    def test_fixed_slippage(self, sample_order: OrderEvent, sample_market: MarketEvent) -> None:
        model = FixedSlippage(cents_per_share=5.0)  # 5 cents
        assert model.calculate_slippage(sample_order, sample_market) == pytest.approx(0.05)

    def test_percentage_slippage(
        self, sample_order: OrderEvent, sample_market: MarketEvent
    ) -> None:
        model = PercentageSlippage(percentage=0.001)  # 0.1%
        # 0.1% of 101 = 0.101
        assert model.calculate_slippage(sample_order, sample_market) == pytest.approx(0.101)

    def test_volume_based_slippage(
        self, sample_order: OrderEvent, sample_market: MarketEvent
    ) -> None:
        model = VolumeBasedSlippage(impact_factor=0.1)
        # 100 shares / 1M volume = 0.0001 * 0.1 = 0.00001
        # 0.00001 * 101 = 0.00101
        slippage = model.calculate_slippage(sample_order, sample_market)
        assert slippage == pytest.approx(0.00101, rel=0.01)

    def test_volume_based_slippage_large_order(self) -> None:
        """Large order has more market impact."""
        model = VolumeBasedSlippage(impact_factor=0.1, max_slippage_pct=0.02)

        large_order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100_000,  # 10% of volume
        )

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1_000_000,
        )

        # 100k / 1M = 0.1 * 0.1 = 0.01 = 1%
        # 1% of 100 = 1.0
        slippage = model.calculate_slippage(large_order, market)
        assert slippage == pytest.approx(1.0)


class TestCommissionModels:
    """Tests for commission models."""

    @pytest.fixture
    def sample_order(self) -> OrderEvent:
        return create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

    def test_zero_commission(self, sample_order: OrderEvent) -> None:
        model = ZeroCommission()
        assert model.calculate_commission(sample_order, 100.0) == 0.0

    def test_per_share_commission(self, sample_order: OrderEvent) -> None:
        model = PerShareCommission(rate=0.01, minimum=1.0)
        # 100 shares * 0.01 = 1.0
        assert model.calculate_commission(sample_order, 100.0) == pytest.approx(1.0)

    def test_per_share_commission_minimum(self) -> None:
        model = PerShareCommission(rate=0.001, minimum=1.0)
        small_order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,  # 10 * 0.001 = 0.01, below minimum
        )
        assert model.calculate_commission(small_order, 100.0) == pytest.approx(1.0)

    def test_tiered_commission(self) -> None:
        model = TieredCommission()  # Default tiers

        # Small order
        small_order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        commission = model.calculate_commission(small_order, 100.0)
        # 100 shares at first tier rate
        assert commission > 0


class TestSimulatedBroker:
    """Tests for SimulatedBroker."""

    def test_submit_order(self) -> None:
        broker = SimulatedBroker()
        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        order_id = broker.submit_order(order)
        assert order_id == order.order_id
        assert len(broker.get_pending_orders()) == 1

    def test_cancel_order(self) -> None:
        broker = SimulatedBroker()
        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        broker.submit_order(order)
        assert broker.cancel_order(order.order_id) is True
        assert len(broker.get_pending_orders()) == 0

    def test_cancel_nonexistent_order(self) -> None:
        broker = SimulatedBroker()
        assert broker.cancel_order("fake_id") is False

    def test_process_market_fills_order(self) -> None:
        broker = SimulatedBroker()
        order = create_order(
            timestamp=datetime(2024, 1, 15, 9, 30),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )

        broker.submit_order(order)

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15, 9, 31),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        assert len(fills) == 1
        assert fills[0].ticker == "AAPL"
        assert fills[0].quantity == 100
        assert fills[0].fill_price == pytest.approx(100.0)  # Open price (default)
        assert len(broker.get_pending_orders()) == 0

    def test_slippage_applied(self) -> None:
        broker = SimulatedBroker(
            slippage_model=FixedSlippage(cents_per_share=10.0),  # 10 cents
        )

        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        broker.submit_order(order)

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        # Close + slippage for buy
        assert fills[0].fill_price == pytest.approx(100.10)
        assert fills[0].slippage == pytest.approx(0.10)

    def test_commission_applied(self) -> None:
        broker = SimulatedBroker(
            commission_model=PerShareCommission(rate=0.01, minimum=0.0),
        )

        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        broker.submit_order(order)

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,
            close=100.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        # 100 shares * 0.01 = 1.0
        assert fills[0].commission == pytest.approx(1.0)

    def test_limit_order_not_filled(self) -> None:
        """Test limit order not filled when price doesn't reach."""
        broker = SimulatedBroker()

        # Buy limit at $95 when price is $100-102
        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=95.0,
        )
        broker.submit_order(order)

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=99.0,  # Doesn't reach 95
            close=101.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        assert len(fills) == 0
        assert len(broker.get_pending_orders()) == 1

    def test_limit_order_filled(self) -> None:
        """Test limit order filled when price reaches."""
        broker = SimulatedBroker()

        # Buy limit at $99 when low is $98
        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            limit_price=99.0,
        )
        broker.submit_order(order)

        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            open=100.0,
            high=102.0,
            low=98.0,  # Reaches 99
            close=101.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        assert len(fills) == 1
        assert fills[0].fill_price == pytest.approx(99.0)

    def test_different_ticker_not_processed(self) -> None:
        """Test that orders for different tickers are not processed."""
        broker = SimulatedBroker()

        order = create_order(
            timestamp=datetime(2024, 1, 15),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
        )
        broker.submit_order(order)

        # Market data for different ticker
        market = MarketEvent(
            timestamp=datetime(2024, 1, 15),
            ticker="MSFT",
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=1_000_000,
        )

        fills = broker.process_market_data(market)

        assert len(fills) == 0
        assert len(broker.get_pending_orders()) == 1
