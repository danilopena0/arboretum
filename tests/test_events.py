"""Tests for the event system."""

from datetime import datetime

import pytest

from backtester.core.events import (
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    SignalEvent,
    SignalType,
)


class TestMarketEvent:
    """Tests for MarketEvent."""

    def test_create_market_event(self, sample_market_event: MarketEvent) -> None:
        """Test basic MarketEvent creation."""
        assert sample_market_event.ticker == "AAPL"
        assert sample_market_event.close == 186.50
        assert sample_market_event.volume == 45_000_000
        assert sample_market_event.event_type == EventType.MARKET

    def test_market_event_is_frozen(self, sample_market_event: MarketEvent) -> None:
        """Test that MarketEvent is immutable."""
        with pytest.raises(AttributeError):
            sample_market_event.close = 200.0  # type: ignore

    def test_market_event_with_adjusted_prices(self, sample_market_event: MarketEvent) -> None:
        """Test price adjustment."""
        adjusted = sample_market_event.with_adjusted_prices(0.5)
        assert adjusted.open == pytest.approx(92.75)
        assert adjusted.close == pytest.approx(93.25)
        assert adjusted.volume == sample_market_event.volume  # Volume unchanged


class TestSignalEvent:
    """Tests for SignalEvent."""

    def test_create_signal_event(self, sample_signal_event: SignalEvent) -> None:
        """Test basic SignalEvent creation."""
        assert sample_signal_event.ticker == "AAPL"
        assert sample_signal_event.signal_type == SignalType.LONG
        assert sample_signal_event.strength == 0.85
        assert sample_signal_event.event_type == EventType.SIGNAL

    def test_signal_event_default_strength(self) -> None:
        """Test default signal strength."""
        signal = SignalEvent(
            timestamp=datetime.now(),
            ticker="AAPL",
            signal_type=SignalType.EXIT,
        )
        assert signal.strength == 1.0

    def test_signal_types(self) -> None:
        """Test all signal types."""
        for signal_type in SignalType:
            signal = SignalEvent(
                timestamp=datetime.now(),
                ticker="TEST",
                signal_type=signal_type,
            )
            assert signal.signal_type == signal_type


class TestOrderEvent:
    """Tests for OrderEvent."""

    def test_create_order_event(self, sample_order_event: OrderEvent) -> None:
        """Test basic OrderEvent creation."""
        assert sample_order_event.ticker == "AAPL"
        assert sample_order_event.side == OrderSide.BUY
        assert sample_order_event.quantity == 100
        assert sample_order_event.event_type == EventType.ORDER

    def test_market_order(self, sample_order_event: OrderEvent) -> None:
        """Test market order detection."""
        assert sample_order_event.is_market_order is True
        assert sample_order_event.is_limit_order is False

    def test_limit_order(self) -> None:
        """Test limit order detection."""
        order = OrderEvent(
            timestamp=datetime.now(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            order_id="order_002",
            limit_price=185.00,
        )
        assert order.is_market_order is False
        assert order.is_limit_order is True
        assert order.limit_price == 185.00


class TestFillEvent:
    """Tests for FillEvent."""

    def test_create_fill_event(self, sample_fill_event: FillEvent) -> None:
        """Test basic FillEvent creation."""
        assert sample_fill_event.ticker == "AAPL"
        assert sample_fill_event.fill_price == 186.55
        assert sample_fill_event.event_type == EventType.FILL

    def test_total_cost_buy(self, sample_fill_event: FillEvent) -> None:
        """Test total cost calculation for buy orders."""
        # 100 shares * $186.55 + $0 commission = $18,655
        assert sample_fill_event.total_cost == pytest.approx(18_655.0)

    def test_total_cost_with_commission(self) -> None:
        """Test total cost with commission."""
        fill = FillEvent(
            timestamp=datetime.now(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=100.0,
            commission=5.0,
            order_id="order_003",
        )
        assert fill.total_cost == pytest.approx(10_005.0)

    def test_total_cost_sell(self) -> None:
        """Test total cost calculation for sell orders."""
        fill = FillEvent(
            timestamp=datetime.now(),
            ticker="AAPL",
            side=OrderSide.SELL,
            quantity=100,
            fill_price=100.0,
            commission=5.0,
            order_id="order_004",
        )
        # 100 shares * $100 - $5 commission = $9,995
        assert fill.total_cost == pytest.approx(9_995.0)

    def test_cost_basis(self) -> None:
        """Test cost basis per share calculation."""
        fill = FillEvent(
            timestamp=datetime.now(),
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=100,
            fill_price=100.0,
            commission=10.0,
            order_id="order_005",
        )
        # ($10,000 + $10) / 100 = $100.10
        assert fill.cost_basis == pytest.approx(100.10)
