"""Execution models: brokers, slippage, and commissions.

Provides abstractions for order execution that can be swapped
between simulated (backtesting) and real (live trading) implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import uuid4

from backtester.core.events import FillEvent, MarketEvent, OrderEvent, OrderSide

# =============================================================================
# Slippage Models
# =============================================================================


class SlippageModel(Protocol):
    """Protocol for slippage calculation."""

    def calculate_slippage(self, order: OrderEvent, market: MarketEvent) -> float:
        """Calculate price slippage for an order.

        Args:
            order: The order being executed
            market: Current market data

        Returns:
            Slippage amount (added to price for buys, subtracted for sells)
        """
        ...


class ZeroSlippage:
    """No slippage - for testing purposes."""

    def calculate_slippage(self, _order: OrderEvent, _market: MarketEvent) -> float:
        return 0.0


class FixedSlippage:
    """Fixed slippage in cents per share.

    Args:
        cents_per_share: Slippage amount in cents (e.g., 1.0 = $0.01)
    """

    def __init__(self, cents_per_share: float = 1.0):
        self.cents_per_share = cents_per_share

    def calculate_slippage(self, _order: OrderEvent, _market: MarketEvent) -> float:
        return self.cents_per_share / 100.0


class PercentageSlippage:
    """Slippage as a percentage of price.

    Args:
        percentage: Slippage percentage (e.g., 0.001 = 0.1%)
    """

    def __init__(self, percentage: float = 0.001):
        self.percentage = percentage

    def calculate_slippage(self, _order: OrderEvent, market: MarketEvent) -> float:
        return market.close * self.percentage


class VolumeBasedSlippage:
    """Slippage that scales with order size relative to volume.

    More realistic model - large orders have more market impact.

    Args:
        impact_factor: Multiplier for volume impact (default 0.1)
        max_slippage_pct: Maximum slippage as percentage of price
    """

    def __init__(self, impact_factor: float = 0.1, max_slippage_pct: float = 0.02):
        self.impact_factor = impact_factor
        self.max_slippage_pct = max_slippage_pct

    def calculate_slippage(self, order: OrderEvent, market: MarketEvent) -> float:
        if market.volume == 0:
            return 0.0

        # Slippage increases with order size as fraction of daily volume
        volume_fraction = order.quantity / market.volume
        slippage_pct = self.impact_factor * volume_fraction

        # Cap at maximum
        slippage_pct = min(slippage_pct, self.max_slippage_pct)

        return market.close * slippage_pct


# =============================================================================
# Commission Models
# =============================================================================


class CommissionModel(Protocol):
    """Protocol for commission calculation."""

    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        """Calculate commission for an order.

        Args:
            order: The order being executed
            fill_price: Price at which order was filled

        Returns:
            Commission amount
        """
        ...


class ZeroCommission:
    """No commission - modern US brokers for stocks."""

    def calculate_commission(self, _order: OrderEvent, _fill_price: float) -> float:
        return 0.0


class PerShareCommission:
    """Commission per share with optional minimum.

    Args:
        rate: Commission per share (e.g., 0.005 = half a cent)
        minimum: Minimum commission per order
        maximum: Maximum commission per order (0 = no max)
    """

    def __init__(self, rate: float = 0.005, minimum: float = 1.0, maximum: float = 0.0):
        self.rate = rate
        self.minimum = minimum
        self.maximum = maximum

    def calculate_commission(self, order: OrderEvent, _fill_price: float) -> float:
        commission = order.quantity * self.rate
        commission = max(commission, self.minimum)
        if self.maximum > 0:
            commission = min(commission, self.maximum)
        return commission


class PerContractCommission:
    """Commission per contract - typically for options.

    Args:
        rate: Commission per contract (e.g., 0.65)
    """

    def __init__(self, rate: float = 0.65):
        self.rate = rate

    def calculate_commission(self, order: OrderEvent, _fill_price: float) -> float:
        return order.quantity * self.rate


class TieredCommission:
    """Volume-based tiered commission structure.

    Args:
        tiers: List of (volume_threshold, rate) tuples, sorted by threshold ascending
               Rate applies to shares above previous tier's threshold
    """

    def __init__(self, tiers: list[tuple[int, float]] | None = None):
        # Default: lower rates for higher volume
        self.tiers = tiers or [
            (0, 0.0035),      # First 300 shares: $0.0035/share
            (300, 0.002),     # 301-3000: $0.002/share
            (3000, 0.001),    # 3001-20000: $0.001/share
            (20000, 0.0005),  # 20001+: $0.0005/share
        ]

    def calculate_commission(self, order: OrderEvent, _fill_price: float) -> float:
        remaining = order.quantity
        commission = 0.0
        prev_threshold = 0

        for threshold, rate in self.tiers:
            if remaining <= 0:
                break

            tier_shares = min(remaining, threshold - prev_threshold) if threshold > 0 else remaining
            if tier_shares > 0:
                commission += tier_shares * rate
                remaining -= tier_shares

            prev_threshold = threshold

        # Handle remaining shares at last tier rate
        if remaining > 0:
            commission += remaining * self.tiers[-1][1]

        return commission


class PercentageCommission:
    """Commission as percentage of trade value.

    Args:
        percentage: Commission rate (e.g., 0.001 = 0.1%)
        minimum: Minimum commission per order
    """

    def __init__(self, percentage: float = 0.001, minimum: float = 0.0):
        self.percentage = percentage
        self.minimum = minimum

    def calculate_commission(self, order: OrderEvent, fill_price: float) -> float:
        commission = order.quantity * fill_price * self.percentage
        return max(commission, self.minimum)


# =============================================================================
# Broker Protocol and Implementations
# =============================================================================


@dataclass
class AccountInfo:
    """Account information summary."""

    cash: float
    positions_value: float
    total_equity: float
    buying_power: float


class Broker(ABC):
    """Abstract broker interface.

    Defines the contract for order execution that can be implemented
    for both simulated (backtesting) and real (live trading) scenarios.
    """

    @abstractmethod
    def submit_order(self, order: OrderEvent) -> str:
        """Submit an order for execution.

        Args:
            order: Order to submit

        Returns:
            Order ID for tracking
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if successfully cancelled
        """
        ...

    @abstractmethod
    def get_pending_orders(self) -> list[OrderEvent]:
        """Get list of pending (unfilled) orders.

        Returns:
            List of pending orders
        """
        ...

    @abstractmethod
    def process_market_data(self, market: MarketEvent) -> list[FillEvent]:
        """Process market data and execute any pending orders.

        Args:
            market: Current market data

        Returns:
            List of fills that occurred
        """
        ...


class SimulatedBroker(Broker):
    """Simulated broker for backtesting.

    Executes orders against historical data with configurable
    slippage and commission models.

    Attributes:
        slippage_model: Model for calculating slippage
        commission_model: Model for calculating commissions
    """

    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
    ):
        """Initialize simulated broker.

        Args:
            slippage_model: Slippage model (default: ZeroSlippage)
            commission_model: Commission model (default: ZeroCommission)
        """
        self.slippage_model = slippage_model or ZeroSlippage()
        self.commission_model = commission_model or ZeroCommission()
        self._pending_orders: dict[str, OrderEvent] = {}
        self._fill_count = 0

    def submit_order(self, order: OrderEvent) -> str:
        """Submit order to pending queue.

        Args:
            order: Order to submit

        Returns:
            Order ID
        """
        self._pending_orders[order.order_id] = order
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if found and cancelled
        """
        if order_id in self._pending_orders:
            del self._pending_orders[order_id]
            return True
        return False

    def get_pending_orders(self) -> list[OrderEvent]:
        """Get list of pending orders.

        Returns:
            List of pending orders
        """
        return list(self._pending_orders.values())

    def process_market_data(self, market: MarketEvent) -> list[FillEvent]:
        """Execute pending orders for this ticker at current market prices.

        Simulates order execution with slippage and commission.

        Args:
            market: Current market data

        Returns:
            List of fills
        """
        fills: list[FillEvent] = []
        orders_to_remove: list[str] = []

        for order_id, order in self._pending_orders.items():
            if order.ticker != market.ticker:
                continue

            # Check if order can be filled
            fill_price = self._calculate_fill_price(order, market)
            if fill_price is None:
                continue  # Order cannot be filled (e.g., limit not reached)

            # Calculate slippage and commission
            slippage = self.slippage_model.calculate_slippage(order, market)
            if order.side == OrderSide.BUY:
                fill_price += slippage
            else:
                fill_price -= slippage

            commission = self.commission_model.calculate_commission(order, fill_price)

            # Create fill event
            fill = FillEvent(
                timestamp=market.timestamp,
                ticker=order.ticker,
                side=order.side,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                order_id=order_id,
                slippage=slippage,
            )
            fills.append(fill)
            orders_to_remove.append(order_id)
            self._fill_count += 1

        # Remove filled orders
        for order_id in orders_to_remove:
            del self._pending_orders[order_id]

        return fills

    def _calculate_fill_price(
        self, order: OrderEvent, market: MarketEvent
    ) -> float | None:
        """Determine fill price for an order.

        Args:
            order: Order to fill
            market: Current market data

        Returns:
            Fill price, or None if order cannot be filled
        """
        if order.is_market_order:
            # Market orders fill at close (simulating execution at bar close)
            return market.close

        # Limit order
        if order.limit_price is None:
            return None

        if order.side == OrderSide.BUY:
            # Buy limit: fill if price drops to limit
            if market.low <= order.limit_price:
                return min(order.limit_price, market.close)
        else:
            # Sell limit: fill if price rises to limit
            if market.high >= order.limit_price:
                return max(order.limit_price, market.close)

        return None

    def reset(self) -> None:
        """Reset broker state."""
        self._pending_orders.clear()
        self._fill_count = 0

    @property
    def total_fills(self) -> int:
        """Total number of fills processed."""
        return self._fill_count


def create_order(
    timestamp: datetime,
    ticker: str,
    side: OrderSide,
    quantity: int,
    limit_price: float | None = None,
) -> OrderEvent:
    """Helper to create an order with auto-generated ID.

    Args:
        timestamp: Order timestamp
        ticker: Stock ticker symbol
        side: BUY or SELL
        quantity: Number of shares
        limit_price: Limit price (None for market order)

    Returns:
        OrderEvent with unique ID
    """
    return OrderEvent(
        timestamp=timestamp,
        ticker=ticker,
        side=side,
        quantity=quantity,
        order_id=str(uuid4()),
        limit_price=limit_price,
    )
