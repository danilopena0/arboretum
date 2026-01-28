"""Portfolio tracking and position management.

Tracks positions, cash, P&L, and trade history throughout a backtest.
"""

from dataclasses import dataclass
from datetime import datetime

import msgspec


class Position(msgspec.Struct):
    """Represents a position in a single security.

    Attributes:
        ticker: Stock ticker symbol
        quantity: Number of shares (positive=long, negative=short)
        avg_cost: Average cost basis per share
        realized_pnl: Realized P&L from closed portions
    """

    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def market_value(self) -> float:
        """Market value at cost basis (use unrealized_pnl for current value)."""
        return abs(self.quantity) * self.avg_cost

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.quantity == 0:
            return 0.0
        return self.quantity * (current_price - self.avg_cost)

    def total_pnl(self, current_price: float) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl(current_price)


@dataclass
class Trade:
    """Record of a single trade execution."""

    timestamp: datetime
    ticker: str
    quantity: int  # positive=buy, negative=sell
    price: float
    commission: float
    slippage: float
    order_id: str

    @property
    def side(self) -> str:
        return "BUY" if self.quantity > 0 else "SELL"

    @property
    def total_value(self) -> float:
        """Total value of trade including commission."""
        return abs(self.quantity) * self.price + self.commission


@dataclass
class PortfolioSnapshot:
    """Point-in-time snapshot of portfolio state."""

    timestamp: datetime
    cash: float
    positions_value: float
    total_equity: float
    realized_pnl: float
    unrealized_pnl: float


class Portfolio:
    """Manages positions, cash, and tracks P&L.

    Thread-safe position tracking with full trade history.

    Attributes:
        initial_capital: Starting cash amount
        cash: Current cash balance
        positions: Dict of ticker -> Position
        trades: List of all executed trades
        snapshots: Historical equity snapshots
    """

    def __init__(self, initial_capital: float = 100_000.0):
        """Initialize portfolio with starting capital.

        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.snapshots: list[PortfolioSnapshot] = []
        self._current_prices: dict[str, float] = {}

    def update_price(self, ticker: str, price: float) -> None:
        """Update the current market price for a ticker.

        Args:
            ticker: Stock ticker symbol
            price: Current market price
        """
        self._current_prices[ticker] = price

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update multiple prices at once.

        Args:
            prices: Dict of ticker -> price
        """
        self._current_prices.update(prices)

    def get_position(self, ticker: str) -> Position:
        """Get position for a ticker (creates empty if not exists).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Position object
        """
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker=ticker)
        return self.positions[ticker]

    def get_quantity(self, ticker: str) -> int:
        """Get current quantity held for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Number of shares (0 if no position)
        """
        if ticker not in self.positions:
            return 0
        return self.positions[ticker].quantity

    def process_fill(
        self,
        timestamp: datetime,
        ticker: str,
        quantity: int,
        fill_price: float,
        commission: float,
        slippage: float,
        order_id: str,
    ) -> None:
        """Process a fill event and update positions.

        Args:
            timestamp: Time of fill
            ticker: Stock ticker symbol
            quantity: Shares filled (positive=buy, negative=sell)
            fill_price: Execution price
            commission: Commission paid
            slippage: Price slippage
            order_id: Order identifier
        """
        self._record_trade(timestamp, ticker, quantity, fill_price, commission, slippage, order_id)
        self._update_cash(quantity, fill_price, commission)
        self._update_position(ticker, quantity, fill_price)
        self._current_prices[ticker] = fill_price

    def _record_trade(
        self,
        timestamp: datetime,
        ticker: str,
        quantity: int,
        price: float,
        commission: float,
        slippage: float,
        order_id: str,
    ) -> None:
        """Record a trade in the trade history."""
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            quantity=quantity,
            price=price,
            commission=commission,
            slippage=slippage,
            order_id=order_id,
        )
        self.trades.append(trade)

    def _update_cash(self, quantity: int, fill_price: float, commission: float) -> None:
        """Update cash balance after a trade."""
        trade_value = quantity * fill_price
        self.cash -= trade_value + commission

    def _update_position(self, ticker: str, quantity: int, fill_price: float) -> None:
        """Update position after a fill."""
        position = self.get_position(ticker)
        old_quantity = position.quantity
        new_quantity = old_quantity + quantity

        if old_quantity == 0:
            self._open_position(ticker, new_quantity, fill_price)
        elif self._is_adding_to_position(old_quantity, quantity):
            self._add_to_position(ticker, position, quantity, new_quantity, fill_price)
        else:
            self._reduce_or_close_position(
                ticker, position, old_quantity, quantity, new_quantity, fill_price
            )

    def _is_adding_to_position(self, old_quantity: int, quantity: int) -> bool:
        """Check if trade adds to existing position (same direction)."""
        return (old_quantity > 0 and quantity > 0) or (old_quantity < 0 and quantity < 0)

    def _open_position(self, ticker: str, quantity: int, fill_price: float) -> None:
        """Open a new position."""
        self.positions[ticker] = Position(
            ticker=ticker,
            quantity=quantity,
            avg_cost=fill_price,
            realized_pnl=0.0,
        )

    def _add_to_position(
        self, ticker: str, position: Position, quantity: int, new_quantity: int, fill_price: float
    ) -> None:
        """Add to an existing position, updating average cost."""
        total_cost = abs(position.quantity) * position.avg_cost + abs(quantity) * fill_price
        new_avg_cost = total_cost / abs(new_quantity)
        self.positions[ticker] = Position(
            ticker=ticker,
            quantity=new_quantity,
            avg_cost=new_avg_cost,
            realized_pnl=position.realized_pnl,
        )

    def _reduce_or_close_position(
        self,
        ticker: str,
        position: Position,
        old_quantity: int,
        quantity: int,
        new_quantity: int,
        fill_price: float,
    ) -> None:
        """Reduce, close, or reverse a position, realizing P&L."""
        # Calculate realized P&L for closed portion
        closed_quantity = min(abs(old_quantity), abs(quantity))
        if old_quantity > 0:
            realized = closed_quantity * (fill_price - position.avg_cost)
        else:
            realized = closed_quantity * (position.avg_cost - fill_price)
        new_realized_pnl = position.realized_pnl + realized

        if new_quantity == 0:
            # Position fully closed
            self.positions[ticker] = Position(
                ticker=ticker, quantity=0, avg_cost=0.0, realized_pnl=new_realized_pnl
            )
        elif abs(new_quantity) < abs(old_quantity):
            # Position reduced but not closed
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=new_quantity,
                avg_cost=position.avg_cost,
                realized_pnl=new_realized_pnl,
            )
        else:
            # Position reversed (closed and opened opposite)
            self.positions[ticker] = Position(
                ticker=ticker,
                quantity=new_quantity,
                avg_cost=fill_price,
                realized_pnl=new_realized_pnl,
            )

    @property
    def positions_value(self) -> float:
        """Total market value of all positions at current prices."""
        total = 0.0
        for ticker, position in self.positions.items():
            if position.quantity != 0:
                price = self._current_prices.get(ticker, position.avg_cost)
                total += position.quantity * price
        return total

    @property
    def total_equity(self) -> float:
        """Total portfolio value (cash + positions)."""
        return self.cash + self.positions_value

    @property
    def realized_pnl(self) -> float:
        """Total realized P&L across all positions."""
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L at current prices."""
        total = 0.0
        for ticker, position in self.positions.items():
            if position.quantity != 0:
                price = self._current_prices.get(ticker, position.avg_cost)
                total += position.unrealized_pnl(price)
        return total

    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl

    @property
    def returns(self) -> float:
        """Total return as decimal (0.10 = 10%)."""
        if self.initial_capital == 0:
            return 0.0
        return (self.total_equity - self.initial_capital) / self.initial_capital

    def take_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Record current portfolio state.

        Args:
            timestamp: Current timestamp

        Returns:
            PortfolioSnapshot with current state
        """
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=self.positions_value,
            total_equity=self.total_equity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_holdings(self) -> dict[str, int]:
        """Get current holdings as ticker -> quantity dict.

        Returns:
            Dict of non-zero positions
        """
        return {ticker: pos.quantity for ticker, pos in self.positions.items() if pos.quantity != 0}

    def can_afford(
        self, _ticker: str, quantity: int, price: float, commission: float = 0.0
    ) -> bool:
        """Check if portfolio has enough cash for a purchase.

        Args:
            _ticker: Stock ticker symbol (unused, for API consistency)
            quantity: Shares to buy (must be positive)
            price: Expected price per share
            commission: Expected commission

        Returns:
            True if affordable
        """
        if quantity <= 0:
            return True  # Sells don't require cash
        required = quantity * price + commission
        return self.cash >= required

    def max_shares_affordable(self, price: float, commission: float = 0.0) -> int:
        """Calculate maximum shares affordable at given price.

        Args:
            price: Price per share
            commission: Commission to reserve

        Returns:
            Maximum whole shares affordable
        """
        available = self.cash - commission
        if available <= 0 or price <= 0:
            return 0
        return int(available / price)

    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.snapshots.clear()
        self._current_prices.clear()
