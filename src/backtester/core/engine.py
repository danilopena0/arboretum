"""Event-driven backtesting engine.

Orchestrates the backtest by processing market events through
strategies, portfolio, and execution in the correct order.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from backtester.core.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderSide,
    SignalEvent,
    SignalType,
)
from backtester.core.execution import SimulatedBroker, create_order
from backtester.core.portfolio import Portfolio, PortfolioSnapshot
from backtester.data.handler import DataHandler

if TYPE_CHECKING:
    from backtester.strategies.base import Strategy


@dataclass
class BacktestConfig:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio value
        position_size: Default position size as fraction of equity (e.g., 0.1 = 10%)
        max_position_size: Maximum position size as fraction of equity
        snapshot_frequency: How often to capture portfolio state for equity curve tracking.
            - "daily": Snapshot at end of each trading day. Best for most backtests.
              Creates one data point per day for equity curves and drawdown analysis.
            - "trade": Snapshot only after each trade fill. Useful for analyzing
              portfolio state at exact trade moments, but produces irregular time series.
            - "both": Snapshot on both daily boundaries AND after each trade.
              Most granular but creates larger result sets. Use when you need to
              analyze both daily performance and exact trade-time portfolio values.
        scale_by_signal_strength: If True, position size is multiplied by signal.strength
            (0.0-1.0). If False, always use full position_size regardless of signal strength.
            Disable this for strategies that don't use variable signal strength.
    """

    initial_capital: float = 100_000.0
    position_size: float = 0.1
    max_position_size: float = 0.25
    snapshot_frequency: str = "daily"
    scale_by_signal_strength: bool = True


@dataclass
class BacktestResult:
    """Results from a completed backtest.

    Attributes:
        config: Configuration used for the backtest
        portfolio: Final portfolio state
        snapshots: Historical portfolio snapshots
        trades: All executed trades
        signals: All signals generated
        start_date: Backtest start date
        end_date: Backtest end date
        bars_processed: Number of market bars processed
    """

    config: BacktestConfig
    portfolio: Portfolio
    snapshots: list[PortfolioSnapshot]
    trades: list
    signals: list[SignalEvent]
    start_date: datetime | None = None
    end_date: datetime | None = None
    bars_processed: int = 0

    @property
    def total_return(self) -> float:
        """Total return as decimal."""
        return self.portfolio.returns

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage."""
        return self.portfolio.returns * 100

    @property
    def final_equity(self) -> float:
        """Final portfolio equity."""
        return self.portfolio.total_equity

    @property
    def num_trades(self) -> int:
        """Total number of trades."""
        return len(self.trades)


class BacktestEngine:
    """Event-driven backtesting engine.

    Processes market data through strategies and executes trades
    via a simulated broker.

    Flow:
    1. Market data arrives (MarketEvent)
    2. Strategy processes market data, may emit SignalEvent
    3. SignalEvent converted to OrderEvent based on position sizing
    4. Broker executes order, generates FillEvent
    5. Portfolio updated with fill
    6. Repeat

    Attributes:
        data_handler: Source of market data
        strategy: Trading strategy
        broker: Order execution handler
        portfolio: Position and P&L tracking
        config: Backtest configuration
    """

    def __init__(
        self,
        data_handler: DataHandler,
        strategy: "Strategy",
        broker: SimulatedBroker | None = None,
        portfolio: Portfolio | None = None,
        config: BacktestConfig | None = None,
    ):
        """Initialize the backtesting engine.

        Args:
            data_handler: Source of market data
            strategy: Trading strategy to run
            broker: Simulated broker (default: new SimulatedBroker)
            portfolio: Portfolio tracker (default: new Portfolio)
            config: Backtest configuration (default: BacktestConfig)
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.config = config or BacktestConfig()
        self.broker = broker or SimulatedBroker()
        self.portfolio = portfolio or Portfolio(self.config.initial_capital)

        self._signals: list[SignalEvent] = []
        self._bars_processed = 0
        self._current_date: datetime | None = None
        self._start_date: datetime | None = None
        self._end_date: datetime | None = None

        # Callbacks
        self._on_bar: Callable[[MarketEvent], None] | None = None
        self._on_signal: Callable[[SignalEvent], None] | None = None
        self._on_fill: Callable[[FillEvent], None] | None = None

    def set_callbacks(
        self,
        on_bar: Callable[[MarketEvent], None] | None = None,
        on_signal: Callable[[SignalEvent], None] | None = None,
        on_fill: Callable[[FillEvent], None] | None = None,
    ) -> None:
        """Set optional callbacks for monitoring.

        Args:
            on_bar: Called for each market bar
            on_signal: Called for each signal generated
            on_fill: Called for each trade fill
        """
        self._on_bar = on_bar
        self._on_signal = on_signal
        self._on_fill = on_fill

    def run(
        self,
        tickers: list[str],
        start: date | datetime,
        end: date | datetime,
        interval: str = "1d",
    ) -> BacktestResult:
        """Run the backtest.

        Args:
            tickers: List of tickers to trade
            start: Backtest start date
            end: Backtest end date
            interval: Data interval

        Returns:
            BacktestResult with performance data
        """
        # Reset state
        self.portfolio.reset()
        self.broker.reset()
        self.strategy.reset()
        self._signals.clear()
        self._bars_processed = 0
        self._start_date = None
        self._end_date = None

        # Initialize strategy with tickers
        self.strategy.set_tickers(tickers)

        # Process market data
        for market_event in self.data_handler.iter_bars(tickers, start, end, interval):
            self._process_bar(market_event)

        # Final snapshot
        if self._end_date:
            self.portfolio.take_snapshot(self._end_date)

        return BacktestResult(
            config=self.config,
            portfolio=self.portfolio,
            snapshots=list(self.portfolio.snapshots),
            trades=list(self.portfolio.trades),
            signals=list(self._signals),
            start_date=self._start_date,
            end_date=self._end_date,
            bars_processed=self._bars_processed,
        )

    def _process_bar(self, market: MarketEvent) -> None:
        """Process a single market bar.

        Args:
            market: Market data event
        """
        # Track dates
        if self._start_date is None:
            self._start_date = market.timestamp
        self._end_date = market.timestamp
        self._bars_processed += 1

        # Update portfolio prices
        self.portfolio.update_price(market.ticker, market.close)

        # Callback
        if self._on_bar:
            self._on_bar(market)

        # Check for new day - take snapshot
        if self._should_snapshot(market.timestamp):
            self.portfolio.take_snapshot(market.timestamp)
        self._current_date = market.timestamp

        # Process any pending orders first
        fills = self.broker.process_market_data(market)
        for fill in fills:
            self._process_fill(fill)

        # Run strategy
        signal = self.strategy.on_market(market)
        if signal:
            self._process_signal(signal, market)

    def _should_snapshot(self, timestamp: datetime) -> bool:
        """Determine if we should take a portfolio snapshot.

        Args:
            timestamp: Current timestamp

        Returns:
            True if snapshot should be taken
        """
        if self.config.snapshot_frequency == "trade":
            return False

        if self._current_date is None:
            return False

        # Different day
        return timestamp.date() != self._current_date.date()

    def _process_signal(self, signal: SignalEvent, market: MarketEvent) -> None:
        """Convert a signal to an order and submit.

        Args:
            signal: Trading signal
            market: Current market data
        """
        self._signals.append(signal)

        if self._on_signal:
            self._on_signal(signal)

        # Convert signal to order
        order = self._signal_to_order(signal, market)
        if order:
            self.broker.submit_order(order)

    def _signal_to_order(self, signal: SignalEvent, market: MarketEvent) -> OrderEvent | None:
        """Convert a signal to an order based on position sizing rules.

        Args:
            signal: Trading signal
            market: Current market data

        Returns:
            OrderEvent or None if no order should be placed
        """
        current_qty = self.portfolio.get_quantity(signal.ticker)
        price = market.close

        if signal.signal_type == SignalType.EXIT:
            # Exit current position
            if current_qty == 0:
                return None

            side = OrderSide.SELL if current_qty > 0 else OrderSide.BUY
            return create_order(
                timestamp=signal.timestamp,
                ticker=signal.ticker,
                side=side,
                quantity=abs(current_qty),
            )

        elif signal.signal_type == SignalType.LONG:
            # Go long
            if current_qty > 0:
                return None  # Already long

            # Close short if any
            if current_qty < 0:
                return create_order(
                    timestamp=signal.timestamp,
                    ticker=signal.ticker,
                    side=OrderSide.BUY,
                    quantity=abs(current_qty),
                )

            # Calculate position size
            target_value = self.portfolio.total_equity * self.config.position_size
            if self.config.scale_by_signal_strength:
                target_value *= signal.strength
            max_value = self.portfolio.total_equity * self.config.max_position_size
            target_value = min(target_value, max_value)

            quantity = int(target_value / price)
            if quantity <= 0:
                return None

            # Check affordability
            if not self.portfolio.can_afford(signal.ticker, quantity, price):
                quantity = self.portfolio.max_shares_affordable(price)
                if quantity <= 0:
                    return None

            return create_order(
                timestamp=signal.timestamp,
                ticker=signal.ticker,
                side=OrderSide.BUY,
                quantity=quantity,
            )

        elif signal.signal_type == SignalType.SHORT:
            # Go short
            if current_qty < 0:
                return None  # Already short

            # Close long if any
            if current_qty > 0:
                return create_order(
                    timestamp=signal.timestamp,
                    ticker=signal.ticker,
                    side=OrderSide.SELL,
                    quantity=current_qty,
                )

            # Calculate position size for short
            target_value = self.portfolio.total_equity * self.config.position_size
            if self.config.scale_by_signal_strength:
                target_value *= signal.strength
            max_value = self.portfolio.total_equity * self.config.max_position_size
            target_value = min(target_value, max_value)

            quantity = int(target_value / price)
            if quantity <= 0:
                return None

            return create_order(
                timestamp=signal.timestamp,
                ticker=signal.ticker,
                side=OrderSide.SELL,
                quantity=quantity,
            )

        return None

    def _process_fill(self, fill: FillEvent) -> None:
        """Process a fill event and update portfolio.

        Args:
            fill: Fill event from broker
        """
        # Determine quantity sign based on side
        quantity = fill.quantity if fill.side == OrderSide.BUY else -fill.quantity

        self.portfolio.process_fill(
            timestamp=fill.timestamp,
            ticker=fill.ticker,
            quantity=quantity,
            fill_price=fill.fill_price,
            commission=fill.commission,
            slippage=fill.slippage,
            order_id=fill.order_id,
        )

        if self._on_fill:
            self._on_fill(fill)

        # Snapshot on trade if configured
        if self.config.snapshot_frequency in ("trade", "both"):
            self.portfolio.take_snapshot(fill.timestamp)
