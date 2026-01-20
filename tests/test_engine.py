"""Tests for backtesting engine."""

from collections.abc import Iterator
from datetime import datetime

from backtester.core.engine import BacktestConfig, BacktestEngine
from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.core.execution import SimulatedBroker
from backtester.core.portfolio import Portfolio
from backtester.data.handler import DataHandler
from backtester.strategies.base import Strategy


class MockDataHandler(DataHandler):
    """Mock data handler for testing."""

    def __init__(self, bars: list[MarketEvent]):
        self._bars = bars
        self._index = 0

    def get_bars(self, tickers, start, end, interval="1d"):
        raise NotImplementedError

    def get_latest_bar(self, _ticker):
        return self._bars[-1] if self._bars else None

    def iter_bars(self, _tickers, _start, _end, _interval="1d") -> Iterator[MarketEvent]:
        yield from self._bars

    def update_bars(self) -> bool:
        if self._index >= len(self._bars) - 1:
            return False
        self._index += 1
        return True


class BuyAndHoldStrategy(Strategy):
    """Simple strategy that buys once and holds."""

    def __init__(self):
        super().__init__("buy_and_hold")
        self._bought: set[str] = set()

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        if event.ticker not in self._bought:
            self._bought.add(event.ticker)
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=event.ticker,
                signal_type=SignalType.LONG,
            )
        return None

    def reset(self) -> None:
        self._bought.clear()


class AlternatingStrategy(Strategy):
    """Strategy that alternates between long and exit."""

    def __init__(self):
        super().__init__("alternating")
        self._count: dict[str, int] = {}

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker
        if ticker not in self._count:
            self._count[ticker] = 0

        self._count[ticker] += 1

        # Buy on bar 5, exit on bar 10, buy on bar 15, etc.
        if self._count[ticker] == 5:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
            )
        elif self._count[ticker] == 10:
            return self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
            )

        return None

    def reset(self) -> None:
        self._count.clear()


def create_test_bars(ticker: str, num_bars: int, start_price: float = 100.0) -> list[MarketEvent]:
    """Create test market data."""
    bars = []
    price = start_price
    for i in range(num_bars):
        # Simple trending price
        price = price * 1.001  # 0.1% increase per bar
        bars.append(
            MarketEvent(
                timestamp=datetime(2024, 1, 1 + i),
                ticker=ticker,
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1_000_000,
            )
        )
    return bars


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_simple_backtest(self) -> None:
        """Test a simple buy and hold backtest."""
        bars = create_test_bars("AAPL", 20)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(
                initial_capital=100_000.0,
                position_size=0.1,
            ),
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 20),
        )

        assert result.bars_processed == 20
        assert result.num_trades >= 1  # At least the initial buy
        assert len(result.signals) >= 1

    def test_multiple_trades(self) -> None:
        """Test a strategy with multiple trades."""
        bars = create_test_bars("AAPL", 15)
        data_handler = MockDataHandler(bars)
        strategy = AlternatingStrategy()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(initial_capital=100_000.0),
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 15),
        )

        # Should have buy on bar 5, exit on bar 10
        assert len(result.signals) == 2

    def test_portfolio_tracking(self) -> None:
        """Test that portfolio is properly tracked."""
        bars = create_test_bars("AAPL", 10, start_price=100.0)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        portfolio = Portfolio(initial_capital=100_000.0)
        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            portfolio=portfolio,
            config=BacktestConfig(initial_capital=100_000.0),
        )

        engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 10),
        )

        # Should have bought shares
        assert portfolio.get_quantity("AAPL") > 0

    def test_callbacks_called(self) -> None:
        """Test that callbacks are invoked."""
        bars = create_test_bars("AAPL", 5)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        bar_count = 0
        signal_count = 0
        fill_count = 0

        def on_bar(_event):
            nonlocal bar_count
            bar_count += 1

        def on_signal(_event):
            nonlocal signal_count
            signal_count += 1

        def on_fill(_event):
            nonlocal fill_count
            fill_count += 1

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
        )
        engine.set_callbacks(on_bar=on_bar, on_signal=on_signal, on_fill=on_fill)

        engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        assert bar_count == 5
        assert signal_count >= 1
        assert fill_count >= 1

    def test_config_position_sizing(self) -> None:
        """Test position sizing configuration."""
        bars = create_test_bars("AAPL", 5, start_price=100.0)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        # 10% position size with 100k capital = 10k target
        # At $100/share = 100 shares
        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            config=BacktestConfig(
                initial_capital=100_000.0,
                position_size=0.1,
            ),
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        portfolio = result.portfolio
        # Should be approximately 100 shares (may vary slightly due to price changes)
        quantity = portfolio.get_quantity("AAPL")
        assert 90 <= quantity <= 110

    def test_slippage_and_commission(self) -> None:
        """Test that slippage and commission affect results."""
        bars = create_test_bars("AAPL", 5, start_price=100.0)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        from backtester.core.execution import FixedSlippage, PerShareCommission

        broker = SimulatedBroker(
            slippage_model=FixedSlippage(cents_per_share=10.0),  # $0.10/share
            commission_model=PerShareCommission(rate=0.01, minimum=1.0),
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            broker=broker,
            config=BacktestConfig(initial_capital=100_000.0),
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        # Check that trades have slippage and commission recorded
        if result.trades:
            trade = result.trades[0]
            assert trade.slippage > 0
            assert trade.commission > 0

    def test_empty_data(self) -> None:
        """Test handling of empty data."""
        data_handler = MockDataHandler([])
        strategy = BuyAndHoldStrategy()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        assert result.bars_processed == 0
        assert result.num_trades == 0

    def test_reset_between_runs(self) -> None:
        """Test that engine resets properly between runs."""
        bars = create_test_bars("AAPL", 5)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
        )

        # First run
        result1 = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        # Second run
        result2 = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 5),
        )

        # Results should be the same
        assert result1.bars_processed == result2.bars_processed
        assert result1.num_trades == result2.num_trades


class TestBacktestResult:
    """Tests for BacktestResult."""

    def test_result_properties(self) -> None:
        """Test result calculation properties."""
        bars = create_test_bars("AAPL", 20, start_price=100.0)
        data_handler = MockDataHandler(bars)
        strategy = BuyAndHoldStrategy()

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
        )

        result = engine.run(
            tickers=["AAPL"],
            start=datetime(2024, 1, 1),
            end=datetime(2024, 1, 20),
        )

        # Check properties don't raise
        _ = result.total_return
        _ = result.total_return_pct
        _ = result.final_equity
        _ = result.num_trades
