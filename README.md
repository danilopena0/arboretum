# Arboretum

A high-performance stock backtesting engine built with Polars, DuckDB, and an event-driven architecture.

## Features

### Core Engine
- **Event-driven architecture**: MarketEvent → SignalEvent → OrderEvent → FillEvent pipeline
- **High-performance data structures**: Uses `msgspec.Struct` for immutable, fast event types
- **Polars-based data handling**: LazyFrames for deferred evaluation and memory efficiency

### Data Layer
- **yfinance integration**: Fetches historical OHLCV data with rate limiting
- **DuckDB caching**: Persistent cache with smart range detection (only fetches missing data)
- **Configurable intervals**: Daily, hourly, minute-level data support

### Portfolio Management
- **Position tracking**: Long/short positions with average cost basis
- **P&L attribution**: Realized and unrealized profit/loss tracking
- **Trade history**: Complete audit trail of all transactions
- **Snapshots**: Point-in-time portfolio state capture

### Execution Models
- **Slippage models**: Zero, Fixed, Percentage, Volume-based
- **Commission models**: Zero, PerShare, PerContract, Tiered, Percentage
- **Simulated broker**: Queue-based order execution with fill simulation
- **Broker protocol**: Abstract interface for future live trading support

### Strategy Framework
- **Abstract base class**: Clean interface for strategy implementation
- **Technical indicators**: SMA, EMA, MACD (both Polars expressions and incremental)
- **Example strategies**: Moving Average Crossover, Dual MA Crossover

## Installation

Requires Python 3.11+

```bash
# Clone the repository
git clone https://github.com/yourusername/arboretum.git
cd arboretum

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## Quick Start

```python
from datetime import date
from backtester.core.engine import BacktestConfig, BacktestEngine
from backtester.core.execution import FixedSlippage, PerShareCommission, SimulatedBroker
from backtester.data.yfinance_handler import YFinanceDataHandler
from backtester.strategies.examples.moving_average_crossover import MovingAverageCrossover

# Setup components
data_handler = YFinanceDataHandler(cache_path="data/cache.duckdb")
strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
broker = SimulatedBroker(
    slippage_model=FixedSlippage(cents_per_share=1.0),
    commission_model=PerShareCommission(rate=0.005, minimum=1.0),
)

# Create engine
engine = BacktestEngine(
    data_handler=data_handler,
    strategy=strategy,
    broker=broker,
    config=BacktestConfig(
        initial_capital=100_000.0,
        position_size=0.2,  # 20% of capital per position
    ),
)

# Run backtest
result = engine.run(
    tickers=["SPY"],
    start=date(2023, 1, 1),
    end=date(2023, 12, 31),
)

# View results
print(f"Final Equity: ${result.final_equity:,.2f}")
print(f"Total Return: {result.total_return_pct:+.2f}%")
print(f"Number of Trades: {result.num_trades}")
```

## Project Structure

```
arboretum/
├── src/
│   └── backtester/
│       ├── core/
│       │   ├── events.py       # Event types (Market, Signal, Order, Fill)
│       │   ├── engine.py       # Backtest engine
│       │   ├── portfolio.py    # Position and portfolio tracking
│       │   └── execution.py    # Broker, slippage, commission models
│       ├── data/
│       │   ├── handler.py      # Abstract data handler
│       │   ├── cache.py        # DuckDB cache layer
│       │   ├── schemas.py      # Polars schema definitions
│       │   └── yfinance_handler.py  # yfinance implementation
│       └── strategies/
│           ├── base.py         # Abstract strategy class
│           ├── indicators/     # Technical indicators
│           └── examples/       # Example strategies
├── tests/
│   ├── test_events.py
│   ├── test_engine.py
│   ├── test_portfolio.py
│   ├── test_execution.py
│   ├── test_data_handler.py
│   └── test_integration.py     # Real data tests
├── data/                       # DuckDB cache storage
├── notebooks/                  # Jupyter notebooks
└── pyproject.toml
```

## Running Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run integration tests (requires internet for yfinance)
pytest tests/test_integration.py -v -m integration

# Run with coverage
pytest tests/ --cov=backtester --cov-report=html
```

## Cache Management

Market data is cached in DuckDB files in the `data/` directory to avoid repeated API calls.

```bash
# Clear all cached data (Linux/Mac)
rm data/*.duckdb

# Clear all cached data (Windows PowerShell)
Remove-Item data/*.duckdb

# Check cache stats in Python
from backtester.data import YFinanceDataHandler
handler = YFinanceDataHandler()
print(handler.get_cache_stats())
```

## Creating a Custom Strategy

```python
from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy

class MyStrategy(Strategy):
    def __init__(self):
        super().__init__("my_strategy")
        self._bought: set[str] = set()

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        # Your strategy logic here
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
```

## Dependencies

- **polars** - High-performance DataFrames
- **duckdb** - Embedded analytics database for caching
- **yfinance** - Yahoo Finance data fetching
- **msgspec** - Fast serialization for events
- **numpy** - Numerical computing
- **numba** - JIT compilation for performance-critical code

## Roadmap

- [ ] Performance metrics (Sharpe, Sortino, Max Drawdown)
- [ ] Tearsheet generation with Plotly
- [ ] Multi-asset portfolio optimization
- [ ] Idea Capture system for research notes

## License

MIT
