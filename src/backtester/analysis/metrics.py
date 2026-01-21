"""Performance metrics for backtest analysis.

Provides functions to calculate risk-adjusted returns, drawdown metrics,
and trade statistics from backtest results.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from backtester.core.portfolio import PortfolioSnapshot, Trade

# =============================================================================
# Data Preparation
# =============================================================================


def snapshots_to_dataframe(snapshots: list[PortfolioSnapshot]) -> pl.DataFrame:
    """Convert portfolio snapshots to a Polars DataFrame.

    Args:
        snapshots: List of PortfolioSnapshot objects

    Returns:
        DataFrame with columns: timestamp, cash, positions_value,
        total_equity, realized_pnl, unrealized_pnl
    """
    if not snapshots:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us"),
                "cash": pl.Float64,
                "positions_value": pl.Float64,
                "total_equity": pl.Float64,
                "realized_pnl": pl.Float64,
                "unrealized_pnl": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "timestamp": [s.timestamp for s in snapshots],
            "cash": [float(s.cash) for s in snapshots],
            "positions_value": [float(s.positions_value) for s in snapshots],
            "total_equity": [float(s.total_equity) for s in snapshots],
            "realized_pnl": [float(s.realized_pnl) for s in snapshots],
            "unrealized_pnl": [float(s.unrealized_pnl) for s in snapshots],
        },
        schema={
            "timestamp": pl.Datetime("us"),
            "cash": pl.Float64,
            "positions_value": pl.Float64,
            "total_equity": pl.Float64,
            "realized_pnl": pl.Float64,
            "unrealized_pnl": pl.Float64,
        },
    )


def trades_to_dataframe(trades: list[Trade]) -> pl.DataFrame:
    """Convert trades to a Polars DataFrame.

    Args:
        trades: List of Trade objects

    Returns:
        DataFrame with trade data
    """
    if not trades:
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime("us"),
                "ticker": pl.Utf8,
                "quantity": pl.Int64,
                "price": pl.Float64,
                "commission": pl.Float64,
                "slippage": pl.Float64,
                "side": pl.Utf8,
                "total_value": pl.Float64,
            }
        )

    return pl.DataFrame(
        {
            "timestamp": [t.timestamp for t in trades],
            "ticker": [t.ticker for t in trades],
            "quantity": [t.quantity for t in trades],
            "price": [t.price for t in trades],
            "commission": [t.commission for t in trades],
            "slippage": [t.slippage for t in trades],
            "side": [t.side for t in trades],
            "total_value": [t.total_value for t in trades],
        }
    )


def calculate_returns(equity_series: pl.Series) -> pl.Series:
    """Calculate period-over-period returns from equity series.

    Args:
        equity_series: Series of equity values

    Returns:
        Series of returns (first value is 0)
    """
    return equity_series.pct_change().fill_null(0.0)


# =============================================================================
# Return Metrics
# =============================================================================


def total_return(
    initial_equity: float,
    final_equity: float,
) -> float:
    """Calculate total return as a decimal.

    Args:
        initial_equity: Starting portfolio value
        final_equity: Ending portfolio value

    Returns:
        Total return (e.g., 0.15 = 15%)
    """
    if initial_equity == 0:
        return 0.0
    return (final_equity - initial_equity) / initial_equity


def cagr(
    initial_equity: float,
    final_equity: float,
    days: int,
) -> float:
    """Calculate Compound Annual Growth Rate.

    Args:
        initial_equity: Starting portfolio value
        final_equity: Ending portfolio value
        days: Number of calendar days in the period

    Returns:
        CAGR as decimal (e.g., 0.12 = 12% annual)
    """
    if initial_equity <= 0 or days <= 0:
        return 0.0

    years = days / 365.0
    if years == 0:
        return 0.0

    if final_equity <= 0:
        return -1.0  # Total loss

    return (final_equity / initial_equity) ** (1 / years) - 1


def annualized_return(returns: pl.Series, periods_per_year: float = 252.0) -> float:
    """Calculate annualized return from a series of periodic returns.

    Args:
        returns: Series of period returns
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized return as decimal
    """
    if returns.is_empty():
        return 0.0

    # Compound returns
    total = (1 + returns).product() - 1
    n_periods = len(returns)

    if n_periods == 0:
        return 0.0

    # Annualize
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0

    return (1 + total) ** (1 / years) - 1


# =============================================================================
# Risk Metrics
# =============================================================================


def volatility(returns: pl.Series, periods_per_year: float = 252.0) -> float:
    """Calculate annualized volatility (standard deviation of returns).

    Args:
        returns: Series of period returns
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized volatility as decimal
    """
    if returns.is_empty() or len(returns) < 2:
        return 0.0

    std = returns.std()
    if std is None:
        return 0.0

    return float(std) * math.sqrt(periods_per_year)  # pyright: ignore[reportArgumentType]


def downside_deviation(
    returns: pl.Series,
    threshold: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate annualized downside deviation (for Sortino ratio).

    Only considers returns below the threshold.

    Args:
        returns: Series of period returns
        threshold: Minimum acceptable return (default 0)
        periods_per_year: Trading periods per year

    Returns:
        Annualized downside deviation
    """
    if returns.is_empty():
        return 0.0

    # Get returns below threshold
    downside = returns.filter(returns < threshold)

    if downside.is_empty():
        return 0.0

    # Calculate downside variance (mean of squared deviations from threshold)
    downside_sq = ((downside - threshold) ** 2).mean()
    if downside_sq is None or downside_sq == 0:
        return 0.0

    return math.sqrt(float(downside_sq)) * math.sqrt(periods_per_year)  # pyright: ignore[reportArgumentType]


# =============================================================================
# Risk-Adjusted Return Metrics
# =============================================================================


def sharpe_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate Sharpe Ratio.

    Sharpe = (annualized_return - risk_free_rate) / annualized_volatility

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year

    Returns:
        Sharpe ratio
    """
    if returns.is_empty() or len(returns) < 2:
        return 0.0

    ann_return = annualized_return(returns, periods_per_year)
    ann_vol = volatility(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate Sortino Ratio.

    Like Sharpe but uses downside deviation instead of total volatility.
    Better for strategies with asymmetric return distributions.

    Args:
        returns: Series of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Sortino ratio
    """
    if returns.is_empty():
        return 0.0

    ann_return = annualized_return(returns, periods_per_year)
    dd = downside_deviation(returns, threshold=0.0, periods_per_year=periods_per_year)

    if dd == 0:
        return 0.0 if ann_return <= risk_free_rate else float("inf")

    return (ann_return - risk_free_rate) / dd


def calmar_ratio(
    returns: pl.Series,
    equity_series: pl.Series,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate Calmar Ratio.

    Calmar = annualized_return / max_drawdown

    Args:
        returns: Series of period returns
        equity_series: Series of equity values
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if returns.is_empty() or equity_series.is_empty():
        return 0.0

    ann_return = annualized_return(returns, periods_per_year)
    max_dd = max_drawdown(equity_series)

    if max_dd == 0:
        return 0.0 if ann_return <= 0 else float("inf")

    return ann_return / max_dd


# =============================================================================
# Drawdown Metrics
# =============================================================================


def drawdown_series(equity_series: pl.Series) -> pl.Series:
    """Calculate drawdown at each point in time.

    Drawdown = (peak - current) / peak

    Args:
        equity_series: Series of equity values

    Returns:
        Series of drawdown values (0 = at peak, positive = below peak)
    """
    if equity_series.is_empty():
        return pl.Series("drawdown", [], dtype=pl.Float64)

    # Running maximum (peak equity)
    peak = equity_series.cum_max()

    # Drawdown as percentage from peak
    dd = (peak - equity_series) / peak

    return dd.fill_nan(0.0).alias("drawdown")


def max_drawdown(equity_series: pl.Series) -> float:
    """Calculate maximum drawdown.

    Args:
        equity_series: Series of equity values

    Returns:
        Maximum drawdown as positive decimal (e.g., 0.15 = 15% drawdown)
    """
    if equity_series.is_empty():
        return 0.0

    dd = drawdown_series(equity_series)
    max_val = dd.max()
    return float(max_val) if max_val is not None else 0.0  # pyright: ignore[reportArgumentType]


def max_drawdown_duration(
    equity_series: pl.Series,
    timestamps: pl.Series,
) -> timedelta:
    """Calculate the longest drawdown duration.

    Duration from peak to recovery (or end if not recovered).

    Args:
        equity_series: Series of equity values
        timestamps: Corresponding timestamps

    Returns:
        Longest drawdown duration
    """
    if equity_series.is_empty() or len(equity_series) < 2:
        return timedelta(0)

    peak = equity_series.cum_max()
    in_drawdown = equity_series < peak

    max_duration = timedelta(0)
    current_start: datetime | None = None

    for _i, (is_dd, ts) in enumerate(zip(in_drawdown, timestamps)):
        if is_dd and current_start is None:
            current_start = ts
        elif not is_dd and current_start is not None:
            duration = ts - current_start
            max_duration = max(max_duration, duration)
            current_start = None

    # Check if still in drawdown at end
    if current_start is not None:
        duration = timestamps[-1] - current_start
        max_duration = max(max_duration, duration)

    return max_duration


def average_drawdown(equity_series: pl.Series) -> float:
    """Calculate average drawdown.

    Args:
        equity_series: Series of equity values

    Returns:
        Average drawdown as decimal
    """
    if equity_series.is_empty():
        return 0.0

    dd = drawdown_series(equity_series)
    mean_val = dd.mean()
    return float(mean_val) if mean_val is not None else 0.0  # pyright: ignore[reportArgumentType]


# =============================================================================
# Trade Metrics
# =============================================================================


@dataclass
class RoundTripTrade:
    """A complete round-trip trade (entry + exit)."""

    ticker: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    commission: float
    side: str  # "LONG" or "SHORT"

    @property
    def return_pct(self) -> float:
        """Return as percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.side == "LONG":
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def duration(self) -> timedelta:
        """Trade duration."""
        return self.exit_time - self.entry_time

    @property
    def is_winner(self) -> bool:
        """Whether trade was profitable."""
        return self.pnl > 0


def extract_round_trips(trades: list[Trade]) -> list[RoundTripTrade]:
    """Extract round-trip trades from trade list.

    Matches entries with exits using FIFO method.

    Args:
        trades: List of all trades

    Returns:
        List of round-trip trades
    """
    if not trades:
        return []

    # Track open positions per ticker
    open_positions: dict[str, list[Trade]] = {}
    round_trips: list[RoundTripTrade] = []

    for trade in sorted(trades, key=lambda t: t.timestamp):
        ticker = trade.ticker

        if ticker not in open_positions:
            open_positions[ticker] = []

        position_trades = open_positions[ticker]

        # Check if this closes an existing position
        if position_trades:
            # Same side = adding to position
            if (position_trades[0].quantity > 0) == (trade.quantity > 0):
                position_trades.append(trade)
            else:
                # Opposite side = closing position
                remaining_qty = abs(trade.quantity)
                exit_price = trade.price
                exit_time = trade.timestamp

                while remaining_qty > 0 and position_trades:
                    entry_trade = position_trades[0]
                    entry_qty = abs(entry_trade.quantity)

                    closed_qty = min(remaining_qty, entry_qty)

                    # Calculate P&L for this portion
                    if entry_trade.quantity > 0:  # Was long
                        pnl = closed_qty * (exit_price - entry_trade.price)
                        side = "LONG"
                    else:  # Was short
                        pnl = closed_qty * (entry_trade.price - exit_price)
                        side = "SHORT"

                    round_trips.append(
                        RoundTripTrade(
                            ticker=ticker,
                            entry_time=entry_trade.timestamp,
                            exit_time=exit_time,
                            entry_price=entry_trade.price,
                            exit_price=exit_price,
                            quantity=closed_qty,
                            pnl=pnl - entry_trade.commission - trade.commission,
                            commission=entry_trade.commission + trade.commission,
                            side=side,
                        )
                    )

                    remaining_qty -= closed_qty

                    if closed_qty >= entry_qty:
                        position_trades.pop(0)
                    # Note: partial fills not fully handled here

        else:
            # Opening new position
            position_trades.append(trade)

    return round_trips


def win_rate(round_trips: list[RoundTripTrade]) -> float:
    """Calculate win rate (percentage of profitable trades).

    Args:
        round_trips: List of round-trip trades

    Returns:
        Win rate as decimal (e.g., 0.55 = 55%)
    """
    if not round_trips:
        return 0.0

    winners = sum(1 for t in round_trips if t.is_winner)
    return winners / len(round_trips)


def profit_factor(round_trips: list[RoundTripTrade]) -> float:
    """Calculate profit factor (gross profit / gross loss).

    Args:
        round_trips: List of round-trip trades

    Returns:
        Profit factor (> 1 means profitable)
    """
    if not round_trips:
        return 0.0

    gross_profit = sum(t.pnl for t in round_trips if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in round_trips if t.pnl < 0))

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def average_win_loss_ratio(round_trips: list[RoundTripTrade]) -> float:
    """Calculate average win / average loss ratio.

    Args:
        round_trips: List of round-trip trades

    Returns:
        Average win/loss ratio
    """
    if not round_trips:
        return 0.0

    winners = [t.pnl for t in round_trips if t.pnl > 0]
    losers = [t.pnl for t in round_trips if t.pnl < 0]

    if not winners or not losers:
        return 0.0

    avg_win = sum(winners) / len(winners)
    avg_loss = abs(sum(losers) / len(losers))

    if avg_loss == 0:
        return float("inf")

    return avg_win / avg_loss


def average_trade_duration(round_trips: list[RoundTripTrade]) -> timedelta:
    """Calculate average trade duration.

    Args:
        round_trips: List of round-trip trades

    Returns:
        Average duration
    """
    if not round_trips:
        return timedelta(0)

    total_seconds = sum(t.duration.total_seconds() for t in round_trips)
    return timedelta(seconds=total_seconds / len(round_trips))


def expectancy(round_trips: list[RoundTripTrade]) -> float:
    """Calculate trade expectancy (expected value per trade).

    Expectancy = (Win Rate * Avg Win) - (Loss Rate * Avg Loss)

    Args:
        round_trips: List of round-trip trades

    Returns:
        Expected P&L per trade
    """
    if not round_trips:
        return 0.0

    return sum(t.pnl for t in round_trips) / len(round_trips)


# =============================================================================
# Benchmark Comparison Metrics
# =============================================================================


def beta(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
) -> float:
    """Calculate beta (sensitivity to benchmark).

    Beta = Cov(portfolio, benchmark) / Var(benchmark)

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns

    Returns:
        Beta coefficient
    """
    if portfolio_returns.is_empty() or benchmark_returns.is_empty():
        return 0.0

    if len(portfolio_returns) != len(benchmark_returns):
        return 0.0

    # Create DataFrame for covariance calculation
    df = pl.DataFrame(
        {
            "portfolio": portfolio_returns,
            "benchmark": benchmark_returns,
        }
    )

    cov_matrix = df.select(
        pl.cov("portfolio", "benchmark").alias("cov"),
        pl.var("benchmark").alias("var"),
    )

    row = cov_matrix.row(0)
    cov, var = row[0], row[1]

    if var is None or var == 0:
        return 0.0

    return cov / var if cov is not None else 0.0


def alpha(
    portfolio_returns: pl.Series,
    benchmark_returns: pl.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> float:
    """Calculate Jensen's alpha (excess return vs CAPM).

    Alpha = Portfolio Return - (Risk Free + Beta * (Benchmark Return - Risk Free))

    Args:
        portfolio_returns: Series of portfolio returns
        benchmark_returns: Series of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized alpha
    """
    if portfolio_returns.is_empty() or benchmark_returns.is_empty():
        return 0.0

    port_ann = annualized_return(portfolio_returns, periods_per_year)
    bench_ann = annualized_return(benchmark_returns, periods_per_year)
    b = beta(portfolio_returns, benchmark_returns)

    expected_return = risk_free_rate + b * (bench_ann - risk_free_rate)
    return port_ann - expected_return


# =============================================================================
# Summary Statistics
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Complete performance metrics summary."""

    # Return metrics
    total_return: float
    cagr: float
    annualized_return: float

    # Risk metrics
    volatility: float
    downside_deviation: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Drawdown metrics
    max_drawdown: float
    max_drawdown_duration_days: float
    average_drawdown: float

    # Trade metrics
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    avg_trade_duration_days: float
    expectancy: float

    # Benchmark metrics (optional)
    beta: float | None = None
    alpha: float | None = None


def calculate_metrics(
    snapshots: list[PortfolioSnapshot],
    trades: list[Trade],
    initial_capital: float,
    benchmark_returns: pl.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: float = 252.0,
) -> PerformanceMetrics:
    """Calculate all performance metrics from backtest results.

    Args:
        snapshots: List of portfolio snapshots
        trades: List of trades
        initial_capital: Starting capital
        benchmark_returns: Optional benchmark return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        PerformanceMetrics with all calculated values
    """
    # Convert to DataFrames
    snap_df = snapshots_to_dataframe(snapshots)

    if snap_df.is_empty():
        return _empty_metrics()

    equity = snap_df["total_equity"]
    timestamps = snap_df["timestamp"]
    returns = calculate_returns(equity)

    # Calculate duration
    duration_days = (timestamps[-1] - timestamps[0]).days if len(timestamps) >= 2 else 0

    final_equity = equity[-1] if len(equity) > 0 else initial_capital

    # Extract round trips
    round_trips = extract_round_trips(trades)

    # Calculate all metrics
    result = PerformanceMetrics(
        # Returns
        total_return=total_return(initial_capital, final_equity),
        cagr=cagr(initial_capital, final_equity, duration_days),
        annualized_return=annualized_return(returns, periods_per_year),
        # Risk
        volatility=volatility(returns, periods_per_year),
        downside_deviation=downside_deviation(returns, periods_per_year=periods_per_year),
        # Risk-adjusted
        sharpe_ratio=sharpe_ratio(returns, risk_free_rate, periods_per_year),
        sortino_ratio=sortino_ratio(returns, risk_free_rate, periods_per_year),
        calmar_ratio=calmar_ratio(returns, equity, periods_per_year),
        # Drawdown
        max_drawdown=max_drawdown(equity),
        max_drawdown_duration_days=max_drawdown_duration(equity, timestamps).days,
        average_drawdown=average_drawdown(equity),
        # Trades
        num_trades=len(round_trips),
        win_rate=win_rate(round_trips),
        profit_factor=profit_factor(round_trips),
        avg_win_loss_ratio=average_win_loss_ratio(round_trips),
        avg_trade_duration_days=average_trade_duration(round_trips).days,
        expectancy=expectancy(round_trips),
    )

    # Benchmark comparison
    if benchmark_returns is not None and not benchmark_returns.is_empty():
        # Align lengths
        min_len = min(len(returns), len(benchmark_returns))
        port_ret = returns.slice(0, min_len)
        bench_ret = benchmark_returns.slice(0, min_len)

        result.beta = beta(port_ret, bench_ret)
        result.alpha = alpha(port_ret, bench_ret, risk_free_rate, periods_per_year)

    return result


def _empty_metrics() -> PerformanceMetrics:
    """Return empty metrics for edge cases."""
    return PerformanceMetrics(
        total_return=0.0,
        cagr=0.0,
        annualized_return=0.0,
        volatility=0.0,
        downside_deviation=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        calmar_ratio=0.0,
        max_drawdown=0.0,
        max_drawdown_duration_days=0.0,
        average_drawdown=0.0,
        num_trades=0,
        win_rate=0.0,
        profit_factor=0.0,
        avg_win_loss_ratio=0.0,
        avg_trade_duration_days=0.0,
        expectancy=0.0,
    )
