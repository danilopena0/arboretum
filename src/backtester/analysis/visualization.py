"""Interactive visualizations for backtest analysis using Plotly.

Provides functions to create various performance charts from backtest results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtester.analysis.metrics import (
    calculate_returns,
    drawdown_series,
    extract_round_trips,
    sharpe_ratio,
    snapshots_to_dataframe,
)

if TYPE_CHECKING:
    from datetime import datetime

    from backtester.core.engine import BacktestResult
    from backtester.core.portfolio import PortfolioSnapshot, Trade


def plot_equity_curve(
    result: BacktestResult | list[PortfolioSnapshot],
    benchmark_equity: list[float] | None = None,
    benchmark_name: str = "Benchmark",
    title: str = "Portfolio Equity Curve",
) -> go.Figure:
    """Plot equity curve over time.

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        benchmark_equity: Optional benchmark equity values (same length as snapshots)
        benchmark_name: Label for benchmark in legend
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty():
        return _empty_figure(title)

    fig = go.Figure()

    # Portfolio equity
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].to_list(),
            y=df["total_equity"].to_list(),
            mode="lines",
            name="Portfolio",
            line={"color": "#2196F3", "width": 2},
            hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
        )
    )

    # Benchmark if provided
    if benchmark_equity and len(benchmark_equity) == len(df):
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"].to_list(),
                y=benchmark_equity,
                mode="lines",
                name=benchmark_name,
                line={"color": "#FF9800", "width": 2, "dash": "dash"},
                hovertemplate="Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode="x unified",
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    return fig


def plot_cumulative_returns(
    result: BacktestResult | list[PortfolioSnapshot],
    benchmark_returns: list[float] | None = None,
    benchmark_name: str = "Benchmark",
    title: str = "Cumulative Returns",
) -> go.Figure:
    """Plot cumulative returns over time.

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        benchmark_returns: Optional benchmark period returns
        benchmark_name: Label for benchmark
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty():
        return _empty_figure(title)

    equity = df["total_equity"]
    initial = equity[0]
    cum_returns = [(e / initial - 1) * 100 for e in equity.to_list()]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].to_list(),
            y=cum_returns,
            mode="lines",
            name="Portfolio",
            line={"color": "#2196F3", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(33, 150, 243, 0.1)",
            hovertemplate="Date: %{x}<br>Return: %{y:+.2f}%<extra></extra>",
        )
    )

    if benchmark_returns and len(benchmark_returns) == len(df):
        # Calculate cumulative benchmark returns
        cum_bench = []
        total = 1.0
        for r in benchmark_returns:
            total *= 1 + r
            cum_bench.append((total - 1) * 100)

        fig.add_trace(
            go.Scatter(
                x=df["timestamp"].to_list(),
                y=cum_bench,
                mode="lines",
                name=benchmark_name,
                line={"color": "#FF9800", "width": 2, "dash": "dash"},
                hovertemplate="Date: %{x}<br>Return: %{y:+.2f}%<extra></extra>",
            )
        )

    # Zero line
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode="x unified",
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    return fig


def plot_returns_distribution(
    result: BacktestResult | list[PortfolioSnapshot],
    bins: int = 50,
    title: str = "Returns Distribution",
) -> go.Figure:
    """Plot histogram of period returns.

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        bins: Number of histogram bins
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty():
        return _empty_figure(title)

    returns = calculate_returns(df["total_equity"])
    returns_pct = [r * 100 for r in returns.to_list()[1:]]  # Skip first (always 0)

    if not returns_pct:
        return _empty_figure(title)

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=returns_pct,
            nbinsx=bins,
            name="Returns",
            marker_color="#2196F3",
            opacity=0.75,
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        )
    )

    # Add mean line
    mean_ret = sum(returns_pct) / len(returns_pct)
    fig.add_vline(
        x=mean_ret, line_dash="dash", line_color="#f44336", annotation_text=f"Mean: {mean_ret:.2f}%"
    )

    # Add zero line
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_drawdown(
    result: BacktestResult | list[PortfolioSnapshot],
    title: str = "Drawdown",
) -> go.Figure:
    """Plot drawdown (underwater) chart.

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty():
        return _empty_figure(title)

    dd = drawdown_series(df["total_equity"])
    dd_pct = [-d * 100 for d in dd.to_list()]  # Negative for underwater plot

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].to_list(),
            y=dd_pct,
            mode="lines",
            name="Drawdown",
            line={"color": "#f44336", "width": 1},
            fill="tozeroy",
            fillcolor="rgba(244, 67, 54, 0.3)",
            hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
        )
    )

    # Find max drawdown point
    min_dd = min(dd_pct)
    min_idx = dd_pct.index(min_dd)
    fig.add_annotation(
        x=df["timestamp"][min_idx],
        y=min_dd,
        text=f"Max: {min_dd:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#f44336",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        showlegend=False,
    )

    # Ensure y-axis goes from negative to 0
    fig.update_yaxes(range=[min(dd_pct) * 1.1, 5])

    return fig


def plot_trades_on_price(
    result: BacktestResult,
    ticker: str,
    price_data: list[tuple[datetime, float]],
    title: str | None = None,
) -> go.Figure:
    """Plot trade entry/exit points on price chart.

    Args:
        result: BacktestResult with trades
        ticker: Ticker symbol to filter trades
        price_data: List of (timestamp, price) tuples
        title: Chart title (default: "{ticker} Trades")

    Returns:
        Plotly Figure object
    """
    if title is None:
        title = f"{ticker} Trades"

    if not price_data:
        return _empty_figure(title)

    timestamps, prices = zip(*price_data)

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(prices),
            mode="lines",
            name="Price",
            line={"color": "#333", "width": 1},
        )
    )

    # Filter trades for this ticker
    ticker_trades = [t for t in result.trades if t.ticker == ticker]

    # Separate buys and sells
    buys = [(t.timestamp, t.price) for t in ticker_trades if t.quantity > 0]
    sells = [(t.timestamp, t.price) for t in ticker_trades if t.quantity < 0]

    if buys:
        buy_times, buy_prices = zip(*buys)
        fig.add_trace(
            go.Scatter(
                x=list(buy_times),
                y=list(buy_prices),
                mode="markers",
                name="Buy",
                marker={"color": "#4CAF50", "size": 10, "symbol": "triangle-up"},
                hovertemplate="Buy<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            )
        )

    if sells:
        sell_times, sell_prices = zip(*sells)
        fig.add_trace(
            go.Scatter(
                x=list(sell_times),
                y=list(sell_prices),
                mode="markers",
                name="Sell",
                marker={"color": "#f44336", "size": 10, "symbol": "triangle-down"},
                hovertemplate="Sell<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
    )

    return fig


def plot_trade_pnl(
    result: BacktestResult | list[Trade],
    title: str = "P&L per Trade",
) -> go.Figure:
    """Plot P&L bar chart for each round-trip trade.

    Args:
        result: BacktestResult or list of trades
        title: Chart title

    Returns:
        Plotly Figure object
    """
    trades = result if isinstance(result, list) else result.trades

    round_trips = extract_round_trips(trades)

    if not round_trips:
        return _empty_figure(title)

    pnls = [rt.pnl for rt in round_trips]
    labels = [f"{rt.ticker} ({rt.exit_time.strftime('%m/%d')})" for rt in round_trips]
    colors = ["#4CAF50" if pnl >= 0 else "#f44336" for pnl in pnls]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=list(range(len(pnls))),
            y=pnls,
            text=[f"${p:,.0f}" for p in pnls],
            textposition="outside",
            marker_color=colors,
            hovertemplate="%{customdata}<br>P&L: $%{y:,.2f}<extra></extra>",
            customdata=labels,
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray")

    fig.update_layout(
        title=title,
        xaxis_title="Trade #",
        yaxis_title="P&L ($)",
        template="plotly_white",
        showlegend=False,
        xaxis={"tickmode": "array", "tickvals": list(range(len(pnls))), "ticktext": labels},
    )

    return fig


def plot_monthly_returns(
    result: BacktestResult | list[PortfolioSnapshot],
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    """Plot monthly returns as a heatmap (year x month grid).

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty():
        return _empty_figure(title)

    # Group by year and month
    import polars as pl

    df = df.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    )

    monthly = (
        df.group_by(["year", "month"])
        .agg(
            pl.col("total_equity").first().alias("start_equity"),
            pl.col("total_equity").last().alias("end_equity"),
        )
        .sort(["year", "month"])
    )

    # Calculate returns
    years = sorted(monthly["year"].unique().to_list())
    months = list(range(1, 13))
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Create return matrix
    z = []
    text = []

    for year in years:
        year_returns = []
        year_text = []
        for month in months:
            row = monthly.filter((pl.col("year") == year) & (pl.col("month") == month))
            if len(row) > 0:
                start = row["start_equity"][0]
                end = row["end_equity"][0]
                ret = (end - start) / start * 100 if start > 0 else 0.0
                year_returns.append(ret)
                year_text.append(f"{ret:+.1f}%")
            else:
                year_returns.append(None)
                year_text.append("")
        z.append(year_returns)
        text.append(year_text)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=month_names,
            y=[str(y) for y in years],
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0, "#f44336"],
                [0.5, "#ffffff"],
                [1, "#4CAF50"],
            ],
            zmid=0,
            colorbar={"title": "Return %"},
            hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Year",
        template="plotly_white",
    )

    return fig


def plot_rolling_sharpe(
    result: BacktestResult | list[PortfolioSnapshot],
    window: int = 63,  # ~3 months
    title: str = "Rolling Sharpe Ratio",
) -> go.Figure:
    """Plot rolling Sharpe ratio over time.

    Args:
        result: BacktestResult or list of PortfolioSnapshot
        window: Rolling window size in periods
        title: Chart title

    Returns:
        Plotly Figure object
    """
    snapshots = result if isinstance(result, list) else result.snapshots

    df = snapshots_to_dataframe(snapshots)
    if df.is_empty() or len(df) < window:
        return _empty_figure(title)

    import polars as pl

    returns = calculate_returns(df["total_equity"])
    timestamps = df["timestamp"].to_list()

    # Calculate rolling Sharpe
    rolling_sharpes = []
    valid_timestamps = []

    for i in range(window, len(returns)):
        window_returns = pl.Series(returns[i - window : i].to_list())
        sr = sharpe_ratio(window_returns, risk_free_rate=0.0, periods_per_year=252)
        rolling_sharpes.append(sr)
        valid_timestamps.append(timestamps[i])

    if not rolling_sharpes:
        return _empty_figure(title)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=valid_timestamps,
            y=rolling_sharpes,
            mode="lines",
            name="Rolling Sharpe",
            line={"color": "#2196F3", "width": 2},
            hovertemplate="Date: %{x}<br>Sharpe: %{y:.2f}<extra></extra>",
        )
    )

    # Reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(
        y=1, line_dash="dash", line_color="#4CAF50", opacity=0.5, annotation_text="Good (1.0)"
    )
    fig.add_hline(
        y=2, line_dash="dash", line_color="#2196F3", opacity=0.5, annotation_text="Excellent (2.0)"
    )

    fig.update_layout(
        title=f"{title} ({window}-day window)",
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        showlegend=False,
    )

    return fig


def plot_portfolio_composition(
    result: BacktestResult,
    title: str = "Portfolio Composition Over Time",
) -> go.Figure:
    """Plot stacked area chart of portfolio composition (cash vs positions).

    Args:
        result: BacktestResult
        title: Chart title

    Returns:
        Plotly Figure object
    """
    df = snapshots_to_dataframe(result.snapshots)
    if df.is_empty():
        return _empty_figure(title)

    timestamps = df["timestamp"].to_list()
    cash = df["cash"].to_list()
    positions = df["positions_value"].to_list()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=cash,
            mode="lines",
            name="Cash",
            stackgroup="one",
            fillcolor="rgba(33, 150, 243, 0.5)",
            line={"color": "#2196F3"},
            hovertemplate="Date: %{x}<br>Cash: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=positions,
            mode="lines",
            name="Positions",
            stackgroup="one",
            fillcolor="rgba(76, 175, 80, 0.5)",
            line={"color": "#4CAF50"},
            hovertemplate="Date: %{x}<br>Positions: $%{y:,.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value ($)",
        template="plotly_white",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "left", "x": 0.01},
        hovermode="x unified",
    )

    return fig


def create_performance_dashboard(
    result: BacktestResult,
    title: str = "Backtest Performance Dashboard",
) -> go.Figure:
    """Create a comprehensive multi-chart dashboard.

    Args:
        result: BacktestResult
        title: Dashboard title

    Returns:
        Plotly Figure with multiple subplots
    """
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Equity Curve",
            "Drawdown",
            "Returns Distribution",
            "P&L per Trade",
            "Rolling Sharpe (63-day)",
            "Portfolio Composition",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    df = snapshots_to_dataframe(result.snapshots)

    if not df.is_empty():
        timestamps = df["timestamp"].to_list()
        equity = df["total_equity"].to_list()

        # 1. Equity curve
        fig.add_trace(
            go.Scatter(
                x=timestamps, y=equity, mode="lines", name="Equity", line={"color": "#2196F3"}
            ),
            row=1,
            col=1,
        )

        # 2. Drawdown
        dd = drawdown_series(df["total_equity"])
        dd_pct = [-d * 100 for d in dd.to_list()]
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=dd_pct,
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line={"color": "#f44336"},
                fillcolor="rgba(244, 67, 54, 0.3)",
            ),
            row=1,
            col=2,
        )

        # 3. Returns distribution
        returns = calculate_returns(df["total_equity"])
        returns_pct = [r * 100 for r in returns.to_list()[1:]]
        if returns_pct:
            fig.add_trace(
                go.Histogram(x=returns_pct, nbinsx=30, name="Returns", marker_color="#2196F3"),
                row=2,
                col=1,
            )

        # 4. Trade P&L
        round_trips = extract_round_trips(result.trades)
        if round_trips:
            pnls = [rt.pnl for rt in round_trips]
            colors = ["#4CAF50" if pnl >= 0 else "#f44336" for pnl in pnls]
            fig.add_trace(
                go.Bar(x=list(range(len(pnls))), y=pnls, name="Trade P&L", marker_color=colors),
                row=2,
                col=2,
            )

        # 5. Rolling Sharpe
        window = 63
        if len(returns) > window:
            import polars as pl

            rolling_sharpes = []
            valid_timestamps = []
            for i in range(window, len(returns)):
                window_returns = pl.Series(returns[i - window : i].to_list())
                sr = sharpe_ratio(window_returns, risk_free_rate=0.0, periods_per_year=252)
                rolling_sharpes.append(sr)
                valid_timestamps.append(timestamps[i])

            fig.add_trace(
                go.Scatter(
                    x=valid_timestamps,
                    y=rolling_sharpes,
                    mode="lines",
                    name="Rolling Sharpe",
                    line={"color": "#2196F3"},
                ),
                row=3,
                col=1,
            )
            fig.add_hline(y=0, line_dash="dot", line_color="gray", row=3, col=1)  # pyright: ignore[reportArgumentType]

        # 6. Portfolio composition
        cash = df["cash"].to_list()
        positions = df["positions_value"].to_list()
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=cash,
                mode="lines",
                name="Cash",
                stackgroup="one",
                fillcolor="rgba(33, 150, 243, 0.5)",
            ),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=positions,
                mode="lines",
                name="Positions",
                stackgroup="one",
                fillcolor="rgba(76, 175, 80, 0.5)",
            ),
            row=3,
            col=2,
        )

    fig.update_layout(
        title=title,
        height=900,
        showlegend=False,
        template="plotly_white",
    )

    return fig


def _empty_figure(title: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 20, "color": "gray"},
    )
    fig.update_layout(title=title, template="plotly_white")
    return fig
