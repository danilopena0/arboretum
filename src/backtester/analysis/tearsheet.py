"""Tearsheet generation for backtest results.

Generates comprehensive performance reports from backtest results
in multiple formats: text, HTML, and dict/JSON.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from backtester.analysis.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    snapshots_to_dataframe,
)

if TYPE_CHECKING:
    from datetime import datetime

    from backtester.core.engine import BacktestResult


@dataclass
class MonthlyReturns:
    """Monthly returns breakdown."""

    year: int
    month: int
    return_pct: float


@dataclass
class YearlyReturns:
    """Yearly returns breakdown."""

    year: int
    return_pct: float


@dataclass
class Tearsheet:
    """Complete tearsheet with all performance data."""

    # Metadata
    start_date: datetime | None
    end_date: datetime | None
    initial_capital: float
    final_equity: float
    bars_processed: int

    # Performance metrics
    metrics: PerformanceMetrics

    # Benchmark comparison (if provided)
    benchmark_ticker: str | None
    benchmark_return: float | None

    # Periodic returns
    monthly_returns: list[MonthlyReturns]
    yearly_returns: list[YearlyReturns]

    def summary(self, include_trades: bool = True) -> str:
        """Generate console-printable text summary.

        Args:
            include_trades: Whether to include trade statistics

        Returns:
            Formatted string for console output
        """
        lines = []
        lines.append("=" * 60)
        lines.append("BACKTEST PERFORMANCE REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Period info
        lines.append("PERIOD")
        lines.append("-" * 40)
        if self.start_date:
            lines.append(f"  Start Date:      {self.start_date.strftime('%Y-%m-%d')}")
        if self.end_date:
            lines.append(f"  End Date:        {self.end_date.strftime('%Y-%m-%d')}")
        lines.append(f"  Bars Processed:  {self.bars_processed:,}")
        lines.append("")

        # Capital
        lines.append("CAPITAL")
        lines.append("-" * 40)
        lines.append(f"  Initial Capital: ${self.initial_capital:,.2f}")
        lines.append(f"  Final Equity:    ${self.final_equity:,.2f}")
        lines.append(f"  Net Profit:      ${self.final_equity - self.initial_capital:,.2f}")
        lines.append("")

        # Returns
        lines.append("RETURNS")
        lines.append("-" * 40)
        lines.append(f"  Total Return:    {self.metrics.total_return * 100:+.2f}%")
        lines.append(f"  CAGR:            {self.metrics.cagr * 100:+.2f}%")
        lines.append(f"  Annual Return:   {self.metrics.annualized_return * 100:+.2f}%")

        if self.benchmark_ticker and self.benchmark_return is not None:
            lines.append(
                f"  Benchmark ({self.benchmark_ticker}): {self.benchmark_return * 100:+.2f}%"
            )
            excess = self.metrics.total_return - self.benchmark_return
            lines.append(f"  Excess Return:   {excess * 100:+.2f}%")
        lines.append("")

        # Risk
        lines.append("RISK")
        lines.append("-" * 40)
        lines.append(f"  Volatility:      {self.metrics.volatility * 100:.2f}%")
        lines.append(f"  Max Drawdown:    {self.metrics.max_drawdown * 100:.2f}%")
        lines.append(f"  Max DD Duration: {self.metrics.max_drawdown_duration_days:.0f} days")
        lines.append(f"  Avg Drawdown:    {self.metrics.average_drawdown * 100:.2f}%")
        lines.append("")

        # Risk-adjusted
        lines.append("RISK-ADJUSTED")
        lines.append("-" * 40)
        lines.append(f"  Sharpe Ratio:    {self.metrics.sharpe_ratio:.2f}")
        lines.append(f"  Sortino Ratio:   {self.metrics.sortino_ratio:.2f}")
        lines.append(f"  Calmar Ratio:    {self.metrics.calmar_ratio:.2f}")

        if self.metrics.beta is not None:
            lines.append(f"  Beta:            {self.metrics.beta:.2f}")
        if self.metrics.alpha is not None:
            lines.append(f"  Alpha:           {self.metrics.alpha * 100:+.2f}%")
        lines.append("")

        # Trade statistics
        if include_trades:
            lines.append("TRADE STATISTICS")
            lines.append("-" * 40)
            lines.append(f"  Number of Trades:   {self.metrics.num_trades}")
            lines.append(f"  Win Rate:           {self.metrics.win_rate * 100:.1f}%")
            lines.append(f"  Profit Factor:      {self._format_ratio(self.metrics.profit_factor)}")
            lines.append(
                f"  Avg Win/Loss:       {self._format_ratio(self.metrics.avg_win_loss_ratio)}"
            )
            lines.append(f"  Avg Trade Duration: {self.metrics.avg_trade_duration_days:.1f} days")
            lines.append(f"  Expectancy:         ${self.metrics.expectancy:,.2f}")
            lines.append("")

        # Yearly returns
        if self.yearly_returns:
            lines.append("YEARLY RETURNS")
            lines.append("-" * 40)
            for yr in self.yearly_returns:
                lines.append(f"  {yr.year}:  {yr.return_pct * 100:+.2f}%")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def _format_ratio(self, value: float) -> str:
        """Format a ratio value, handling infinity."""
        if value == float("inf"):
            return "∞"
        elif value == float("-inf"):
            return "-∞"
        return f"{value:.2f}"

    def to_dict(self) -> dict:
        """Convert tearsheet to dictionary for JSON serialization.

        Returns:
            Dictionary with all tearsheet data
        """
        result = {
            "metadata": {
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "initial_capital": self.initial_capital,
                "final_equity": self.final_equity,
                "bars_processed": self.bars_processed,
            },
            "returns": {
                "total_return": self.metrics.total_return,
                "cagr": self.metrics.cagr,
                "annualized_return": self.metrics.annualized_return,
            },
            "risk": {
                "volatility": self.metrics.volatility,
                "downside_deviation": self.metrics.downside_deviation,
                "max_drawdown": self.metrics.max_drawdown,
                "max_drawdown_duration_days": self.metrics.max_drawdown_duration_days,
                "average_drawdown": self.metrics.average_drawdown,
            },
            "risk_adjusted": {
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "sortino_ratio": self.metrics.sortino_ratio,
                "calmar_ratio": self.metrics.calmar_ratio,
                "beta": self.metrics.beta,
                "alpha": self.metrics.alpha,
            },
            "trades": {
                "num_trades": self.metrics.num_trades,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "avg_win_loss_ratio": self.metrics.avg_win_loss_ratio,
                "avg_trade_duration_days": self.metrics.avg_trade_duration_days,
                "expectancy": self.metrics.expectancy,
            },
            "benchmark": {
                "ticker": self.benchmark_ticker,
                "return": self.benchmark_return,
            },
            "monthly_returns": [asdict(m) for m in self.monthly_returns],
            "yearly_returns": [asdict(y) for y in self.yearly_returns],
        }

        # Replace infinity values with strings for JSON
        return cast("dict[str, Any]", _replace_infinity(result))

    def to_json(self, indent: int = 2) -> str:
        """Convert tearsheet to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent)

    def to_html(self, filepath: str | None = None) -> str:
        """Generate HTML report.

        Args:
            filepath: Optional path to write HTML file

        Returns:
            HTML string
        """
        html = _generate_html_report(self)

        if filepath:
            with Path(filepath).open("w", encoding="utf-8") as f:
                f.write(html)

        return html


def _replace_infinity(obj: dict | list | float) -> dict | list | float | str:
    """Recursively replace infinity values with strings."""
    if isinstance(obj, dict):
        return {k: _replace_infinity(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_infinity(item) for item in obj]
    elif isinstance(obj, float):
        if obj == float("inf"):
            return "Infinity"
        elif obj == float("-inf"):
            return "-Infinity"
    return obj


def _calculate_monthly_returns(snapshots_df: pl.DataFrame) -> list[MonthlyReturns]:
    """Calculate monthly returns from snapshots.

    Args:
        snapshots_df: DataFrame with timestamp and total_equity columns

    Returns:
        List of monthly returns
    """
    if snapshots_df.is_empty():
        return []

    # Add year and month columns
    df = snapshots_df.with_columns(
        pl.col("timestamp").dt.year().alias("year"),
        pl.col("timestamp").dt.month().alias("month"),
    )

    # Get first and last equity for each month
    monthly = (
        df.group_by(["year", "month"])
        .agg(
            pl.col("total_equity").first().alias("start_equity"),
            pl.col("total_equity").last().alias("end_equity"),
        )
        .sort(["year", "month"])
    )

    results = []
    for row in monthly.iter_rows(named=True):
        start_eq = row["start_equity"]
        end_eq = row["end_equity"]
        ret = (end_eq - start_eq) / start_eq if start_eq > 0 else 0.0
        results.append(MonthlyReturns(year=row["year"], month=row["month"], return_pct=ret))

    return results


def _calculate_yearly_returns(snapshots_df: pl.DataFrame) -> list[YearlyReturns]:
    """Calculate yearly returns from snapshots.

    Args:
        snapshots_df: DataFrame with timestamp and total_equity columns

    Returns:
        List of yearly returns
    """
    if snapshots_df.is_empty():
        return []

    # Add year column
    df = snapshots_df.with_columns(pl.col("timestamp").dt.year().alias("year"))

    # Get first and last equity for each year
    yearly = (
        df.group_by("year")
        .agg(
            pl.col("total_equity").first().alias("start_equity"),
            pl.col("total_equity").last().alias("end_equity"),
        )
        .sort("year")
    )

    results = []
    for row in yearly.iter_rows(named=True):
        start_eq = row["start_equity"]
        end_eq = row["end_equity"]
        ret = (end_eq - start_eq) / start_eq if start_eq > 0 else 0.0
        results.append(YearlyReturns(year=row["year"], return_pct=ret))

    return results


def _generate_html_report(tearsheet: Tearsheet) -> str:
    """Generate HTML report from tearsheet.

    Args:
        tearsheet: Tearsheet object

    Returns:
        HTML string
    """
    m = tearsheet.metrics

    # Monthly returns heatmap data
    monthly_html = ""
    if tearsheet.monthly_returns:
        years = sorted({mr.year for mr in tearsheet.monthly_returns})
        monthly_html = "<h3>Monthly Returns</h3><table class='monthly-returns'>"
        monthly_html += "<tr><th>Year</th>"
        for month in range(1, 13):
            monthly_html += f"<th>{_month_abbr(month)}</th>"
        monthly_html += "<th>Total</th></tr>"

        for year in years:
            monthly_html += f"<tr><td class='year'>{year}</td>"
            year_total = 1.0
            for month in range(1, 13):
                mr = next(
                    (x for x in tearsheet.monthly_returns if x.year == year and x.month == month),
                    None,
                )
                if mr:
                    val = mr.return_pct * 100
                    year_total *= 1 + mr.return_pct
                    color_class = "positive" if val >= 0 else "negative"
                    monthly_html += f"<td class='{color_class}'>{val:+.1f}%</td>"
                else:
                    monthly_html += "<td>-</td>"
            year_return = (year_total - 1) * 100
            color_class = "positive" if year_return >= 0 else "negative"
            monthly_html += f"<td class='total {color_class}'>{year_return:+.1f}%</td></tr>"
        monthly_html += "</table>"

    benchmark_html = ""
    if tearsheet.benchmark_ticker and tearsheet.benchmark_return is not None:
        excess = tearsheet.metrics.total_return - tearsheet.benchmark_return
        benchmark_html = (
            f"""
        <h3>Benchmark Comparison</h3>
        <table class="metrics-table">
            <tr><td>Benchmark</td><td>{tearsheet.benchmark_ticker}</td></tr>
            <tr><td>Benchmark Return</td><td>{tearsheet.benchmark_return * 100:+.2f}%</td></tr>
            <tr><td>Excess Return</td><td>{excess * 100:+.2f}%</td></tr>
            <tr><td>Beta</td><td>{m.beta:.2f}</td></tr>
            <tr><td>Alpha</td><td>{m.alpha * 100:+.2f}%</td></tr>
        </table>
        """
            if m.beta is not None and m.alpha is not None
            else ""
        )

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Backtest Performance Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #2196F3;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .metrics-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metrics-card h3 {{
            margin-top: 0;
            color: #2196F3;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .metrics-table td {{
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }}
        .metrics-table td:first-child {{
            color: #666;
        }}
        .metrics-table td:last-child {{
            text-align: right;
            font-weight: 500;
        }}
        .summary-box {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
        }}
        .stat {{
            text-align: center;
            padding: 20px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }}
        .stat-value.positive {{ color: #4CAF50; }}
        .stat-value.negative {{ color: #f44336; }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .monthly-returns {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .monthly-returns th, .monthly-returns td {{
            padding: 8px;
            text-align: center;
            border: 1px solid #eee;
        }}
        .monthly-returns th {{
            background: #f5f5f5;
            font-weight: 500;
        }}
        .monthly-returns .year {{
            font-weight: bold;
            background: #f5f5f5;
        }}
        .monthly-returns .positive {{
            background: rgba(76, 175, 80, 0.2);
        }}
        .monthly-returns .negative {{
            background: rgba(244, 67, 54, 0.2);
        }}
        .monthly-returns .total {{
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            color: #999;
            font-size: 0.9em;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>Backtest Performance Report</h1>

    <div class="summary-box">
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-value {"positive" if m.total_return >= 0 else "negative"}">{m.total_return * 100:+.2f}%</div>
                <div class="stat-label">Total Return</div>
            </div>
            <div class="stat">
                <div class="stat-value">{m.sharpe_ratio:.2f}</div>
                <div class="stat-label">Sharpe Ratio</div>
            </div>
            <div class="stat">
                <div class="stat-value negative">{m.max_drawdown * 100:.2f}%</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
            <div class="stat">
                <div class="stat-value">{m.win_rate * 100:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metrics-card">
            <h3>Period</h3>
            <table class="metrics-table">
                <tr><td>Start Date</td><td>{tearsheet.start_date.strftime("%Y-%m-%d") if tearsheet.start_date else "-"}</td></tr>
                <tr><td>End Date</td><td>{tearsheet.end_date.strftime("%Y-%m-%d") if tearsheet.end_date else "-"}</td></tr>
                <tr><td>Bars Processed</td><td>{tearsheet.bars_processed:,}</td></tr>
            </table>
        </div>

        <div class="metrics-card">
            <h3>Capital</h3>
            <table class="metrics-table">
                <tr><td>Initial Capital</td><td>${tearsheet.initial_capital:,.2f}</td></tr>
                <tr><td>Final Equity</td><td>${tearsheet.final_equity:,.2f}</td></tr>
                <tr><td>Net Profit</td><td>${tearsheet.final_equity - tearsheet.initial_capital:,.2f}</td></tr>
            </table>
        </div>

        <div class="metrics-card">
            <h3>Returns</h3>
            <table class="metrics-table">
                <tr><td>Total Return</td><td>{m.total_return * 100:+.2f}%</td></tr>
                <tr><td>CAGR</td><td>{m.cagr * 100:+.2f}%</td></tr>
                <tr><td>Annualized Return</td><td>{m.annualized_return * 100:+.2f}%</td></tr>
            </table>
        </div>

        <div class="metrics-card">
            <h3>Risk</h3>
            <table class="metrics-table">
                <tr><td>Volatility</td><td>{m.volatility * 100:.2f}%</td></tr>
                <tr><td>Max Drawdown</td><td>{m.max_drawdown * 100:.2f}%</td></tr>
                <tr><td>Max DD Duration</td><td>{m.max_drawdown_duration_days:.0f} days</td></tr>
                <tr><td>Avg Drawdown</td><td>{m.average_drawdown * 100:.2f}%</td></tr>
            </table>
        </div>

        <div class="metrics-card">
            <h3>Risk-Adjusted</h3>
            <table class="metrics-table">
                <tr><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
                <tr><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
                <tr><td>Calmar Ratio</td><td>{m.calmar_ratio:.2f}</td></tr>
            </table>
        </div>

        <div class="metrics-card">
            <h3>Trade Statistics</h3>
            <table class="metrics-table">
                <tr><td>Number of Trades</td><td>{m.num_trades}</td></tr>
                <tr><td>Win Rate</td><td>{m.win_rate * 100:.1f}%</td></tr>
                <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
                <tr><td>Avg Win/Loss</td><td>{m.avg_win_loss_ratio:.2f}</td></tr>
                <tr><td>Avg Trade Duration</td><td>{m.avg_trade_duration_days:.1f} days</td></tr>
                <tr><td>Expectancy</td><td>${m.expectancy:,.2f}</td></tr>
            </table>
        </div>
    </div>

    {benchmark_html}

    {monthly_html}

    <div class="footer">
        Generated by Arboretum Backtesting Engine
    </div>
</body>
</html>"""

    return html


def _month_abbr(month: int) -> str:
    """Get month abbreviation."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months[month - 1]


def generate_tearsheet(
    result: BacktestResult,
    benchmark: str | None = None,
    benchmark_returns: pl.Series | None = None,
    risk_free_rate: float = 0.0,
) -> Tearsheet:
    """Generate a tearsheet from backtest results.

    Args:
        result: BacktestResult from engine.run()
        benchmark: Benchmark ticker symbol (for display only)
        benchmark_returns: Series of benchmark returns (aligned with snapshots)
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino calculations

    Returns:
        Tearsheet with complete performance analysis

    Example:
        result = engine.run(...)
        tearsheet = generate_tearsheet(result, benchmark="SPY")
        print(tearsheet.summary())
        tearsheet.to_html("report.html")
    """
    # Calculate metrics
    metrics = calculate_metrics(
        snapshots=result.snapshots,
        trades=result.trades,
        initial_capital=result.config.initial_capital,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    )

    # Convert snapshots to DataFrame for period returns
    snapshots_df = snapshots_to_dataframe(result.snapshots)

    # Calculate periodic returns
    monthly = _calculate_monthly_returns(snapshots_df)
    yearly = _calculate_yearly_returns(snapshots_df)

    # Calculate benchmark return if provided
    benchmark_total_return = None
    if benchmark_returns is not None and not benchmark_returns.is_empty():
        benchmark_total_return = float((1 + benchmark_returns).product() - 1)

    return Tearsheet(
        start_date=result.start_date,
        end_date=result.end_date,
        initial_capital=result.config.initial_capital,
        final_equity=result.final_equity,
        bars_processed=result.bars_processed,
        metrics=metrics,
        benchmark_ticker=benchmark,
        benchmark_return=benchmark_total_return,
        monthly_returns=monthly,
        yearly_returns=yearly,
    )
