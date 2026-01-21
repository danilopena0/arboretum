"""Momentum indicators using Polars expressions.

These functions work with Polars DataFrames for vectorized computation
during data preparation. For bar-by-bar computation during backtesting,
use the incremental versions.
"""

import polars as pl


def rsi(column: str, period: int = 14) -> pl.Expr:
    """Relative Strength Index expression.

    Args:
        column: Column name to compute RSI over
        period: RSI period (default 14)

    Returns:
        Polars expression for RSI
    """
    delta = pl.col(column).diff()
    gain = delta.clip(lower_bound=0)
    loss = (-delta).clip(lower_bound=0)

    # Use Wilder's smoothing (exponential with alpha = 1/period)
    avg_gain = gain.ewm_mean(alpha=1.0 / period, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0 / period, adjust=False)

    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rate_of_change(column: str, period: int = 10) -> pl.Expr:
    """Rate of Change (ROC) expression.

    Measures percentage change over a period.

    Args:
        column: Column name
        period: Lookback period

    Returns:
        Polars expression for ROC
    """
    return (pl.col(column) / pl.col(column).shift(period) - 1.0) * 100.0


def momentum(column: str, period: int = 10) -> pl.Expr:
    """Price Momentum expression.

    Measures absolute price change over a period.

    Args:
        column: Column name
        period: Lookback period

    Returns:
        Polars expression for momentum
    """
    return pl.col(column) - pl.col(column).shift(period)


def add_rsi(
    df: pl.DataFrame, column: str = "close", period: int = 14, alias: str = "rsi"
) -> pl.DataFrame:
    """Add RSI column to DataFrame.

    Args:
        df: Input DataFrame
        column: Column to compute RSI over
        period: RSI period
        alias: Output column name

    Returns:
        DataFrame with RSI column added
    """
    return df.with_columns(rsi(column, period).alias(alias))


def add_roc(
    df: pl.DataFrame, column: str = "close", period: int = 10, alias: str | None = None
) -> pl.DataFrame:
    """Add Rate of Change column to DataFrame.

    Args:
        df: Input DataFrame
        column: Column to compute ROC over
        period: Lookback period
        alias: Output column name (default: roc_{period})

    Returns:
        DataFrame with ROC column added
    """
    output_name = alias or f"roc_{period}"
    return df.with_columns(rate_of_change(column, period).alias(output_name))


# =============================================================================
# Incremental versions for bar-by-bar computation
# =============================================================================


class IncrementalRSI:
    """Incremental RSI calculator using Wilder's smoothing method.

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Uses exponential smoothing with alpha = 1/period (Wilder's method).
    """

    def __init__(self, period: int = 14):
        """Initialize RSI calculator.

        Args:
            period: RSI period (default 14)
        """
        self.period = period
        self.alpha = 1.0 / period
        self._prev_close: float | None = None
        self._avg_gain: float | None = None
        self._avg_loss: float | None = None
        self._count = 0
        self._initial_gains: list[float] = []
        self._initial_losses: list[float] = []

    def update(self, close: float) -> float | None:
        """Update with new close price and return current RSI.

        Args:
            close: Closing price

        Returns:
            RSI value (0-100) if enough data, None otherwise
        """
        if self._prev_close is None:
            self._prev_close = close
            return None

        # Calculate change
        change = close - self._prev_close
        self._prev_close = close

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        self._count += 1

        # Initial period: accumulate gains/losses for first SMA
        if self._count <= self.period:
            self._initial_gains.append(gain)
            self._initial_losses.append(loss)

            if self._count == self.period:
                # Initialize with SMA of first period
                self._avg_gain = sum(self._initial_gains) / self.period
                self._avg_loss = sum(self._initial_losses) / self.period
            else:
                return None
        else:
            # Wilder's smoothing: new_avg = alpha * current + (1 - alpha) * prev_avg
            self._avg_gain = self.alpha * gain + (1.0 - self.alpha) * self._avg_gain  # type: ignore
            self._avg_loss = self.alpha * loss + (1.0 - self.alpha) * self._avg_loss  # type: ignore

        # Calculate RSI
        if self._avg_loss == 0:
            return 100.0 if self._avg_gain > 0 else 50.0

        rs = self._avg_gain / self._avg_loss  # type: ignore
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def value(self) -> float | None:
        """Current RSI value."""
        if self._avg_gain is None or self._avg_loss is None:
            return None

        if self._avg_loss == 0:
            return 100.0 if self._avg_gain > 0 else 50.0

        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data for a valid RSI."""
        return self._count >= self.period

    def reset(self) -> None:
        """Reset the calculator."""
        self._prev_close = None
        self._avg_gain = None
        self._avg_loss = None
        self._count = 0
        self._initial_gains.clear()
        self._initial_losses.clear()


class IncrementalROC:
    """Incremental Rate of Change calculator.

    ROC = (current_price / price_n_periods_ago - 1) * 100
    """

    def __init__(self, period: int = 10):
        """Initialize ROC calculator.

        Args:
            period: Lookback period
        """
        self.period = period
        self._prices: list[float] = []

    def update(self, price: float) -> float | None:
        """Update with new price and return current ROC.

        Args:
            price: Current price

        Returns:
            ROC value (percentage) if enough data, None otherwise
        """
        self._prices.append(price)

        if len(self._prices) > self.period + 1:
            self._prices.pop(0)

        if len(self._prices) <= self.period:
            return None

        old_price = self._prices[0]
        if old_price == 0:
            return None

        return (price / old_price - 1.0) * 100.0

    @property
    def value(self) -> float | None:
        """Current ROC value."""
        if len(self._prices) <= self.period:
            return None

        old_price = self._prices[0]
        if old_price == 0:
            return None

        return (self._prices[-1] / old_price - 1.0) * 100.0

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._prices) > self.period

    def reset(self) -> None:
        """Reset the calculator."""
        self._prices.clear()


class IncrementalMomentum:
    """Incremental Price Momentum calculator.

    Tracks returns over a lookback period for momentum ranking.
    """

    def __init__(self, period: int = 20):
        """Initialize Momentum calculator.

        Args:
            period: Lookback period for momentum calculation
        """
        self.period = period
        self._prices: list[float] = []

    def update(self, price: float) -> float | None:
        """Update with new price and return momentum (return over period).

        Args:
            price: Current price

        Returns:
            Return over lookback period as decimal (not percentage), None if not ready
        """
        self._prices.append(price)

        if len(self._prices) > self.period + 1:
            self._prices.pop(0)

        if len(self._prices) <= self.period:
            return None

        old_price = self._prices[0]
        if old_price == 0:
            return None

        return (price - old_price) / old_price

    @property
    def value(self) -> float | None:
        """Current momentum value (return over period)."""
        if len(self._prices) <= self.period:
            return None

        old_price = self._prices[0]
        if old_price == 0:
            return None

        return (self._prices[-1] - old_price) / old_price

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._prices) > self.period

    def reset(self) -> None:
        """Reset the calculator."""
        self._prices.clear()
