"""Moving average indicators using Polars expressions.

These functions work with Polars DataFrames for vectorized computation
during data preparation. For bar-by-bar computation during backtesting,
use the incremental versions.
"""

import polars as pl


def sma(column: str, window: int) -> pl.Expr:
    """Simple Moving Average expression.

    Args:
        column: Column name to compute SMA over
        window: Window size

    Returns:
        Polars expression for SMA
    """
    return pl.col(column).rolling_mean(window_size=window)


def ema(column: str, span: int) -> pl.Expr:
    """Exponential Moving Average expression.

    Args:
        column: Column name to compute EMA over
        span: EMA span (period)

    Returns:
        Polars expression for EMA
    """
    return pl.col(column).ewm_mean(span=span)


def wma(column: str, window: int) -> pl.Expr:
    """Weighted Moving Average expression.

    More recent values have higher weights.

    Args:
        column: Column name to compute WMA over
        window: Window size

    Returns:
        Polars expression for WMA
    """
    # Create weights: 1, 2, 3, ..., window
    weights = list(range(1, window + 1))
    weight_sum = sum(weights)

    # Normalize weights
    normalized_weights = [w / weight_sum for w in weights]

    return pl.col(column).rolling_map(
        lambda s: sum(v * w for v, w in zip(s, normalized_weights)),
        window_size=window,
    )


def add_sma(df: pl.DataFrame, column: str, window: int, alias: str | None = None) -> pl.DataFrame:
    """Add SMA column to DataFrame.

    Args:
        df: Input DataFrame
        column: Column to compute SMA over
        window: Window size
        alias: Output column name (default: {column}_sma_{window})

    Returns:
        DataFrame with SMA column added
    """
    output_name = alias or f"{column}_sma_{window}"
    return df.with_columns(sma(column, window).alias(output_name))


def add_ema(df: pl.DataFrame, column: str, span: int, alias: str | None = None) -> pl.DataFrame:
    """Add EMA column to DataFrame.

    Args:
        df: Input DataFrame
        column: Column to compute EMA over
        span: EMA span
        alias: Output column name (default: {column}_ema_{span})

    Returns:
        DataFrame with EMA column added
    """
    output_name = alias or f"{column}_ema_{span}"
    return df.with_columns(ema(column, span).alias(output_name))


# =============================================================================
# Incremental versions for bar-by-bar computation
# =============================================================================


class IncrementalSMA:
    """Incremental Simple Moving Average calculator.

    Efficiently computes SMA bar-by-bar using a rolling buffer.
    """

    def __init__(self, window: int):
        """Initialize SMA calculator.

        Args:
            window: Window size
        """
        self.window = window
        self._values: list[float] = []
        self._sum: float = 0.0

    def update(self, value: float) -> float | None:
        """Update with new value and return current SMA.

        Args:
            value: New value

        Returns:
            SMA value if window is full, None otherwise
        """
        self._values.append(value)
        self._sum += value

        if len(self._values) > self.window:
            self._sum -= self._values.pop(0)

        if len(self._values) < self.window:
            return None

        return self._sum / self.window

    @property
    def value(self) -> float | None:
        """Current SMA value."""
        if len(self._values) < self.window:
            return None
        return self._sum / self.window

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data for a valid SMA."""
        return len(self._values) >= self.window

    def reset(self) -> None:
        """Reset the calculator."""
        self._values.clear()
        self._sum = 0.0


class IncrementalEMA:
    """Incremental Exponential Moving Average calculator.

    Efficiently computes EMA bar-by-bar.
    """

    def __init__(self, span: int):
        """Initialize EMA calculator.

        Args:
            span: EMA span
        """
        self.span = span
        self.multiplier = 2.0 / (span + 1)
        self._value: float | None = None
        self._count = 0

    def update(self, value: float) -> float | None:
        """Update with new value and return current EMA.

        Args:
            value: New value

        Returns:
            EMA value
        """
        self._count += 1

        if self._value is None:
            self._value = value
        else:
            self._value = (value - self._value) * self.multiplier + self._value

        return self._value

    @property
    def value(self) -> float | None:
        """Current EMA value."""
        return self._value

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data for a reasonably accurate EMA."""
        # EMA needs roughly span periods to stabilize
        return self._count >= self.span

    def reset(self) -> None:
        """Reset the calculator."""
        self._value = None
        self._count = 0


class MACD:
    """MACD (Moving Average Convergence Divergence) indicator.

    Computed bar-by-bar using incremental EMAs.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        """Initialize MACD calculator.

        Args:
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        """
        self.fast_ema = IncrementalEMA(fast)
        self.slow_ema = IncrementalEMA(slow)
        self.signal_ema = IncrementalEMA(signal)
        self._macd_value: float | None = None
        self._signal_value: float | None = None

    def update(self, value: float) -> tuple[float | None, float | None, float | None]:
        """Update with new value.

        Args:
            value: New price value

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        fast = self.fast_ema.update(value)
        slow = self.slow_ema.update(value)

        if fast is not None and slow is not None:
            self._macd_value = fast - slow
            self._signal_value = self.signal_ema.update(self._macd_value)

            if self._signal_value is not None:
                histogram = self._macd_value - self._signal_value
                return (self._macd_value, self._signal_value, histogram)

        return (None, None, None)

    @property
    def macd(self) -> float | None:
        """Current MACD line value."""
        return self._macd_value

    @property
    def signal(self) -> float | None:
        """Current signal line value."""
        return self._signal_value

    @property
    def histogram(self) -> float | None:
        """Current histogram value."""
        if self._macd_value is None or self._signal_value is None:
            return None
        return self._macd_value - self._signal_value

    @property
    def is_ready(self) -> bool:
        """Whether MACD is ready (has valid signal line)."""
        return self._signal_value is not None

    def reset(self) -> None:
        """Reset the calculator."""
        self.fast_ema.reset()
        self.slow_ema.reset()
        self.signal_ema.reset()
        self._macd_value = None
        self._signal_value = None
