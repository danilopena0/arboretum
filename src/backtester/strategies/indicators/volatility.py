"""Volatility indicators using Polars expressions.

These functions work with Polars DataFrames for vectorized computation
during data preparation. For bar-by-bar computation during backtesting,
use the incremental versions.
"""

import math

import polars as pl


def std_dev(column: str, window: int) -> pl.Expr:
    """Rolling standard deviation expression.

    Args:
        column: Column name
        window: Window size

    Returns:
        Polars expression for rolling std dev
    """
    return pl.col(column).rolling_std(window_size=window)


def std_dev_zero_mean(column: str, window: int) -> pl.Expr:
    """Rolling standard deviation assuming zero mean.

    Computes sqrt(sum(x^2) / n) instead of sqrt(sum((x - mean)^2) / n).
    Useful for return series where expected return is assumed to be zero,
    or when you want to measure raw volatility without centering.

    Args:
        column: Column name
        window: Window size

    Returns:
        Polars expression for zero-mean rolling std dev
    """
    return (pl.col(column).pow(2).rolling_mean(window_size=window)).sqrt()


def atr(window: int = 14) -> pl.Expr:
    """Average True Range expression.

    Requires columns: high, low, close

    Args:
        window: ATR period (default 14)

    Returns:
        Polars expression for ATR
    """
    # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    prev_close = pl.col("close").shift(1)
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return tr.rolling_mean(window_size=window)


def bollinger_bands(column: str, window: int = 20, num_std: float = 2.0) -> dict[str, pl.Expr]:
    """Bollinger Bands expressions.

    Args:
        column: Column name (typically 'close')
        window: SMA window (default 20)
        num_std: Number of standard deviations (default 2.0)

    Returns:
        Dict with 'middle', 'upper', 'lower' expressions
    """
    middle = pl.col(column).rolling_mean(window_size=window)
    std = pl.col(column).rolling_std(window_size=window)

    return {
        "middle": middle,
        "upper": middle + (std * num_std),
        "lower": middle - (std * num_std),
    }


def add_atr(df: pl.DataFrame, window: int = 14, alias: str = "atr") -> pl.DataFrame:
    """Add ATR column to DataFrame.

    Args:
        df: Input DataFrame with high, low, close columns
        window: ATR period
        alias: Output column name

    Returns:
        DataFrame with ATR column added
    """
    return df.with_columns(atr(window).alias(alias))


def add_bollinger_bands(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 20,
    num_std: float = 2.0,
    prefix: str = "bb",
) -> pl.DataFrame:
    """Add Bollinger Bands columns to DataFrame.

    Args:
        df: Input DataFrame
        column: Column to compute bands over
        window: SMA window
        num_std: Number of standard deviations
        prefix: Prefix for output columns

    Returns:
        DataFrame with bb_middle, bb_upper, bb_lower columns
    """
    bands = bollinger_bands(column, window, num_std)
    return df.with_columns(
        bands["middle"].alias(f"{prefix}_middle"),
        bands["upper"].alias(f"{prefix}_upper"),
        bands["lower"].alias(f"{prefix}_lower"),
    )


# =============================================================================
# Incremental versions for bar-by-bar computation
# =============================================================================


class IncrementalStdDev:
    """Incremental standard deviation calculator using Welford's algorithm."""

    def __init__(self, window: int):
        """Initialize std dev calculator.

        Args:
            window: Window size
        """
        self.window = window
        self._values: list[float] = []
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> float | None:
        """Update with new value and return current std dev.

        Args:
            value: New value

        Returns:
            Standard deviation if window is full, None otherwise
        """
        self._values.append(value)

        if len(self._values) > self.window:
            self._values.pop(0)

        if len(self._values) < self.window:
            return None

        # Recalculate (simpler than incremental for windowed)
        mean = sum(self._values) / self.window
        variance = sum((x - mean) ** 2 for x in self._values) / self.window
        return math.sqrt(variance)

    @property
    def value(self) -> float | None:
        """Current std dev value."""
        if len(self._values) < self.window:
            return None
        mean = sum(self._values) / self.window
        variance = sum((x - mean) ** 2 for x in self._values) / self.window
        return math.sqrt(variance)

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._values) >= self.window

    def reset(self) -> None:
        """Reset the calculator."""
        self._values.clear()


class IncrementalStdDevZeroMean:
    """Incremental standard deviation calculator assuming zero mean.

    Computes sqrt(sum(x^2) / n) - useful for return series where
    expected return is assumed to be zero.
    """

    def __init__(self, window: int):
        """Initialize std dev calculator.

        Args:
            window: Window size
        """
        self.window = window
        self._values: list[float] = []
        self._sum_sq: float = 0.0

    def update(self, value: float) -> float | None:
        """Update with new value and return current std dev.

        Args:
            value: New value

        Returns:
            Standard deviation if window is full, None otherwise
        """
        self._values.append(value)
        self._sum_sq += value * value

        if len(self._values) > self.window:
            removed = self._values.pop(0)
            self._sum_sq -= removed * removed

        if len(self._values) < self.window:
            return None

        return math.sqrt(self._sum_sq / self.window)

    @property
    def value(self) -> float | None:
        """Current std dev value."""
        if len(self._values) < self.window:
            return None
        return math.sqrt(self._sum_sq / self.window)

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._values) >= self.window

    def reset(self) -> None:
        """Reset the calculator."""
        self._values.clear()
        self._sum_sq = 0.0


class IncrementalATR:
    """Incremental Average True Range calculator."""

    def __init__(self, window: int = 14):
        """Initialize ATR calculator.

        Args:
            window: ATR period
        """
        self.window = window
        self._tr_values: list[float] = []
        self._prev_close: float | None = None
        self._atr: float | None = None

    def update(self, high: float, low: float, close: float) -> float | None:
        """Update with new bar and return current ATR.

        Args:
            high: Bar high
            low: Bar low
            close: Bar close

        Returns:
            ATR value if window is full, None otherwise
        """
        # Calculate true range
        if self._prev_close is None:
            tr = high - low
        else:
            tr = max(
                high - low,
                abs(high - self._prev_close),
                abs(low - self._prev_close),
            )

        self._prev_close = close
        self._tr_values.append(tr)

        if len(self._tr_values) > self.window:
            self._tr_values.pop(0)

        if len(self._tr_values) < self.window:
            return None

        self._atr = sum(self._tr_values) / self.window
        return self._atr

    @property
    def value(self) -> float | None:
        """Current ATR value."""
        return self._atr

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._tr_values) >= self.window

    def reset(self) -> None:
        """Reset the calculator."""
        self._tr_values.clear()
        self._prev_close = None
        self._atr = None


class IncrementalBollingerBands:
    """Incremental Bollinger Bands calculator."""

    def __init__(self, window: int = 20, num_std: float = 2.0):
        """Initialize Bollinger Bands calculator.

        Args:
            window: SMA window
            num_std: Number of standard deviations
        """
        self.window = window
        self.num_std = num_std
        self._values: list[float] = []

    def update(self, value: float) -> tuple[float | None, float | None, float | None]:
        """Update with new value.

        Args:
            value: New price value

        Returns:
            Tuple of (middle, upper, lower) or (None, None, None)
        """
        self._values.append(value)

        if len(self._values) > self.window:
            self._values.pop(0)

        if len(self._values) < self.window:
            return (None, None, None)

        mean = sum(self._values) / self.window
        variance = sum((x - mean) ** 2 for x in self._values) / self.window
        std = math.sqrt(variance)

        return (mean, mean + std * self.num_std, mean - std * self.num_std)

    @property
    def middle(self) -> float | None:
        """Current middle band (SMA)."""
        if len(self._values) < self.window:
            return None
        return sum(self._values) / self.window

    @property
    def is_ready(self) -> bool:
        """Whether we have enough data."""
        return len(self._values) >= self.window

    def reset(self) -> None:
        """Reset the calculator."""
        self._values.clear()
