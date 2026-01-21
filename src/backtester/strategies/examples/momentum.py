"""Price Momentum Strategies.

Momentum strategies based on the empirical observation that assets with
strong recent performance tend to continue performing well over intermediate
timeframes.

Variants:
1. Simple Momentum: Enter when momentum is positive and above threshold
2. Absolute Momentum: Enter when momentum exceeds risk-free rate proxy
"""

from backtester.core.events import MarketEvent, SignalEvent, SignalType
from backtester.strategies.base import Strategy
from backtester.strategies.indicators.momentum import IncrementalMomentum
from backtester.strategies.indicators.moving_averages import IncrementalSMA


class PriceMomentum(Strategy):
    """Simple Price Momentum Strategy.

    Goes long when price momentum (return over lookback period) is positive
    and exceeds a minimum threshold. Exits when momentum turns negative.

    Attributes:
        lookback_period: Period for momentum calculation (default 20 days)
        entry_threshold: Minimum momentum to enter (default 0.0, i.e., positive)
        exit_threshold: Momentum level to exit (default 0.0)
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 0.0,
        exit_threshold: float = 0.0,
        strategy_id: str = "price_momentum",
    ):
        """Initialize the strategy.

        Args:
            lookback_period: Number of bars for momentum calculation
            entry_threshold: Minimum momentum (as decimal) to enter long
            exit_threshold: Momentum level below which to exit
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold

        # Per-ticker indicators and state
        self._momentum: dict[str, IncrementalMomentum] = {}
        self._prev_momentum: dict[str, float | None] = {}
        self._position: dict[str, bool] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        """Initialize indicators for each ticker.

        Args:
            tickers: List of tickers to trade
        """
        super().set_tickers(tickers)
        for ticker in tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._prev_momentum[ticker] = None
            self._position[ticker] = False

    def reset(self) -> None:
        """Reset strategy state."""
        for ticker in self._tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._prev_momentum[ticker] = None
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        """Process market data and generate signals.

        Args:
            event: Market data event

        Returns:
            Signal if momentum condition met, None otherwise
        """
        ticker = event.ticker

        # Initialize if needed
        if ticker not in self._momentum:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._prev_momentum[ticker] = None
            self._position[ticker] = False

        # Update momentum
        current_momentum = self._momentum[ticker].update(event.close)

        if current_momentum is None:
            return None

        self._prev_momentum[ticker] = current_momentum

        signal = None

        # Entry: momentum crosses above entry threshold
        if not self._position[ticker]:
            if current_momentum > self.entry_threshold:
                # Signal strength based on momentum magnitude
                strength = min(1.0, abs(current_momentum) * 5 + 0.5)
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=strength,
                )
                self._position[ticker] = True

        # Exit: momentum falls below exit threshold
        elif current_momentum <= self.exit_threshold:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class AbsoluteMomentum(Strategy):
    """Absolute Momentum Strategy (Time-Series Momentum).

    Only enters long when the asset's momentum exceeds a risk-free rate proxy.
    This provides downside protection by avoiding assets with weak or negative
    momentum.

    Attributes:
        lookback_period: Period for momentum calculation
        risk_free_rate: Annual risk-free rate (converted to period rate)
    """

    def __init__(
        self,
        lookback_period: int = 252,  # ~1 year of trading days
        risk_free_rate: float = 0.02,  # 2% annual
        strategy_id: str = "absolute_momentum",
    ):
        """Initialize the strategy.

        Args:
            lookback_period: Number of bars for momentum calculation
            risk_free_rate: Annual risk-free rate
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        # Convert annual rate to period rate (assuming daily bars)
        self.period_threshold = risk_free_rate * (lookback_period / 252)

        self._momentum: dict[str, IncrementalMomentum] = {}
        self._position: dict[str, bool] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = False

    def reset(self) -> None:
        for ticker in self._tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._momentum:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = False

        current_momentum = self._momentum[ticker].update(event.close)

        if current_momentum is None:
            return None

        signal = None

        # Entry: momentum exceeds risk-free threshold
        if not self._position[ticker] and current_momentum > self.period_threshold:
            # Strength based on excess return over risk-free
            excess_return = current_momentum - self.period_threshold
            strength = min(1.0, excess_return * 2 + 0.5)
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
            )
            self._position[ticker] = True

        # Exit: momentum falls below risk-free threshold
        elif self._position[ticker] and current_momentum <= self.period_threshold:
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class TrendFollowingMomentum(Strategy):
    """Trend Following Momentum with MA Filter.

    Combines momentum with a trend filter using moving averages.
    Only takes long positions when:
    1. Momentum is positive
    2. Price is above the long-term moving average (trend filter)

    This reduces whipsaws by requiring both momentum and trend confirmation.

    Attributes:
        momentum_period: Period for momentum calculation
        ma_period: Period for trend filter moving average
    """

    def __init__(
        self,
        momentum_period: int = 20,
        ma_period: int = 50,
        strategy_id: str = "trend_momentum",
    ):
        """Initialize the strategy.

        Args:
            momentum_period: Bars for momentum calculation
            ma_period: Bars for trend filter MA
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.momentum_period = momentum_period
        self.ma_period = ma_period

        self._momentum: dict[str, IncrementalMomentum] = {}
        self._ma: dict[str, IncrementalSMA] = {}
        self._position: dict[str, bool] = {}

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._momentum[ticker] = IncrementalMomentum(self.momentum_period)
            self._ma[ticker] = IncrementalSMA(self.ma_period)
            self._position[ticker] = False

    def reset(self) -> None:
        for ticker in self._tickers:
            self._momentum[ticker] = IncrementalMomentum(self.momentum_period)
            self._ma[ticker] = IncrementalSMA(self.ma_period)
            self._position[ticker] = False

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._momentum:
            self._momentum[ticker] = IncrementalMomentum(self.momentum_period)
            self._ma[ticker] = IncrementalSMA(self.ma_period)
            self._position[ticker] = False

        # Update indicators
        current_momentum = self._momentum[ticker].update(event.close)
        current_ma = self._ma[ticker].update(event.close)

        # Need both indicators ready
        if current_momentum is None or current_ma is None:
            return None

        price = event.close
        signal = None

        # Trend is up if price > MA
        trend_up = price > current_ma
        # Momentum is positive
        momentum_positive = current_momentum > 0

        # Entry: both momentum and trend confirm
        if not self._position[ticker] and trend_up and momentum_positive:
            # Stronger signal when momentum and trend strongly aligned
            strength = min(1.0, abs(current_momentum) * 5 + 0.5)
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.LONG,
                strength=strength,
            )
            self._position[ticker] = True

        # Exit: either momentum turns negative OR price falls below MA
        elif self._position[ticker] and (not trend_up or not momentum_positive):
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = False

        return signal


class DualMomentum(Strategy):
    """Dual Momentum Strategy with long and short positions.

    Goes long when momentum is positive and above threshold,
    goes short when momentum is negative and below threshold.

    WARNING: Shorting based on negative momentum can be risky as
    prices can continue falling (momentum can persist).
    """

    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 0.02,  # 2% for entry
        strategy_id: str = "dual_momentum",
    ):
        """Initialize the strategy.

        Args:
            lookback_period: Bars for momentum calculation
            entry_threshold: Minimum abs(momentum) for entry
            strategy_id: Strategy identifier
        """
        super().__init__(strategy_id)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold

        self._momentum: dict[str, IncrementalMomentum] = {}
        self._position: dict[str, str] = {}  # "long", "short", "flat"

    def set_tickers(self, tickers: list[str]) -> None:
        super().set_tickers(tickers)
        for ticker in tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = "flat"

    def reset(self) -> None:
        for ticker in self._tickers:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = "flat"

    def on_market(self, event: MarketEvent) -> SignalEvent | None:
        ticker = event.ticker

        if ticker not in self._momentum:
            self._momentum[ticker] = IncrementalMomentum(self.lookback_period)
            self._position[ticker] = "flat"

        current_momentum = self._momentum[ticker].update(event.close)

        if current_momentum is None:
            return None

        position = self._position[ticker]
        signal = None

        # Entry signals when flat
        if position == "flat":
            if current_momentum > self.entry_threshold:
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, abs(current_momentum) * 5 + 0.5),
                )
                self._position[ticker] = "long"

            elif current_momentum < -self.entry_threshold:
                signal = self.create_signal(
                    timestamp=event.timestamp,
                    ticker=ticker,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, abs(current_momentum) * 5 + 0.5),
                )
                self._position[ticker] = "short"

        # Exit signals when momentum reverses
        elif (position == "long" and current_momentum <= 0) or (
            position == "short" and current_momentum >= 0
        ):
            signal = self.create_signal(
                timestamp=event.timestamp,
                ticker=ticker,
                signal_type=SignalType.EXIT,
                strength=1.0,
            )
            self._position[ticker] = "flat"

        return signal
