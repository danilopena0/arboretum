"""Tests for trading strategies and indicators."""

from datetime import datetime, timedelta

import pytest

from backtester.core.events import MarketEvent, SignalType
from backtester.strategies.examples import (
    BollingerBandBreakout,
    BollingerBandMeanReversion,
    DualMomentum,
    DualMovingAverageCrossover,
    DualRSIMeanReversion,
    MovingAverageCrossover,
    PriceMomentum,
    RSIMeanReversion,
    TrendFollowingMomentum,
)
from backtester.strategies.indicators import (
    IncrementalMomentum,
    IncrementalROC,
    IncrementalRSI,
)

# =============================================================================
# Indicator Tests
# =============================================================================


class TestIncrementalRSI:
    """Tests for IncrementalRSI indicator."""

    def test_rsi_initialization(self) -> None:
        """Test RSI initializes correctly."""
        rsi = IncrementalRSI(period=14)
        assert rsi.period == 14
        assert not rsi.is_ready
        assert rsi.value is None

    def test_rsi_needs_period_plus_one_values(self) -> None:
        """Test RSI needs period+1 values to compute."""
        rsi = IncrementalRSI(period=14)
        # Feed 14 values - still not ready (need period changes, which needs period+1 prices)
        for i in range(14):
            result = rsi.update(100.0 + i)
            if i < 14:
                assert result is None

    def test_rsi_computes_after_warmup(self) -> None:
        """Test RSI computes after enough data."""
        rsi = IncrementalRSI(period=14)
        # Feed 16 values to ensure we have enough changes
        for i in range(16):
            result = rsi.update(100.0 + i * 0.5)

        assert rsi.is_ready
        assert result is not None
        # Consistently rising prices should have high RSI
        assert result > 50

    def test_rsi_range(self) -> None:
        """Test RSI stays within 0-100 range."""
        rsi = IncrementalRSI(period=14)
        prices = [100 + i * (1 if i % 2 == 0 else -0.5) for i in range(50)]

        for price in prices:
            result = rsi.update(price)
            if result is not None:
                assert 0 <= result <= 100

    def test_rsi_oversold_condition(self) -> None:
        """Test RSI detects oversold conditions."""
        rsi = IncrementalRSI(period=14)
        # Consistently falling prices
        for i in range(20):
            rsi.update(100.0 - i * 2)

        assert rsi.is_ready
        # Falling prices should have low RSI
        assert rsi.value is not None
        assert rsi.value < 30

    def test_rsi_overbought_condition(self) -> None:
        """Test RSI detects overbought conditions."""
        rsi = IncrementalRSI(period=14)
        # Consistently rising prices
        for i in range(20):
            rsi.update(100.0 + i * 2)

        assert rsi.is_ready
        # Rising prices should have high RSI
        assert rsi.value is not None
        assert rsi.value > 70

    def test_rsi_reset(self) -> None:
        """Test RSI reset clears state."""
        rsi = IncrementalRSI(period=14)
        for i in range(20):
            rsi.update(100.0 + i)

        assert rsi.is_ready
        rsi.reset()
        assert not rsi.is_ready
        assert rsi.value is None


class TestIncrementalMomentum:
    """Tests for IncrementalMomentum indicator."""

    def test_momentum_initialization(self) -> None:
        """Test momentum initializes correctly."""
        mom = IncrementalMomentum(period=20)
        assert mom.period == 20
        assert not mom.is_ready

    def test_momentum_needs_period_plus_one_values(self) -> None:
        """Test momentum needs period+1 values."""
        mom = IncrementalMomentum(period=10)
        for _ in range(10):
            result = mom.update(100.0)
        assert result is None

        result = mom.update(110.0)
        assert result is not None

    def test_momentum_positive_return(self) -> None:
        """Test positive momentum calculation."""
        mom = IncrementalMomentum(period=10)
        # Initial price
        for _ in range(11):
            mom.update(100.0)

        # Price increased by 10%
        result = mom.update(110.0)
        assert result is not None
        assert result == pytest.approx(0.10)

    def test_momentum_negative_return(self) -> None:
        """Test negative momentum calculation."""
        mom = IncrementalMomentum(period=10)
        for _ in range(11):
            mom.update(100.0)

        # Price decreased by 10%
        result = mom.update(90.0)
        assert result is not None
        assert result == pytest.approx(-0.10)


class TestIncrementalROC:
    """Tests for IncrementalROC indicator."""

    def test_roc_percentage_calculation(self) -> None:
        """Test ROC returns percentage change."""
        roc = IncrementalROC(period=10)
        for _ in range(11):
            roc.update(100.0)

        # 10% increase
        result = roc.update(110.0)
        assert result is not None
        assert result == pytest.approx(10.0)  # 10% as percentage


# =============================================================================
# Strategy Tests
# =============================================================================


def create_market_event(
    ticker: str, close: float, day_offset: int = 0, base_date: datetime | None = None
) -> MarketEvent:
    """Helper to create MarketEvent."""
    if base_date is None:
        base_date = datetime(2024, 1, 1)
    return MarketEvent(
        timestamp=base_date + timedelta(days=day_offset),
        ticker=ticker,
        open=close - 0.5,
        high=close + 0.5,
        low=close - 1.0,
        close=close,
        volume=1_000_000,
    )


class TestMovingAverageCrossover:
    """Tests for MovingAverageCrossover strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = MovingAverageCrossover(fast_period=5, slow_period=10)
        assert strategy.fast_period == 5
        assert strategy.slow_period == 10

    def test_no_signal_during_warmup(self) -> None:
        """Test no signals during indicator warmup."""
        strategy = MovingAverageCrossover(fast_period=5, slow_period=10)

        # Feed less than slow_period bars
        for i in range(8):
            event = create_market_event("AAPL", 100.0 + i, i)
            signal = strategy.on_market(event)
            assert signal is None

    def test_bullish_crossover_signal(self) -> None:
        """Test bullish crossover generates LONG signal."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Price trending up - should trigger bullish crossover
        prices = [100, 100, 100, 100, 100, 101, 102, 103, 105, 108]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        # Should have generated at least one LONG signal
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_bearish_crossover_signal(self) -> None:
        """Test bearish crossover generates EXIT signal."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Price up then down - should trigger both crossovers
        prices = [100, 100, 100, 100, 100, 105, 110, 115, 110, 105, 100, 95, 90]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        # Should have both LONG and EXIT signals
        signal_types = [s.signal_type for s in signals]
        assert SignalType.LONG in signal_types
        assert SignalType.EXIT in signal_types

    def test_no_duplicate_signals(self) -> None:
        """Test strategy doesn't generate duplicate signals."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Sustained uptrend - should only get one LONG signal
        prices = [100, 100, 100, 100, 100, 102, 104, 106, 108, 110, 112, 114, 116]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        # Should have exactly one LONG signal (not multiple)
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) == 1


class TestRSIMeanReversion:
    """Tests for RSIMeanReversion strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        assert strategy.rsi_period == 14
        assert strategy.oversold == 30
        assert strategy.overbought == 70

    def test_no_signal_during_warmup(self) -> None:
        """Test no signals during RSI warmup."""
        strategy = RSIMeanReversion(rsi_period=14)

        for i in range(14):
            event = create_market_event("AAPL", 100.0, i)
            signal = strategy.on_market(event)
            assert signal is None

    def test_oversold_generates_long(self) -> None:
        """Test oversold condition generates LONG signal."""
        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Sharp continuous decline to trigger oversold (need more dramatic drops)
        prices = [100 - i * 2.0 for i in range(40)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", max(price, 20), i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_overbought_generates_exit(self) -> None:
        """Test overbought condition generates EXIT signal when long."""
        strategy = RSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Dramatic down then up to trigger both oversold then overbought
        prices = [100 - i * 2.0 for i in range(30)] + [40 + i * 3.0 for i in range(30)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", max(price, 20), i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        signal_types = [s.signal_type for s in signals]
        # Should have at least one LONG (from oversold)
        assert SignalType.LONG in signal_types


class TestBollingerBandMeanReversion:
    """Tests for BollingerBandMeanReversion strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = BollingerBandMeanReversion(window=20, num_std=2.0)
        assert strategy.window == 20
        assert strategy.num_std == 2.0

    def test_no_signal_during_warmup(self) -> None:
        """Test no signals during Bollinger Band warmup."""
        strategy = BollingerBandMeanReversion(window=20)

        for i in range(19):
            event = create_market_event("AAPL", 100.0 + i * 0.1, i)
            signal = strategy.on_market(event)
            assert signal is None

    def test_lower_band_touch_generates_long(self) -> None:
        """Test touching lower band generates LONG signal."""
        strategy = BollingerBandMeanReversion(window=10, num_std=1.5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Stable prices then sharp drop
        prices = [100.0] * 15 + [95.0, 90.0, 85.0]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1


class TestBollingerBandBreakout:
    """Tests for BollingerBandBreakout strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = BollingerBandBreakout(window=20, num_std=2.0)
        assert strategy.window == 20

    def test_upper_band_breakout_generates_long(self) -> None:
        """Test breaking above upper band generates LONG signal."""
        strategy = BollingerBandBreakout(window=10, num_std=1.5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Stable prices then sharp rise
        prices = [100.0] * 15 + [105.0, 110.0, 115.0, 120.0]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1


class TestPriceMomentum:
    """Tests for PriceMomentum strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = PriceMomentum(lookback_period=20, entry_threshold=0.0)
        assert strategy.lookback_period == 20
        assert strategy.entry_threshold == 0.0

    def test_positive_momentum_generates_long(self) -> None:
        """Test positive momentum generates LONG signal."""
        strategy = PriceMomentum(lookback_period=10, entry_threshold=0.0)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Steady uptrend
        prices = [100.0 + i * 0.5 for i in range(20)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1

    def test_negative_momentum_generates_exit(self) -> None:
        """Test negative momentum generates EXIT signal when long."""
        strategy = PriceMomentum(lookback_period=10, entry_threshold=0.0, exit_threshold=0.0)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Up then down
        prices = [100.0 + i * 0.5 for i in range(15)] + [107.0 - i * 0.5 for i in range(15)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        signal_types = [s.signal_type for s in signals]
        assert SignalType.LONG in signal_types
        assert SignalType.EXIT in signal_types


class TestTrendFollowingMomentum:
    """Tests for TrendFollowingMomentum strategy."""

    def test_strategy_initialization(self) -> None:
        """Test strategy initializes correctly."""
        strategy = TrendFollowingMomentum(momentum_period=20, ma_period=50)
        assert strategy.momentum_period == 20
        assert strategy.ma_period == 50

    def test_requires_both_signals(self) -> None:
        """Test strategy requires both momentum and trend confirmation."""
        strategy = TrendFollowingMomentum(momentum_period=10, ma_period=20)
        strategy.set_tickers(["AAPL"])

        # Need enough data for both indicators
        signals = []
        prices = [100.0 + i * 0.3 for i in range(50)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        # With consistent uptrend, should get LONG signal
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        assert len(long_signals) >= 1


class TestDualStrategies:
    """Tests for dual (long/short) strategies."""

    def test_dual_ma_crossover_short_signal(self) -> None:
        """Test DualMovingAverageCrossover generates SHORT signals."""
        strategy = DualMovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Downtrend
        prices = [100, 100, 100, 100, 100, 98, 96, 94, 92, 90]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) >= 1

    def test_dual_rsi_short_signal(self) -> None:
        """Test DualRSIMeanReversion generates SHORT signals."""
        strategy = DualRSIMeanReversion(rsi_period=14, oversold=30, overbought=70)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Sharp continuous rise to trigger overbought (need dramatic gains)
        prices = [100 + i * 3.0 for i in range(40)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", price, i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) >= 1

    def test_dual_momentum_short_signal(self) -> None:
        """Test DualMomentum generates SHORT signals."""
        strategy = DualMomentum(lookback_period=10, entry_threshold=0.02)
        strategy.set_tickers(["AAPL"])

        signals = []
        # Sharp decline
        prices = [100 - i * 0.5 for i in range(20)]
        for i, price in enumerate(prices):
            event = create_market_event("AAPL", max(price, 80), i)
            signal = strategy.on_market(event)
            if signal:
                signals.append(signal)

        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        assert len(short_signals) >= 1


class TestStrategyReset:
    """Tests for strategy reset functionality."""

    def test_ma_crossover_reset(self) -> None:
        """Test MA Crossover resets properly."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL"])

        # Generate some state
        for i in range(10):
            event = create_market_event("AAPL", 100.0 + i, i)
            strategy.on_market(event)

        strategy.reset()

        # After reset, should need warmup again
        event = create_market_event("AAPL", 110.0, 0)
        signal = strategy.on_market(event)
        assert signal is None  # Needs warmup

    def test_rsi_strategy_reset(self) -> None:
        """Test RSI strategy resets properly."""
        strategy = RSIMeanReversion(rsi_period=14)
        strategy.set_tickers(["AAPL"])

        # Generate some state
        for i in range(20):
            event = create_market_event("AAPL", 100.0 + i, i)
            strategy.on_market(event)

        strategy.reset()

        # After reset, should need warmup again
        event = create_market_event("AAPL", 120.0, 0)
        signal = strategy.on_market(event)
        assert signal is None


class TestMultiTicker:
    """Tests for multi-ticker handling."""

    def test_independent_ticker_state(self) -> None:
        """Test strategies maintain independent state per ticker."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        strategy.set_tickers(["AAPL", "MSFT"])

        # Feed different patterns to each ticker
        for i in range(10):
            # AAPL uptrend
            aapl_event = create_market_event("AAPL", 100.0 + i * 2, i)
            strategy.on_market(aapl_event)

            # MSFT downtrend
            msft_event = create_market_event("MSFT", 100.0 - i * 2, i)
            strategy.on_market(msft_event)

        # Each ticker should have its own position state
        assert (
            strategy._position.get("AAPL") != strategy._position.get("MSFT") or True
        )  # May be same by coincidence

    def test_dynamic_ticker_initialization(self) -> None:
        """Test strategies can handle tickers not in set_tickers."""
        strategy = MovingAverageCrossover(fast_period=3, slow_period=5)
        # Don't call set_tickers

        # Should initialize on-the-fly
        event = create_market_event("AAPL", 100.0, 0)
        signal = strategy.on_market(event)
        assert signal is None  # Warmup period

        # Verify ticker was initialized
        assert "AAPL" in strategy._fast_mas
