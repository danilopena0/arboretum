"""Integration tests with real market data."""

from datetime import date

import pytest

from backtester.core.engine import BacktestConfig, BacktestEngine
from backtester.core.execution import FixedSlippage, PerShareCommission, SimulatedBroker
from backtester.data.yfinance_handler import YFinanceDataHandler
from backtester.strategies.examples.moving_average_crossover import MovingAverageCrossover


@pytest.mark.integration
class TestIntegrationWithRealData:
    """Integration tests using real yfinance data."""

    def test_ma_crossover_spy_one_year(self) -> None:
        """Test MA crossover strategy on 1 year of SPY data."""
        # Setup
        data_handler = YFinanceDataHandler(cache_path="data/test_cache.duckdb")
        strategy = MovingAverageCrossover(fast_period=10, slow_period=30)
        broker = SimulatedBroker(
            slippage_model=FixedSlippage(cents_per_share=1.0),
            commission_model=PerShareCommission(rate=0.0, minimum=0.0),  # Zero commission
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            strategy=strategy,
            broker=broker,
            config=BacktestConfig(
                initial_capital=100_000.0,
                position_size=0.2,  # 20% position size
            ),
        )

        # Run backtest
        result = engine.run(
            tickers=["SPY"],
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
        )

        # Assertions
        assert result.bars_processed > 200  # Should have ~252 trading days
        assert result.num_trades > 0  # Should have some trades
        assert result.final_equity > 0  # Should still have money

        # Print summary
        print(f"\n{'='*50}")
        print("SPY MA Crossover Backtest Results (2023)")
        print(f"{'='*50}")
        print(f"Initial Capital:  ${result.config.initial_capital:,.2f}")
        print(f"Final Equity:     ${result.final_equity:,.2f}")
        print(f"Total Return:     {result.total_return_pct:+.2f}%")
        print(f"Bars Processed:   {result.bars_processed}")
        print(f"Number of Trades: {result.num_trades}")
        print(f"Signals Generated: {len(result.signals)}")
        print(f"{'='*50}\n")

    def test_cache_hit_performance(self) -> None:
        """Test that cached data is used on second run."""
        data_handler = YFinanceDataHandler(cache_path="data/test_cache.duckdb")

        # First call - may hit API
        lf1 = data_handler.get_bars(["SPY"], date(2023, 6, 1), date(2023, 6, 30))
        df1 = lf1.collect()

        # Second call - should use cache
        lf2 = data_handler.get_bars(["SPY"], date(2023, 6, 1), date(2023, 6, 30))
        df2 = lf2.collect()

        # Data should be identical
        assert len(df1) == len(df2)
        assert df1["close"].to_list() == df2["close"].to_list()

        # Check cache stats
        stats = data_handler.get_cache_stats()
        assert stats["total_rows"] > 0
        print(f"\nCache stats: {stats}")
