"""Example trading strategies."""

from backtester.strategies.examples.bollinger_band import (
    BollingerBandBreakout,
    BollingerBandMeanReversion,
    DualBollingerBand,
)
from backtester.strategies.examples.momentum import (
    AbsoluteMomentum,
    DualMomentum,
    PriceMomentum,
    TrendFollowingMomentum,
)
from backtester.strategies.examples.moving_average_crossover import (
    DualMovingAverageCrossover,
    MovingAverageCrossover,
)
from backtester.strategies.examples.rsi_mean_reversion import (
    DualRSIMeanReversion,
    RSIMeanReversion,
)

__all__ = [
    "AbsoluteMomentum",
    "BollingerBandBreakout",
    "BollingerBandMeanReversion",
    "DualBollingerBand",
    "DualMomentum",
    "DualMovingAverageCrossover",
    "DualRSIMeanReversion",
    "MovingAverageCrossover",
    "PriceMomentum",
    "RSIMeanReversion",
    "TrendFollowingMomentum",
]
