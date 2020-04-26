from pandas import DataFrame

from server.strategies.main import TradingStrategy
from server.configs.strategies import Crossover as CrossoverConfig
from typing import Callable

from server.strategies.models import MovingAverageType, TradingDecision


class CrossoverStrategy(TradingStrategy):
    """
    Deploys the crossover strategy

    Description:
        Uses two moving averages (one short (MAs) and one long (MAl)). Whenever MAs crosses MAl (in a positive
         direction) then this is a buy opportunity. Whenever MAl crosses MAs, it's a sell opportunity.
    """

    # defines the division of short / long from the last state
    last_state_div = None  # type: float

    def __init__(self, config):
        # type: (CrossoverConfig) -> None

        super().__init__()
        self.short = self._assign_moving_average(config.short)
        self.long = self._assign_moving_average(config.long)

    @classmethod
    def _assign_moving_average(cls, type):
        # type: (MovingAverageType) -> Callable
        # assigns the moving average function based on name

        if type == MovingAverageType.EXP:
            return cls.exp_moving_average
        elif type == MovingAverageType.NORMAL:
            return cls.moving_average
        else:
            raise NotImplementedError


    def _strategy(self, df):
        # type: (DataFrame) -> TradingDecision
        short_ma = self.short(df)
        long_ma = self.long(df)

        current_state_div = short_ma / long_ma
        last_state = self.last_state_div
        self.last_state_div = current_state_div

        if current_state_div > 1 and last_state < 1:
            return TradingDecision.BUY
        elif current_state_div < 1 and last_state > 1:
            return TradingDecision.SELL
        else:
            return TradingDecision.WAIT
