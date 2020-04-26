
from pandas import DataFrame
from server.strategies.models import TradingDecision
from server.strategies.exceptions import GeneralTradingStrategyError

class TradingStrategy:
    """
    The base strategy class. All strategies should inherit this class.
    """

    def __init__(self):
        pass

    def predict(self, df):
        # type: (DataFrame) -> TradingDecision
        return self._strategy(df)

    def _strategy(self, df):
        # type: (DataFrame) -> TradingDecision
        # This function is a placeholder, this should be defined in all sub classes separately
        raise GeneralTradingStrategyError

    @classmethod
    def exp_moving_average(cls, df, window):
        # type: (DataFrame, int) -> float
        # TODO: Make this return a value rather than a df
        return df.ewm(span=window).mean()

    @classmethod
    def moving_average(cls, df, window):
        # type: (DataFrame, int) -> float
        # TODO: Make this return a value rather than a df
        return df.rolling(window).mean()

