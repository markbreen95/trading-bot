from server.strategies.models import MovingAverageType


class GeneralTradingStrategyError(Exception):

    def __str__(self):
        return "You cannot use the default TradingStrategy class as a strategy."

class InvalidMovingAverageType(Exception):

    type = None  # type: MovingAverageType

    def __init__(self, type):
        # type: (MovingAverageType) -> None
        self.type = type

    def __str__(self):
        # type: () -> str
        return "%s is not a valid moving average type" % self.type
