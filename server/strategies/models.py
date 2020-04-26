from enum import Enum

class TradingDecision(Enum):
    BUY = "buy"
    SELL = "sell"
    WAIT = "wait"

class MovingAverageType(Enum):
    EXP = "exponential"
    NORMAL = "normal"