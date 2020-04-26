from enum import Enum

class SupportedExchanges(Enum):
    BINANCE = "binance"
    NOT_SUPPORTED = "not_supported"


class SupportedSymbols(Enum):
    BTCUSDT = "BTCUSDT"
    NOT_SUPPORTED = "not_supported"


class Intervals(Enum):
    M1 = "1m"
    M3 = "3m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"
    NOT_SUPPORTED = "not_supported"


class BinanceDataSources(Enum):
    BTCUSDT = "binance:" + SupportedSymbols.BTCUSDT.value
    NOT_SUPPORTED = "binance:" + SupportedSymbols.NOT_SUPPORTED.value