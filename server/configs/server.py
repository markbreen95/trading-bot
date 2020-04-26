
from server.configs.base import Bot, DataSource
from server.configs.strategies import Crossover
from server.strategies.models import MovingAverageType
from server.configs.exchanges import SupportedExchanges, SupportedSymbols, Intervals, BinanceDataSources

# DEFINE ALL NEW BOTS UNDER HERE
# ADD A NAMED TUPLE FOR EACH ENTRY

data_sources = [
    DataSource(
        exchange=SupportedExchanges.BINANCE,
        symbol=SupportedSymbols.BTCUSDT,
        interval=[Intervals.M15, Intervals.H1]
    )
]

bots = [
    Bot(
        name="SampleBot",
        strategy=Crossover(
            short=MovingAverageType.EXP,
            long=MovingAverageType.NORMAL,
            buy_cutoff=1,
            sell_cutoff=1
        ),
        sources=[BinanceDataSources.BTCUSDT],
        interval=Intervals.H1
    )
]