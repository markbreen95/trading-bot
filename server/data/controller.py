from typing import Dict, Any

from server.binance.manager import BinanceManager
from server.binance.stream import BinanceStream
from server.configs.server import data_sources
from server.configs.base import DataSource
from server.configs.exchanges import SupportedExchanges, SupportedSymbols


class DataController:
    """
    All data sources are managed by this controller.

    To access exchange data from the controller:
        data.(exchange).[(symbol)][(interval)]
        data.binance['BTCUSDT']['15m'] for example
    """
    binance = {}  # type: Dict[str, BinanceStream]

    def __init__(self, binance):
        # type: (BinanceManager) -> None

        self.binance_manager = binance
        for source in data_sources:
            self._init_source(source)

    def _init_source(self, source):
        # type: (DataSource) -> None

        if source.exchange == SupportedExchanges.BINANCE:
            self._init_binance(source)
        else:
            raise NotImplementedError

    def _init_binance(self, source):
        # type: (DataSource) -> None
        if source.symbol.value not in self.binance:
            self.binance[source.symbol.value] = {}

        new_stream = BinanceStream(
            self.binance_manager.client,
            source.symbol
        )
        new_stream.kline_with_firebase()
        self.binance[source.symbol.value][source.interval.value] = new_stream
