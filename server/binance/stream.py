

from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from binance.client import Client
from typing import Callable, Dict, Any

from server.binance.exceptions import StreamNotFreshError
from server.configs.exchanges import SupportedSymbols, Intervals


class BinanceStream:
    """
    This class defines a binance data stream that's setup using BinanceSocketManager
    """

    bm = None  # type: BinanceSocketManager
    symbol = None  # type: str
    key = None  # type: str
    fresh = True  # type: bool

    def __init__(self, client, symbol):
        # type: (Client, SupportedSymbols) -> None
        self.bm = BinanceSocketManager(client)
        self.symbol = symbol.value

    def start_kline(self, callback=None, interval=KLINE_INTERVAL_1HOUR):
        # type: (Callable, str) -> None
        if not callback:
            callback = self._store_data
        self.key = self.bm.start_kline_socket(self.symbol, callback, interval=interval)

    def kline_with_firebase(self, interval=Intervals.H1):
        # type: (Intervals) -> None
        self.key = self.bm.start_kline_socket(self.symbol, self._update_firebase, interval=interval)

    def _update_firebase(self):
        # type: () -> None
        raise NotImplementedError

    def refresh(self, client):
        # type: (Client) -> None
        self.bm = BinanceSocketManager(client)
        self.fresh = True

    def start(self):
        # type: () -> None
        if self.fresh:
            self.fresh = False
            self.bm.start()
        else:
            raise StreamNotFreshError

    def close(self):
        # type: () -> None
        self.bm.close()

    def _store_data(self, data):
        # type: (Dict[str, Any]) -> None
        self.data = data
