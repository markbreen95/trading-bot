from typing import Callable

from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR

from server.configs.exchanges import SupportedSymbols
from server.secrets.binance import ROSS_BINANCE_KEY, ROSS_BINANCE_SECRET
from server.binance.stream import BinanceStream


class BinanceManager:
    """
    This class is our wrapper for the binance library, and provides api usage for its functionality.
    """

    def __init__(self):
        self.client = Client(ROSS_BINANCE_KEY, ROSS_BINANCE_SECRET)

    def stream(self, symbol):
        # type: (SupportedSymbols) -> BinanceStream
        """
        For creating a general stream
        """
        return BinanceStream(self.client, symbol)

    def kline_stream(self, symbol, callback, interval=KLINE_INTERVAL_1HOUR):
        # type: (SupportedSymbols, Callable, str) -> BinanceStream
        """
        Initialises the stream and returns it. Stream can be started with stream.start()
        """

        stream = BinanceStream(self.client, symbol)
        stream.start_kline(callback, interval)

        return stream

    def refresh_stream(self, stream):
        # type: (BinanceStream) -> None
        stream.refresh(self.client)
