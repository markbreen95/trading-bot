
import pickle
from server.binance.stream import BinanceStream

class Bot:

    def __init__(self, name, path, data_source="binance:BNBBTC", stream="kline"):
        # type: (str, str) -> None
        self.name = name
        self.model = pickle.load(path)
        exchange, symbol = data_source.split(":")
        if exchange != "binance":
            raise NotImplementedError("%s is not a supported exchange." % exchange)

        if stream != "kline":
            raise NotImplementedError("%s is not a supported stream type." % stream)


    def _parse_kline_data(self, data):
        # type: (str) -> None

