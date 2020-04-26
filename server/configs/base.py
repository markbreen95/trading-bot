from collections import namedtuple

Bot = namedtuple("BotConfig", [
    "name",
    "strategy",
    "sources",
    "interval",
])

DataSource = namedtuple("DataSource", [
    "exchange",
    "symbol",
    "interval",
])