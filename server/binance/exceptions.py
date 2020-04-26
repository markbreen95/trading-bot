class StreamNotFreshError(Exception):
    """
    Raised whenever a BinanceStream object has already had its stream started
    """
    def __str__(self):
        # type: () -> str
        return "A BinanceStream object cannot be restarted, you must create a new object or refresh."