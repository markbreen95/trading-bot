# -*- coding: utf-8 -*-
from server.binance.manager import BinanceManager
from server.data.controller import DataController
from server.endpoints.controller import EndpointController
from server.bots.controller import BotController


class Server:
    
    """
    Server is the server controller that controls all parts of the server.
    
    To start the server, run Server.start()
    
    Associated Classes:
        BotController: Controls all bots
        EndpointController: Controls all endpoints/websockets

    Guide:
        - bots are setup in server.py
    """
    
    bot_controller = None  # type: BotController
    endpoint_controller = None  # type: EndpointController
    binance = None  # type: BinanceManager

    def start(self):
        # type: () -> None
        self.bot_controller.start()
        self.endpoint_controller.start()
        pass
    
    def __init__(self):
        # type: () -> None

        # Initialise data sources first
        self.binance = BinanceManager()
        self.data = DataController(self.binance)

        pass
    
    
