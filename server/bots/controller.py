#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from server.binance.manager import BinanceManager


class BotController:
    """
    The BotController class is the class that controls and runs the actual bot
    """
    
    def __init__(self, binance_manager):
        # type: (BinanceManager) -> None
        self.binance_manager = binance_manager


    def add(self, name, path):
        # type: () -> None
        """
        Adds bots to controller
        """

