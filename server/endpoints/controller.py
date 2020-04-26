#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from server.endpoints.models import EndpointData


class EndpointController:
    """
    The EndpointController class controls all endpoints/websockets that commmunicate with client devices
    
    """

    endpoints = []  # type: List[Endpoint]
    data = []  # type: List[EndpointData]

    def __init__(self):
        # type: () -> None
        pass

    def start(self):
        # type: () -> None
        """
        Activates each endpoint
        """

        for e, d in zip(self.endpoints, self.data):
            e.start(d)

