from typing import Dict, Any

from server.endpoints.models import EndpointType, EndpointData

class Endpoint:
    """
    The class that defines all endpoints for use in the server.

    This includes both websockets and http endpoints
    """

    type = None  # type: EndpointType
    path = None  # type: str
    data = None  # type: Dict[str, Any]

    def __init__(self, type, path):
        # type: (EndPointType, str) -> None
        self.type = type
        self.path = path

    def start(self, data):
        # type: (EndpointData) -> None
        pass

