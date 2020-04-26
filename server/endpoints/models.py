from enum import Enum
from typing import Dict, Any


class EndpointType(Enum):
    WS = "web_socket"
    WH = "web_hook"

EndpointData  = Dict[str, Any]