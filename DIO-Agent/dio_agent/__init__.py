"""Public package surface for DIO-Agent."""

from dio_agent._version import __version__
from dio_agent.config import Config
from dio_agent.controller import DIOAgent

__all__ = [
    "Config",
    "DIOAgent",
    "__version__",
]
