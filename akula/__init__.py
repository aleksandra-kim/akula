__all__ = (
    "DATA_DIR",
    "generate_virtual_markets_datapackage",
)

from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / "data"

from .version import version as __version__
from .virtual_markets import generate_virtual_markets_datapackage
