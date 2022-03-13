__all__ = (
    "DATA_DIR",
    "generate_markets_datapackage",
)

from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / "data"

from .version import version as __version__
from .markets import generate_markets_datapackage
from .parameterized_exchanges import generate_parameterized_exchanges_datapackage
from .combustion import generate_liquid_fuels_combustion_correlated_samples

