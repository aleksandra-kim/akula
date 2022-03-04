__all__ = (
    "DATA_DIR",
    "generate_implicit_markets_datapackage",
)

from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / "data"

from .version import version as __version__
from .implicit_markets import generate_implicit_markets_datapackage
from .parameterized_exchanges import generate_parameterized_exchanges_datapackage
from .combustion import generate_liquid_fuels_combustion_correlated_samples

