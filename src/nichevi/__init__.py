from importlib.metadata import version

from ._components import DirichletDecoder, NicheDecoder
from ._de_utils import (
    adjusted_nearest_neighbors,
    corrupt_counts,
)
from ._model_v11 import nicheSCVI
from ._module_v11 import nicheVAE

__all__ = [
    "nicheSCVI",
    "nicheVAE",
    "NicheDecoder",
    "DirichletDecoder",
    "corrupt_counts",
    "adjusted_nearest_neighbors",
]

__version__ = version("niche-VI")
