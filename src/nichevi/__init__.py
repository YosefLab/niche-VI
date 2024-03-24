from importlib.metadata import version

from ._model_v11 import nicheSCVI
from ._module_v11 import nicheVAE

from ._components import NicheDecoder, DirichletDecoder


from ._de_utils import (
    corrupt_counts,
)

__all__ = [
    "nicheSCVI",
    "nicheVAE",
    "NicheDecoder",
    "DirichletDecoder",
    "corrupt_counts",
]

__version__ = version("niche-VI")
