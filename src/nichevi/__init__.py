from importlib.metadata import version

from ._components import DirichletDecoder, NicheDecoder
from ._constants import NICHEVI_REGISTRY_KEYS
from ._model import nicheVI
from ._module import nicheVAE

__all__ = [
    "nicheVI",
    "nicheVAE",
    "NicheDecoder",
    "DirichletDecoder",
]

__version__ = version("niche-VI")
