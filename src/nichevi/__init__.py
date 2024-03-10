from importlib.metadata import version

from ._model_v11 import nicheSCVI
from ._module_v11 import nicheVAE

from ._components import NicheDecoder, DirichletDecoder

__all__ = ["nicheSCVI", "nicheVAE", "NicheDecoder", "DirichletDecoder"]

__version__ = version("niche-VI")
