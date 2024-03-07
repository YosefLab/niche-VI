from importlib.metadata import version

from ._model_v11 import nicheSCVI
from ._module_v11 import nicheVAE

__all__ = ["nicheSCVI", "nicheVAE"]

__version__ = version("niche-VI")
