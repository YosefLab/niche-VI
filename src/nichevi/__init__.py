from importlib.metadata import version

from ._model import nicheSCVI
from ._module import nicheVAE

__all__ = ["nicheSCVI", "nicheVAE"]

__version__ = version("niche-VI")
