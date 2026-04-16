from importlib.metadata import version

from .ops import imread, imwrite

__version__ = version("imrw")
__all__ = ["imread", "imwrite", "__version__"]
