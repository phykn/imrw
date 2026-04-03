import numpy as np

from typing import Any
from pathlib import Path
from PIL import Image


def imread(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def imwrite(path: str | Path, img: np.ndarray, **kwargs: Any) -> None:
    Image.fromarray(img).save(path, **kwargs)
