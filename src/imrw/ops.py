from pathlib import Path

import numpy as np
from PIL import Image


def imread(path: str | Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def imwrite(path: str | Path, img: np.ndarray, **kwargs) -> None:
    if img.dtype != np.uint8:
        raise ValueError(f"imwrite expects a uint8 numpy array, got dtype={img.dtype}")

    if img.ndim not in (2, 3):
        raise ValueError(f"imwrite expects a 2D or 3D array, got shape={img.shape}")

    if img.ndim == 3:
        if img.shape[2] not in (1, 3, 4):
            raise ValueError(
                f"imwrite expects channel count in {{1, 3, 4}}, got shape={img.shape}"
            )
        if img.shape[2] == 1:
            img = img[:, :, 0]

    Image.fromarray(img).convert("RGB").save(path, **kwargs)
