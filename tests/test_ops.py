import numpy as np
import pytest
from pathlib import Path
from imio import imread, imwrite

def test_im_ops(tmp_path: Path):
    path = tmp_path / "test.png"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]

    imwrite(path, img)
    assert path.exists()

    loaded = imread(path)
    assert np.array_equal(img, loaded)
