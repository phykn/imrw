import numpy as np
import pytest
from pathlib import Path
from imio import imread, imwrite


def test_im_ops_roundtrip(tmp_path: Path):
    path = tmp_path / "test.png"
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[0, 0] = [255, 0, 0]

    imwrite(path, img)
    assert path.exists()

    loaded = imread(path)
    assert np.array_equal(img, loaded)


def test_imwrite_rejects_non_uint8(tmp_path: Path):
    path = tmp_path / "bad_dtype.png"
    img = np.zeros((10, 10, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="uint8"):
        imwrite(path, img)


def test_imwrite_rejects_invalid_ndim(tmp_path: Path):
    path = tmp_path / "bad_ndim.png"
    img = np.zeros((4, 4, 3, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="2D or 3D"):
        imwrite(path, img)


def test_imwrite_rejects_invalid_channel_count(tmp_path: Path):
    path = tmp_path / "bad_channels.png"
    img = np.zeros((10, 10, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="channel count"):
        imwrite(path, img)
