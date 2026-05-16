import numpy as np
import pytest
from pathlib import Path
from PIL import Image
from imrw import imread, imwrite


def pillow_rgb_array(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


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


def test_rgba_reads_as_rgb(tmp_path: Path):
    path = tmp_path / "rgba_rt.png"
    img = np.zeros((10, 10, 4), dtype=np.uint8)
    img[..., 3] = 255
    img[0, 0] = [255, 0, 0, 128]

    imwrite(path, img)
    loaded = imread(path)

    assert loaded.shape == (10, 10, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(img[..., :3], loaded)


def test_grayscale_2d_reads_as_rgb(tmp_path: Path):
    path = tmp_path / "gray_rt.png"
    img = np.full((10, 10), 128, dtype=np.uint8)
    img[0, 0] = 42

    imwrite(path, img)
    loaded = imread(path)

    assert loaded.shape == (10, 10, 3)
    assert loaded.dtype == np.uint8
    expected = np.stack([img] * 3, axis=-1)
    assert np.array_equal(expected, loaded)


def test_grayscale_3d_reads_as_rgb(tmp_path: Path):
    path = tmp_path / "gray3d_rt.png"
    img = np.full((10, 10, 1), 128, dtype=np.uint8)

    imwrite(path, img)
    loaded = imread(path)

    assert loaded.shape == (10, 10, 3)
    expected = np.stack([img[:, :, 0]] * 3, axis=-1)
    assert np.array_equal(expected, loaded)


def test_uint16_tiff_reads_like_pillow_rgb_convert(tmp_path: Path):
    path = tmp_path / "16bit.tif"
    data = np.linspace(0, 65535, 100 * 100, dtype=np.uint16).reshape(100, 100)
    Image.fromarray(data).save(path)

    loaded = imread(path)
    expected = pillow_rgb_array(path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(expected, loaded)


def test_int32_tiff_reads_like_pillow_rgb_convert(tmp_path: Path):
    path = tmp_path / "32bit.tif"
    data = np.array([[0, 1, 255, 256, 65535]], dtype=np.int32)
    Image.fromarray(data, mode="I").save(path)

    loaded = imread(path)
    expected = pillow_rgb_array(path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(expected, loaded)


def test_float_tiff_reads_like_pillow_rgb_convert(tmp_path: Path):
    path = tmp_path / "float.tif"
    data = np.array([[0.0, 0.5, 1.0, 255.0]], dtype=np.float32)
    Image.fromarray(data, mode="F").save(path)

    loaded = imread(path)
    expected = pillow_rgb_array(path)

    assert loaded.shape == expected.shape
    assert loaded.dtype == np.uint8
    assert np.array_equal(expected, loaded)


def test_fake_tif_extension_reads_correctly(tmp_path: Path):
    path = tmp_path / "fake.tif"
    data = np.zeros((50, 50, 3), dtype=np.uint8)
    data[0, 0] = [255, 0, 0]
    data[1, 1] = [0, 128, 0]
    Image.fromarray(data).save(path, format="PNG")

    loaded = imread(path)

    assert loaded.shape == (50, 50, 3)
    assert loaded.dtype == np.uint8
    assert np.array_equal(data, loaded)
