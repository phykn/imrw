# imrw
Minimal image I/O library for Python.

## Features
- Read/Write images via `Pillow` & `numpy`.
- Clean and simple interface.
- `imread` always returns an RGB (`H x W x 3`) numpy array.
- `imwrite` expects a `uint8` array with shape `H x W`, `H x W x 1`, `H x W x 3`, or `H x W x 4`.

## Quick Start
```bash
pip install imrw
```

```python
from imrw import imread, imwrite

image = imread("input.png")
imwrite("output.png", image)
```
