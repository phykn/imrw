# imrw
Minimal image I/O library for Python.

## Features
- Read/Write images via `Pillow` & `numpy`.
- Clean and simple interface.
- `imread` uses Pillow's RGB conversion and returns a `uint8` numpy array with shape `H x W x 3`, regardless of the source format (grayscale, RGBA, CMYK, palette, etc.).
- `imwrite` accepts a `uint8` array with shape `H x W`, `H x W x 1`, `H x W x 3`, or `H x W x 4` and always saves as RGB.

## Quick Start
```bash
pip install imrw
```

```python
from imrw import imread, imwrite

image = imread("input.png")
imwrite("output.png", image)
```
