# imio
Minimal image I/O library for Python.

## Features
- Read/Write images via `Pillow` & `numpy`.
- Clean and simple interface.

## Quick Start
```bash
pip install .
```

```python
from imio import imread, imwrite

image = imread("input.png")
imwrite("output.png", image)
```