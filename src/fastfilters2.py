"""Image filters for computing feature maps."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version(__package__)
del version

import math
import numpy

import _fastfilters2 as _ff


def _kernels(scale):
    r0 = math.ceil(3 * scale)
    r1 = math.ceil(3.5 * scale)
    r2 = math.ceil(4 * scale)

    x02 = numpy.square(numpy.arange(-r0, r0 + 1, dtype="d"))
    x1 = numpy.arange(-r1, r1 + 1, dtype="d")
    x22 = numpy.square(numpy.arange(-r2, r2 + 1, dtype="d"))

    scale2 = scale * scale

    a0 = 1 / (math.sqrt(math.tau) * scale)
    a1 = -a0 / scale2
    a2 = -a1 / scale2
    b = -0.5 / scale2

    k0 = a0 * numpy.exp(b * x02)
    k1 = a1 * x1 * numpy.exp(b * x1 * x1)
    k2 = (a1 + a2 * x22) * numpy.exp(b * x22)

    k0 /= k0.sum()
    k1 /= abs((x1 * k1).sum())
    k2 /= (x22 * (k2 - k2.mean())).sum() / 2

    return k0.astype("f"), k1.astype("f"), k2.astype("f")


_SCALES = 0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0
_KERNELS = {s: _kernels(s) for s in _SCALES}


def call(arr, scale, *, kernels=_KERNELS):
    try:
        k = kernels[scale]
    except KeyError:
        k = kernels[scale] = _kernels(scale)
    arr = numpy.asarray(arr, dtype="f")
    out = numpy.empty((*arr.shape, 3), dtype="f")
    _ff.call(arr, out, *k)
    return out
