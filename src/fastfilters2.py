"""Image filters for computing feature maps."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version(__package__)
del version

import numpy

import _fastfilters2 as _ff


def call(arr, scale):
    arr = numpy.asarray(arr, dtype=numpy.float32)
    out = numpy.empty((*arr.shape, 3), dtype=numpy.float32)
    _ff.call(arr, scale, out)
    return out
