import fastfilters as ff1
import numpy
import pytest

import fastfilters2 as ff2

SCALES = 0.3, 0.7, 1, 1.6, 3.5, 5, 10
# SCALES = 10,


@pytest.fixture
def arr():
    seed = 42
    rng = numpy.random.default_rng(seed=seed)
    shape = (512,) * 2
    # shape = (64,) * 3
    dtype = numpy.uint8
    dmax = numpy.iinfo(dtype).max
    a = rng.integers(low=0, high=dmax, size=shape, dtype=dtype, endpoint=True)
    # Large values could trigger hidden problems with numerical precision,
    # which is what we want in testing.
    a.ravel()[0] = dmax
    a.ravel()[a.size // 2] = dmax
    return a


def assert_equal(a, b):
    # For now, comparing w.r.t ULP doesn't work because some filters
    # significantly differ from the original implementation, but
    # the difference is still within an acceptable absolute range.
    # numpy.testing.assert_array_almost_equal_nulp(a, b, nulp=6)
    numpy.testing.assert_allclose(a, b, rtol=0, atol=1e-4)


def ff1_call(arr, scale):
    outputs = [
        ff1.gaussianSmoothing(arr, scale),
        ff1.gaussianGradientMagnitude(arr, scale),
        ff1.laplacianOfGaussian(arr, scale),
    ]
    # This is a bit unfair to ff1 due to additional stacking and reshaping
    # work, but it should be negligible for large input sizes.
    outputs = [a.reshape((*arr.shape, -1)) for a in outputs]
    return numpy.concatenate(outputs, axis=-1)


@pytest.mark.parametrize("scale", SCALES)
def bench_gaussian_ff1(benchmark, arr, scale):
    benchmark(ff1_call, arr, scale)


@pytest.mark.parametrize("scale", SCALES)
def bench_gaussian_ff2(benchmark, arr, scale):
    benchmark(ff2.call, arr, scale)


@pytest.mark.parametrize("scale", SCALES)
def test_gaussian(arr, scale):
    res1 = ff1.gaussianSmoothing(arr, scale)
    res2 = ff2.call(arr, scale)[..., 0]
    assert_equal(res1, res2)


@pytest.mark.parametrize("scale", SCALES)
def test_gradient_magnitude(arr, scale):
    res1 = ff1.gaussianGradientMagnitude(arr, scale)
    res2 = ff2.call(arr, scale)[..., 1]
    assert_equal(res1, res2)


@pytest.mark.parametrize("scale", SCALES)
def test_laplacian(arr, scale):
    res1 = ff1.laplacianOfGaussian(arr, scale)
    res2 = ff2.call(arr, scale)[..., 2]
    assert_equal(res1, res2)
