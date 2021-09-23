# fastfilters2

Image filters for computing feature maps.

## Build

### Prerequisites

* [miniconda][miniconda] or some other compatible installer
* [Halide][halide]

### Configure

```shell
conda env create --name ff2
conda activate ff2
python -m pip install --editable .
inv configure /path/to/installed/llvm/distribution
```

### Run tests

```shell
inv test
```

### Run benchmarks

```shell
inv bench
```

[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[halide]: https://github.com/halide/Halide#getting-halide
