import shlex
import shutil

from invoke import task

BUILD_DIR = "build"


@task
def configure(c, cmake_prefix_path, generator="Ninja"):
    """Configure CMake build."""
    cmake_prefix_path = shlex.quote(cmake_prefix_path)
    generator = shlex.quote(generator)
    c.run(
        f"cmake -S . -B {BUILD_DIR} -G {generator} -DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}"
    )


@task
def clean(_c):
    """Delete CMake build."""
    shutil.rmtree(BUILD_DIR)


@task
def build(c):
    """Recompile filters and Python extension."""
    c.run(f"cmake --build {BUILD_DIR} --target install")


@task(build)
def bench(c, save=False):
    """Run benchmarks."""
    cmd = "pytest --benchmark-enable --benchmark-only --benchmark-group-by=param:scale"
    if save:
        cmd += " --benchmark-autosave"
    c.run(cmd, pty=True)


@task(build)
def test(c):
    """Run tests."""
    c.run("pytest --benchmark-skip", pty=True)


@task
def fmt(c):
    """Format source code."""
    c.run("clang-format -i --verbose fastfilters2.c gen.cpp")
    c.run("isort .")
    c.run("black .")
