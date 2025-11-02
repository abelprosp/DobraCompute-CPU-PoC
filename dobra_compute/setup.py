
from setuptools import setup, Extension
from setuptools import setup
import sys
import sysconfig
import os

try:
    import pybind11
except Exception as e:
    print("pybind11 is required at build time. Install with: pip install pybind11", file=sys.stderr)
    raise

include_dirs = [pybind11.get_include()]

extra_compile_args = ["-O3", "-march=native", "-fopenmp"]
extra_link_args = ["-fopenmp"]

ext_modules = [
    Extension(
        "dobra_qgemm",
        sources=["qgemm.cpp"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    )
]

setup(
    name="dobra_qgemm",
    version="0.1.0",
    description="DobraCompute PoC: int8 GEMM kernel with OpenMP",
    ext_modules=ext_modules,
    zip_safe=False,
)
