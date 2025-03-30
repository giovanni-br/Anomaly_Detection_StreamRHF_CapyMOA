from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("random_histogram_forest.pyx"),
    include_dirs=[np.get_include()],
)

