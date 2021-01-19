from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

examples_extension = Extension(
    name="example",
    sources=["example.pyx"],
    libraries=["example"],
    library_dirs=["lib"],
    include_dirs=["lib"]
)

setup(
    name='example',
    ext_modules=cythonize([examples_extension])
)
