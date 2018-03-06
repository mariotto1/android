from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = [Extension(
    'ngrams',
    ["ngrams.pyx"],
    extra_compile_args=['-std=c++0x', '-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++")]

setup(
    name='ngrams',
    ext_modules=cythonize(ext, language="c++"),
)
