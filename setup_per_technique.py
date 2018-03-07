from distutils.core import setup, Extension
from Cython.Build import cythonize

ext = [Extension(
    'ngrams_per_technique',
    ["ngrams_per_technique.pyx"],
    extra_compile_args=['-std=c++0x', '-fopenmp'],
    extra_link_args=['-fopenmp'],
    language="c++")]

setup(
    name='ngrams_per_technique',
    ext_modules=cythonize(ext, language="c++"),
)
