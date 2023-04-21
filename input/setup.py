from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
from distutils.sysconfig import get_python_lib
import sys


setup(name="pyHB",
    ext_modules=cythonize(Extension("integrator_cython",
                                    ["integrator_cython.pyx"],extra_compile_args=["-O3", "-fopenmp"],
                                    extra_link_args=["-O3 -fopenmp"],
                                    include_dirs = [numpy.get_include()]),
                                    compiler_directives={'language_level' : "3"},
                                    annotate=True
                                    )
    )
