#!/usr/bin/python3
from distutils.core import setup, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs

setup(
    name = "ricochet",
    version = "0.1",
    author = "Karol Zieba",
    author_email = "notkarol@gmail.com",
    description = "Solve a sliding puzzle game",
    ext_modules=[Extension("ricochet", ["ricochet.c"], extra_compile_args=['-std=c99'])],
    include_dirs=get_numpy_include_dirs(),
)
