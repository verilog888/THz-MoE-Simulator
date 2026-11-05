import os
os.environ.setdefault('CC','gcc')
os.environ.setdefault('CXX','g++')
from setuptools import Extension,setup
import pybind11
setup(ext_modules=[Extension('vdie_moe_sim._native',['vdie_moe_sim/native/core.cpp'],
                             include_dirs=[pybind11.get_include()],language='c++',
                             extra_compile_args=['-O3','-g0','-std=c++20'])])
