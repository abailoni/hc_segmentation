
from setuptools import setup

# setup(name='Neuro-skunkworks',
#       packages=['skunkworks',],
#      )




# from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    name='segm_utils_CY',
    version='0.01',
    description='Some segmentation tools',
    author='Alberto Bailoni',
    author_email='alberto.bailoni@iwr.uni-heidelberg.de',
    ext_modules = cythonize("./long_range_hc/criteria/learned_HC/utils/segm_utils_CY.pyx"),
    include_dirs=[numpy.get_include()]
)