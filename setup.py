#from setuptools import setup
from distutils.core import setup, Extension
import os


# module1 = Extension('orphics.tools.deg2hp',
#                     sources = ['src/deg2healpix.c'],
#                     include_dirs=[os.environ['CFITSIO_DIR']+'/include',os.environ['HEALPIX']+'/include'],
#                     libraries=['cfitsio','chealpix'],
#                     library_dirs=[os.environ['CFITSIO_DIR']+'/lib',os.environ['HEALPIX']+'/lib'] )

setup(name='orphics',
      version='0.1',
      description='Cosmology Analysis',
      url='https://github.com/msyriac/orphics',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['orphics'],
      zip_safe=False)#,
#       ext_modules = [module1])
