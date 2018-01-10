from distutils.core import setup, Extension
import os



setup(name='orphics',
      version='0.1',
      description='Cosmology Analysis',
      url='https://github.com/msyriac/orphics',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['orphics'],
      package_dir={'orphics':'orphics'},
      zip_safe=False)
