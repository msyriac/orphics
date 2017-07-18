from distutils.core import setup, Extension
import os



setup(name='orphics',
      version='0.1',
      description='Cosmology Analysis',
      url='https://github.com/msyriac/orphics',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['orphics','orphics.tools','orphics.theory','orphics.analysis'],
      package_dir={'orphics':'orphics','orphics.tools':'orphics/tools','orphics.theory':'orphics/theory','orphics.analysis':'orphics/analysis'},
      zip_safe=False)
