''' Setup file '''
from os import path
from distutils import extension

from setuptools import setup
import numpy

package_name = 'logpar'
cfile = path.join(package_name, 'constrained_ahc.c')

ahc_module = package_name + '.constrained_ahc'
cli_module = package_name + '.cli'
utils_module = package_name + '.utils'

ext_modules = [extension.Extension(ahc_module, [cfile], 
                                   include_dirs=[numpy.get_include()])]

setup(name=package_name,
      version='0.2',
      description='Tools to parcellate connectivity matrices',
      url='http://github.com/AthenaEPI/logpar',
      author='Gallardo Diez, Guillermo Alejandro',
      author_email='gallardo@cbs.mpg.de',
      include_package_data=True,
      ext_modules=ext_modules,
      packages=[package_name, cli_module, utils_module],
      scripts=['scripts/cifti_parcellate', 'scripts/extract_parcellation',
               'scripts/cifti_average'],
      zip_safe=False)
