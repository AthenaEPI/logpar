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
      description='Tools to simplify the process of making tractography,\
                   create connectomes and parcellate subjects in the HCP',
      url='http://github.com/AthenaEPI/logpar',
      author='Gallardo Diez, Guillermo Alejandro',
      author_email='guillermo-gallardo.diez@inria.fr',
      include_package_data=True,
      ext_modules=ext_modules,
      packages=[package_name, cli_module, utils_module],
      scripts=['bin/cifti_parcellate', 'bin/extract_parcellation',
               'bin/cifti_average', 'bin/seeds_from_labeled_volume',
               'bin/vmgenerator', 'bin/resample_volume_nilearn',
               'bin/trkgenerator', 'bin/basic_hagmann'],
      zip_safe=False)
