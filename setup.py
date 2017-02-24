''' Setup file '''
from os import path
from distutils import extension

from setuptools import setup
import numpy

package_name = 'logpar'
cfile = path.join(package_name, 'constrained_ahc.c')
pyxfile = path.join(package_name, 'constrained_ahc.pyx')

ahc_module = package_name + '.constrained_ahc'
cli_module = package_name + '.cli'
utils_module = package_name + '.utils'

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [extension.Extension(ahc_module, [pyxfile])]
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules += [extension.Extension(ahc_module, [cfile], 
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
               'bin/cifti_average'],
      zip_safe=False)
