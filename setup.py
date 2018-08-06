import sys
from setuptools import setup, Extension, find_packages
import numpy as np
import numpy.distutils.core


setup(
        name = 'grgrlib',
        version = '0.0.0alpha',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Various functions & libraries for economic dynamic analysis',
        packages = find_packages(),
        install_requires=[
            # 'pandas',
            #'slycot',
            'sympy',
            'scipy',
            'numpy',
         ],
        py_modules=['grgrlib','pyzlb'],
   )
