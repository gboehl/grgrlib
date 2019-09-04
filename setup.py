from setuptools import setup, find_packages

setup(
        name = 'grgrlib',
        version = '0.01',
        author='Gregor Boehl',
        author_email='admin@gregorboehl.com',
        description='Various insanely helpful functions',
        packages = find_packages(),
        install_requires=[
            'sympy',
            'scipy',
            'pathos',
            'matplotlib',
            'numpy',
            'numba'
         ],
   )
