from setuptools import setup, find_packages
from os import path

# read the contents of the README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/gboehl/grgrlib",
    name='grgrlib',
    version='0.1.14',
    author='Gregor Boehl',
    author_email='admin@gregorboehl.com',
    license='MIT',
    description='Various insanely helpful functions',
    classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    extras_require={
     'njitted': ['numba'],
     'jaxed': ['jax', 'jaxlib'],
     'full': ['numba', 'jax', 'jaxlib'],
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'sympy',
    ],
    include_package_data=True,
)
