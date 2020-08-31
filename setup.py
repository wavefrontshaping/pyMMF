#!/usr/bin/env python


from distutils.core import setup
import os


long_description='Please read the documentation on Github.'
if os.path.exists('README.txt'):
    long_description = open('README.txt').read()


setup(name='pyMMF',
    version='0.4',
    description='Multimode optical fiber simulation package.',
    author='Sebastien M. Popoff',
    author_email='sebastien.popoff@espci.psl.eu',
    url='https://www.wavefrontshaping.net',
    license = 'MIT',
    #py_modules = ['pyMMF'],
    packages = ['pyMMF','pyMMF.solvers'],
    long_description=long_description,
    classifiers=[
		"Programming Language :: Python :: 2",
		"Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
    ],
    install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'numba',
      ],
    )
