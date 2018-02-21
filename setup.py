#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
from setuptools import setup

DESCRIPTION = open('README_short.txt').read()
LONG_DESCRIPTION = open('README_long.txt').read()

setup(
	author = 'chidi Ugonna',
	author_email = 'chidi_ugonna@hotmail.com',
	name = 'nklab-neuro-pipelines',
	version = '0.0.1',
	description = DESCRIPTION,
	long_description = LONG_DESCRIPTION,
	url = 'https://github.com/chidiugonna/nklab-neuro-pipelines',
	platforms = ['OS Independent'],
	license = 'MIT License',
	zip_safe=False
	classifiers= [ 
	    'Development Status :: 3 - Alpha',
	    'License :: OSI Approved :: MIT license',
	    'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.3'
	], #need to understand more about this
	packages = ['nklab-neuro-pipelines'],
	install_requires = [
             'numpy>=1.12.0',
             'matplotlib>=2.0.0'
        ]
	)

