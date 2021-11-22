# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:28:38 2021

@author: Robert
"""

from setuptools import setup

setup(name='copied_project',
      version='1.0',
      # list folders, not files
      packages=['src'],
      scripts=[],# sample from capitalize package: 'capitalize/bin/cap_script.py'
      package_data={'copied_project':['data/raw','data/processed','data/sql']},#sample from capitalize package:'capitalize': ['data/cap_data.txt']
      )
