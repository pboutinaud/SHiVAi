#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setuptools file for the shivautils package.
@author: Yann Rio
@organization: Fealinx
"""
from os.path import join
from setuptools import setup, find_packages

setup(name='shivautils',
      version='0.0.1',
      packages=find_packages(),
      author="Yann Rio",
      author_email="yrio@fealinx.com",
      description="Basic processing tools on MRI images",
      scripts=[join("scripts", "preprocess.py"),
               join("scripts", "run_preproc_wf.py"),
               join("scripts", "preproc_wf_adapted_percentile.py"),
               join("scripts", "preproc_pretrained_brainmask.py"),
               join("scripts", "script_wf_pretrained_brainmask.py"),
               join("scripts", "slicer_run_preprocessing.py") ],
      install_requires=[
            "nibabel>=4.0.2",
            "numpy>=1.21.5",
            "scipy>=1.7.3",
            "importlib",
            "argparse>=1.1",
            "nipype>=1.7.0",
            "networkx==2.8.7",
            "bokeh",
            ],

      classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
                 ]
      )
