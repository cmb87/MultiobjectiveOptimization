#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(
    name="moop",
    version="0.0.0.2",
    description="Multiobjective Optimization Package",
    author=["Christian Peeren"],
    author_email=["christian.peeren@gmail.com"],
    url="https://github.com/cmb87/MultiobjectiveOptimization",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    install_requires=[
        "matplotlib",
        "scipy",
        "numpy",
    ],
)
