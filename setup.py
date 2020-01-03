#!/usr/bin/env python

from distutils.core import setup

from setuptools import find_packages

setup(
    name="moop",
    version="0.0.0.1",
    description="Multiobjective Optimization Package",
    author=["Christian Peeren", "Matthias Huels"],
    author_email=["christian.peeren@gmail.com", "matze.huels@gmail.com"],
    url="https://github.com/cmb87/MultiobjectiveOptimization",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    install_requires=[
        "pip==19.1.1",
        "matplotlib==3.1.2",
        "scipy ==1.4.1",
        "numpy == 1.18.0",
    ],
)
