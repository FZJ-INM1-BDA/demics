#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

readme = ""
package_name = "demics"
requirements = []
setup_requirements = []
test_requirements = []

setup(
    author="Eric Upschulte",
    author_email='e.upschulte@fz-juelich.de',
    classifiers=[],
    description="Detection of Microstructures",
    install_requires=requirements,
    long_description=readme,
    keywords=package_name,
    name=package_name,
    packages=find_packages(include=[package_name]),
    setup_requires=setup_requirements,
    version='0.1.0',
    zip_safe=False
)
