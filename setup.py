#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme = ""
package_name = "demics"
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
    # packages=['demics', 'demics.system', 'demics.meta', 'demics.ops', 'demics.controller', 'demics.data',
    #           'demics.environment', 'demics.tensor'],
    packages=find_packages(include=[package_name]),
    setup_requires=setup_requirements,
    version='1.0.0-alpha',
    zip_safe=False
)
