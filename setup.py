# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='vdp',
    version='0.1.0',
    description='Vision Data Processor for training image classifiers',
    long_description=readme,
    author='Patrick D. Weber',
    author_email='patrick310@gmail.com',
    url='https://github.com/patrick310/sc-vision',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)