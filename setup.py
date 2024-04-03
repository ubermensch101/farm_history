from setuptools import setup, find_packages
import subprocess

dep_file_name = 'requirements.txt'

dependencies = [line.strip() for line in open(dep_file_name).readlines() if line.strip()]

setup(
    name='dolr',
    version='1.0',
    packages=find_packages(),
    install_requires=dependencies
)