from setuptools import setup, find_packages

setup(
    name='flantastic',
    version='1.0.0',
    packages=find_packages(include=['flantastic', 'flantastic*'])
)