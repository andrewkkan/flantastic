from setuptools import setup, find_packages

setup(
    name='flantastic',
    python_requires='>=3.9',
    version='1.0.0',
    url='https://github.com/andrewkkan/flantastic',
    description='Mixture of datasets intended for multi-task training, by combining pre-packaged or custom packaged datasets with prompt templates into a single dataset.',
    long_description=open('README.md').read(),
    package_dir = {"": "src"},
    packages = find_packages("src"),
    install_requires=[
        'datasets',
        'promptsource @ git+https://github.com/bigscience-workshop/promptsource.git'
    ]
)
