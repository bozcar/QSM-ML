from importlib_metadata import entry_points
from setuptools import setup, find_packages

setup(
    name='local_QSM_ML',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['tensorflow', 'numpy', 'matplotlib'],
    entry_points={
        'scripts': [
            ''
        ]
    }
)