from setuptools import setup, find_packages


setup(
    name='solids',
    packages=find_packages(),
    version='0.0.1',
    install_requires=['sympy', 'numpy', 'matplotlib', 'scipy'],
    python_requires='>=3.7, <3.8'
)