from setuptools import setup, find_packages

setup(
    name='solids',
    author='Mark Todisco',
    packages=find_packages(),
    version='0.0.1',
    install_requires=[
        'sympy',
        'numpy',
        'matplotlib',
        'IPython',
        'Sphinx'
    ],
    python_requires='>=3.7, <3.8'
)
