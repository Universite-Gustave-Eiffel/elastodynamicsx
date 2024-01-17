import setuptools
import subprocess
# import os

with open("Readme.md", "r") as fh:
    long_description = fh.read()


class MakeTheDoc(setuptools.Command):
    description = "Generate Documentation Pages using Sphinx"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """The command to run when users invoke python setup.py doc"""
        subprocess.run(
            ['sphinx-build docsrc docs'], shell=True)


class SafetyChecks(setuptools.Command):
    description = "Run all tests (flake8, mypy, pytest)"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        """The command to run when users invoke python setup.py doc"""
        subprocess.run(
            ['echo "\n\t##########\n\t-> running flake8..." && flake8 && '
             + 'echo "\n\t##########\n\t-> running mypy..." && mypy test/ && mypy elastodynamicsx/ && '
             + 'echo "\n\t##########\n\t-> running pytest..." && pytest'], shell=True)


setuptools.setup(
    name='ElastodynamiCSx',
    author="Pierric Mora",
    author_email="pierric.mora@univ-eiffel.fr",
    version="0.2.2",
    packages=setuptools.find_packages(),
    url='https://github.com/Universite-Gustave-Eiffel/elastodynamicsx',
    license='MIT License',
    description='Elastodynamics with FEniCSx/DOLFINx',
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=[],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Linux",
        "Operating System :: MacOS",
        ],  # noqa
    cmdclass={
        'doc': MakeTheDoc,  # allow user to build the doc with python setup.py doc
        'tests': SafetyChecks,  # allow user to run all tests with python setup.py test
        },  # noqa
    install_requires=[
        'numpy',
        'matplotlib',
        'pyvista',
        'sphinx', 'sphinx-rtd-theme', 'myst-parser', 'nbsphinx', 'sphinx-tabs', 'jupyter-sphinx',  # for sphinx doc
        ],  # noqa
    python_requires='>=3')
