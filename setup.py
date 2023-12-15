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
            ['sphinx-build docs/ docs/_build'], shell=True)


setuptools.setup(
    name='ElastodynamiCSx',
    author="Pierric Mora",
    author_email="pierric.mora@univ-eiffel.fr",
    version="0.2.1",
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
        },  # noqa
    install_requires=[
        'numpy',
        'matplotlib',
        'pyvista',
        'sphinx', 'sphinx-rtd-theme', 'myst-parser', 'nbsphinx',  # to generate the sphinx doc
        ],  # noqa
    python_requires='>=3')
