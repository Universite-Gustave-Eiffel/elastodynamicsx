import setuptools
import os

with open("Readme.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name='ElastodynamiCSx',
    author="Pierric Mora",
    author_email="pierric.mora@univ-eiffel.fr",
    version="0.1",
    packages=setuptools.find_packages(),
    #url='https://gitlab.com/geoendgroup/ElastodynamiCSx/ElastodynamiCSx',
    license='MIT License',
    description='elastodynamics with Fenicsx/dolfinx',
    long_description=long_description,
    long_description_content_type="text/markdown",
    scripts=[],
    #scripts=[
    #    # TODO: custom the list of python executable files to add to the path at install
    #    os.path.join('Instrumentation', 'Viewers', 'SignalViewer0D.py'),  # works
    #    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        ],
    install_requires=[
        # === "official" packages
        'numpy', 'scipy',
        'matplotlib', 'pyvista',
        #'dolfinx >= 0.4.1', ufl
        'mpi4py', 'petsc4py',
        ],
    python_requires='>=3')
