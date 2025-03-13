""" IonSim: simulating trapped ion physics in python """

from setuptools import setup, find_packages

# with open("ionsim/_version.py") as f:
#     code = compile(f.read(), "ionsim/_version.py", 'exec')
#     exec(code)

classifiers = """\
Development Status :: 2 - Beta
Intended Audience :: Science/Research
License :: Other/Proprietary License
Programming Language :: Python
Topic :: Scientific/Engineering :: Physics
Operating System :: Microsoft :: Windows
Operating System :: MacOS :: MacOS X
Operating System :: Unix
"""

setup(
    name='IonSim',
    # version=__version__,
    description='Tools for simulating trapped ion physics.',
    # long_description=open('README.md').read(),
    author='Brandon Ruzic',
    author_email='bruzic@sandia.gov',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    # package_data={  'ionsim.physics'  :['data/*.data'],
    #                 'ionsim.tests'    :['functionTestData/*']},
    install_requires=['numpy','scipy','matplotlib', "csaps"],
    platforms=['any'],
    classifiers=filter(None, classifiers.split("\n")),
)
