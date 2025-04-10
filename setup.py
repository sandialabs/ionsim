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
    use_scm_version=True,
    setup_requires=['setuptools>=42', 'setuptools_scm'],
    description='Tools for simulating trapped ion physics.',
    # long_description=open('README.md').read(),
    author='Brandon Ruzic',
    author_email='bruzic@sandia.gov',
    package_dir={'':'src'},
    packages=find_packages(where='src'),
    # package_data={  'ionsim.physics'  :['data/*.data'],
    #                 'ionsim.tests'    :['functionTestData/*']},
    install_requires=['numpy','scipy','matplotlib', "csaps", "icecream", "pyyaml", "nptyping"],
    platforms=['any'],
    classifiers=filter(None, classifiers.split("\n")),
    package_data={
        'ionsim.atomic_config_data': ['*.yaml'],  # Include all YAML files in the atomic_config_data package
    },
)
