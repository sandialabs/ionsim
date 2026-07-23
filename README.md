IonSim
=======

## Quickstart

If you already have a github account and a virtual environment
capable of installing packages from pypi.org, follow these
instructions in a bash terminal with the appropriate virtual
environment active. If not, the full instructions are below.

```bash
# Clone the repo (the below URL is also available from the "Code" button on the project page in github).
git clone git@github.com:sandialabs/ionsim.git

# Create a virtual environment: 
python -m venv myvenv

# Activate the virtual environment.
source myvenv/bin/activate

# Install the project and its dependencies in editable mode. This means any changes to
# the code in this directory will be reflected when running programs in this virtual
# environment without reinstalling ionsim.
pip install -e ionsim
```

## Installation 

### Create a virtual environment

First, make sure you have a version of Python downloaded from
python.org, homebrew, apt, yum, etc. depending on your system. IonSim
is intended to work with all current versions of Python, but support
for newer releases may lag by a few months.

#### Windows virtual environments

Assuming the binary downloaded to
`~/AppData/Local/Programs/Python/Python313/python.exe`, run the
following commands (change myvenv to the name you want to call the
virtual environment):

```bash
# Create the virtual environment
~/AppData/Local/Programs/Python/Python313/python.exe -m venv myvenv

# Activate the virtual environment.
source myvenv/Scripts/activate
```

#### Linux/Mac virtual environments

Exact instructions may vary by system.

Typing `python` or `python3` at the command prompt will use the
system's default Python. If you want a more recent version, install it
with your system's package manager. It will then often be called
something like `python313`. Run the following from the command line:

```bash
# Create the virtual environment. Use the python version (python, python3,
# python313, etc.) you wish to the virtual environment to have.
python -m venv myvenv

# Activate the virtual environment.
source myvenv/bin/activate
```


### Install IonSim

In a bash terminal, activate your virtual environment. Now type the
following commands to download then install IonSim. This will create a
directory called ionsim in the current directory.

```bash
# Clone the repo (the below URL is also available from the "Code" button on the project page in github: https://github.com/sandialabs/ionsim#).
git clone https://github.com/sandialabs/ionsim.git 

# Install the project and its dependencies in editable mode. This means any changes to
# the code in this directory will be reflected when running programs in this virtual
# environment without reinstalling ionsim.
pip install -e ionsim
```


### Example Usage
In the IonSim project directory, navigate to the `examples` directory 
and use your version of python to run the MS gate example. 
This represents a coherent simulation of a Molmer-Sorenson gate:

```bash
# Go to the ionsim directory
cd examples/

# "Check out" the branch with gate set tomography analysis capabilities:
python example_simulated_multimode_MS_gate.py
```
