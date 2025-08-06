IonSim
=======

## Installation on SRN

Installation on the SRN is largely a matter of dealing with the proxy server and properly setting up your gitlab-ex account.

### Create an ssh key

In a bash terminal (download git bash on Windows), run ssh-keygen and
accept all the default options by pressing enter. This will create a
private/public key pair (by default in `~/.ssh/rsa_id` and
`~/.ssh/rsa_id.pub`), and we will copy the public key into gitlab.

### Create a virtual environment (optional)

First, make sure you have a version of Python downloaded from
python.org, homebrew, apt, yum, etc. depending on your system. IonSim
is intended to work with all current versions of Python, but support
for newer releases may lag by a few months.

#### Windows

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

#### Linux/Mac

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

### Set up Nexus

Sandia's network proxy makes it difficult to use pypi.org, the general
source of all Python packages. As a workaround, Sandia maintains a
mirror inside the SRN, `nexus.web.sandia.gov`. Use the below bash
commands to configure pip. Make sure you have activated the virtual
environment created above.

```bash
# Unset any bash environment variables for the proxy
unset http_proxy
unset https_proxy

# Unset any previous sandia proxy configuration in pip
pip config unset global.proxy

# Set Sandia Nexus pip configuration variables
pip config set global.index "https://nexus.web.sandia.gov/repository/pypi-proxy/pypi"
pip config set global.index-url "https://nexus.web.sandia.gov/repository/pypi-proxy/simple"

# If you want to add our private Nexus.
# However, pip will ask for a user and password and you must be a member of wg-ioncontrol-users or wg-qscout
pip config set global.extra-index-url "https://resnexus.web.sandia.gov/repository/iontraps/simple"

# Add trusted hosts if you get certificate errors.
pip config set global.trusted-host "pypi.org files.pythonhosted.org nexus.web.sandia.gov resnexus.web.sandia.gov"
```

### Upload ssh key to gitlab

Make sure you have a gitlab-ex account and have been added to the
ionsim project. Also, make sure you type
`https://gitlab-ex.sandia.gov` rather than simply
`gitlab-ex.sandia.gov` to avoid a server timeout.

Once logged into the gitlab web site, click on your profile picture,
then preferences. On the left margin there will be an SSH Keys
option. Click that, then the "Add new key" button. Copy the key from
earlier (located in `~/.ssh/id_rsa.pub`) into the "Key" text
box. Click the "Add key" button.

### Install IonSim

In a bash terminal, activate your virtual environment. Now type the
following commands to download then install IonSim. This will create a
directory called ionsim in the current directory.

```bash
# Clone the repo (the below URL is also available from the "Code" button on the project page in gitlab).
git clone git@gitlab-ex.sandia.gov:bruzic/ionsim.git

# Install the project and its dependencies in editable mode. This means any changes to
# the code in this directory will be reflected when running programs in this virtual
# environment without reinstalling ionsim.
pip install -e ionsim
```
