IonSim
=======

## Quickstart

If you already have a gitlab-ex account and a virtual environment
capable of installing packages from pypi.org, follow these
instructions in a bash terminal with the appropriate virtual
environment active. If not, the full instructions are below.

```bash
# Clone the repo (the below URL is also available from the "Code" button on the project page in gitlab).
git clone git@gitlab-ex.sandia.gov:bruzic/ionsim.git

# Install the project and its dependencies in editable mode. This means any changes to
# the code in this directory will be reflected when running programs in this virtual
# environment without reinstalling ionsim.
pip install -e ionsim
```

## Installation on SRN

Installation on the SRN is largely a matter of dealing with the proxy server and properly setting up your gitlab-ex account.

### Create an ssh key

**These instructions require a bash shell. On Windows we recommend
using git bash, which comes with a download of git.**

See [gitlab-ex's ssh
instructions](https://gitlab-ex.sandia.gov/help/user/ssh.md) for a
thorough and up-to-date explanation of ssh keys. The following
instructions have been known to work but may not apply to all systems
or for all time.

In a bash terminal, run `ssh-keygen -t rsa` and accept all the default options
by pressing enter. This will create a private/public key pair (by
default in `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`), and we will copy
the public key into gitlab in a later step.

### Create a virtual environment (optional)

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

### Set up Nexus (Optional)

**Before performing these commands, try the following `pip` command. If
no errors occur, you do not need to set up Nexus.**

```bash
# Test connectivity to pypi.org
pip install --dry-run requests
```

In the past, Sandia's network proxy made it difficult to use pypi.org,
the general source of all Python packages. As a workaround, Sandia
maintains a mirror inside the SRN, `nexus.web.sandia.gov`. Use the
below bash commands to configure pip to use Nexus. Make sure you have
activated the virtual environment created above.

```bash
# Unset any bash environment variables for the proxy
unset http_proxy
unset https_proxy

# Unset any previous sandia proxy configuration in pip. It is ok if this command errors.
pip config unset global.proxy

# Set Sandia Nexus pip configuration variables
pip config set global.index "https://nexus.web.sandia.gov/repository/pypi-proxy/pypi"
pip config set global.index-url "https://nexus.web.sandia.gov/repository/pypi-proxy/simple"

# Add trusted hosts if you get certificate errors.
pip config set global.trusted-host "pypi.org files.pythonhosted.org nexus.web.sandia.gov resnexus.web.sandia.gov"
```

### Upload ssh key to gitlab

Make sure you have a gitlab-ex account and have been added to the
ionsim project. In Firefox, make sure you type
`https://gitlab-ex.sandia.gov` rather than simply
`gitlab-ex.sandia.gov` to avoid a server timeout.

Once logged into the gitlab web site, click on your profile picture,
then preferences. On the left margin there will be an SSH Keys
option. Click that, then the "Add new key" button. Copy the key from
earlier (located in `~/.ssh/id_rsa.pub`) into the "Key" text
box. Click the "Add key" button.

Test your connection by running `ssh -T git@gitlab-ex.sandia.gov`.
You should see a welcome message without being prompted for a password.

If there is a problem with your connection, as has sometimes happened when running the above instruction on a Mac, try adding the following lines to your `~/.ssh/config` file:

```
PubkeyAcceptedAlgorithms +ssh-rsa
IdentityFile ~/.ssh/id_rsa
```

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
