#!/usr/bin/env bash

# Set the script to fail if any command fails.
set -euo pipefail

# Set target python version from the .python-version file.
PYTHON_VERSION=$(cat .python-version)

# Check if target python version has been installed by pyenv. If not, install it.
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    pyenv install "$PYTHON_VERSION"
fi

# To address the issue that poetry will stuck at the keyring confirmation step.
# Reproduce: `poetry install -vvv`
# https://github.com/python-poetry/poetry/issues/8623#issuecomment-1793624371
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
# keyring --disable

# Set local python version to the target version.
pyenv local "$PYTHON_VERSION"

# Set poetry virtualenv to the target version.
poetry env use "$PYTHON_VERSION"

# Install dependencies.
poetry install

poetry run jupyter lab --no-browser --port=8888
