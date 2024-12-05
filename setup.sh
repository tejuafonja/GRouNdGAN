#!/bin/bash

SOURCE_PATH="${BASH_SOURCE[0]:-${(%):-%x}}"

RELATIVE_PATH="$(dirname "$SOURCE_PATH")"
ABSOLUTE_PATH="$(realpath "${RELATIVE_PATH}")"

source "${ABSOLUTE_PATH}"/config.sh
source "${ABSOLUTE_PATH}"/modules.sh

# python3.9 
# PYTHON_PATH="/p/project1/hai_diffprivtab/local/python3.9/bin/python3.9"
PYTHON_PATH=/usr/bin/python3.9 
$PYTHON_PATH -m venv --prompt "$ENV_NAME" --system-site-packages "${ENV_DIR}"

source "${ABSOLUTE_PATH}"/activate.sh

# python3.9 
# $PYTHON_PATH -m pip install -r "${ABSOLUTE_PATH}"/requirements.txt 
