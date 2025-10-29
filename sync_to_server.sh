#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Syncing from '$SCRIPT_DIR/*'"
rsync --exclude="trades" --exclude="bots" --exclude="__pycache__" --exclude=".*" --exclude="logs" --exclude="data" -r "$SCRIPT_DIR/" restic-phil:/mnt/raid2017/pi/pumpfun-bonkfun-bot/

