#!/bin/bash

DATA_SOURCE=$HOME/data
DATA_TARGET=$SLURM_TMPDIR/src/
SRC_SOURCE=./
SRC_TARGET=$SLURM_TMPDIR/src/
API_KEYS=$HOME/init-keys.sh

# copy datasets to SLURM_TMPDIR
for SUBDIR in "$@"
do
    if ! [ -d "$DATA_TARGET/$SUBDIR/" ]; then
        mkdir -p $DATA_TARGET/$SUBDIR
        echo "Copying $SUBDIR ..."
        cp -r $DATA_SOURCE/$SUBDIR $DATA_TARGET/
        # extract any archives
        for FILE in "$DATA_TARGET/$SUBDIR/*.tar.gz"
        do
            tar -xzf $FILE -C $DATA_TARGET/$SUBDIR
        done
    fi
done

# load modules
module load python/3.10

# load api keys
source $API_KEYS

# copy source
mkdir -p $SRC_TARGET
cd $SRC_SOURCE
GIT_WORK_TREE=$SRC_TARGET git checkout @{-1} -f
echo "Git: checked out code"
GIT_WORK_TREE=$SRC_TARGET git status
GIT_WORK_TREE=$SRC_TARGET git log -1 --oneline

# make virtual env under .venv
cd $SRC_TARGET
# this converts pyproject.toml from poetry
uvx pdm import pyproject.toml
# install dependencies
uv sync
source $SRC_TARGET/.venv/bin/activate
