#!/usr/bin/env bash

# link ./dataset to $HOME/.dataset directory via symbolic link
printf "Creating symbolic link to dataset directory..."
# get directory name of current file
curdir=$(dirname "$0")
printf '\nln -s ${curdir}/dataset $HOME/.dataset\n'
ln -s ${curdir}/dataset $HOME/.dataset