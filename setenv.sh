#!/bin/bash

full_path=$(realpath "${BASH_SOURCE[0]}")
dir_path=$(dirname "$full_path")
export PYTHONPATH=$dir_path
