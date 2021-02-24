#!/usr/bin/env bash
# Run unit tests provided for the libquantum codebase.

set -o nounset
set -o errexit
set -o xtrace

cd ..

python3 -m unittest tests/test_quantum.py
