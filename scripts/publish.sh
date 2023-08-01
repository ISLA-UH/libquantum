#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "usage: ./publish.sh <token>"
  exit 1
fi

TOKEN=${1}

set -o nounset
set -o errexit
set -o xtrace

cd ..
# Build the distributions
#python3 setup.py sdist bdist_wheel
python3 -m build .

# Upload the distributions to PyPi
twine upload -r pypi -u __token__ -p ${TOKEN} --skip-existing dist/*

# Create a git tag for this version
VERSION="v$(python -c 'import toml; print(toml.load("pyproject.toml")["project"]["version"])')"
git tag -a ${VERSION} -m"Release ${VERSION}"
git push origin ${VERSION}
