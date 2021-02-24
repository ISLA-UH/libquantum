#!/usr/bin/env bash

if [ -z "$1" ]
then
  echo "usage: ./publish.sh <user> <password>"
  exit 1
fi

if [ -z "$2" ]
then
  echo "usage: ./publish.sh <user> <password>"
  exit 1
fi

USER=${1}
PASS=${2}

set -o nounset
set -o errexit
set -o xtrace

cd ..
# Build the distributions
python3 setup.py sdist bdist_wheel

# Upload the distributions to PyPi
twine upload -r pypi -u ${USER} -p ${PASS} --skip-existing dist/*

# Create a git tag for this version
VERSION="v$(python3 setup.py --version)"
git tag -a ${VERSION} -m"Release ${VERSION}"
git push origin ${VERSION}
