#!/bin/bash

set -e

for version in cp312 cp311 cp310 cp39 cp38
do
    CMAKE_BUILD_PARALLEL_LEVEL=8 /opt/python/${version}-${version}/bin/python -m pip wheel --no-deps . -w /wheels/${version} -v
    auditwheel repair /wheels/${version}/*.whl -w many_linux_wheels/
done