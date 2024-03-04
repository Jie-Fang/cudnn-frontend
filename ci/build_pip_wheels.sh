#!/bin/bash

/opt/python/cp312-cp312/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp312 -v
/opt/python/cp311-cp311/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp311 -v
/opt/python/cp310-cp310/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp310 -v
/opt/python/cp39-cp39/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp39 -v
/opt/python/cp38-cp38/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp38 -v

auditwheel repair /wheels/cp312/*.whl -w many_linux_wheels/
auditwheel repair /wheels/cp311/*.whl -w many_linux_wheels/
auditwheel repair /wheels/cp310/*.whl -w many_linux_wheels/
auditwheel repair /wheels/cp39/*.whl -w  many_linux_wheels/
auditwheel repair /wheels/cp38/*.whl -w  many_linux_wheels/
