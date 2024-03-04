# See https://github.com/pypa/manylinux
FROM quay.io/pypa/manylinux_2_28_x86_64

ARG CUDA_VERSION="12-3"

# Install cudnn and cuda toolkit
RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
RUN dnf clean all
RUN dnf -y module install nvidia-driver:open-dkms --skip-broken
RUN dnf -y install cuda-toolkit-${CUDA_VERSION}
RUN dnf -y install cudnn

# RUN /opt/python/cp312-cp312/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp312
# RUN /opt/python/cp311-cp311/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp311
# RUN /opt/python/cp310-cp310/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp310
# RUN /opt/python/cp39-cp39/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp39
# RUN /opt/python/cp38-cp38/bin/python -m pip wheel --no-deps /builds/cudnn/cudnn_frontend/ -w /wheels/cp8

# RUN auditwheel repair /wheels/cp312/*.whl -w /output/
# RUN auditwheel repair /wheels/cp311/*.whl -w /output/
# RUN auditwheel repair /wheels/cp310/*.whl -w /output/
# RUN auditwheel repair /wheels/cp39/*.whl -w /output/
# RUN auditwheel repair /wheels/cp38/*.whl -w /output/