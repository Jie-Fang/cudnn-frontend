#!/bin/bash

set -e

DATE_FOLDER=`echo $(date '+%Y-%m-%d')`

for version in cp312 cp311 cp310 cp39 cp38
do
    echo "Building for ${version}"
    /opt/python/${version}-${version}/bin/python -m venv ${version}_env
    source ${version}_env/bin/activate
    CMAKE_BUILD_PARALLEL_LEVEL=8 /opt/python/${version}-${version}/bin/python -m pip wheel --no-deps . -w /wheels/${version} -v
    deactivate
    auditwheel repair /wheels/${version}/*.whl -w many_linux_wheels/
    wheel=`ls many_linux_wheels/*${version}*.whl`
    wheel_name=`echo ${wheel} | cut -d / -f2`
    if [[ $CI_COMMIT_BRANCH == "main" ]]; then
        echo "main branch" 
        curl -u agopal:$JFROG_API_KEY -T  ${wheel} https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/cudnn_frontend/main/${DATE_FOLDER}/${wheel_name}
    elif [[ $CI_COMMIT_BRANCH == "develop" ]]; then 
        echo "develop branch"
        curl -u agopal:${JFROG_API_KEY} -T  ${wheel} https://urm.nvidia.com/artifactory/hw-cudnn-generic/CUDNN/cudnn_frontend/develop/latest/${wheel_name}
    else 
       echo $CI_COMMIT_BRANCH
       echo "Not posting to artifactory"
    fi
done
