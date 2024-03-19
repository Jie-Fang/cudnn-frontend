# rebuild docker images and tag them
# note that while CUDA and cuDNN versions are fixed,
# apt packages and pypi packages will use the latest versions
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.5.0.96_11.7.1 -f dockers/Dockerfile --build-arg CUDA_VERSION=11.7.1 --build-arg CUDNN_VERSION=8.5.0.96 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.6.0.163_11.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION=11.8.0 --build-arg CUDNN_VERSION=8.6.0.163 --build-arg DLFW_MONTH=22.10 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.7.0.84_11.8.0 -f dockers/Dockerfile --build-arg CUDA_VERSION=11.8.0 --build-arg CUDNN_VERSION=8.7.0.84 --build-arg DLFW_MONTH=23.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.8.1.3_12.0.1 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.0.1 --build-arg CUDNN_VERSION=8.8.1.3 --build-arg DLFW_MONTH=23.03 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.2.26_12.1.0 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.1.0 --build-arg CUDNN_VERSION=8.9.2.26 --build-arg DLFW_MONTH=24.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.5.29_12.2.2 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.2.2 --build-arg CUDNN_VERSION=8.9.5.29 --build-arg DLFW_MONTH=24.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.7.29_12.2.2 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.2.2 --build-arg CUDNN_VERSION=8.9.7.29 --build-arg DLFW_MONTH=24.01 .
docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.0.0.312_12.3.1 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.3.1 --build-arg CUDNN_VERSION=9.0.0.312 --build-arg DLFW_MONTH=24.02 .

docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_12.3.1 -f dockers/Dockerfile --build-arg CUDA_VERSION=12.3.1 --build-arg SKIP_CUDNN=true --build-arg DLFW_MONTH=24.02 .

#############################################
############# RUN WITH CAUTION ##############
#############################################
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.5.0.96_11.7.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.6.0.163_11.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.7.0.84_11.8.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.8.1.3_12.0.1
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.2.26_12.1.0
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.5.29_12.2.2
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_8.9.7.29_12.2.2
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_9.0.0.312_12.3.1

docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:cudnn_12.3.1