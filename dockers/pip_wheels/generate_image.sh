docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:pip_wheels_cuda_13.0_cudnn_9.15.0 -f dockers/pip_wheels/Dockerfile --build-arg CUDA_VERSION=13-0 --build-arg CUDNN_VERSION=9.15.0 .
docker buildx build --platform linux/arm64 --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:pip_wheels_aarch64_cuda_13.0_cudnn_9.15.0 -f dockers/pip_wheels/Dockerfile.aarch64 --build-arg CUDA_VERSION=13-0 --build-arg CUDNN_VERSION=9.15.0 . --push

docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:pip_wheels_cuda_13.0_cudnn_9.15.0
