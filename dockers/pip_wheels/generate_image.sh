docker image build --no-cache -t gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:pip_wheels_cuda_12.8_cudnn_9.8.0 -f dockers/pip_wheels/Dockerfile --build-arg CUDA_VERSION=12-8 --build-arg CUDNN_VERSION=9.8.0 .
docker push gitlab-master.nvidia.com:5005/cudnn/cudnn_frontend:pip_wheels_cuda_12.8_cudnn_9.8.0
