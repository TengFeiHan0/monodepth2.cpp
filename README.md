# Monodepth2.cpp
This is a pure C++ implementation of a very popular depth estimation network named monodepth2. This entire project is totally based on Libtorch. 

## torchscript demo

1. download related libraries including OpenCV and [libtorch](https://pytorch.org/get-started/locally/).
2. download my converted torchscript model or convert your trained model into torchscript by yourself. If you don't familiar with torchscript currently, please check the offical [docs](https://pytorch.org/tutorials/advanced/cpp_export.html)
3. prepare a sample image and change its path in main.cpp
4. if you don't have available gpus, please annotate CUDA options in  ```CMakeLists.txt ```
## runtime
Model | Language | 3D Packing | Inference time / im |Link
--- |:---:|:---:|:---:|---:
packnet_32 | litorch | Yes | 328ms | [download](https://drive.google.com/file/d/14BLOVAMV5ZQeq7tbI1b6GJ9erkSvCszF/view?usp=sharing)
packnet_32 | python | Yes | 3.5s | [download](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_K.pth.tar)

## dockerfile
If you're familiar with docker, you could run this [project](https://hub.docker.com/repository/docker/tengfei2503/maskrcnn-benchmark.cpp) withour the need of installing those libraries. please remember  install nvidia-docker because our projects needs to use gpus.

## Converted models

you could follow ```to_jit.py ``` to create your own torchscript model and use my converted model directly. We provide three different converted models as below.\
[monodepth2](https://drive.google.com/open?id=1kZ0H_dWjjv07TvmgbENdhPnkuKQEXt9g)(`FP32`)\
[packnet-sfm](https://drive.google.com/file/d/1wesXmbRr9z4mTD6Ox7aquOsp4Vom6Dkg/view?usp=sharing)(`FP16`)

## ONNX
we also offer a onnx file that could be accerlated with TensorRT. The related demo code will be released soon.\
[packnet-sfm]()(ONNX)
