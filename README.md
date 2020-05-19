# Monodepth2.cpp

## ⏳ Training



**Monocular training:**
```shell

```

 
## 💾 KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
## To Do
- [] add train & inference script
- [] add kitti dataloader
- [] distributed training
- [x] add loss function
- [] add log printing

## References
- [monodepth2](https://github.com/nianticlabs/monodepth2)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [maskrcn-benchmark.cpp](https://github.com/lsrock1/maskrcnn_benchmark.cpp)

# Torchscript demo

1. download related libraries including OpenCV and [libtorch](https://pytorch.org/get-started/locally/).
2. download my converted torchscript model or convert your trained model into torchscript by yourself. If you don't familiar with torchscript currently, please check the offical [docs](https://pytorch.org/tutorials/advanced/cpp_export.html)
3. prepare a sample image and change its path in main.cpp
4. if you don't have available gpus, please annotate CUDA options in  ```CMakeLists.txt ```
## runtime
Model | Language | 3D Packing | Inference time / im |Link
--- |:---:|:---:|:---:|---:
packnet_32 | litorch | Yes |  | [download](https://drive.google.com/file/d/14BLOVAMV5ZQeq7tbI1b6GJ9erkSvCszF/view?usp=sharing)
packnet_32 | python | Yes |  | [download](https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet_MR_selfsup_K.pth.tar)

## dockerfile
If you're familiar with docker, you could run this [project](https://hub.docker.com/repository/docker/tengfei2503/maskrcnn-benchmark.cpp) withour the need of installing those libraries. please remember  install nvidia-docker because our projects needs to use gpus.

## converted models

you could follow ```to_jit.py ``` to create your own torchscript model and use my converted model directly. We provide three different converted models as below.\
[monodepth2](https://drive.google.com/open?id=1kZ0H_dWjjv07TvmgbENdhPnkuKQEXt9g)(`FP32`)\
[packnet-sfm](https://drive.google.com/file/d/1wesXmbRr9z4mTD6Ox7aquOsp4Vom6Dkg/view?usp=sharing)(`FP16`)

## onnx
we also offer a onnx file that could be accerlated with TensorRT. The related demo code will be released soon.\
[packnet-sfm]()(ONNX)

