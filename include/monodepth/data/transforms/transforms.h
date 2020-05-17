#pragma once
#include <torch/torch.h>
#include <torch/data/example.h>
#include <torch/data/transforms/tensor.h>
#include <datasets/cityscapes_datasets.h>
namespace monodepth{
namespace data{

template<typename InputType, typename OutputType>
class ImageTransform : public torch::data::transforms::Transform<torch::data::Example<InputType, DICT>, torch::data::Example<OutputType, DICT>>{
  
public:  
  virtual torch::data::Example<OutputType, DICT> operator()(torch::data::Example<InputType, DICT> input) = 0;

  torch::data::Example<OutputType, DICT> apply(torch::data::Example<InputType, DICT> input) override{
    return (*this)(input);
  }
};

using MatToTensorTransform = ImageTransform<cv::Mat, torch::Tensor>;
using MatToMatTransform = ImageTransform<cv::Mat, cv::Mat>;
using TensorToTensorTransform = ImageTransform<torch::Tensor, torch::Tensor>;


class Resize : public MatToMatTransform{

public:
  Resize(int min_size, int max_size):min_size_(min_size), max_size_(max_size){};
  std::pair<int, int> get_size(std::pair<int, int> image_size);
  torch::data::Example<cv::Mat, DICT> operator()(torch::data::Example<cv::Mat, DICT> input) override;

private:
  int min_size_;
  int max_size_;
};

class RandomHorizontalFlip : public MatToMatTransform{

public:
  RandomHorizontalFlip(float prob = 0.5):prob_(prob){};
  torch::data::Example<cv::Mat, DICT> operator()(torch::data::Example<cv::Mat, DICT> input) override;

private:
  float prob_;
};

class RandomVerticalFlip : public MatToMatTransform{

public:
  RandomVerticalFlip(float prob = 0.5):prob_(prob){};
  torch::data::Example<cv::Mat, DICT> operator()(torch::data::Example<cv::Mat, DICT> input) override;

private:
  float prob_;
};

class ToTensor : public MatToTensorTransform{

public:
  torch::data::Example<torch::Tensor, DICT> operator()(torch::data::Example<cv::Mat, DICT> input) override;
};

class Normalize : public TensorToTensorTransform{

public:
  Normalize(torch::ArrayRef<float> mean, torch::ArrayRef<float> stddev, bool to_bgr255);
  torch::data::Example<torch::Tensor, DICT> operator()(torch::data::Example<torch::Tensor, DICT> input) override;

private:
  torch::Tensor mean, stddev;
  bool to_bgr255_;
};

class Compose : public MatToTensorTransform{

public:
  Compose(std::vector<std::shared_ptr<MatToMatTransform>> MtoMtransforms,
                 std::vector<std::shared_ptr<TensorToTensorTransform>> TtoTtransforms);
  torch::data::Example<torch::Tensor, DICT> operator()(torch::data::Example<cv::Mat, DICT> input) override;

private:
  std::vector<std::shared_ptr<MatToMatTransform>> MtoMtransforms_;
  ToTensor to_tensor;
  std::vector<std::shared_ptr<TensorToTensorTransform>> TtoTtransforms_;
};

}//data
}//monodepth