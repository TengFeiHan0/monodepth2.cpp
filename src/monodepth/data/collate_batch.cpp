#include "collate_batch.h"
#include <iostream>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include <tuple>

namespace monodepth{
namespace data{

BatchCollator::BatchCollator(int size_divisible) :size_divisible_(size_divisible){}

batch BatchCollator::apply_batch(std::vector<torch::data::Example<torch::Tensor, DICT>> examples)
{
  std::vector<torch::Tensor> tensors;
  std::vector<DICT> imagedata;
  tensors.reserve(examples.size());
  
  for(auto& example : examples){
   
    tensors.push_back(example.data);
    imagedata.push_back(example.target);
    
  }
  
  return std::make_tuple(tensors, imagedata);
}

}
}