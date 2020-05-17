#pragma once

#include <torch/data/example.h>
#include <torch/data/transforms/collate.h>
#include "datasets/cityscapes_datasets.h"

namespace monodepth{
namespace data{

using batch =std::tuple<std::vector<torch::Tensor>, std::vector<DICT>>;

struct BatchCollator : public torch::data::transforms::Collation<batch, std::vector<torch::data::Example<torch::Tensor, DICT>>>{

  BatchCollator(int size_divisible);
  batch apply_batch(std::vector<torch::data::Example<torch::Tensor, DICT>> examples) override;

  int size_divisible_;

};

}
}