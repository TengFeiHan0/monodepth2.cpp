#pragma once
#include "samplers/samplers.h"
#include <torch/data/samplers/base.h>

#include "datasets/cityscapes_datasets.h"
namespace monodepth{
namespace data{

std::shared_ptr<torch::data::samplers::Sampler<>> make_data_sampler(int dataset_size, bool shuffle, bool is_distributed);
std::vector<int> _quantize(std::vector<int> bins, std::vector<int> x);
// std::vector<float> _compute_aspect_ratios(COCODataset dataset);
std::shared_ptr<torch::data::samplers::Sampler<>> make_batch_data_sampler(CityScapesDataset dataset, 
                                                                          bool is_train,
                                                                          bool is_distributed,
                                                                          int start_iter);

}
}