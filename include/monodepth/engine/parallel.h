#pragma once
#include <torch/torch.h>
#ifdef WITH_CUDA
#include <torch/nn/parallel/data_parallel.h>
#endif
#include "datasets/cityscapes_datasets.h"
#include "estimator/monodepth2.h"

namespace monodepth{
namespace engine{

std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<monodepth::modeling::SelfDepthModel>& modules,
    const std::vector<torch::Tensor> &inputs,
    const std::vector<monodepth::data::DICT> &targets,
    const torch::optional<std::vector<torch::Device>>& devices = torch::nullopt);


std::map<std::string, torch::Tensor> data_parallel(
    std::shared_ptr<monodepth::modeling::SelfDepthModel> &module,
    torch::Tensor input,
    monodepth::data::DICT &target,
    torch::optional<std::vector<torch::Device>> devices = torch::nullopt,
    torch::optional<torch::Device> output_device = torch::nullopt,
    int64_t dim = 0);

}
}