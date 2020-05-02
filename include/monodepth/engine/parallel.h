#pragma once
#include <torch/torch.h>
#ifdef WITH_CUDA
#include <torch/nn/parallel/data_parallel.h>
#endif
#include <modeling.h>
#include <image_list.h>
#include <bounding_box.h>


namespace monodepth{
namespace engine{

std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<monodepth::modeling::GeneralizedRCNN>& modules,
    const std::vector<monodepth::structures::ImageList>& inputs,
    const std::vector<std::vector<monodepth::structures::BoxList>>& targets,
    const torch::optional<std::vector<torch::Device>>& devices = torch::nullopt);


std::pair<torch::Tensor, std::map<std::string, torch::Tensor>> data_parallel(
    monodepth::modeling::GeneralizedRCNN module,
    monodepth::structures::ImageList images, 
    std::vector<monodepth::structures::BoxList> targets,
    torch::optional<std::vector<torch::Device>> devices = torch::nullopt,
    torch::optional<torch::Device> output_device = torch::nullopt,
    int64_t dim = 0);

}
}