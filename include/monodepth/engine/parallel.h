// #pragma once
// #include <torch/torch.h>
// #ifdef WITH_CUDA
// #include <torch/nn/parallel/data_parallel.h>
// #endif



// namespace monodepth{
// namespace engine{

// std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
//     std::vector<monodepth::modeling::Monodepth>& modules,
    
//     const torch::optional<std::vector<torch::Device>>& devices = torch::nullopt);


// std::pair<torch::Tensor, std::map<std::string, torch::Tensor>> data_parallel(
//     monodepth::modeling::Monodepth module,
//     torch::optional<std::vector<torch::Device>> devices = torch::nullopt,
//     torch::optional<torch::Device> output_device = torch::nullopt,
//     int64_t dim = 0);

// }
// }