#include "parallel.h"
#include <iostream>


namespace monodepth{
namespace engine{

#ifdef WITH_CUDA

std::vector<std::map<std::string, torch::Tensor>> parallel_apply(
    std::vector<torch::nn::ModuleHolder<monodepth::modeling::SelfDepthModel>>& modules,
    const std::vector<torch::Tensor> &inputs,
    const std::vector<monodepth::data::DICT> &targets,
    const torch::optional<std::vector<torch::Device>>& devices) {
    TORCH_CHECK(
        modules.size() == inputs.size(), "Must have as many inputs as modules");
    if (devices) {
      TORCH_CHECK(
          modules.size() == devices->size(),
          "Must have as many devices as modules");
    }

  std::vector<std::map<std::string, torch::Tensor>> outputs(modules.size());
  std::mutex mutex;

  // std::exception_ptr can be passed between threads:
  // > An instance of std::exception_ptr may be passed to another function,
  // > possibly on another thread, where the exception may be rethrown [...].
  // https://en.cppreference.com/w/cpp/error/exception_ptr
  std::exception_ptr exception;
  at::parallel_for(
      /*begin=*/0,
      /*end=*/modules.size(),
      /*grain_size=*/1,
      [&modules,&devices, &outputs, &mutex, &exception](
          int64_t index, int64_t stop) {
        for (; index < stop; ++index) {
          try {
            std::map<std::string, torch::Tensor> loss_map = modules[index]->forward<std::map<std::string, torch::Tensor>>(inputs[index], targets[index]);
          
            loss_map = loss_map.to(devices ? (*devices)[index] : inputs[index].device());
            std::lock_guard<std::mutex> lock(mutex);
            outputs[index] = loss_map;
          } catch (...) {
            std::lock_guard<std::mutex> lock(mutex);
            if (!exception) {
              exception = std::current_exception();
            }
          }
        }
      });

  if (exception) {
    std::rethrow_exception(exception);
  }

  return outputs;
}
#endif

std::map<std::string, torch::Tensor> data_parallel(
    std::shared_ptr<monodepth::modeling::SelfDepthModel> &module,
    torch::Tensor input, 
    monodepth::data::DICT &target,
    torch::optional<std::vector<torch::Device>> devices,
    torch::optional<torch::Device> output_device,
    int64_t dim) {
  if (!devices) {
    const auto device_count = torch::cuda::device_count();

    TORCH_CHECK(
        device_count > 0, "Expected at least one CUDA device to be available");
    devices = std::vector<torch::Device>();
    devices->reserve(device_count);
    for (size_t index = 0; index < device_count; ++index) {
      devices->emplace_back(torch::kCUDA, index);
    }
  }
  if (!output_device) {
    output_device = devices->front();
  }

  if (devices->size() >= 1) {
    // torch::Tensor loss = torch::zeros({1}).to(devices->front());
    module->to(devices->front());
    input = input.to(devices->front());

    target["tensor_pre"].cuda();
    target["tensor_cur"].cuda();
    target["tensor_next"].cuda();
    target["intrinsics"].cuda();
    
    
    auto loss_map = module->forward<std::map<std::string, torch::Tensor>>(input, target).to(*output_device);

    return loss_map;
  }

#ifdef WITH_CUDA
  torch::autograd::Scatter scatter(*devices, /*chunk_sizes=*/torch::nullopt, dim);
  
  
  auto scattered_inputs = torch::fmap<torch::Tensor>(scatter.apply({std::move(input)}));
  
  auto scattered_targets = torch::fmap<monodepth::data::DICT>(scatter.apply({std::move(targets)}));
  auto replicas = torch::nn::parallel::replicate<monodepth::modeling::SelfDepthModel>(module, *devices);
  auto outputs = parallel_apply(replicas, scattered_inputs, scattered_targets, *devices);
 
  return torch::autograd::Gather(*output_device, dim)
      .apply(torch::fmap<torch::autograd::Variable>(std::move(outputs)))
      .front();
#else
  AT_ERROR("data_parallel not supported without CUDA");
  return  std::map<std::string, torch::Tensor>{};
#endif
}

}
}