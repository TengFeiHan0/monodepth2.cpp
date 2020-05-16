#pragma once
#include <torch/torch.h>


namespace monodepth{
namespace modeling{

class BackboneImpl : public torch::nn::Module{

public:
  explicit BackboneImpl(torch::nn::Sequential backbone, int64_t out_channels);
  std::vector<torch::Tensor> forward(torch::Tensor x);
  int64_t get_out_channels();
  std::shared_ptr<BackboneImpl> clone(torch::optional<torch::Device> device = torch::nullopt) const;

private:
  torch::nn::Sequential backbone_{nullptr};
  int64_t out_channels_{0};
};

TORCH_MODULE(Backbone);

Backbone BuildResnetBackbone();

Backbone BuildBackbone();

}
}