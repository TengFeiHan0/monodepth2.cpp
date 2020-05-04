#include <torch/torch.h>
#include <iostream>


struct Convblock : torch::nn::Module {
  torch::nn::Conv2d conv1;
  torch::nn::GroupNorm gn1;
  
  Convblock(int64_t inplanes, int64_t planes)
      : conv1(conv_options(inplanes, planes, 3, 1)),
        gn1(16,planes)
        {
    register_module("conv1", conv1);
    register_module("gn1", gn1);
  }

  torch::Tensor forward(torch::Tensor x) {
  
    x = conv1->forward(x);
    x = gn1->forward(x);
    x = torch::elu(x, inplace=true);
    return x;
  }
};