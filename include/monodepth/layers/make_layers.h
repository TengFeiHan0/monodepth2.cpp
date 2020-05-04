#pragma once
#include <torch/torch.h>

namespace rcnn {
namespace layers{

torch::nn::Sequential ConvWithKaimingUniform(/*NO GN use_gn=false, */
                                             bool use_elu,
                                             int64_t in_channels,
                                             int64_t out_channels,
                                             );

torch::nn::Linear MakeFC(int64_t dim_in, int64_t hidden_dim);

torch::nn::Sequential MakeConv3x3(int64_t in_channels,
                                  int64_t out_channels,
                                  bool use_refl=true,
                                  bool kaiming_init=true);

} // namespace layers
} // namespace rcnn
