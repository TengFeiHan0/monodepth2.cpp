#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include "backbone/backbone.h"

namespace monodepth{
    namespace modeling{
        struct PoseDecoder : public torch::nn::Module{

            public:
              PoseDecoderImpl();

              template<typename T>
              T forward(std::vector<torch::Tensor> inputs);
            private:
              Backbone backbone; 
              torch::nn::Conv2d squeeze_0{nullptr};
              torch::nn::Conv2d pose_0{nullptr};
              torch::nn::Conv2d pose_1{nullptr};
              torch::nn::Conv2d pose_2{nullptr}; 
              int64_t num_frames;

              int64_t out_channels;

        };

        TORCH_MODULE(PoseDecoder);

        template<>
        std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PoseDecoderImpl::forward(std::vector<torch::Tensor> inputs);

        PoseDecoder BuildPoseDecoderModule();

    }
}