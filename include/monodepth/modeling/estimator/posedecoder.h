#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include "backbone/backbone.h"

namespace monodepth{
    namespace modeling{
        struct PoseDecoderImpl : public torch::nn::Module{

            public:
              PoseDecoderImpl();

              
            std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> forward(std::vector<torch::Tensor> inputs);
            private:
              //note: all variables should be initialized with nullptr(Backbone backbone{nullptr}), 
              //instead of Backbone backbone;
              Backbone backbone{nullptr};       
              torch::nn::Conv2d squeeze_0{nullptr};
              torch::nn::Conv2d pose_0{nullptr};
              torch::nn::Conv2d pose_1{nullptr};
              torch::nn::Conv2d pose_2{nullptr}; 
              int64_t num_frames{0};

              int64_t out_channels{0};

        };

        TORCH_MODULE(PoseDecoder);

        
        PoseDecoder BuildPoseDecoderModule();

    }
}