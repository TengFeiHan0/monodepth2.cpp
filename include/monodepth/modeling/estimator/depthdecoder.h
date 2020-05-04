#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include "backbone/backbone.h"

namespace monodepth{
    namespace modeling{
        struct DepthDecoderImpl : torch::nn::Module {
            public:
                DepthDecoderImpl(); 
                
                template<typename T>
                T forward(torch::Tensor x);

            private:
                Backbone backbone;     
                
        };
        TORCH_MODULE(DepthDecoder);

        template<>
        std::vector<torch::Tensor> MonodepthImpl::forward(torch::Tensor x);

        template<>
        torch::Tensor MonodepthImpl::forward(torch::Tensor x);

        DepthDecoder BuildDepthDecoderModule();
    }
}