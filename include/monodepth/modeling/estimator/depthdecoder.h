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

                std::vector<torch::nn::Sequential> up_Conv_0;
                std::vector<torch::nn::Sequential> up_Conv_1;
                std::vector<torch::nn::Sequential> disp_Conv;

                std::vector<int64_t> num_ch_enc;
                std::vector<int64_t> num_ch_dec;
                int64_t num_ch_in;
                int64_t num_ch_out;
                bool use_skips;


                
        };
        

        template<>
        std::vector<torch::Tensor> DepthDecoderImpl::forward(torch::Tensor x);

        template<>
        torch::Tensor DepthDecoderImpl::forward(torch::Tensor x);

        TORCH_MODULE(DepthDecoder);

        DepthDecoder BuildDepthDecoderModule();
    }
}