#pragma once
#include <torch/torch.h>
#include "estimator/depthdecoder.h"
#include "estimator/posedecoder.h"

namespace monodepth{
    namespace modeling{
        
        class MonodepthImpl : public torch::nn::Module{

          public:
            MonodepthImpl();

            template <typename T>
            T forward(torch::Tensor x);

            
            private:
            DepthDecoder depthdecoder;
            PoseDecoder posedecoder;

           
            
        };

        template<>
        std::vector<torch::Tensor> MonodepthImpl::forward(torch::Tensor x);
        
        template<>
        torch::Tensor MonodepthImpl::forward(torch::Tensor x);

        TORCH_MODULE(Monodepth);

        
    }
}