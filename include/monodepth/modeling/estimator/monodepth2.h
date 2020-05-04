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
            T forward(torch::Tensor x );

            std::map<std::string, torch::Tensor> compute_losses(std::map<std::string, torch::Tensor> &inputs, std::map<std::string, torch::Tensor> &outputs);
            
            private:
            DepthDecoder depthdecoder;
            PoseDecoder posedecoder;

            void generate_image_pred(std::map<std::string, torch::Tensor> &inputs, std::map<std::string, torch::Tensor> &outputs);

            torch::Tensor compute_reprojection_loss(torch::Tensor pred, torch::Tensor target);
            
        };
        TORCH_MODULE(Monodepth);

        template<>
        std::vector<torch::Tensor> MonodepthImpl::forward(torch::Tensor x);
        
        template<>
        torch::Tensor MonodepthImpl::forward(torch::Tensor x);

        class SSIMImpl : public torch::nn::Module{
            public:
              SSIMImpl();

              torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

        };
        TORCH_MODULE(SSIM);

    
    }
}