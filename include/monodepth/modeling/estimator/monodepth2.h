#pragma once
#include <torch/torch.h>
#include "estimator/depthdecoder.h"
#include "estimator/posedecoder.h"
#include "estimator/mp_losses.h"
#include "datasets/cityscapes_datasets.h"
namespace monodepth{
    namespace modeling{
        
        class SelfDepthModel : public torch::nn::Module{

          public:
            SelfDepthModel();

            template <typename T>
            T forward(torch::Tensor input, monodepth::data::DICT target);

            
            private:
            DepthDecoder depthdecoder;
            PoseDecoder posedecoder;

            MultiViewPhotometricLoss mp_loss;

           
            
        };

        template<>
        std::map<std::string, torch::Tensor> SelfDepthModel::forward(torch::Tensor input, monodepth::data::DICT target);
        
        template<>
        torch::Tensor SelfDepthModel::forward(torch::Tensor input, monodepth::data::DICT target);

        
        SelfDepthModel BuildModel();

        
    }
}