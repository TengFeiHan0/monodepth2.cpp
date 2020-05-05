#include"estimator/monodepth2.h"

namespace monodepth{
    namespace modeling{
        MonodepthImpl::MonodepthImpl()
          :posedecoder(register_module("posedecoder", BuildPoseDecoderModule())),
           depthdecoder(register_module("depthdecoder", BuildDepthDecoderModule())){};

        template <>
        torch::Tensor MonodepthImpl::forward( torch::Tensor x){
            assert(!is_training())
            std::vector<torch::Tensor> features = backbone->forward(x);
            std::vector<torch::Tensor> disps = depthdecoder->forward(features);

            return disps[0];
        };

        template <>
        std::vector<torch::Tensor> MonodepthImpl::forward( torch::Tensor x){
            assert(is_training())
            std::vector<torch::Tensor> features = backbone->forward(x);
            std::vector<torch::Tensor> disps = depthdecoder->forward(features);

            return disps;
        };

       

    }
}