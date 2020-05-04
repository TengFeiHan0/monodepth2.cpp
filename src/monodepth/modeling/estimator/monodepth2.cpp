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

        torch::Tensor compute_reprojection_loss(torch::Tensor pred, torch::Tensor target){
            torch::Tensor abs_diff = torch::abs(target - pred);
            auto l1_loss = abs_diff.mean(1, True);

            torch::Tensor ssim_loss = SSIM(pred, target);
            ssim_loss = ssim_loss.mean(1, True);

            torch::Tensor reprojection_loss = 0.85*ssim_loss +0.15*l1_loss;

            return reprojection_loss;

        };
        
            

    }
}