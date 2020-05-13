
#include <torch/torch.h>

namespace monodepth{
    namespace modeling{

        class MultiViewPhotometricLossImpl : public torch::nn::Module{

            public:
              MultiViewPhotometricLossImpl();

              std::map<std::string, torch::Tensor> forward(torch::Tensor image, std::vector<torch::Tensor>&context, 
                                     std::vector<torch::Tensor> &inv_depths, torch::Tensor K, torch::Tensor ref_K, std::vector<torch::Tensor> &poses);

            private:
              std::vector<torch::Tensor> warp_ref_image(std::vector<torch::Tensor> &inv_depths, torch::Tensor ref_image, torch::Tensor K, torch::Tensor ref_K, torch::Tensor pose);

              std::vector<torch::Tensor> calc_photometric_loss(std::vector<torch::Tensor> &t_est, std::vector<torch::Tensor> &images);

              torch::Tensor reduce_photometric_loss(std::vector<std::vector<torch::Tensor>> &photometric_losses);

              torch::Tensor reduce_function(std::vector<torch::Tensor> &losses);

              torch::Tensor calc_smooth_loss(std::vector<torch::Tensor> &inv_depths, std::vector<torch::Tensor> &images);

        };

        TORCH_MODULE(MultiViewPhotometricLoss);

        



    }
}