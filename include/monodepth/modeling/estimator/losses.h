#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace modeling{
    namespace modeling{
    
        torch::Tensor compute_reprojection_loss(torch::Tensor pred, torch::Tensor target);

        std::map<std::string, torch::Tensor> compute_losses(std::map<std::string, torch::Tensor> &inputs, std::map<std::string, torch::Tensor> &outputs);
        
        class SSIMImpl : public torch::nn::Module{
            public:
              SSIMImpl();

              torch::Tensor forward(torch::Tensor x, torch::Tensor y);
            private:
              torch::nn::AvgPool2d mu_x_pool{nullptr};
              torch::nn::AvgPool2d mu_y_pool{nullptr};
              torch::nn::AvgPool2d sig_x_pool{nullptr};
              torch::nn::AvgPool2d sig_y_pool{nullptr};
              torch::nn::AvgPool2d sig_xy_pool{nullptr};
              torch::nn::ReflectionPad2d refl{nullptr};

              int64_t C1;
              int64_t C2;  

        };
        TORCH_MODULE(SSIM);
    }
}