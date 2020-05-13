#include <map>
#include <string>
#include <vector>
#include <torch/torch.h>

namespace monodepth{
    namespace modeling{
    
        torch::Tensor compute_reprojection_loss(torch::Tensor pred, torch::Tensor target);

        torch::Tensor get_smooth_loss(torch::Tensor disp, torch::Tensor img);

        std::map<std::string, torch::Tensor> compute_losses(std::map<std::string, torch::Tensor> &inputs, std::map<std::string, torch::Tensor> &outputs);
        
        
    }
}