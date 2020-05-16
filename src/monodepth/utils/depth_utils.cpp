#include <torch/torch.h>

namespace monodepth{
    namespace utils{

        torch::Tensor inv2depths( torch::Tensor depth){

             return 1./ depth.clamp(1e-6, 200.0);
        };

        std::vector<torch::Tensor> inv2depths(std::vector<torch::Tensor> &depths){
            for(auto d: depths){
                  d = inv2depths(d);
            }
        };

        
    }
}