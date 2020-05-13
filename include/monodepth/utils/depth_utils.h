#include <torch/torch.h>
#include <iostream>

namespace monodepth{
    namespace utils{

       
        torch::Tensor inv2depths( torch::Tensor depth);

        std::vector<torch::Tensor> inv2depths(std::vector<torch::Tensor> &depths);
        



    }
}