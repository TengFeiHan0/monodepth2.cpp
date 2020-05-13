#include <torch/torch.h>

namespace monodepth{
    namespace utils{
        std::vector<torch::Tensor> match_scales(torch::Tensor image, std::vector<torch::Tensor> &targets, int num_scales);
        

    }
}