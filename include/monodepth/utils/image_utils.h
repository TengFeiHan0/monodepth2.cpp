#include <torch/torch.h>
#include <iostream>
namespace monodepth{
    namespace utils{
        torch::Tensor interpolate_image(torch::Tensor image, std::vector<int64_t> &shape);
        
        std::vector<torch::Tensor> match_scales(torch::Tensor image, std::vector<torch::Tensor> &targets, int num_scales);
        
        torch::Tensor SSIM(torch::Tensor x, torch::Tensor y, float C1, float C2);

        torch::nn::AvgPool2d pool2d{nullptr};

        torch::nn::ReflectionPad2d refl{nullptr};
    }
}