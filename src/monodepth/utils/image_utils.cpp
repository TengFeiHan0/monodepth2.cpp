#include "image_utils.h"
#include <torch/torch.h>

namespace monodepth{
    namespace utils{

        torch::Tensor interpolate_image(torch::Tensor image,c10::IntArrayRef &shape){

            if(shape == image.sizes()){
                return image;
            }else{
                int64_t h, w = shape[2],shape[3];
                return torch::nn::functional::interpolate(image, torch::nn::functional::InterpolateFuncOptions().size({h,w}).mode(torch::kBilinear));
            }

        };

        std::vector<torch::Tensor> match_scales(torch::Tensor image, std::vector<torch::Tensor> &targets, int num_scales){
            std::vector<torch::Tensor> images;

            for(int i = 0; i < num_scales; ++i){
                auto target_shape = targets[i].sizes();
                if(image.sizes() == target_shape){
                    images.push_back(image);
                }else{
                    images.push_back(interpolate_image(image, target_shape));
                }
            }

        }
        
        torch::Tensor SSIM(torch::Tensor x, torch::Tensor y, float C1, float C2){
            
            torch::nn::AvgPool2d pool2d{nullptr};

            torch::nn::ReflectionPad2d refl{nullptr};

            pool2d = torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions({3,3}).stride({1,1}));

            refl= torch::nn::ReflectionPad2d(
            torch::nn::ReflectionPad2dOptions({1,1,1,1}));

            x = refl->forward(x);
            y = refl->forward(y);

            auto mu_x = pool2d->forward(x);
            auto mu_y = pool2d->forward(y);

            auto sigma_x = pool2d->forward(x*x) - mu_x*mu_x;  
            auto sigma_y = pool2d->forward(y*y) - mu_y*mu_y;
            auto sigma_xy = pool2d->forward(x*y) - mu_x*mu_y;

            auto SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
            auto SSIM_d = (mu_x*mu_x+ mu_y*mu_y + C1) * (sigma_x + sigma_y + C2);

            return torch::clamp((1 - SSIM_n/SSIM_d)/2, 0,1);
        }

    }
}