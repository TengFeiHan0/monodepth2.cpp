
#include <torch/torch.h>
#include <iostream>
#include <memory>
namespace monodepth{
  
    namespace utils{

        class Camera: public torch::nn::Module{
            public:
              Camera(torch::Tensor K, torch::Tensor Tcw);

              std::shared_ptr<Camera> scaled(float scale_factor);
              
              torch::Tensor reconstruct(torch::Tensor depth, char frame);

              torch::Tensor project(torch::Tensor X, char frame);


            private:
              torch::Tensor K_;
              torch::Tensor Tcw_;
              
        };

        

        torch::Tensor scale_intrinsic(torch::Tensor K, float x_scale, float y_scale);

        torch::Tensor view_synthesis( torch::Tensor ref_image, torch::Tensor depth, std::shared_ptr<Camera> &ref_cam, std::shared_ptr<Camera> &cam);

        
    }
}