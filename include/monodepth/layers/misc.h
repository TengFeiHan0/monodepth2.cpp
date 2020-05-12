#include <torch/torch.h>
#include <vector>

namespace monodepth{
    namespace layers {
        torch::Tensor transformation_from_parameters(torch::Tensor axisangle, torch::Tensor translation, bool invert);
        
        torch::Tensor get_translation_matrix(torch::Tensor translation);
        torch::Tensor rot_from_axisangle(torch::Tensor axis);

        std::tuple<torch::Tensor, torch::Tensor> disp_to_depth(torch::Tensor, int64_t min_depth, int64_t max_depth);
        
        class BackprojectDepthImpl : public torch::nn::Module {
            public:
              BackprojectDepthImpl(int batch_size, int height, int width);
            private:
              int batch_size_;
              int height_;
              int width_;  

        };
        TORCH_MODULE(BackprojectDepth);

        class Project3DImpl : public torch::nn::Module {
            public:
              Project3DImpl(int batch_size, int height, int width, int64_t eps);

              torch::Tensor forward(torch::Tensor points, torch::Tensor K, torch::Tensor T);
            private:
              int batch_size_;
              int height_;
              int width_;
              int64_t eps_;  

        };
        TORCH_MODULE(Project3D);
    }
}