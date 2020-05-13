#include <torch/torch.h>


namespace monodepth{
    namespace data{
        struct CameraImpl: public torch::nn::Module{
            public:
              CameraImpl(torch::Tensor K, torch::Tensor Tcw);
            
              
              

        };
        TORCH_MODULE(Camera);

        torch::Tensor view_synthesis( torch::Tensor ref_image, torch::Tensor depth, Camera &ref_cam, Camera &cam);

        
    }
}