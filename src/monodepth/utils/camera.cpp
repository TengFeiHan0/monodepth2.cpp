
#include <torch/torch.h>
#include "camera.h"
#include <memory>
namespace monodepth{

    namespace utils{
        Camera::Camera(torch::Tensor K, torch::Tensor Tcw): K_(K), Tcw_(Tcw){}

        std::shared_ptr<Camera> Camera::scaled(float scale_factor){
            K_ = scale_intrinsic(K_, scale_factor, scale_factor);
            std::shared_ptr<monodepth::utils::Camera> res_cam = std::make_shared<Camera>(K_, Tcw_);
            return res_cam;
        }

        torch::Tensor scale_intrinsic(torch::Tensor K, float x_scale, float y_scale){
            K[0][0] *=x_scale;
            K[1][1] *=y_scale;
            K[0][2] = (K[0][2] +0.5)*x_scale -0.5;
            K[1][2] = (K[1][2] +0.5)*y_scale -0.5;

            return K;
        }
      
        torch::Tensor view_synthesis( torch::Tensor ref_image, torch::Tensor depth, std::shared_ptr<Camera> &ref_cam, std::shared_ptr<Camera> &cam){
            //       """
            // Synthesize an image from another plus a depth map.

            // Parameters
            // ----------
            // ref_image : torch.Tensor [B,3,H,W]
            //     Reference image to be warped
            // depth : torch.Tensor [B,1,H,W]
            //     Depth map from the original image
            // ref_cam : Camera
            //     Camera class for the reference image
            // cam : Camera
            //     Camera class for the original image
            // mode : str
            //     Interpolation mode
            // padding_mode : str
            //     Padding mode for interpolation

            // Returns
            // -------
            // ref_warped : torch.Tensor [B,3,H,W]
            //     Warped reference image in the original frame of reference
            // """
            assert(depth.size(1)==1);
            // auto world_points = cam->reconstruct(depth, 'w');
            // auto ref_coords = ref_cam->project(world_points, 'w');

            // return torch::nn::functional::grid_sample(
            //     ref_image, ref_coords, 
            //     torch::nn::functional::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros);
           
        }

        
    }
}