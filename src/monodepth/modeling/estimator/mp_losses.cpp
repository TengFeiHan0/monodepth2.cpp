#include "estimator/mp_losses.h"
#include "camera.h"
#include "depth_utils.h"
#include "image_utils.h"
#include "misc.h"
namespace monodepth{
    namespace modeling{

        MultiViewPhotometricLossImpl::MultiViewPhotometricLossImpl(){};
         
        // """
        //     Warps a reference image to produce a reconstruction of the original one.

        //     Parameters
        //     ----------
        //     inv_depths : torch.Tensor [B,1,H,W]
        //         Inverse depth map of the original image
        //     ref_image : torch.Tensor [B,3,H,W]
        //         Reference RGB image
        //     K : torch.Tensor [B,3,3]
        //         Original camera intrinsics
        //     ref_K : torch.Tensor [B,3,3]
        //         Reference camera intrinsics
        //     pose : Pose
        //         Original -> Reference camera transformation

        //     Returns
        //     -------
        //     ref_warped : torch.Tensor [B,3,H,W]
        //         Warped reference image (reconstructing the original one)
        // """                 
        std::vector<torch::Tensor> MultiViewPhotometricLossImpl::warp_ref_image(std::vector<torch::Tensor> &inv_depths, torch::Tensor ref_image, torch::Tensor K, 
                             torch::Tensor ref_K, torch::Tensor pose){

            auto shape_op  =  ref_image.sizes();

            auto B = shape_op[0], H = shape_op[2], W = shape_op[3];
            int64_t device = ref_image.get_device(); 
            std::vector<monodepth::data::Camera> cams, ref_cams;
            std::vector<torch::Tensor> depths, ref_warped;

            std::vector<torch::Tensor> ref_images = monodepth::utils::match_scales(ref_image, inv_depths, 4);
            for(int i=0; i< 4; i++){
                auto depth_shape = inv_depths[i].sizes();
                auto DH = depth_shape[2], DW = depth_shape[3];

                float scale_factor = DW / float(W);
                cams.push_back(Camera(K= K.float()).scaled(scale_factor).to(device));
                ref_cams.push_back(Camera(K=ref_K.float(), Tcw=pose).scaled(scale_factor).to(device));

                depths.push_back(monodepth::utils::inv2depths(inv_depths[i]));

                ref_warped.push_back(view_synthesis(
                        ref_images[i], depths[i], ref_cams[i], cams[i]));
            }

            return ref_warped;
        };

        std::vector<torch::Tensor> MultiViewPhotometricLossImpl::calc_photometric_loss(std::vector<torch::Tensor> &t_est, std::vector<torch::Tensor> &images){
            
            std::vector<torch::Tensor> l1_loss, ssim_loss, photometric_loss;
            for(int i = 0; i < 4; i++){
                
                l1_loss.push_back(at::abs(t_est[i]-images[i]));
                ssim_loss.push_back(SSIM->forward(t_est[i], images[i]));
                photometric_loss.push_back( 0.85 * ssim_loss[i].mean(1, true) +
                                0.15 * l1_loss[i].mean(1, true));
            }
            for(int i = 0; i < 4; i++){
                auto mean= photometric_loss[i].mean();
                auto std = photometric_loss[i].std();
                photometric_loss[i] = torch::clamp(photometric_loss[i],0, (mean.item() + 0.5*std).item());
            }

            return photometric_loss;
        };

        torch::Tensor MultiViewPhotometricLossImpl::reduce_function(std::vector<torch::Tensor> &losses){

            return (std::get<0>(torch::cat(losses,1).min(1, true))).mean();
        };

        torch::Tensor MultiViewPhotometricLossImpl::reduce_photometric_loss(std::vector<std::vector<torch::Tensor>> &photometric_losses){

            
            torch::Tensor photometric_loss;
            
            for(int i = 0; i < photometric_losses.size(); i++){

                photometric_loss = torch::sum(reduce_function(photometric_losses[i])); 
            }

            return photometric_loss;       
        };
        
        std::map<std::string, torch::Tensor> MultiViewPhotometricLossImpl::forward(torch::Tensor image, std::vector<torch::Tensor> &context, std::vector<torch::Tensor> &inv_depths,
                            torch::Tensor K, torch::Tensor ref_K, std::vector<torch::Tensor> &poses){

            //       """
            //         Calculates training photometric loss.

            //         Parameters
            //         ----------
            //         image : torch.Tensor [B,3,H,W]
            //             Original image
            //         context : list of torch.Tensor [B,3,H,W]
            //             Context containing a list of reference images
            //         inv_depths : list of torch.Tensor [B,1,H,W]
            //             Predicted depth maps for the original image, in all scales
            //         K : torch.Tensor [B,3,3]
            //             Original camera intrinsics
            //         ref_K : torch.Tensor [B,3,3]
            //             Reference camera intrinsics
            //         poses : list of Pose
            //             Camera transformation between original and context
            //         return_logs : bool
            //             True if logs are saved for visualization
            //         progress : float
            //             Training percentage

            //         Returns
            //         -------
            //         losses_and_metrics : dict
            //             Output dictionary
            // """
            std::vector<std::vector<torch::Tensor>> photometric_losses(4);
            std::vector<torch::Tensor> images = monodepth::utils::match_scales(image, inv_depths, 4);

            std::vector<torch::Tensor> ref_warped, photometric_loss, ref_images, unwarped_image_loss;
            for(int i = 0; i < 4; i++){
                //Calculate warped images
                ref_warped = warp_ref_image(inv_depths, context[i], K, ref_K, poses[i]);
                //Calculate and store image loss
                photometric_loss = calc_photometric_loss(ref_warped, images);
               
                ref_images = monodepth::utils::match_scales(context[i], inv_depths, 4);
                //Calculate and store unwarped image loss
                unwarped_image_loss = calc_photometric_loss(ref_images, images);
                 for(int j = 0; j < 4; j++){
                    photometric_losses[j].push_back(photometric_loss[j]);
                    photometric_losses[j].push_back(unwarped_image_loss[j]);
                };
            }
            torch::Tensor loss;
            //Calculate reduced photometric loss
            loss = reduce_photometric_loss(photometric_losses);
            
            std::map<std::string, torch::Tensor> losses;
            losses["loss"] =loss.unsqueeze(0);

            return losses;
        }

    }
}