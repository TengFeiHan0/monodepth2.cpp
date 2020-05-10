#include <torch/torch.h>
#include "losses.h"
#include "defaults.h"

using namespace F = torch::nn::functional;
namespace monodepth{
    namespace modeling{
        
        torch::Tensor get_smooth_loss(torch::Tensor disp, torch::Tensor img){

            torch::Tensor grad_disp_x = torch::abs(disp[:, :, :, :-1] - disp[:, :, :, 1:]);
            torch::Tensor grad_disp_y = torch::abs(disp[:, :, :-1, :] - disp[:, :, 1:, :]);

            torch::Tensor grad_img_x = torch.mean(torch::abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True);
            torch::Tensor grad_img_y = torch.mean(torch::abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True);

            grad_disp_x *= at::exp(-grad_img_x);
            grad_disp_y *= at::exp(-grad_img_y);

            return grad_disp_x.mean() + grad_disp_y.mean();
        }



        std::map<std::string, torch::Tensor> compute_losses( std::vector<std::vector<torch::Tensor>> &inputs,
                std::vector<torch::Tensor> &disps, std::vector<std::vector<torch::Tensor>> &recontructed_img){

            std::map<std::string, torch::Tensor> losses;
            torch::Tensor total_loss =0;
            for(int i=0; i < 4; i++){
                torch::Tensor loss =0;
                std::vector<torch::Tensor> reprojection_losses;
                int source_scale =i;
                torch::Tensor disp = disps[i];
                torch::Tensor color = inputs[i][0];
                torch::Tensor target = inputs[source_scale][0];
                for(int j = 1; j <2; ++j){
                    torch::Tensor pred = recontructed_img[i][j];
                    reprojection_losses.push_back(compute_reprojection_loss(pred, target));
                }
                torch::Tensor reprojection_loss = at::cat(at::TensorList(reprojection_losses),1);

                std::vector<torch::Tensor> identity_reprojection_losses;
                for(int j=1; j<=2; ++j){
                    torch::Tensor pred = inputs[source_scale][j];
                    identity_reprojection_losses.push_back(compute_reprojection_loss(pred, target));
                }
                torch::Tensor identity_reprojection_loss = at::cat(at::TensorList(identity_reprojection_losses),1);
                
                identity_reprojection_loss +=torch::randn(identity_reprojection_loss.shape).cuda()*0.00001;

                torch::Tensor combined = at::cat((identity_reprojection_loss, reprojection_loss), 1);
                if(combined.shape[1]==1){
                    torch::Tensor to_optimize = combined;
                }else{
                    std::tuple optimize_and_idxs = at::min(combined, 1);
                    torch::Tensor to_optimize = std::get<0>(optimize_and_idxs);
                    torch::Tensor idxs = std::get<1>(optimize_and_idxs);
                }
                loss += to_optimize.mean();

                torch::Tensor mean_disp = disp.mean(2,True).mean(3, True);
                torch::Tensor norm_disp = disp / (mean_disp +1e-7);
                torch::Tensor smooth_loss = get_smooth_loss(norm_disp, color);
                int64_t smooth_weight =  monodepth::config::GetCFG<int64_t>({"MODEL","DEPTH", "SMOOTH_WEIGHT"});

                loss += smooth_weight * smooth_loss / (2**i);
                total_loss +=loss;
            }
            total_loss /=4;
            losses["loss"] = total_loss

            return losses;
        };
        
        std::vector<std::vector<torch::Tensor>>  generate_image_pred( 
                     std::vector<std::vector<torch::Tensor>> &inputs,std::vector<torch::Tensor> &disps, 
                     std::map<int, torch::Tensor> &cam_T_cam){

                std::vector<std::vector<torch::Tensor>> sample_img;
                std::vector<std::vector<torch::Tensor>> recontructed_img;
            
                for(int i= 0; i<4; i++){
                    torch::Tensor disp = disps[i];
                    int source_scale = i; 
                    int64_t min_depth =  monodepth::config::GetCFG<int64_t>({"MODEL","DEPTH", "MIN_DEPTH"});
                    int64_t max_depth = monodepth::config::GetCFG<int64_t>({"MODEL","DEPTH", "MAX_DEPTH"});

                    std::tuple tuple_depth = disp_to_depth(disp, min_depth, max_depth);
                    torch::Tensor depth = std::get<1>(tuple_depth);

                    for(int j=1; j<=2; j++){
                        torch::Tensor T = cam_T_cam[j];
                        int64_t min_depth = 
                        torch::Tensor cam_points = backproject_depth[source_scale]->forward(depth, inv_K[source_scale]);
                        torch::Tensor pix_coords = project_3d[source_scale]->forward(cam_points, K[source_scale], T);

                        sample_img[i][j]= pix_coords;
                        recontructed_img[i][j] = F::grid_sample(inputs[i][j], sample_img[i][j],
                            F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder)) 
                        );
                    }


                }
                return reconstructed_img;

        };

        

        torch::Tensor compute_reprojection_loss(torch::Tensor pred, torch::Tensor target){
            torch::Tensor abs_diff = torch::abs(target - pred);
            auto l1_loss = abs_diff.mean(1, True);

            torch::Tensor ssim_loss = monodepth::modeling::SSIM->forward(pred, target);

            ssim_loss = ssim_loss.mean(1, True);

            torch::Tensor reprojection_loss = 0.85*ssim_loss +0.15*l1_loss;

            return reprojection_loss;

        };
        
        SSIMImpl::SSIMImpl(){
            mu_x_pool = register_module("mu_x_pool", torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({3,3}).stride({1,1})
            ));
            mu_y_pool = register_module("mu_y_pool",torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({3,3}).stride({1,1})
            ));
            sig_x_pool = register_module("sig_x_pool",torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({3,3}).stride({1,1})
            ));
            sig_y_pool = register_module("sig_y_pool",torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({3,3}).stride({1,1})
            ));
            sig_xy_pool = register_module("sig_xy_pool",torch::nn::AvgPool2d(
                torch::nn::AvgPool2dOptions({3,3}).stride({1,1})
            ));

            refl = register_module("refl", torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({1,1,1,1})
            ));

            C1 = 0.01**2;
            C2 = 0.03**2;

        };

        torch::Tensor SSIMImpl->forward(torch::Tensor x, torch::Tensor y){
            x = refl->forward(x);
            y = refl->forward(y);

           auto mu_x = mu_x_pool->forward(x);
           auto mu_y = mu_y_pool->forward(y);

           auto sigma_x = sig_x_pool->forward(x**2) - mu_x**2;  
           auto sigma_y = sig_y_pool->forward(y**2) - mu_y**2;
           auto sigma_xy = sig_xy_pool->forward(x*y) - mu_x*mu_y;

           auto SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2);
           auto SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2);

           return torch::clamp((1 - SSIM_n/SSIM_d)/2, 0,1);

        }

            
    }
}