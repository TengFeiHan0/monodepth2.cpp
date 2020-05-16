#include "make_layers.h"
#include "defaults.h"
#include "estimator/posedecoder.h"

namespace monodepth{
    namespace modeling{

        PoseDecoderImpl::PoseDecoderImpl(){

            num_frames = monodepth::config::GetCFG<int64_t>({"MODEL", "DEPTH", "NUM_FRAMES"});
            out_channels = monodepth::config::GetCFG<int64_t>({"MODEL", "DEPTH","OUT_CHANNELS"});

            backbone = register_module("backbone", BuildBackbone());

            squeeze_0 = register_module("squeeze_0", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, 256, 1)));
            
            pose_0    = register_module("pose_0", torch::nn::Conv2d(torch::nn::Conv2dOptions(256,256, 3)
                            .stride(1)
                            .padding(1)
                            ));

            pose_1 = register_module("pose_1", torch::nn::Conv2d( torch::nn::Conv2dOptions(256, 256, 3)
                            .stride(1)
                            .padding(1)));
                    
            pose_2 = register_module("pose2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 6*num_frames,1)));          
        }
              
        std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>> PoseDecoderImpl::forward(std::vector<torch::Tensor> inputs){
            
            auto x = at::cat(at::TensorList(inputs),1);
            std::vector<torch::Tensor> features = backbone->forward(x);
            std::vector<torch::Tensor> cat_features;
            for(auto x : features){
                cat_features.push_back(torch::relu(squeeze_0->forward(x)));
            }
            auto out = at::cat(at::TensorList(cat_features),1);

            out = torch::relu(pose_0->forward(out));
            out = torch::relu(pose_1->forward(out));
            out =pose_2->forward(out);
            out = out.mean(3).mean(2);

            out = 0.01 * out.view({-1, num_frames, 1, 6});
            std::vector<torch::Tensor> axisangle;
            axisangle.push_back(out[0]);
            axisangle.push_back(out[1]);
            axisangle.push_back(out[2]);
            std::vector<torch::Tensor> translation;
            translation.push_back(out[3]);
            translation.push_back(out[4]);
            translation.push_back(out[5]);

            return std::make_tuple(axisangle, translation);
        }
    }
}