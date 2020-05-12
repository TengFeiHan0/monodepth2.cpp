#include "make_layers.h"
#include "defaults.h"
#include "estimator/depthdecoder.h"

namespace monodepth{
    namespace modeling{
        DepthDecoderImpl::DepthDecoderImpl():backbone(register_module("backbone", BuildBackbone())){

            num_ch_enc = monodepth::config::GetCFG<std::vector<int64_t>>({"MODEL","DEPTH", "NUM_CH_ENC"});
            num_ch_dec = monodepth::config::GetCFG<std::vector<int64_t>>({"MODEL","DEPTH", "NUM_CH_DEC"});
            use_skips = monodepth::config::GetCFG<bool>({"MODEL", "DEPTH", "USE_SKIPS"});

            for(int i=4; i>=0; i--) {
                if(i==4){
                    num_ch_in = num_ch_enc[num_ch_enc.size()-1];
                }
                else{
                    num_ch_in = num_ch_dec[i+1];
                }
                num_ch_out = num_ch_dec[i];

                up_Conv_0.push_back(
                    register_module("upconv_0"+std::to_string(i), 
                    monodepth::layers::ConvWithKaimingUniform(true, num_ch_in, num_ch_out))
                    );
                num_ch_in = num_ch_dec[i];
                if(use_skips && i>0){
                    num_ch_in = num_ch_enc[i-1];
                }
                num_ch_out = num_ch_dec[i];
                up_Conv_1.push_back(
                    register_module("upconv_1"+std::to_string(i), 
                    monodepth::layers::ConvWithKaimingUniform(true, num_ch_in, num_ch_out))
                    );
            }

            for(int i = 1; i <4; ++i){
                disp_Conv.push_back(
                    register_module("disconv"+std::to_string(i), 
                    monodepth::layers::MakeConv3x3(num_ch_dec[i], 1, true, false))
                );
            }


        };

        template <>
        std::vector<torch::Tensor> DepthDecoderImpl::forward(torch::Tensor x){
            assert(is_training());
            std::vector<torch::Tensor> disps;
            
            std::vector<torch::Tensor> features = backbone->forward(x);

            x = features[features.size()-1];

            for(int i=4; i>=0; i--){
                x = up_Conv_0[i]->forward(x);
                x =  torch::upsample_nearest2d(x, {x.size(2)*2, x.size(3)*2});
                if(use_skips && i>0){
                    x +=features[i-1];
                }
                x = at::cat(x,1);
                x = up_Conv_1[i]->forward(x);
                if(i>0){
                    disps.push_back(
                        at::sigmoid(disp_Conv[i]->forward(x))
                    );
                }
                
            }

            return disps;
        }

        template <>
        torch::Tensor DepthDecoderImpl::forward(torch::Tensor x){
            assert(!is_training());
            std::vector<torch::Tensor> disps;
            
            std::vector<torch::Tensor> features = backbone->forward(x);

             x = features[features.size()-1];

            for(int i=4; i>=0; i--){
                x = up_Conv_0[i]->forward(x);
                x =  torch::upsample_nearest2d(x, {x.size(2)*2, x.size(3)*2});
                if(use_skips && i>0){
                    x +=features[i-1];
                }
                x = at::cat(x,1);
                x = up_Conv_1[i]->forward(x);
                if(i>0){
                    disps.push_back(
                        at::sigmoid(disp_Conv[i]->forward(x))
                    );
                }
                
            }

            return disps[0];
        }

        DepthDecoder BuildDepthDecoderModule(){
            return DepthDecoder();
        }
    }
}


