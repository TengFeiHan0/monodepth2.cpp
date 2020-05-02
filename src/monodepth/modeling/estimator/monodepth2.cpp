#include <torch/torch.h>
#include <iostream>

struct DepthDecoder : torch::nn::Module {
    DepthDecoder(){

        for(int i=4; i>=0; i--) {
            if(i==4){
                num_ch_in = num_ch_enc[num_ch_enc.size()-1];
            }
            else{
                num_ch_in = num_ch_dec[i+1];
            }
            num_ch_out = num_ch_dec[i];

            decoder_conv.push_back(
                register_module("upconv_0"+std::to_string(i), conv_block(use_relu, num_ch_in, num_ch_out, 3, 1, 1))
                );
            num_ch_in = num_ch_dec[i];
            if(use_skips && i>0){
                num_ch_in = num_ch_enc[i-1];
            }
            num_ch_out = num_ch_dec[i];
            decoder_conv.push_back(
                register_module("upconv_1"+std::to_string(i), conv_block(use_relu, num_ch_in, num_ch_out, 3, 1, 1))
                );
        }

        for(int i = 0; i <4; ++i){
            decoder_conv.push_back(
                register_module("disconv"+std::to_string(i), conv_block(use_relu, num_ch_dec[i+1], 1, 3, 1, 1))
            );
        }

        static torch::Tensor sig =  at::sigmoid();
    }

    torch::Tensor forward(std::vector<torch::Tensor> inputs){
        auto x = inputs[inputs.size()-1];

        for(int i=4; i>=0; i--){
            x = 

        }


    }



}