#include <make_layers.h>
#include <conv2d.h>

namespace F = torch::nn::functional;

namespace monodepth {
namespace layers {

torch::nn::Sequential ConvWithKaimingUniform(/*NO GN use_gn=false, */
                                             bool use_elu,
                                             int64_t in_channels,
                                             int64_t out_channels
                                            ){
  torch::nn::Sequential module;
  module->push_back(torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({1,1,1,1})));
  module->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                        .bias(true)));              
  
  module->push_back(torch::nn::Functional(torch::relu)); 
  for (auto &param : module->named_parameters()) {
    if (param.key().find("weight") != std::string::npos) {
      torch::nn::init::kaiming_uniform_(param.value(), 1);
    }
    else if (param.key().find("bias") != std::string::npos) {
      torch::nn::init::zeros_(param.value());
    }
  }
  return module;
}



torch::nn::Sequential MakeConv3x3(int64_t in_channels,
                                  int64_t out_channels,
                                  bool use_refl,
                                  bool kaiming_init) {
  torch::nn::Sequential module;

  if(use_refl){

    module->push_back(torch::nn::ReflectionPad2d(
                torch::nn::ReflectionPad2dOptions({1,1,1,1})));
  }
  else{
    module->push_back(torch::nn::ZeroPad2d(
                torch::nn::ZeroPad2dOptions({1,1,1,1})
    ));
  }
  
  module->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                                        .bias(true)));
  if (kaiming_init) {
    for(auto &param : module->named_parameters()){
      if(param.key().find("weight") != std::string::npos) {
        torch::nn::init::kaiming_uniform_(param.value(), 1);
      }
      else if(param.key().find("bias") != std::string::npos){
        torch::nn::init::zeros_(param.value());
      }
    }
  }
  else {
    for (auto &param : module->named_parameters()) {
      if (param.key().find("weight") != std::string::npos) {
        torch::nn::init::normal_(param.value(), 0.01);
      }
      else if (param.key().find("bias") != std::string::npos) {
        torch::nn::init::zeros_(param.value());
      }
    } 
  }

  return module;
}

} // namespace layers
} // namespace monodepth
