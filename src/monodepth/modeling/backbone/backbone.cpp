#include "backbone/backbone.h"
#include "backbone/resnet.h"
#include "backbone/vovnet.h"


#include <make_layers.h>
#include <registry.h>
#include <defaults.h>


namespace monodepth{
namespace modeling{

BackboneImpl::BackboneImpl(torch::nn::Sequential backbone, int64_t out_channels) : backbone_(register_module("backbone", backbone)), out_channels_(out_channels){};

std::vector<torch::Tensor> BackboneImpl::forward(torch::Tensor x){
  return backbone_->forward<std::vector<torch::Tensor>>(x);
}

int64_t BackboneImpl::get_out_channels(){
  return out_channels_;
}

std::shared_ptr<BackboneImpl> BackboneImpl::clone(torch::optional<torch::Device> device) const{
  torch::NoGradGuard no_grad;
  torch::nn::Sequential backbone_copy;
  auto backbone_clone = std::dynamic_pointer_cast<torch::nn::SequentialImpl>(backbone_->clone());
  static_cast<torch::nn::SequentialImpl&>(*backbone_copy) = std::move(*backbone_clone);
  std::shared_ptr<BackboneImpl> copy = std::make_shared<BackboneImpl>(backbone_copy, out_channels_);
  if(device.has_value())
    copy->to(device.value());
  return copy;
}

Backbone BuildResnetBackbone(){
  torch::nn::Sequential model;
  auto body = ResNet();
  model->push_back(body);
  auto backbone = Backbone(model, monodepth::config::GetCFG<int64_t>({"MODEL", "RESNETS", "BACKBONE_OUT_CHANNELS"}));
  return backbone;
}

Backbone BuildBackbone(){
  std::string name = monodepth::config::GetCFG<std::string>({"MODEL", "BACKBONE", "CONV_BODY"});
  monodepth::registry::backbone build_function = monodepth::registry::BACKBONES(name);
  Backbone model = build_function();
  return model;
}

}
}