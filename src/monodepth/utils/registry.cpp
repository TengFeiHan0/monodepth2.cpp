#include "registry.h"
#include <map>
#include <cassert>

#include <backbone/backbone.h>


namespace monodepth{
namespace registry{

backbone BACKBONES(std::string conv_body){
  std::map<std::string, backbone> backbone_builder_map{
    {"R-50-C4", monodepth::modeling::BuildResnetBackbone},
    {"R-50-C5", monodepth::modeling::BuildResnetBackbone},
    {"R-101-C4", monodepth::modeling::BuildResnetBackbone},
    {"R-101-C5", monodepth::modeling::BuildResnetBackbone},
  };
  assert(backbone_builder_map.count(conv_body));
  return backbone_builder_map.find(conv_body)->second;
}

}
}

