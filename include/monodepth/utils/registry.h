#pragma once
#include <torch/torch.h>

#include <string>
#include "backbone/backbone.h"

namespace monodepth{
namespace registry{

using backbone = monodepth::modeling::Backbone (*) (void);
backbone BACKBONES(std::string conv_body);

//hard code
}
}

