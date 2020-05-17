#pragma once
#include <torch/torch.h>
#include <torch/data/samplers/base.h>
#include "datasets/cityscapes_datasets.h"
#include "samplers/build.h"
#include "samplers/samplers.h"

#include "transforms/build.h"
#include "transforms/transforms.h"


namespace monodepth{
namespace engine{

void do_train();

}
}