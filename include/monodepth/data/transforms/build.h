#pragma once
#include "transforms/transforms.h"


namespace monodepth{
namespace data{

Compose BuildTransforms(bool is_train=true);

}
}