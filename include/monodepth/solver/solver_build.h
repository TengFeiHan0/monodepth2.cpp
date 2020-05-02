#pragma once
#include "optimizer.h"
#include "lr_scheduler.h"
#include "detector/generalized_rcnn.h"


namespace monodepth{
namespace solver{

ConcatOptimizer MakeOptimizer(monodepth::modeling::GeneralizedRCNN& model);

ConcatScheduler MakeLRScheduler(ConcatOptimizer& optimizer, int last_epoch);

}
}