#pragma once
#include "optimizer.h"
#include "lr_scheduler.h"
#include "estimator/monodepth2.h"


namespace monodepth{
namespace solver{

ConcatOptimizer MakeOptimizer(monodepth::modeling::Monodepth& model);

ConcatScheduler MakeLRScheduler(ConcatOptimizer& optimizer, int last_epoch);

}
}