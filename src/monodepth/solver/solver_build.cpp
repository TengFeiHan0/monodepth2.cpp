#include "solver_build.h"
#include "defaults.h"


namespace monodepth{
namespace solver{

ConcatOptimizer MakeOptimizer(std::shared_ptr<monodepth::modeling::SelfDepthModel> &model){
  return ConcatOptimizer(model->named_parameters(), 
                        monodepth::config::GetCFG<double>({"SOLVER", "BASE_LR"}),
                        monodepth::config::GetCFG<double>({"SOLVER", "MOMENTUM"}),
                        monodepth::config::GetCFG<double>({"SOLVER", "WEIGHT_DECAY"}),
                        monodepth::config::GetCFG<double>({"SOLVER", "WEIGHT_DECAY_BIAS"}));
}

ConcatScheduler MakeLRScheduler(ConcatOptimizer& optimizer, int last_epoch){
  std::string method = monodepth::config::GetCFG<std::string>({"SOLVER", "WARMUP_METHOD"});
  return ConcatScheduler(optimizer, 
                        monodepth::config::GetCFG<std::vector<int64_t>>({"SOLVER", "STEPS"}),
                        monodepth::config::GetCFG<float>({"SOLVER", "GAMMA"}),
                        monodepth::config::GetCFG<float>({"SOLVER", "WARMUP_FACTOR"}),
                        monodepth::config::GetCFG<int64_t>({"SOLVER", "WARMUP_ITERS"}),
                        method,
                        last_epoch);
}

}
}