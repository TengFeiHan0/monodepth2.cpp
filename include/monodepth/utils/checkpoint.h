#pragma once
#include <torch/torch.h>
#include "estimator/monodepth2.h"
#include <solver_build.h>
#include <memory>

namespace monodepth{
namespace utils{

class Checkpoint{

public:
  Checkpoint(monodepth::modeling::SelfDepthModel & model, 
             monodepth::solver::ConcatOptimizer& optimizer, 
             monodepth::solver::ConcatScheduler& scheduler, 
             std::string save_dir);

  int load(std::string weight_path);
  void save(std::string name, int iteration);
  bool has_checkpoint();
  std::string get_checkpoint_file();
  int load_from_checkpoint();
  void write_checkpoint_file(std::string name);

  static void load(monodepth::modeling::SelfDepthModel& model, std::string save_dir, std::string weight_dir);

private:
  monodepth::modeling::SelfDepthModel& model_;
  monodepth::solver::ConcatOptimizer& optimizer_;
  monodepth::solver::ConcatScheduler& scheduler_;
  std::string save_dir_;
};

}
}