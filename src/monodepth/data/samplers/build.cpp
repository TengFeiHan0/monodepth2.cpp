#include "samplers/build.h"
#include <memory>
#include "bisect.h"
#include "samplers/samplers.h"

#include <defaults.h>


namespace monodepth {
namespace data {

std::shared_ptr<torch::data::samplers::Sampler<>> make_data_sampler(int dataset_size, bool shuffle, bool is_distributed) {
  if(is_distributed){
    return std::shared_ptr<torch::data::samplers::Sampler<>>(new torch::data::samplers::DistributedRandomSampler(dataset_size));
  }
  if (shuffle)
    return std::shared_ptr<torch::data::samplers::Sampler<>> (new torch::data::samplers::RandomSampler(dataset_size));
  else
    return std::shared_ptr<torch::data::samplers::Sampler<>> (new torch::data::samplers::SequentialSampler(dataset_size));
}

std::vector<int> _quantize(std::vector<float> x, std::vector<float> bins) {
  std::vector<int> quantized;
  for (auto& i : x)
    quantized.push_back(static_cast<int>(monodepth::utils::bisect_right(bins, i)));
  return quantized;
}

std::shared_ptr<torch::data::samplers::Sampler<>> make_batch_data_sampler(CityScapesDataset dataset, 
                                                                          bool is_train,
                                                                          bool is_distributed,
                                                                          int start_iter) {
  std::shared_ptr<torch::data::samplers::Sampler<>> batch_sampler;
  int64_t images_per_batch;
  bool shuffle = true;
  int num_iters = -1;
  if (is_train) {
    images_per_batch = monodepth::config::GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
    shuffle = true;
    num_iters = monodepth::config::GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
  }
  else {
    images_per_batch = monodepth::config::GetCFG<int64_t>({"TEST", "IMS_PER_BATCH"});
    shuffle = false;
    //no distributed
    start_iter = 0;
  }
  bool aspect_grouping = monodepth::config::GetCFG<bool>({"DATALOADER", "ASPECT_RATIO_GROUPING"});

  std::shared_ptr<torch::data::samplers::Sampler<>> sampler = make_data_sampler(dataset.size().value(), shuffle, is_distributed);
  if (!aspect_grouping) {
    batch_sampler = sampler;
  }
  if (num_iters != -1) {
    batch_sampler = std::make_shared<IterationBasedBatchSampler>(batch_sampler, num_iters, start_iter);
  }

  return batch_sampler;
}
                                                                        

} // namespace data
} // namespace monodepth
