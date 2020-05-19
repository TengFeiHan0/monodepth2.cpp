#include "build.h"
#include "paths_catalog.h"
#include "tovec.h"

#include <cassert>
#include <type_traits>
#include <iostream>
#include "collate_batch.h"
#include "defaults.h"
#include "samplers/samplers.h"
#include "samplers/build.h"

#include "transforms/build.h"
#include "transforms/transforms.h"

#include <torch/data/dataloader/stateful.h>
#include <torch/data/dataloader/stateless.h>

#include <torch/csrc/utils/memory.h>
#include <torch/csrc/utils/variadic.h>



namespace monodepth{
namespace data{

CityScapesDataset BuildCityDataset(std::vector<std::string> dataset_list){
  assert(dataset_list.size() == 1);
  monodepth::config::DatasetCatalog dataset_catalog = monodepth::config::DatasetCatalog();
  std::string dataset_name, img_dir, ann_file;
  std::tie(dataset_name, img_dir, ann_file) = dataset_catalog[dataset_list[0]];
  if(dataset_name.compare("CityScapesDataset")==0)
    return monodepth::data::CityScapesDataset(img_dir);
  else
    assert(false);   
}


// template<>
// std::unique_ptr<torch::data::StatelessDataLoader<CityScapesDataset, torch::data::samplers::RandomSampler>> MakeDataLoader(bool is_train , int start_iter){

//   std::vector<std::string> dataset_list;
//   if(is_train){
//     auto dataset = monodepth::config::GetCFG<std::string>({"DATASETS", "TRAIN"});
//     dataset_list.push_back(dataset);
//     }
//   else{
//     auto dataset = monodepth::config::GetCFG<std::string>({"DATASETS", "TEST"});
//     //dataset_list = monodepth::config::tovec(dataset.get());
//     dataset_list.push_back(dataset);
//   }
    

//   Compose transforms = BuildTransforms(is_train);
//   BatchCollator collate = BatchCollator(monodepth::config::GetCFG<int>({"DATALOADER", "SIZE_DIVISIBILITY"}));
  
//   int64_t images_per_batch;
//   if(is_train){
//     images_per_batch = monodepth::config::GetCFG<int64_t>({"SOLVER", "IMS_PER_BATCH"});
//   }
//   else{
//     images_per_batch = monodepth::config::GetCFG<int64_t>({"TEST", "IMS_PER_BATCH"});
//   }
//   CityScapesDataset city = BuildDataset(dataset_list);
//   auto dataset = city;//.map(transforms); //.map(collate)*?;
//   std::shared_ptr<torch::data::samplers::Sampler<>> sampler = make_batch_data_sampler(city, is_train, start_iter);


//   torch::data::DataLoaderOptions options(images_per_batch);
//   options.workers(monodepth::config::GetCFG<int64_t>({"DATALOADER", "NUM_WORKERS"}));
//   int num_iter = monodepth::config::GetCFG<int64_t>({"SOLVER", "MAX_ITER"});
//   return torch::data::make_data_loader(std::move(dataset), *dynamic_cast<torch::data::samplers::RandomSampler*>(sampler.get()), options);

// }

}
}