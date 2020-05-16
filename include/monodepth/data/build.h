#pragma once
#include <torch/torch.h>
#include <torch/data/samplers/base.h>
#include "datasets/cityscapes_datasets.h"


namespace monodepth{
namespace data{

//supports only one dataset
//TODO Concat dataset
CityScapesDataset BuildDataset(std::vector<std::string> dataset_list);

template<typename T>
T MakeDataLoader(bool is_train=true , bool is_distributed=false, int start_iter=0);


}
}