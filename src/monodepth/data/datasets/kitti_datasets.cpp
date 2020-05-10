#include "kitti_datasets.h"

namespace monodepth{
    namespace data{
        KITTIDataset::KITTIDataset(std::string data_path):data_path_(data_path){
            

        }

        torch::data::Example<std::map<std::string, torch::Tensor>> KITTIDataset::get(size_t index){
            

        }
    }

}