#pragma once

#include <vector>
#include <tuple>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


namespace monodepth{
    namespace data{

        class KITTIDataset : public torch::data::datasets<KITTIDataset>{
            public:
              explicit KITTIDataset(std::string data_path) : data_path_(data_path){};

              torch::data::Example<> get(size_t index) override{};

              torch::optional<size_t> size() const override {};

            private:
              std::string data_path_;

        }; 

    }
}