#pragma once

#include <vector>
#include <tuple>
#include <string>
#include <map>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>


namespace monodepth{
    namespace data{

        class KITTIDataset : public torch::data::dataset<KITTIDataset>{
            public:
              explicit KITTIDataset(std::string data_path);

              torch::data::Example<std::map<std::string, cv::Mat>> get(size_t index) override;

              torch::optional<size_t> size() const override;

            private:
              std::string data_path_;

        }; 

    }
}