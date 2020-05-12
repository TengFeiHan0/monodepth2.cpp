#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>
#include <torch/types.h>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <defaults.h>
namespace monodepth{
    namespace data{

        struct Image{
            cv::Mat img_pre;
            cv::Mat img_next;
            int img_width;
            int img_height;
            std::vector<float> img_K;
        };

        class CityScapesDataset: public torch::data::datasets::Dataset<CityScapesDataset, torch::data::Example<cv::Mat, monodepth::data::Image>> {
            public:
                CityScapesDataset(std::string data_path);
                
                torch::data::Example<cv::Mat, monodepth::data::Image> get(size_t index) override;

                torch::optional<size_t> size() const override;

            private:
              std::string data_path_;
              std::vector<cv::String> all_img_path;

        };
    }
}