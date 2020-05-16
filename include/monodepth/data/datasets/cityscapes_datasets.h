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

        struct ImageData{

            int64_t img_idx;

            cv::Mat img_pre;
            cv::Mat img_cur;
            cv::Mat img_next;

            torch::Tensor tensor_pre;
            torch::Tensor tensor_cur;
            torch::Tensor tensor_next;

            int img_width;
            int img_height;
            std::vector<std::vector<float>> img_K;
        };

        class CityScapesDataset: public torch::data::datasets::Dataset<CityScapesDataset, torch::data::Example<cv::Mat, monodepth::data::ImageData>> {
            public:
                CityScapesDataset(std::string data_path);
                
                torch::data::Example<cv::Mat, monodepth::data::ImageData> get(size_t index) override;

                torch::optional<size_t> size() const override;

                torch::Tensor img2Tensor (cv::Mat &input_mat);

            private:
              std::string data_path_;
              std::vector<cv::String> all_img_path;

        };
    }
}