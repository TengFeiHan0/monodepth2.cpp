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

        // struct DICT{
             
        //     torch::Tensor tensor_pre;
        //     torch::Tensor tensor_cur;
        //     torch::Tensor tensor_next;
        //     //augmented image
        //     torch::Tensor aug_pre;
        //     torch::Tensor aug_next;

        //     torch::Tensor img_K;

        //     // torch::Tensor img_invK;
        // };
        typedef std::map<std::string, torch::Tensor> DICT;
        class CityScapesDataset: public torch::data::datasets::Dataset<CityScapesDataset, torch::data::Example<cv::Mat, DICT>> {
            public:
                CityScapesDataset(std::string data_path);
                
                torch::data::Example<cv::Mat, DICT> get(size_t index) override;

                torch::optional<size_t> size() const override;

                torch::Tensor img2Tensor (cv::Mat &input_mat);

            private:
              std::string data_path_;
              std::vector<cv::String> all_img_path;

        };
    }
}