#include "datasets/cityscapes_datasets.h"
#include <iostream>
#include <opencv2/opencv.hpp>
namespace monodepth{
    namespace data {

        

        CityScapesDataset::CityScapesDataset(std::string data_path):data_path_(data_path){
            std::string img_ext = monodepth::config::GetCFG<std::string>({"DATASETS", "IMG_EXT"});
            std::string  img_path = data_path_ + img_ext;
            cv::glob(img_path, all_img_path, true);
            sort(all_img_path.begin(), all_img_path.end());
            
        }

        torch::Tensor CityScapesDataset::img2Tensor(cv::Mat &input_mat){
            
            input_mat.convertTo(input_mat,CV_32FC1,1/255.);
            
            //[0, 1]
            //transform a cv::Mat into a tensor
            torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1,input_mat.rows, input_mat.cols,3}, torch::kF32);
            tensor_image = tensor_image.permute({0,3,1,2});
            return tensor_image;
        }

        torch::data::Example<cv::Mat, DICT>  CityScapesDataset::get(size_t index){
      
            std::map<std::string, torch::Tensor> target;
            cv::Mat img_pre, img_cur, img_next;
            // input.img_height = monodepth::config::GetCFG<int>({"DATASETS", "IMG_HEIGHT"});
            // input.img_width = monodepth::config::GetCFG<int>({"DATASETS", "IMG_WIDTH"});

            img_cur = cv::imread(all_img_path[index]);
            target["tensor_cur"]= img2Tensor(img_cur);
            //special case at the first and last image
            if(index == 0){
                 img_pre = cv::imread(all_img_path[0]);
                 img_next = cv::imread(all_img_path[index+1]);

                 target["tensor_pre"] = img2Tensor(img_pre);
                 target["tensor_next"] = img2Tensor(img_next);
                 torch::data::Example<cv::Mat, DICT> value{img_cur, target};
                 return value;
            }else if(index == monodepth::data::CityScapesDataset::size()){

                img_pre = cv::imread(all_img_path[index-1]);
                img_next = cv::imread(all_img_path[index]);
                
                target["tensor_pre"] = img2Tensor(img_pre);
                target["tensor_next"] = img2Tensor(img_next);
                torch::data::Example<cv::Mat, DICT> value{img_cur, target};
                return value;
            }
            
            img_pre = cv::imread(all_img_path[index-1]);
            img_next = cv::imread(all_img_path[index+1]);

            float array[] = {1.105, 0, 0.537, 0, 
                           0, 2.212, 0.501, 0, 
                           0, 0, 1, 0,
                           0, 0, 0, 1};
            auto options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA, 1);          
            torch::Tensor intrinsics = torch::from_blob(array, 16,  options);
            target["intrinsics"] = intrinsics.reshape({4,4});
            
            target["tensor_pre"] = img2Tensor(img_pre);
            target["tensor_next"] = img2Tensor(img_next);
            torch::data::Example<cv::Mat, DICT> value{img_cur, target};
            return value; 
        }

        torch::optional<size_t>  CityScapesDataset::size() const{
            return all_img_path.size();
        } 
    }
}