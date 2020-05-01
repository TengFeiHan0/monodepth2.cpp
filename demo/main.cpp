#include <iostream>
#include <chrono>
#include <time.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace std::chrono; 

int main(int argc,char**argv)
{
    //load a pytorch model
    torch::jit::script::Module decoder = torch::jit::load("/home/fei/monodepth2.cpp/demo/packnet_fp16.pt");
    decoder.to(at::kCUDA);//put this model on GPU
    cv::Mat src=cv::imread("/home/fei/monodepth2.cpp/demo/test_image.png");
    cv::Mat input_mat;
    int w=640;
    int h=192;
    
    cv::resize(src,input_mat,cv::Size(w,h));
    //[0, 255]
    input_mat.convertTo(input_mat,CV_32FC1,1/255.);
    
    //[0, 1]
    //transform a cv::Mat into a tensor
    torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1,input_mat.rows, input_mat.cols,3}, torch::kF32);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.to(at::kCUDA);
    //[0,1]
    std::vector<torch::IValue> batch;//inputs
    batch.push_back(tensor_image);
   
    auto start = std::chrono::high_resolution_clock::now();
    auto outputs= decoder.forward(batch);// inference
    auto tensor_output = outputs.toTensor();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start); 

    // It should be known that it takes longer time at first time
    std::cout << "inference taken : " << duration.count() << " ms" << endl; 
    
    
    tensor_output = tensor_output.permute({0,2,3,1});//change the dimentions of tensor
    // tensor_output=tensor_output.squeeze(0).detach().permute({1,2,0});
    // tensor_output=tensor_output.mul(255).clamp(0,255).to(torch::kU8);
    tensor_output=tensor_output.to(torch::kCPU);
    
    cv::Mat disp=cv::Mat(h,w,CV_16FC1,tensor_output.data_ptr());
   
    double minVal; double maxVal; 
    cv::minMaxLoc(disp, &minVal, &maxVal); 
    disp /= maxVal;
    
    cv::resize(disp,disp,cv::Size(src.cols,src.rows));
    disp/=255;

    disp.convertTo(disp,CV_8UC1);
    cv::cvtColor(disp,disp,cv::COLOR_GRAY2BGR);
  
    src.push_back(disp);
    //cv::resize(src,src,cv::Size(),0.5,0.5);
    
    cv::imwrite("src.jpg",disp);
    
    cout<<"Done"<<endl;
    return 0;
}
