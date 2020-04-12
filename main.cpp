#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
int main(int argc,char**argv)
{
    torch::jit::script::Module decoder = torch::jit::load("/opt/depth/model.pt");
    decoder.to(at::kCUDA);
    cv::Mat src=cv::imread("/opt/depth/test_image.png");
    cv::Mat input_mat;
    int w=640;
    int h=192;
    
    cv::resize(src,input_mat,cv::Size(w,h));
    input_mat.convertTo(input_mat,CV_32FC3,1./255.);
    torch::Tensor tensor_image = torch::from_blob(input_mat.data, {1,input_mat.rows, input_mat.cols,3}, torch::kF32);
    tensor_image = tensor_image.permute({0,3,1,2});
    tensor_image = tensor_image.to(at::kCUDA);

    std::vector<torch::IValue> batch;
    batch.push_back(tensor_image);
    //batch.clear();
    
    auto outputs= decoder.forward(batch);
    auto outputs_tu = outputs.toTuple();
    auto tensor_output = outputs_tu->elements()[3].toTensor().to(at::kCPU);
   
    tensor_output = tensor_output.permute({0,3,2,1});

    cv::Mat disp=cv::Mat(h,w,CV_32FC1,tensor_output.data_ptr());
    cv::resize(disp,disp,cv::Size(src.cols,src.rows));
    disp*=512;

    disp.convertTo(disp,CV_8UC1);
    cv::cvtColor(disp,disp,cv::COLOR_GRAY2BGR);
    src.push_back(disp);
    cv::resize(src,src,cv::Size(),0.5,0.5);
    // vector<cv::Mat> channels={disp,disp,disp};
    // cv::merge(channels,disp);
    //cv::resize(src,src,cv::Size(),0.5,0.5);
    // cv::imshow("result",disp);
    cv::imwrite("src.jpg",src);
    

    cout<<"Done"<<endl;
    return 0;
}