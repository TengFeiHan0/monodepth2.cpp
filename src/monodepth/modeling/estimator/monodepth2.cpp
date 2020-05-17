#include"estimator/monodepth2.h"
#include "datasets/cityscapes_datasets.h"
namespace monodepth{
    namespace modeling{
        SelfDepthModel::SelfDepthModel()
          :posedecoder(BuildPoseDecoderModule()),
           depthdecoder(BuildDepthDecoderModule()){}

        template<>
        std::map<std::string, torch::Tensor> SelfDepthModel::forward(torch::Tensor input, monodepth::data::DICT target){
            assert(is_training());
            std::map<std::string, torch::Tensor> losses;

            std::vector<torch::Tensor> disps;
            disps = depthdecoder->forward<std::vector<torch::Tensor>>(input);

            std::vector<torch::Tensor> pose_inputs;
            pose_inputs.push_back(input);
            pose_inputs.push_back(target["tensor_pre"]);
            pose_inputs.push_back(target["tensor_next"]);

            std::vector<torch::Tensor> poses;
            poses = posedecoder->forward<std::vector<torch::Tensor>>(pose_inputs);
            

            std::vector<torch::Tensor> context;
            context.push_back(target["tensor_pre"]);
            context.push_back(target["tensor_next"]);
            losses = mp_loss->forward(input, context , disps, target["intrinsics"], target["intrinsics"].inverse(), poses);

            return losses;
        }


        template<>
        torch::Tensor SelfDepthModel::forward(torch::Tensor input, monodepth::data::DICT target){
            assert(!is_training());
            std::vector<torch::Tensor> disps = depthdecoder->forward<std::vector<torch::Tensor>>(input);

            // std::vector<torch::Tensor> pose_inputs;
            // pose_inputs.push_back(input);
            // pose_inputs.push_back(target.tensor_pre);
            // pose_inputs.push_back(target.tensor_next)

            // std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>>  poses = posedecoder->forward(pose_inputs);
            return disps[0];
        }  
        
        
        SelfDepthModel BuildModel(){
            return SelfDepthModel();
        }

       

    }
}