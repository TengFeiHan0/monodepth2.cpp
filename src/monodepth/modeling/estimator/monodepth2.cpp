#include"estimator/monodepth2.h"

namespace monodepth{
    namespace modeling{
        MonodepthImpl::MonodepthImpl()
          :posedecoder(register_module("posedecoder", BuildPoseDecoderModule())),
           depthdecoder(register_module("depthdecoder", BuildDepthDecoderModule())){}

        
        

       

    }
}