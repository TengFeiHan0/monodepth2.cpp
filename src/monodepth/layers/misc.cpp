#include <torch/torch.h>
#include <iostream>
#include <cassert>

namespace monodepth {
  namespace layers {

    std::tuple<torch::Tensor, torch::Tensor> disp_to_depth(torch::Tensor disp, int64_t min_depth, int64_t max_depth){
        int64_t min_disp = 1 / max_depth;
        int64_t max_disp = 1 / min_depth;
        torch::Tensor scaled_disp = min_disp + (max_disp - min_disp)*disp;
        torch::Tensor depth = 1 / scaled_disp;

        return std::make_tuple(scaled_disp, depth);

    }

    at::Tensor get_translation_matrix(at::Tensor translation){

      at::Tensor T = at::zeros({translation.size(0), 4, 4});
      T.to(translation.device());
      at::Tensor t = translation.contiguous().view({-1, 3, 1});

      T[0][0] = 1;
      T[1][1] = 1;
      T[2][2] = 1;
      T[3][3] = 1;
      T[3][0,2]=t;

      return T;
    }

    torch::Tensor rot_from_axisangle(torch::Tensor axis){
      torch::Tensor angle = at::norm(axis, 2,2, true);
      axis = axis / (angle +1e-7);
      torch::Tensor ca = at::cos(angle);
      torch::Tensor sa = at::sin(angle);
      torch::Tensor C = 1 - ca;

      torch::Tensor x = axis[0].unsqueeze(1);
      torch::Tensor y = axis[1].unsqueeze(1);
      torch::Tensor z = axis[2].unsqueeze(1);

      torch::Tensor xs = x * sa;
      torch::Tensor ys = y * sa;
      torch::Tensor zs = z * sa;
      torch::Tensor xC = x * C;
      torch::Tensor yC = y * C;
      torch::Tensor zC = z * C;
      torch::Tensor xyC = x * yC;
      torch::Tensor yzC = y * zC;
      torch::Tensor zxC = z * xC;

      torch::Tensor rot = at::zeros({axis.size(0), 4, 4});
      rot.to(axis.device());

      rot[0][0] = at::squeeze(x * xC + ca);
      rot[0][1] = at::squeeze(xyC - zs);
      rot[0][2] = at::squeeze(zxC + ys);
      rot[1][0] = at::squeeze(xyC + zs);
      rot[1][1] = at::squeeze(y * yC + ca);
      rot[1][2] = at::squeeze(yzC - xs);
      rot[2][0] = at::squeeze(zxC - ys);
      rot[2][1] = at::squeeze(yzC + xs);
      rot[2][2] = at::squeeze(z * zC + ca);
      rot[3][3] = 1;

      return rot;
    }
    
    torch::Tensor transformation_from_parameters(torch::Tensor axisangle, torch::Tensor translation, bool invert){
      torch::Tensor R = rot_from_axisangle(axisangle);
      torch::Tensor t = translation.clone();
      
      if(invert){
        R = R.transpose(1,2);
        t *= -1;
      }

      torch::Tensor T = get_translation_matrix(t);
      torch::Tensor M_tensor;
      if(invert){
        M_tensor = at::matmul(R, T);
      }
      else{
        M_tensor = at::matmul(T, R);
      }

      return M_tensor;
    }
        
    // BackprojectDepthImpl::BackprojectDepthImpl(int batch_size, int height, int width):batch_size_(batch_size), height_(height), width_(width){
    //   std::vector<torch::Tensor> meshGrid = at::meshgrid(width_, height_);

    //   torch::Tensor id_coords = at::stack(at::TensorList(meshGrid),dim=0);

    // }

    // Project3DImpl::Project3DImpl(int batch_size, int height, int width, int64_t eps):
    //     batch_size_(batch_size), height_(height), width_(width),eps_(eps){}
      
    // torch::Tensor Project3DImpl::forward(torch::Tensor points, torch::Tensor K, torch::Tensor T){
    //   torch::Tensor P = at::matmul(K, T);
    //   torch::Tensor cam_points = at::matmul(P, points);

    //   torch::Tensor pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps_);
    //   pix_coords = pix_coords.view(batch_size_, 2, height_, width_);
    //   pix_coords = pix_coords.permute(0, 2, 3, 1);
    //   pix_coords[0] /= width_ - 1;
    //   pix_coords[1] /= height_ - 1;
    //   pix_coords = (pix_coords - 0.5) * 2;

    //   return pix_coords;
    // }

  }
}