# Copyright 2020 Toyota Research Institute.  All rights reserved.

"""
Evaluation script for depth estimation from monocular images
Takes a pretrained pytorch model and calculates error metrics
==========================================================
Usage:
make docker-evaluate-depth \
MODEL=path/to/model.pth.tar (pretrained depth network) \
INPUT_PATH=/path/to/split (for now only a KITTI split file or an image folder are supported) \
DEPTH_TYPE=depth type used for evaluation [velodyne] \
CROP=crop used in depth evaluation [garg] \
SAVE_OUTPUT=/path/to/output (path where the output images and predicted depths will be saved)
==========================================================
"""
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_sfm_utils import load_dispnet_with_args



parser = argparse.ArgumentParser(description='TRI Monocular Depth Inference Script')
parser.add_argument("--pretrained_model", type=str, help="pretrained model path")

args = parser.parse_args()

# Loads disp_net model and checkpoint_args
disp_net, checkpoint_args = load_dispnet_with_args(args)

# Iterate and compute depth
disp_net.cuda()
disp_net.eval()
example = torch.rand(1, 3,640, 192).cuda()
traced_script_module = torch.jit.trace(disp_net, example)
# 保存模型
traced_script_module.save("packnet.pt")


