
"""
this script is used for converting your pytorch model into torchscript that could be used under CPP environment. 
The following code is extended from  ``` scripts/evaluate_depth.py ``` of  [packnet-sfm](https://github.com/TRI-ML/packnet-sfm),
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


